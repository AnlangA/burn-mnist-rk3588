use crate::data::load_png_as_mnist_28x28;
use burn::nn::{
    Linear, LinearConfig, PaddingConfig2d, Relu,
    conv::{Conv2d, Conv2dConfig},
    pool::{MaxPool2d, MaxPool2dConfig},
};
use burn::prelude::*;
use burn::store::{BurnpackStore, ModuleSnapshot, SafetensorsStore};
pub use burn::tensor::Device;
use burn::tensor::{TensorData, activation::softmax};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendKind {
    Cpu,
    Gpu,
    GpuVulkan,
    GpuGl,
}

impl BackendKind {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" | "flex" => Ok(Self::Cpu),
            "gpu" | "wgpu" | "auto-gpu" => Ok(Self::Gpu),
            "vulkan" | "vk" | "gpu-vulkan" => Ok(Self::GpuVulkan),
            "gl" | "gles" | "opengl" | "gpu-gl" => Ok(Self::GpuGl),
            other => Err(format!(
                "unknown backend '{other}', expected cpu|gpu|vulkan|gl"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::GpuVulkan => "vulkan",
            Self::GpuGl => "gl",
        }
    }
}

pub fn resolve_device(kind: BackendKind) -> Result<(Device, String), String> {
    match kind {
        BackendKind::Cpu => {
            #[cfg(feature = "cpu")]
            {
                let dev = Device::flex();
                Ok((dev, "Burn Flex CPU (SIMD + Rayon)".to_string()))
            }
            #[cfg(not(feature = "cpu"))]
            {
                Err("CPU backend feature 'cpu' is not enabled at compile time".into())
            }
        }
        BackendKind::Gpu => {
            #[cfg(feature = "gpu")]
            {
                use burn::tensor::wgpu::WgpuBackend;
                if let Ok(dev) = Device::wgpu_options()
                    .graphics_api(WgpuBackend::Vulkan)
                    .init()
                {
                    return Ok((dev, "Burn WGPU (Vulkan)".to_string()));
                }
                if let Ok(dev) = Device::wgpu_options().graphics_api(WgpuBackend::Gl).init() {
                    return Ok((dev, "Burn WGPU (OpenGL ES)".to_string()));
                }
                let dev = Device::wgpu(Default::default());
                let desc = format!("Burn WGPU ({dev:?})");
                Ok((dev, desc))
            }
            #[cfg(not(feature = "gpu"))]
            {
                Err("GPU backend feature 'gpu' is not enabled at compile time".into())
            }
        }
        BackendKind::GpuVulkan => {
            #[cfg(feature = "gpu")]
            {
                use burn::tensor::wgpu::WgpuBackend;
                let dev = Device::wgpu_options()
                    .graphics_api(WgpuBackend::Vulkan)
                    .init()
                    .map_err(|e| format!("failed to initialize WGPU Vulkan backend: {e:?}"))?;
                Ok((dev, "Burn WGPU (Vulkan)".to_string()))
            }
            #[cfg(not(feature = "gpu"))]
            {
                Err("GPU backend feature 'gpu' is not enabled at compile time".into())
            }
        }
        BackendKind::GpuGl => {
            #[cfg(feature = "gpu")]
            {
                use burn::tensor::wgpu::WgpuBackend;
                let dev = Device::wgpu_options()
                    .graphics_api(WgpuBackend::Gl)
                    .init()
                    .map_err(|e| format!("failed to initialize WGPU OpenGL ES backend: {e:?}"))?;
                Ok((dev, "Burn WGPU (OpenGL ES)".to_string()))
            }
            #[cfg(not(feature = "gpu"))]
            {
                Err("GPU backend feature 'gpu' is not enabled at compile time".into())
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prediction {
    pub digit: usize,
    pub confidence: f32,
    pub probabilities: [f32; 10],
}

/// Convolutional Neural Network for 28x28 MNIST digit classification.
///
/// Architecture:
/// - `conv1`: 1 -> 16 channels, 3x3 Valid (`28x28 -> 26x26`), ReLU, MaxPool 2x2 (`26x26 -> 13x13`)
/// - `conv2`: 16 -> 32 channels, 3x3 Valid (`13x13 -> 11x11`), ReLU, MaxPool 2x2 (`11x11 -> 5x5`)
/// - `fc1`: `32 * 5 * 5 = 800 -> 128`, ReLU
/// - `fc2`: `128 -> 10` logits
#[derive(Module, Debug)]
pub struct MnistCnn {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
    pub pool1: MaxPool2d,
    pub pool2: MaxPool2d,
    pub fc1: Linear,
    pub fc2: Linear,
    pub act: Relu,
}

impl MnistCnn {
    pub fn new(device: &Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([1, 16], [3, 3])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            conv2: Conv2dConfig::new([16, 32], [3, 3])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            pool1: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            pool2: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            fc1: LinearConfig::new(32 * 5 * 5, 128).init(device),
            fc2: LinearConfig::new(128, 10).init(device),
            act: Relu::new(),
        }
    }

    /// Forward pass returning raw logits `[batch_size, 10]`.
    pub fn forward_logits(&self, input: Tensor<4>) -> Tensor<2> {
        let [batch_size, _, _, _] = input.dims();
        let x = self.conv1.forward(input);
        let x = self.act.forward(x);
        let x = self.pool1.forward(x);

        let x = self.conv2.forward(x);
        let x = self.act.forward(x);
        let x = self.pool2.forward(x);

        let x = x.reshape([batch_size, 32 * 5 * 5]);
        let x = self.fc1.forward(x);
        let x = self.act.forward(x);
        self.fc2.forward(x)
    }

    /// Forward pass returning normalized softmax probabilities `[batch_size, 10]`.
    pub fn forward_probabilities(&self, input: Tensor<4>) -> Tensor<2> {
        let logits = self.forward_logits(input);
        softmax(logits, 1)
    }

    pub fn save_burnpack(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create dir {}: {e}", parent.display()))?;
        }
        let mut store = BurnpackStore::from_file(path)
            .auto_extension(false)
            .overwrite(true);
        self.save_into(&mut store)
            .map_err(|e| format!("failed to save burnpack to {}: {e}", path.display()))
    }

    pub fn save_safetensors(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create dir {}: {e}", parent.display()))?;
        }
        let mut store = SafetensorsStore::from_file(path).overwrite(true);
        self.save_into(&mut store)
            .map_err(|e| format!("failed to save safetensors to {}: {e}", path.display()))
    }

    pub fn load_burnpack_file(path: &Path, device: &Device) -> Result<Self, String> {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(path).auto_extension(false);
        let result = model
            .load_from(&mut store)
            .map_err(|e| format!("failed to load burnpack from {}: {e}", path.display()))?;
        if !result.missing.is_empty() || !result.errors.is_empty() {
            return Err(format!(
                "incomplete burnpack load from {}: missing={:?}, errors={:?}",
                path.display(),
                result.missing,
                result.errors
            ));
        }
        Ok(model)
    }

    pub fn load_burnpack_static(bytes: &'static [u8], device: &Device) -> Result<Self, String> {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_static(bytes);
        let result = model
            .load_from(&mut store)
            .map_err(|e| format!("failed to load static burnpack: {e}"))?;
        if !result.missing.is_empty() || !result.errors.is_empty() {
            return Err(format!(
                "incomplete static burnpack load: missing={:?}, errors={:?}",
                result.missing, result.errors
            ));
        }
        Ok(model)
    }

    pub fn predict_28x28(
        &self,
        normalized_pixels: &[f32; 784],
        device: &Device,
    ) -> Result<Prediction, String> {
        let input = Tensor::<4>::from_data(
            TensorData::new(normalized_pixels.to_vec(), [1, 1, 28, 28]),
            device,
        );
        let probs_tensor = self.forward_probabilities(input);
        let probs_vec = probs_tensor
            .to_data()
            .try_to_vec::<f32>()
            .map_err(|e| format!("failed to read probabilities tensor: {e:?}"))?;
        if probs_vec.len() != 10 {
            return Err(format!(
                "expected 10 probabilities, got {}",
                probs_vec.len()
            ));
        }
        let mut probabilities = [0.0f32; 10];
        let mut sum = 0.0f32;
        for (i, &p) in probs_vec.iter().enumerate() {
            let clean = if p.is_finite() && p >= 0.0 { p } else { 0.0 };
            probabilities[i] = clean;
            sum += clean;
        }
        if sum > 0.0 {
            for p in &mut probabilities {
                *p /= sum;
            }
        } else {
            probabilities = [0.1; 10];
        }
        let (digit, &confidence) = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap_or((0, &0.1));
        Ok(Prediction {
            digit,
            confidence,
            probabilities,
        })
    }

    pub fn predict_png(&self, path: &Path, device: &Device) -> Result<Prediction, String> {
        let pixels = load_png_as_mnist_28x28(path)?;
        self.predict_28x28(&pixels, device)
    }
}

/// Parse a directory of Linux `/proc/<pid>/fdinfo` files and sum `drm-engine-panthor`
/// nanoseconds across unique `drm-client-id` contexts.
///
/// Mesa's `PanVK` (Vulkan) and `Panfrost` (OpenGL ES) drivers duplicate the DRM render node
/// file descriptor multiple times within the same process (sharing the same `drm-client-id`).
/// Deduplicating by `drm-client-id` avoids counting the same hardware DRM client multiple times.
pub fn parse_panthor_fdinfo_dir(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut clients: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for entry in entries.flatten() {
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let mut is_panthor = false;
        let mut client_id: Option<String> = None;
        let mut engine_ns: Option<u64> = None;

        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("drm-driver:") {
                if rest.trim() == "panthor" {
                    is_panthor = true;
                }
            } else if let Some(rest) = line.strip_prefix("drm-client-id:") {
                let id = rest.trim();
                if !id.is_empty() {
                    client_id = Some(id.to_string());
                }
            } else if let Some(rest) = line.strip_prefix("drm-engine-panthor:")
                && let Some(ns_str) = rest.split_whitespace().next()
                && let Ok(ns) = ns_str.parse::<u64>()
            {
                engine_ns = Some(ns);
            }
        }

        if is_panthor && let Some(ns) = engine_ns {
            let key = client_id.unwrap_or_else(|| entry.file_name().to_string_lossy().into_owned());
            clients
                .entry(key)
                .and_modify(|prev| *prev = (*prev).max(ns))
                .or_insert(ns);
        }
    }
    clients.values().copied().sum()
}

/// Read RK3588 Mali-G610 (`panthor` DRM driver) cumulative hardware GPU execution time in nanoseconds
/// for the current process from `/proc/<pid>/fdinfo`, deduplicated by `drm-client-id`.
/// Returns 0 on non-panthor hosts.
pub fn read_panthor_gpu_ns() -> u64 {
    let pid = std::process::id();
    parse_panthor_fdinfo_dir(Path::new(&format!("/proc/{pid}/fdinfo")))
}
