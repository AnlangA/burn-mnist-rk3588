//! Rockchip RK3588 NPU (`RKNPU` / `librknnrt.so`) inference backend.
//!
//! Dynamically loads `librknnrt.so` at runtime via `dlopen`/`dlsym` and executes
//! compiled `.rknn` models across the 3 hardware NPU cores (`Core 0`, `Core 1`, `Core 2`)
//! on RK3588 via `/dev/rknpu`.

use crate::data::load_png_as_mnist_28x28;
use crate::model::Prediction;
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

pub const RKNB_MAGIC: &[u8; 8] = b"RKNB\x01\x00\x00\x00";

const RKNN_NPU_CORE_0: u32 = 1;
const RKNN_NPU_CORE_1: u32 = 2;
const RKNN_NPU_CORE_2: u32 = 4;
const RKNN_CORE_MASKS: [u32; 3] = [RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2];

const RKNN_TENSOR_FLOAT32: i32 = 0;
const RKNN_TENSOR_NHWC: i32 = 1;
const RKNN_QUERY_SDK_VERSION: u32 = 5;

type RknnContext = u64;

#[repr(C)]
#[derive(Clone, Copy)]
struct RknnSdkVersion {
    api_version: [u8; 256],
    drv_version: [u8; 256],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RknnInput {
    index: u32,
    buf: *mut c_void,
    size: u32,
    pass_through: u8,
    type_: i32,
    fmt: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RknnOutput {
    want_float: u8,
    is_prealloc: u8,
    index: u32,
    buf: *mut c_void,
    size: u32,
}

type FnRknnInit = unsafe extern "C" fn(
    context: *mut RknnContext,
    model: *mut c_void,
    size: u32,
    flag: u32,
    extend: *mut c_void,
) -> c_int;
type FnRknnDestroy = unsafe extern "C" fn(context: RknnContext) -> c_int;
type FnRknnQuery =
    unsafe extern "C" fn(context: RknnContext, cmd: u32, info: *mut c_void, size: u32) -> c_int;
type FnRknnInputsSet =
    unsafe extern "C" fn(context: RknnContext, n_inputs: u32, inputs: *mut RknnInput) -> c_int;
type FnRknnRun = unsafe extern "C" fn(context: RknnContext, extend: *mut c_void) -> c_int;
type FnRknnOutputsGet = unsafe extern "C" fn(
    context: RknnContext,
    n_outputs: u32,
    outputs: *mut RknnOutput,
    extend: *mut c_void,
) -> c_int;
type FnRknnSetCoreMask = unsafe extern "C" fn(context: RknnContext, core_mask: u32) -> c_int;

#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

#[cfg(target_os = "linux")]
const RTLD_NOW: c_int = 2;
#[cfg(all(unix, not(target_os = "linux")))]
const RTLD_NOW: c_int = 2;

struct RknnApi {
    handle: *mut c_void,
    rknn_init: FnRknnInit,
    rknn_destroy: FnRknnDestroy,
    rknn_query: FnRknnQuery,
    rknn_inputs_set: FnRknnInputsSet,
    rknn_run: FnRknnRun,
    rknn_outputs_get: FnRknnOutputsGet,
    rknn_set_core_mask: FnRknnSetCoreMask,
}

// Safety: `librknnrt.so` functions are thread-safe across distinct `RknnContext` handles.
unsafe impl Send for RknnApi {}
unsafe impl Sync for RknnApi {}

impl RknnApi {
    #[cfg(unix)]
    fn load() -> Result<Arc<Self>, String> {
        let candidates = [
            "librknnrt.so",
            "/usr/lib/librknnrt.so",
            "/usr/local/lib/librknnrt.so",
            "/usr/lib/aarch64-linux-gnu/librknnrt.so",
        ];
        let mut handle: *mut c_void = std::ptr::null_mut();
        for cand in &candidates {
            let c_str = CString::new(*cand).unwrap();
            let h = unsafe { dlopen(c_str.as_ptr(), RTLD_NOW) };
            if !h.is_null() {
                handle = h;
                break;
            }
        }
        if handle.is_null() {
            let err_msg = unsafe {
                let p = dlerror();
                if p.is_null() {
                    "unknown dlopen error".to_string()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            return Err(format!("failed to dlopen librknnrt.so: {err_msg}"));
        }

        unsafe fn load_sym<T>(handle: *mut c_void, name: &str) -> Result<T, String> {
            let c_name = CString::new(name).unwrap();
            let ptr = unsafe { dlsym(handle, c_name.as_ptr()) };
            if ptr.is_null() {
                return Err(format!("missing symbol '{name}' in librknnrt.so"));
            }
            Ok(unsafe { std::mem::transmute_copy(&ptr) })
        }

        unsafe {
            let rknn_init = load_sym::<FnRknnInit>(handle, "rknn_init")?;
            let rknn_destroy = load_sym::<FnRknnDestroy>(handle, "rknn_destroy")?;
            let rknn_query = load_sym::<FnRknnQuery>(handle, "rknn_query")?;
            let rknn_inputs_set = load_sym::<FnRknnInputsSet>(handle, "rknn_inputs_set")?;
            let rknn_run = load_sym::<FnRknnRun>(handle, "rknn_run")?;
            let rknn_outputs_get = load_sym::<FnRknnOutputsGet>(handle, "rknn_outputs_get")?;
            let rknn_set_core_mask = load_sym::<FnRknnSetCoreMask>(handle, "rknn_set_core_mask")?;

            Ok(Arc::new(Self {
                handle,
                rknn_init,
                rknn_destroy,
                rknn_query,
                rknn_inputs_set,
                rknn_run,
                rknn_outputs_get,
                rknn_set_core_mask,
            }))
        }
    }

    #[cfg(not(unix))]
    fn load() -> Result<Arc<Self>, String> {
        Err("RKNN NPU backend is only supported on Unix/Linux targets".into())
    }
}

impl Drop for RknnApi {
    fn drop(&mut self) {
        #[cfg(unix)]
        if !self.handle.is_null() {
            unsafe {
                dlclose(self.handle);
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RknbProfileEntry {
    pub name: String,
    pub batch_size: usize,
    pub input_width: usize,
    pub int8: bool,
    pub offset: usize,
    pub size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RknbManifest {
    profiles: Vec<RknbProfileEntry>,
}

/// Parse an RKNB multi-profile bundle if the trailer magic is present, or return
/// a single fallback entry spanning the entire `.rknn` file.
pub fn parse_rknb_bundle(data: &[u8], default_int8: bool) -> Result<Vec<RknbProfileEntry>, String> {
    if data.len() >= 24 && &data[data.len() - 8..] == RKNB_MAGIC {
        let manifest_offset =
            u64::from_le_bytes(data[data.len() - 24..data.len() - 16].try_into().unwrap()) as usize;
        let manifest_len =
            u64::from_le_bytes(data[data.len() - 16..data.len() - 8].try_into().unwrap()) as usize;
        if manifest_offset
            .checked_add(manifest_len)
            .is_none_or(|end| end > data.len() - 24)
        {
            return Err("corrupt RKNB bundle manifest bounds".into());
        }
        let manifest: RknbManifest =
            serde_json::from_slice(&data[manifest_offset..manifest_offset + manifest_len])
                .map_err(|e| format!("failed to parse RKNB manifest JSON: {e}"))?;
        for p in &manifest.profiles {
            if p.offset
                .checked_add(p.size)
                .is_none_or(|end| end > data.len())
                || p.size == 0
            {
                return Err(format!("invalid RKNB profile slice bounds for {}", p.name));
            }
        }
        if manifest.profiles.is_empty() {
            return Err("RKNB manifest contains zero profiles".into());
        }
        return Ok(manifest.profiles);
    }

    if data.len() < 16 {
        return Err("RKNN model file is too small".into());
    }
    Ok(vec![RknbProfileEntry {
        name: if default_int8 {
            "int8_b1".to_string()
        } else {
            "fp16_b1".to_string()
        },
        batch_size: 1,
        input_width: 28,
        int8: default_int8,
        offset: 0,
        size: data.len(),
    }])
}

/// Resolve the `.rknn` model path from a user-supplied `--model` path (which may point to `.bpk`,
/// `.safetensors`, `.onnx`, or `.rknn`).
pub fn resolve_rknn_model_path(model_path: &Path, prefer_int8_standalone: bool) -> PathBuf {
    if model_path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("rknn"))
        && model_path.is_file()
    {
        return model_path.to_path_buf();
    }

    let parent = model_path.parent().unwrap_or_else(|| Path::new("."));
    let candidates = [
        parent.join("mnist_cnn.rknn"),
        PathBuf::from("artifacts/mnist_cnn.rknn"),
        PathBuf::from("/opt/burn-mnist-rk3588/model/mnist_cnn.rknn"),
        PathBuf::from("/usr/share/digit-recognizer/model/mnist_cnn.rknn"),
    ];
    for c in &candidates {
        if c.is_file() {
            return c.clone();
        }
    }
    if prefer_int8_standalone {
        let int8_cands = [
            parent.join("mnist_cnn_int8.rknn"),
            PathBuf::from("artifacts/mnist_cnn_int8.rknn"),
            PathBuf::from("/opt/burn-mnist-rk3588/model/mnist_cnn_int8.rknn"),
            PathBuf::from("/usr/share/digit-recognizer/model/mnist_cnn_int8.rknn"),
        ];
        for c in &int8_cands {
            if c.is_file() {
                return c.clone();
            }
        }
    }
    parent.join("mnist_cnn.rknn")
}

struct RknnSingleCtx {
    api: Arc<RknnApi>,
    ctx: RknnContext,
    batch_size: usize,
    input_width: usize,
    in_buf: Vec<f32>,
    out_buf: Vec<f32>,
}

// Safety: each `RknnSingleCtx` owns a dedicated `RknnContext` and private `in_buf`/`out_buf`.
unsafe impl Send for RknnSingleCtx {}

impl RknnSingleCtx {
    fn new(
        api: Arc<RknnApi>,
        model_bytes: &[u8],
        batch_size: usize,
        input_width: usize,
        core_mask: u32,
    ) -> Result<Self, String> {
        let mut owned = model_bytes.to_vec();
        let mut ctx: RknnContext = 0;
        let ret = unsafe {
            (api.rknn_init)(
                &mut ctx,
                owned.as_mut_ptr().cast::<c_void>(),
                owned.len() as u32,
                0,
                std::ptr::null_mut(),
            )
        };
        if ret != 0 || ctx == 0 {
            return Err(format!(
                "rknn_init(batch={batch_size}, width={input_width}) failed with code {ret}"
            ));
        }
        if core_mask != 0 {
            let _ = unsafe { (api.rknn_set_core_mask)(ctx, core_mask) };
        }
        Ok(Self {
            api,
            ctx,
            batch_size,
            input_width,
            in_buf: vec![0.0f32; batch_size * 28 * input_width],
            out_buf: vec![0.0f32; batch_size * 10],
        })
    }

    fn query_sdk_version(&self) -> Option<(String, String)> {
        let mut ver = RknnSdkVersion {
            api_version: [0u8; 256],
            drv_version: [0u8; 256],
        };
        let ret = unsafe {
            (self.api.rknn_query)(
                self.ctx,
                RKNN_QUERY_SDK_VERSION,
                (&mut ver as *mut RknnSdkVersion).cast::<c_void>(),
                std::mem::size_of::<RknnSdkVersion>() as u32,
            )
        };
        if ret != 0 {
            return None;
        }
        let cstr_to_string = |buf: &[u8; 256]| -> String {
            let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            String::from_utf8_lossy(&buf[..end]).trim().to_string()
        };
        Some((
            cstr_to_string(&ver.api_version),
            cstr_to_string(&ver.drv_version),
        ))
    }

    /// Pack `count` (<= `self.batch_size`) images of 784 floats (`28x28`) into `self.in_buf`,
    /// execute `rknn_inputs_set` + `rknn_run` + `rknn_outputs_get`, and write softmax
    /// probabilities into `dst_probs` (`count * 10` floats).
    fn run_subbatch(
        &mut self,
        src_pixels: &[f32],
        count: usize,
        dst_probs: &mut [f32],
    ) -> Result<(), String> {
        let count = count.min(self.batch_size);
        if count == 0 {
            return Ok(());
        }
        if self.input_width == 28 {
            let n_floats = count * 784;
            self.in_buf[..n_floats].copy_from_slice(&src_pixels[..n_floats]);
        } else {
            // Pad each 28-element row to `input_width` (32) for zero-Slice NPU DMA alignment
            let w = self.input_width;
            for b in 0..count {
                let src_img = &src_pixels[b * 784..(b + 1) * 784];
                let dst_img = &mut self.in_buf[b * 28 * w..(b + 1) * 28 * w];
                for y in 0..28 {
                    let s_row = &src_img[y * 28..(y + 1) * 28];
                    let d_row = &mut dst_img[y * w..y * w + 28];
                    d_row.copy_from_slice(s_row);
                }
            }
        }

        let mut inp = RknnInput {
            index: 0,
            buf: self.in_buf.as_mut_ptr().cast::<c_void>(),
            size: (self.in_buf.len() * std::mem::size_of::<f32>()) as u32,
            pass_through: 0,
            type_: RKNN_TENSOR_FLOAT32,
            fmt: RKNN_TENSOR_NHWC,
        };
        let ret = unsafe { (self.api.rknn_inputs_set)(self.ctx, 1, &mut inp) };
        if ret != 0 {
            return Err(format!("rknn_inputs_set failed with code {ret}"));
        }

        let ret = unsafe { (self.api.rknn_run)(self.ctx, std::ptr::null_mut()) };
        if ret != 0 {
            return Err(format!("rknn_run failed with code {ret}"));
        }

        let mut out = RknnOutput {
            want_float: 1,
            is_prealloc: 1,
            index: 0,
            buf: self.out_buf.as_mut_ptr().cast::<c_void>(),
            size: (self.out_buf.len() * std::mem::size_of::<f32>()) as u32,
        };
        let ret =
            unsafe { (self.api.rknn_outputs_get)(self.ctx, 1, &mut out, std::ptr::null_mut()) };
        if ret != 0 {
            return Err(format!("rknn_outputs_get failed with code {ret}"));
        }

        for b in 0..count {
            let logits = &self.out_buf[b * 10..(b + 1) * 10];
            let probs = softmax_row_10(logits);
            dst_probs[b * 10..(b + 1) * 10].copy_from_slice(&probs);
        }
        Ok(())
    }
}

impl Drop for RknnSingleCtx {
    fn drop(&mut self) {
        if self.ctx != 0 {
            unsafe {
                let _ = (self.api.rknn_destroy)(self.ctx);
            }
            self.ctx = 0;
        }
    }
}

struct WorkerShared {
    state: Mutex<WorkerState>,
    cv_work: Condvar,
    cv_done: Condvar,
}

struct WorkerState {
    generation: u64,
    completed_gen: u64,
    src_ptr: usize,
    dst_ptr: usize,
    count: usize,
    result: Result<(), String>,
    shutdown: bool,
}

struct PersistentCoreWorker {
    shared: Arc<WorkerShared>,
    handle: Option<JoinHandle<()>>,
}

impl PersistentCoreWorker {
    fn spawn(mut core_ctx: RknnSingleCtx) -> Self {
        let shared = Arc::new(WorkerShared {
            state: Mutex::new(WorkerState {
                generation: 0,
                completed_gen: 0,
                src_ptr: 0,
                dst_ptr: 0,
                count: 0,
                result: Ok(()),
                shutdown: false,
            }),
            cv_work: Condvar::new(),
            cv_done: Condvar::new(),
        });
        let shared_clone = Arc::clone(&shared);
        let handle = thread::spawn(move || {
            let mut seen_gen = 0u64;
            loop {
                let (src_ptr, dst_ptr, count, next_gen) = {
                    let mut guard = shared_clone.state.lock().unwrap();
                    while guard.generation == seen_gen && !guard.shutdown {
                        guard = shared_clone.cv_work.wait(guard).unwrap();
                    }
                    if guard.shutdown {
                        return;
                    }
                    seen_gen = guard.generation;
                    (guard.src_ptr, guard.dst_ptr, guard.count, seen_gen)
                };

                let res = if count > 0 {
                    let src =
                        unsafe { std::slice::from_raw_parts(src_ptr as *const f32, count * 784) };
                    let dst =
                        unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut f32, count * 10) };
                    core_ctx.run_subbatch(src, count, dst)
                } else {
                    Ok(())
                };

                let mut guard = shared_clone.state.lock().unwrap();
                guard.result = res;
                guard.completed_gen = next_gen;
                shared_clone.cv_done.notify_one();
            }
        });

        Self {
            shared,
            handle: Some(handle),
        }
    }

    unsafe fn dispatch(&self, src: &[f32], dst: &mut [f32], count: usize) -> u64 {
        let mut guard = self.shared.state.lock().unwrap();
        let next_gen = guard.generation + 1;
        guard.src_ptr = src.as_ptr() as usize;
        guard.dst_ptr = dst.as_mut_ptr() as usize;
        guard.count = count;
        guard.generation = next_gen;
        self.shared.cv_work.notify_one();
        next_gen
    }

    fn wait(&self, target_gen: u64) -> Result<(), String> {
        let mut guard = self.shared.state.lock().unwrap();
        while guard.completed_gen < target_gen {
            guard = self.shared.cv_done.wait(guard).unwrap();
        }
        guard.result.clone()
    }
}

impl Drop for PersistentCoreWorker {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.shared.state.lock() {
            guard.shutdown = true;
            self.shared.cv_work.notify_all();
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// 3-Core NPU execution tier where Core 0 runs on the caller thread and Cores 1 & 2
/// run on persistent worker threads concurrently.
struct RknnThreeCoreTier {
    sub_batch_size: usize,
    core0: RknnSingleCtx,
    worker1: PersistentCoreWorker,
    worker2: PersistentCoreWorker,
}

impl RknnThreeCoreTier {
    fn new(
        api: Arc<RknnApi>,
        model_bytes: &[u8],
        sub_batch_size: usize,
        input_width: usize,
    ) -> Result<Self, String> {
        let core0 = RknnSingleCtx::new(
            Arc::clone(&api),
            model_bytes,
            sub_batch_size,
            input_width,
            RKNN_CORE_MASKS[0],
        )?;
        let core1 = RknnSingleCtx::new(
            Arc::clone(&api),
            model_bytes,
            sub_batch_size,
            input_width,
            RKNN_CORE_MASKS[1],
        )?;
        let core2 = RknnSingleCtx::new(
            api,
            model_bytes,
            sub_batch_size,
            input_width,
            RKNN_CORE_MASKS[2],
        )?;
        Ok(Self {
            sub_batch_size,
            core0,
            worker1: PersistentCoreWorker::spawn(core1),
            worker2: PersistentCoreWorker::spawn(core2),
        })
    }

    /// Execute up to `3 * sub_batch_size` samples across Core 0, Core 1, and Core 2 in parallel.
    fn run_wave(
        &mut self,
        src_pixels: &[f32],
        total_count: usize,
        dst_probs: &mut [f32],
    ) -> Result<(), String> {
        let cap = self.sub_batch_size;
        let c0 = total_count.div_ceil(3).min(cap);
        let rem1 = total_count.saturating_sub(c0);
        let c1 = rem1.div_ceil(2).min(cap);
        let c2 = rem1.saturating_sub(c1).min(cap);

        let (src0, src_rest) = src_pixels.split_at(c0 * 784);
        let (src1, src2) = src_rest.split_at(c1 * 784);

        let (dst0, dst_rest) = dst_probs.split_at_mut(c0 * 10);
        let (dst1, dst2) = dst_rest.split_at_mut(c1 * 10);

        let gen1 = if c1 > 0 {
            Some(unsafe { self.worker1.dispatch(src1, dst1, c1) })
        } else {
            None
        };
        let gen2 = if c2 > 0 {
            Some(unsafe { self.worker2.dispatch(src2, dst2, c2) })
        } else {
            None
        };

        let res0 = self.core0.run_subbatch(src0, c0, dst0);

        if let Some(g1) = gen1 {
            self.worker1.wait(g1)?;
        }
        if let Some(g2) = gen2 {
            self.worker2.wait(g2)?;
        }
        res0
    }
}

pub struct RknnEngine {
    api: Arc<RknnApi>,
    bundle_bytes: Vec<u8>,
    profiles: Vec<RknbProfileEntry>,
    int8_mode: bool,
    b1_core0: RknnSingleCtx,
    tier_b6: Option<RknnThreeCoreTier>,
    tier_b22: Option<RknnThreeCoreTier>,
    tier_b86: Option<RknnThreeCoreTier>,
    description: String,
}

impl RknnEngine {
    pub fn load(model_path: &Path, int8_mode: bool) -> Result<Self, String> {
        let resolved_path = resolve_rknn_model_path(model_path, int8_mode);
        let bundle_bytes = std::fs::read(&resolved_path)
            .map_err(|e| format!("failed to read RKNN model {}: {e}", resolved_path.display()))?;
        let profiles = parse_rknb_bundle(&bundle_bytes, int8_mode)?;
        let api = RknnApi::load()?;

        let b1_entry = profiles
            .iter()
            .find(|p| p.int8 == int8_mode && p.batch_size == 1)
            .or_else(|| profiles.iter().find(|p| p.batch_size == 1))
            .cloned()
            .ok_or_else(|| "no batch=1 profile found in RKNN model".to_string())?;

        let b1_slice = &bundle_bytes[b1_entry.offset..b1_entry.offset + b1_entry.size];
        let b1_core0 = RknnSingleCtx::new(
            Arc::clone(&api),
            b1_slice,
            b1_entry.batch_size,
            b1_entry.input_width,
            RKNN_NPU_CORE_0,
        )?;

        let freq_mhz = read_rknpu_freq_hz() / 1_000_000;
        let dtype_label = if b1_entry.int8 { "INT8" } else { "FP16" };
        let description = match b1_core0.query_sdk_version() {
            Some((rt_ver, drv_ver)) => {
                let rt_short = rt_ver.split_whitespace().next().unwrap_or(&rt_ver);
                if freq_mhz > 0 {
                    format!(
                        "RK3588 NPU 3-Core ({dtype_label}, librknnrt {rt_short}, drv {drv_ver}, {freq_mhz} MHz)"
                    )
                } else {
                    format!(
                        "RK3588 NPU 3-Core ({dtype_label}, librknnrt {rt_short}, drv {drv_ver})"
                    )
                }
            }
            None => format!("RK3588 NPU 3-Core ({dtype_label})"),
        };

        Ok(Self {
            api,
            bundle_bytes,
            profiles,
            int8_mode: b1_entry.int8,
            b1_core0,
            tier_b6: None,
            tier_b22: None,
            tier_b86: None,
            description,
        })
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    fn ensure_tier(&mut self, target_sub_bs: usize) -> Result<bool, String> {
        let already = match target_sub_bs {
            6 => self.tier_b6.is_some(),
            22 => self.tier_b22.is_some(),
            86 => self.tier_b86.is_some(),
            _ => false,
        };
        if already {
            return Ok(true);
        }
        let Some(entry) = self
            .profiles
            .iter()
            .find(|p| p.int8 == self.int8_mode && p.batch_size == target_sub_bs)
            .cloned()
        else {
            return Ok(false);
        };
        let slice = &self.bundle_bytes[entry.offset..entry.offset + entry.size];
        let tier = match RknnThreeCoreTier::new(
            Arc::clone(&self.api),
            slice,
            entry.batch_size,
            entry.input_width,
        ) {
            Ok(t) => t,
            Err(_first_err) => {
                // Free other cached multi-core tiers to reclaim /dev/rknpu CMA memory and retry.
                self.tier_b6 = None;
                self.tier_b22 = None;
                self.tier_b86 = None;
                RknnThreeCoreTier::new(
                    Arc::clone(&self.api),
                    slice,
                    entry.batch_size,
                    entry.input_width,
                )?
            }
        };
        match target_sub_bs {
            6 => self.tier_b6 = Some(tier),
            22 => self.tier_b22 = Some(tier),
            86 => self.tier_b86 = Some(tier),
            _ => {}
        }
        Ok(true)
    }

    /// Run single-sample `28x28` inference on NPU Core 0.
    pub fn predict_28x28(&mut self, normalized_pixels: &[f32; 784]) -> Result<Prediction, String> {
        let mut probs = [0.0f32; 10];
        self.b1_core0
            .run_subbatch(normalized_pixels, 1, &mut probs)?;
        Ok(prediction_from_probabilities(probs))
    }

    pub fn predict_png(&mut self, path: &Path) -> Result<Prediction, String> {
        let pixels = load_png_as_mnist_28x28(path)?;
        self.predict_28x28(&pixels)
    }

    /// Run batch inference for `batch_size` samples (`samples.len() == batch_size * 784`),
    /// returning normalized probabilities of length `batch_size * 10`.
    pub fn forward_probabilities_batch(
        &mut self,
        samples: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        if batch_size == 0 {
            return Err("batch_size must be > 0".into());
        }
        if samples.len() != batch_size * 784 {
            return Err(format!(
                "expected {} floats for batch_size={batch_size}, got {}",
                batch_size * 784,
                samples.len()
            ));
        }

        let mut out_probs = vec![0.0f32; batch_size * 10];
        if batch_size == 1 {
            self.b1_core0.run_subbatch(samples, 1, &mut out_probs)?;
            return Ok(out_probs);
        }

        let candidate_tiers: &[usize] = if batch_size <= 18 {
            &[6]
        } else if batch_size <= 66 {
            &[22, 6]
        } else {
            &[86, 22, 6]
        };

        for &target_sub_bs in candidate_tiers {
            if let Ok(true) = self.ensure_tier(target_sub_bs) {
                let tier = match target_sub_bs {
                    6 => self.tier_b6.as_mut().unwrap(),
                    22 => self.tier_b22.as_mut().unwrap(),
                    _ => self.tier_b86.as_mut().unwrap(),
                };
                let wave_cap = tier.sub_batch_size * 3;
                let mut offset = 0usize;
                while offset < batch_size {
                    let count = (batch_size - offset).min(wave_cap);
                    tier.run_wave(
                        &samples[offset * 784..(offset + count) * 784],
                        count,
                        &mut out_probs[offset * 10..(offset + count) * 10],
                    )?;
                    offset += count;
                }
                return Ok(out_probs);
            }
        }

        // Fallback if using a single-profile b=1 .rknn file without RKNB bundle tiers
        for b in 0..batch_size {
            self.b1_core0.run_subbatch(
                &samples[b * 784..(b + 1) * 784],
                1,
                &mut out_probs[b * 10..(b + 1) * 10],
            )?;
        }
        Ok(out_probs)
    }
}

pub fn softmax_row_10(logits: &[f32]) -> [f32; 10] {
    let mut max_v = f32::NEG_INFINITY;
    for &x in logits.iter().take(10) {
        if x.is_finite() && x > max_v {
            max_v = x;
        }
    }
    if !max_v.is_finite() {
        return [0.1; 10];
    }
    let mut probs = [0.0f32; 10];
    let mut sum = 0.0f32;
    for (i, &x) in logits.iter().take(10).enumerate() {
        let e = if x.is_finite() {
            (x - max_v).exp()
        } else {
            0.0
        };
        probs[i] = e;
        sum += e;
    }
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    } else {
        probs = [0.1; 10];
    }
    probs
}

pub fn prediction_from_probabilities(probabilities: [f32; 10]) -> Prediction {
    let (digit, &confidence) = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap_or((0, &0.1));
    Prediction {
        digit,
        confidence,
        probabilities,
    }
}

pub fn read_sysfs_u64(path: &Path) -> u64 {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// Read cumulative RK3588 NPU hardware busy time in nanoseconds across all 3 NPU cores
/// from `/sys/module/rknpu/parameters/hw_busy_ns`.
pub fn read_rknpu_busy_ns() -> u64 {
    read_sysfs_u64(Path::new("/sys/module/rknpu/parameters/hw_busy_ns"))
}

/// Read cumulative RK3588 NPU hardware jobs completed from `/sys/module/rknpu/parameters/hw_jobs`.
pub fn read_rknpu_jobs() -> u64 {
    read_sysfs_u64(Path::new("/sys/module/rknpu/parameters/hw_jobs"))
}

/// Read current RK3588 NPU clock frequency in Hz from `/sys/module/rknpu/parameters/npu_freq_hz`.
pub fn read_rknpu_freq_hz() -> u64 {
    read_sysfs_u64(Path::new("/sys/module/rknpu/parameters/npu_freq_hz"))
}
