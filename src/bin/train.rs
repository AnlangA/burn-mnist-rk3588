use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, ModuleOptimizer};
use burn::prelude::*;
use burn_mnist_rk3588::data::{MnistSplit, load_test_split, load_train_split};
use burn_mnist_rk3588::model::{BackendKind, MnistCnn, resolve_device};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f32,
    pub train_accuracy: f32,
    pub test_loss: f32,
    pub test_accuracy: f32,
    pub epoch_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingSummary {
    pub backend_requested: String,
    pub backend_resolved: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub train_samples: usize,
    pub test_samples: usize,
    pub total_training_seconds: f64,
    pub final_train_loss: f32,
    pub final_train_accuracy: f32,
    pub final_test_loss: f32,
    pub final_test_accuracy: f32,
    pub burnpack_path: String,
    pub safetensors_path: String,
    pub history: Vec<EpochMetrics>,
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn shuffle(&mut self, slice: &mut [usize]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

fn evaluate(
    model: &MnistCnn,
    split: &MnistSplit,
    batch_size: usize,
    device: &Device,
) -> Result<(f32, f32), String> {
    let eval_model = model.clone().to_device(device);
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let indices: Vec<usize> = (0..split.len()).collect();
    let mut total_loss = 0.0f64;
    let mut correct = 0usize;
    let mut total = 0usize;

    for chunk in indices.chunks(batch_size.max(1)) {
        let (images, targets) = split.batch(chunk, device)?;
        let logits = eval_model.forward_logits(images);
        let loss = loss_fn.forward(logits.clone(), targets);
        let loss_val = loss
            .to_data()
            .try_to_vec::<f32>()
            .map_err(|e| format!("failed to read eval loss: {e:?}"))?[0];
        let logits_vec = logits
            .to_data()
            .try_to_vec::<f32>()
            .map_err(|e| format!("failed to read eval logits: {e:?}"))?;

        for (row_idx, &sample_idx) in chunk.iter().enumerate() {
            let row = &logits_vec[row_idx * 10..(row_idx + 1) * 10];
            let pred = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(d, _)| d)
                .unwrap_or(0);
            if pred == usize::from(split.samples[sample_idx].label) {
                correct += 1;
            }
        }
        total_loss += f64::from(loss_val) * (chunk.len() as f64);
        total += chunk.len();
    }

    let avg_loss = if total > 0 {
        (total_loss / (total as f64)) as f32
    } else {
        0.0
    };
    let accuracy = if total > 0 {
        (correct as f32) / (total as f32)
    } else {
        0.0
    };
    Ok((avg_loss, accuracy))
}

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "Usage: train [--data-dir DIR] [--artifact-dir DIR] [--backend cpu|gpu|vulkan|gl]\n\
             [--epochs N] [--batch-size N] [--lr FLOAT] [--max-train N] [--max-test N]"
        );
        return Ok(());
    }

    let data_dir =
        PathBuf::from(parse_arg(&args, "--data-dir").unwrap_or_else(|| "data/mnist".into()));
    let artifact_dir =
        PathBuf::from(parse_arg(&args, "--artifact-dir").unwrap_or_else(|| "artifacts".into()));
    let backend_str = parse_arg(&args, "--backend").unwrap_or_else(|| "cpu".into());
    let epochs: usize = parse_arg(&args, "--epochs")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let batch_size: usize = parse_arg(&args, "--batch-size")
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let lr: f64 = parse_arg(&args, "--lr")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-3);
    let max_train: Option<usize> = parse_arg(&args, "--max-train").and_then(|v| v.parse().ok());
    let max_test: Option<usize> = parse_arg(&args, "--max-test").and_then(|v| v.parse().ok());

    let backend_kind = BackendKind::parse(&backend_str)?;
    let (base_device, backend_desc) = resolve_device(backend_kind)?;
    let train_device = base_device.clone().autodiff();

    println!(
        "[train] Backend: {} ({}), epochs={}, batch_size={}, lr={}",
        backend_kind.as_str(),
        backend_desc,
        epochs,
        batch_size,
        lr
    );
    println!(
        "[train] Loading MNIST dataset from {} ...",
        data_dir.display()
    );
    let train_split = load_train_split(Path::new(&data_dir), max_train)?;
    let test_split = load_test_split(Path::new(&data_dir), max_test)?;
    println!(
        "[train] Loaded {} training samples and {} test samples",
        train_split.len(),
        test_split.len()
    );

    let mut model = MnistCnn::new(&train_device);
    let loss_fn = CrossEntropyLossConfig::new().init(&train_device);
    let mut optim: ModuleOptimizer = AdamConfig::new().init();

    let mut indices: Vec<usize> = (0..train_split.len()).collect();
    let mut rng = SplitMix64::new(0x2026_1108_3588_cafe);
    let mut history = Vec::with_capacity(epochs);
    let total_start = Instant::now();

    for epoch in 1..=epochs {
        let epoch_start = Instant::now();
        rng.shuffle(&mut indices);

        let mut running_loss = 0.0f64;
        let mut running_correct = 0usize;
        let mut running_total = 0usize;

        for (step, chunk) in indices.chunks(batch_size.max(1)).enumerate() {
            let (images, targets) = train_split.batch(chunk, &train_device)?;
            let logits = model.forward_logits(images);
            let loss = loss_fn.forward(logits.clone(), targets);

            let loss_val = loss
                .to_data()
                .try_to_vec::<f32>()
                .map_err(|e| format!("failed to read train loss: {e:?}"))?[0];
            let logits_vec = logits
                .to_data()
                .try_to_vec::<f32>()
                .map_err(|e| format!("failed to read train logits: {e:?}"))?;

            for (row_idx, &sample_idx) in chunk.iter().enumerate() {
                let row = &logits_vec[row_idx * 10..(row_idx + 1) * 10];
                let pred = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(d, _)| d)
                    .unwrap_or(0);
                if pred == usize::from(train_split.samples[sample_idx].label) {
                    running_correct += 1;
                }
            }

            running_loss += f64::from(loss_val) * (chunk.len() as f64);
            running_total += chunk.len();

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);

            if (step + 1) % 100 == 0 {
                println!(
                    "  [epoch {epoch}/{epochs} step {}] train_loss={:.4} train_acc={:.2}%",
                    step + 1,
                    running_loss / (running_total as f64),
                    100.0 * (running_correct as f64) / (running_total as f64)
                );
            }
        }

        let train_loss = (running_loss / (running_total.max(1) as f64)) as f32;
        let train_accuracy = (running_correct as f32) / (running_total.max(1) as f32);
        let (test_loss, test_accuracy) = evaluate(&model, &test_split, 256, &base_device)?;
        let epoch_seconds = epoch_start.elapsed().as_secs_f64();

        println!(
            "[epoch {epoch}/{epochs}] train_loss={train_loss:.4} train_acc={:.2}% | test_loss={test_loss:.4} test_acc={:.2}% ({epoch_seconds:.2}s)",
            train_accuracy * 100.0,
            test_accuracy * 100.0
        );

        history.push(EpochMetrics {
            epoch,
            train_loss,
            train_accuracy,
            test_loss,
            test_accuracy,
            epoch_seconds,
        });
    }

    let total_training_seconds = total_start.elapsed().as_secs_f64();
    std::fs::create_dir_all(&artifact_dir)?;

    let eval_model = model.to_device(&base_device);
    let bpk_path = artifact_dir.join("mnist_cnn.bpk");
    let st_path = artifact_dir.join("mnist_cnn.safetensors");
    let metrics_path = artifact_dir.join("training_metrics.json");

    eval_model.save_burnpack(&bpk_path)?;
    eval_model.save_safetensors(&st_path)?;

    let last = history.last().cloned().unwrap_or(EpochMetrics {
        epoch: 0,
        train_loss: 0.0,
        train_accuracy: 0.0,
        test_loss: 0.0,
        test_accuracy: 0.0,
        epoch_seconds: 0.0,
    });

    let summary = TrainingSummary {
        backend_requested: backend_kind.as_str().to_string(),
        backend_resolved: backend_desc,
        epochs,
        batch_size,
        learning_rate: lr,
        train_samples: train_split.len(),
        test_samples: test_split.len(),
        total_training_seconds,
        final_train_loss: last.train_loss,
        final_train_accuracy: last.train_accuracy,
        final_test_loss: last.test_loss,
        final_test_accuracy: last.test_accuracy,
        burnpack_path: bpk_path.display().to_string(),
        safetensors_path: st_path.display().to_string(),
        history,
    };

    std::fs::write(&metrics_path, serde_json::to_string_pretty(&summary)?)?;
    println!(
        "[train] Saved model artifacts:\n  - {}\n  - {}\n  - {}",
        bpk_path.display(),
        st_path.display(),
        metrics_path.display()
    );
    Ok(())
}
