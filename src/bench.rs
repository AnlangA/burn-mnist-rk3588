use crate::data::{load_png_as_mnist_28x28, load_test_split};
use crate::model::{BackendKind, MnistCnn, Prediction, read_panthor_gpu_ns, resolve_device};
use burn::prelude::*;
use burn::tensor::TensorData;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchBenchResult {
    pub batch_size: usize,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub mean_batch_ms: f64,
    pub p50_batch_ms: f64,
    pub p95_batch_ms: f64,
    pub p99_batch_ms: f64,
    pub per_sample_ms: f64,
    pub throughput_samples_per_sec: f64,
    pub panthor_gpu_total_ms: f64,
    pub panthor_gpu_per_batch_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplePredictionRecord {
    pub file_name: String,
    pub expected_digit: Option<usize>,
    pub predicted_digit: usize,
    pub confidence: f32,
    pub correct: Option<bool>,
    pub latency_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendBenchReport {
    pub backend_requested: String,
    pub backend_resolved: String,
    pub model_load_ms: f64,
    pub single_sample_warmup_ms: f64,
    pub test_set_samples: usize,
    pub test_set_accuracy: Option<f32>,
    pub test_set_warmup_ms: Option<f64>,
    pub test_set_eval_ms: Option<f64>,
    pub png_samples_total: usize,
    pub png_samples_correct: usize,
    pub png_samples_accuracy: Option<f32>,
    pub batch_benchmarks: Vec<BatchBenchResult>,
    pub png_predictions: Vec<SamplePredictionRecord>,
}

pub fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * pct).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub fn digit_from_filename(path: &Path) -> Option<usize> {
    let name = path.file_name()?.to_str()?;
    let tail = name.strip_prefix('d')?;
    let d = tail.chars().next()?.to_digit(10)? as usize;
    (d < 10).then_some(d)
}

pub fn run_single_image_inference(
    kind: BackendKind,
    model_path: &Path,
    image_path: &Path,
) -> Result<(Prediction, f64, String), String> {
    let (device, backend_desc) = resolve_device(kind)?;
    let model = MnistCnn::load_burnpack_file(model_path, &device)?;
    let pixels = load_png_as_mnist_28x28(image_path)?;
    let t0 = Instant::now();
    let pred = model.predict_28x28(&pixels, &device)?;
    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok((pred, latency_ms, backend_desc))
}

pub fn run_backend_bench(
    kind: BackendKind,
    model_path: &Path,
    data_dir: Option<&Path>,
    images_dir: Option<&Path>,
    iterations: usize,
) -> Result<BackendBenchReport, String> {
    let iterations = iterations.max(1);
    let (device, backend_desc) = resolve_device(kind)?;
    println!(
        "=== Benchmarking backend: {} ({}) ===",
        kind.as_str(),
        backend_desc
    );

    let load_start = Instant::now();
    let model = MnistCnn::load_burnpack_file(model_path, &device)?;
    let model_load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    // Warm up single-sample [1, 1, 28, 28] pipeline so PNG sample latencies reflect steady state.
    let single_warmup_start = Instant::now();
    let _ = model.predict_28x28(&[0.0f32; 784], &device)?;
    let single_sample_warmup_ms = single_warmup_start.elapsed().as_secs_f64() * 1000.0;

    let mut test_set_samples = 0usize;
    let mut test_set_accuracy = None;
    let mut test_set_warmup_ms = None;
    let mut test_set_eval_ms = None;

    if let Some(dir) = data_dir
        && dir.is_dir()
        && let Ok(test_split) = load_test_split(dir, None)
    {
        test_set_samples = test_split.len();
        let indices: Vec<usize> = (0..test_split.len()).collect();

        // Warm up full-batch (256) and tail-batch (e.g. 16) shapes prior to timing 10k evaluation.
        let warmup_start = Instant::now();
        if let Some(first_chunk) = indices.chunks(256).next() {
            let (w_img, _) = test_split.batch(first_chunk, &device)?;
            let _ = model
                .forward_probabilities(w_img)
                .to_data()
                .try_to_vec::<f32>()
                .map_err(|e| format!("test warmup error: {e:?}"))?;
        }
        if let Some(last_chunk) = indices.chunks(256).next_back()
            && last_chunk.len() != 256
        {
            let (w_img, _) = test_split.batch(last_chunk, &device)?;
            let _ = model
                .forward_probabilities(w_img)
                .to_data()
                .try_to_vec::<f32>()
                .map_err(|e| format!("test tail warmup error: {e:?}"))?;
        }
        let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;
        test_set_warmup_ms = Some(warmup_ms);

        let eval_start = Instant::now();
        let mut correct = 0usize;
        for chunk in indices.chunks(256) {
            let (images, _) = test_split.batch(chunk, &device)?;
            let probs = model.forward_probabilities(images);
            let probs_vec = probs
                .to_data()
                .try_to_vec::<f32>()
                .map_err(|e| format!("failed to read test probs: {e:?}"))?;
            for (row_idx, &sample_idx) in chunk.iter().enumerate() {
                let row = &probs_vec[row_idx * 10..(row_idx + 1) * 10];
                let pred = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(d, _)| d)
                    .unwrap_or(0);
                if pred == usize::from(test_split.samples[sample_idx].label) {
                    correct += 1;
                }
            }
        }
        let elapsed_ms = eval_start.elapsed().as_secs_f64() * 1000.0;
        let acc = (correct as f32) / (test_split.len().max(1) as f32);
        test_set_accuracy = Some(acc);
        test_set_eval_ms = Some(elapsed_ms);
        println!(
            "  [MNIST 10k test] accuracy={:.2}% ({}/{}) eval={:.2} ms (warmup={:.2} ms)",
            acc * 100.0,
            correct,
            test_split.len(),
            elapsed_ms,
            warmup_ms
        );
    }

    let mut png_predictions = Vec::new();
    let mut png_samples_total = 0usize;
    let mut png_samples_correct = 0usize;

    if let Some(img_dir) = images_dir
        && img_dir.is_dir()
    {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(img_dir)
            .map_err(|e| format!("failed to read {}: {e}", img_dir.display()))?
            .filter_map(|e| e.ok().map(|x| x.path()))
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("png"))
            })
            .collect();
        entries.sort();

        for path in entries {
            let pixels = load_png_as_mnist_28x28(&path)?;
            let t0 = Instant::now();
            let pred = model.predict_28x28(&pixels, &device)?;
            let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let expected = digit_from_filename(&path);
            let is_correct = expected.map(|e| e == pred.digit);
            png_samples_total += 1;
            if is_correct == Some(true) {
                png_samples_correct += 1;
            }
            png_predictions.push(SamplePredictionRecord {
                file_name: path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("sample.png")
                    .to_string(),
                expected_digit: expected,
                predicted_digit: pred.digit,
                confidence: pred.confidence,
                correct: is_correct,
                latency_ms,
            });
        }
        if png_samples_total > 0 {
            let mean_png_ms = png_predictions.iter().map(|p| p.latency_ms).sum::<f64>()
                / (png_samples_total as f64);
            println!(
                "  [PNG samples] accuracy={:.2}% ({}/{}) mean_latency={:.3} ms",
                (png_samples_correct as f64) * 100.0 / (png_samples_total as f64),
                png_samples_correct,
                png_samples_total,
                mean_png_ms
            );
        }
    }

    let png_samples_accuracy = if png_samples_total > 0 {
        Some((png_samples_correct as f32) / (png_samples_total as f32))
    } else {
        None
    };

    let batch_sizes = [1usize, 16, 64, 256];
    let mut batch_benchmarks = Vec::with_capacity(batch_sizes.len());

    for &bs in &batch_sizes {
        let iters = if bs == 1 {
            iterations.max(1)
        } else if bs <= 64 {
            (iterations / 2).max(20)
        } else {
            (iterations / 4).max(10)
        };
        let warmup = (iters / 5).max(5);
        let dummy_data: Vec<f32> = (0..(bs * 784))
            .map(|i| ((i % 255) as f32 / 255.0 - 0.1307) / 0.3081)
            .collect();

        for _ in 0..warmup {
            let input = Tensor::<4>::from_data(
                TensorData::new(dummy_data.clone(), [bs, 1, 28, 28]),
                &device,
            );
            let out = model.forward_probabilities(input);
            let _ = out
                .to_data()
                .try_to_vec::<f32>()
                .map_err(|e| format!("warmup sync error: {e:?}"))?;
        }

        let mut times_ms = Vec::with_capacity(iters);
        let gpu_before_ns = read_panthor_gpu_ns();
        for _ in 0..iters {
            let t0 = Instant::now();
            let input = Tensor::<4>::from_data(
                TensorData::new(dummy_data.clone(), [bs, 1, 28, 28]),
                &device,
            );
            let out = model.forward_probabilities(input);
            let vals = out
                .to_data()
                .try_to_vec::<f32>()
                .map_err(|e| format!("bench sync error: {e:?}"))?;
            std::hint::black_box(vals);
            times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let gpu_after_ns = read_panthor_gpu_ns();
        let panthor_gpu_total_ms =
            (gpu_after_ns.saturating_sub(gpu_before_ns) as f64) / 1_000_000.0;
        let panthor_gpu_per_batch_ms = panthor_gpu_total_ms / (iters as f64);

        times_ms.sort_by(|a, b| a.total_cmp(b));
        let sum_ms: f64 = times_ms.iter().sum();
        let mean_batch_ms = sum_ms / (iters as f64);
        let p50_batch_ms = percentile(&times_ms, 0.50);
        let p95_batch_ms = percentile(&times_ms, 0.95);
        let p99_batch_ms = percentile(&times_ms, 0.99);
        let per_sample_ms = mean_batch_ms / (bs as f64);
        let throughput_samples_per_sec = 1000.0 / per_sample_ms;

        println!(
            "  [batch={bs:3}] mean={mean_batch_ms:.3}ms p50={p50_batch_ms:.3}ms p95={p95_batch_ms:.3}ms p99={p99_batch_ms:.3}ms | per_sample={per_sample_ms:.4}ms | throughput={throughput_samples_per_sec:.1} img/s | panthor_gpu={panthor_gpu_per_batch_ms:.3}ms/batch"
        );

        batch_benchmarks.push(BatchBenchResult {
            batch_size: bs,
            iterations: iters,
            warmup_iterations: warmup,
            mean_batch_ms,
            p50_batch_ms,
            p95_batch_ms,
            p99_batch_ms,
            per_sample_ms,
            throughput_samples_per_sec,
            panthor_gpu_total_ms,
            panthor_gpu_per_batch_ms,
        });
    }

    Ok(BackendBenchReport {
        backend_requested: kind.as_str().to_string(),
        backend_resolved: backend_desc,
        model_load_ms,
        single_sample_warmup_ms,
        test_set_samples,
        test_set_accuracy,
        test_set_warmup_ms,
        test_set_eval_ms,
        png_samples_total,
        png_samples_correct,
        png_samples_accuracy,
        batch_benchmarks,
        png_predictions,
    })
}

pub fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}
