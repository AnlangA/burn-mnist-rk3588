//! Dedicated RK3588 NPU inference and benchmark binary (`RKNPU` / `librknnrt.so` 3-Core FP16 & INT8).
use burn_mnist_rk3588::bench::{parse_arg, run_backend_bench, run_single_image_inference};
use burn_mnist_rk3588::model::BackendKind;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "Usage: mnist-npu [--model PATH] [--backend npu|npu-int8|npu-all] [--data-dir DIR]\n\
             [--images-dir DIR] [--image PNG_PATH] [--iterations N] [--output-json PATH]"
        );
        return Ok(());
    }

    let model_path = PathBuf::from(
        parse_arg(&args, "--model").unwrap_or_else(|| "artifacts/mnist_cnn.rknn".into()),
    );
    let backend_arg = parse_arg(&args, "--backend").unwrap_or_else(|| "npu-all".into());
    let data_dir = parse_arg(&args, "--data-dir").map(PathBuf::from);
    let images_dir = parse_arg(&args, "--images-dir").map(PathBuf::from);
    let single_image = parse_arg(&args, "--image").map(PathBuf::from);
    let iterations: usize = parse_arg(&args, "--iterations")
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let output_json = parse_arg(&args, "--output-json").map(PathBuf::from);

    if let Some(img_path) = single_image {
        let kind = if backend_arg == "npu-all" {
            BackendKind::Npu
        } else {
            BackendKind::parse(&backend_arg)?
        };
        let (pred, latency_ms, desc) = run_single_image_inference(kind, &model_path, &img_path)?;
        println!(
            "[{}] {} -> digit={} confidence={:.2}% latency={:.3}ms",
            desc,
            img_path.display(),
            pred.digit,
            pred.confidence * 100.0,
            latency_ms
        );
        return Ok(());
    }

    let backends: Vec<BackendKind> = if backend_arg == "npu-all" {
        vec![BackendKind::Npu, BackendKind::NpuInt8]
    } else {
        vec![BackendKind::parse(&backend_arg)?]
    };

    let mut reports = Vec::new();
    for kind in backends {
        match run_backend_bench(
            kind,
            &model_path,
            data_dir.as_deref(),
            images_dir.as_deref(),
            iterations,
        ) {
            Ok(rep) => reports.push(rep),
            Err(err) => {
                eprintln!("NPU backend {} failed: {err}", kind.as_str());
                if backend_arg != "npu-all" {
                    return Err(err.into());
                }
            }
        }
    }

    if let Some(json_path) = output_json {
        if let Some(parent) = json_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&json_path, serde_json::to_string_pretty(&reports)?)?;
        println!("Saved NPU benchmark JSON to {}", json_path.display());
    }

    Ok(())
}
