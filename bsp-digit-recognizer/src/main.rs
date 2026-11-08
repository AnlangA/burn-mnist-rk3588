// SPDX-License-Identifier: GPL-3.0-only
//! Digit recognizer showcase: pick a handwritten sample, preview it and
//! show recognition probabilities powered by the trained Burn CNN model
//! (supporting both CPU Flex and GPU WGPU Mali-G610 panthor backends).

mod model;

use burn_mnist_rk3588::model::BackendKind;
use slint::{ComponentHandle, Model, ModelRc, VecModel};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

slint::include_modules!();

const DEFAULT_IMAGE_DIRS: &[&str] = &[
    "/usr/share/digit-recognizer/images",
    "/opt/burn-mnist-rk3588/images",
    "assets/images",
];

/// Where SIGUSR1 screenshots go when no explicit path is configured; the
/// systemd unit provides this writable RuntimeDirectory.
const DEFAULT_CAPTURE_PATH: &str = "/run/digit-recognizer/screenshot.png";

fn print_help() {
    println!(
        "digit-recognizer [--images-dir DIR] [--backend cpu|gpu|vulkan|gl|npu|npu-int8] [--benchmark] [--capture FILE.png] [--help]\n\
         Handwritten digit recognition showcase for the ATK-DLRK3588.\n\
         Renders on the LinuxKMS/FemtoVG stack like slint-dashboard and runs\n\
         live Burn CNN inference on CPU (Flex SIMD+Rayon) or GPU (WGPU Mali-G610).\n\
         --capture saves one GPU screenshot shortly after start-up and exits.\n\
         SIGUSR1 saves a screenshot to $DIGIT_RECOGNIZER_CAPTURE (default {DEFAULT_CAPTURE_PATH})."
    );
}

fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a String> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|i| args.get(i + 1))
}

fn image_dir(args: &[String]) -> PathBuf {
    if let Some(dir) = arg_value(args, "--images-dir") {
        return PathBuf::from(dir);
    }
    DEFAULT_IMAGE_DIRS
        .iter()
        .map(PathBuf::from)
        .find(|dir| dir.is_dir())
        .unwrap_or_else(|| PathBuf::from("assets/images"))
}

fn parse_backend_arg(args: &[String]) -> Result<Option<BackendKind>, String> {
    match arg_value(args, "--backend") {
        Some(val) => BackendKind::parse(val).map(Some),
        None => Ok(None),
    }
}

fn png_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|x| x.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("png"))
        })
        .collect();
    entries.sort();
    Ok(entries)
}

fn run_headless_benchmark(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let entries = png_files(dir)?;

    let live = model::model_live();
    let backend = model::backend_label();
    println!(
        "[digit-recognizer] model_live={} backend=\"{}\" images_dir={}",
        live,
        backend,
        dir.display()
    );
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut total_ms = 0.0f64;

    for path in &entries {
        let t0 = Instant::now();
        let pred = model::predict(path);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let expected = model::digit_from_name(path);
        let ok = expected == Some(pred.digit);
        if ok {
            correct += 1;
        }
        total += 1;
        total_ms += elapsed_ms;
        println!(
            "  {} -> predicted={} expected={:?} conf={:.2}% latency={:.3}ms",
            path.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
            pred.digit,
            expected,
            pred.probabilities[pred.digit] * 100.0,
            elapsed_ms
        );
    }
    let avg_ms = if total > 0 {
        total_ms / (total as f64)
    } else {
        0.0
    };
    println!(
        "[digit-recognizer] Summary: accuracy={}/{} ({:.1}%), mean_latency={:.3}ms/img",
        correct,
        total,
        if total > 0 {
            100.0 * (correct as f64) / (total as f64)
        } else {
            0.0
        },
        avg_ms
    );
    Ok(())
}

fn load_samples(dir: &Path) -> Vec<DigitSample> {
    let entries = match png_files(dir) {
        Ok(entries) => entries,
        Err(err) => {
            eprintln!("cannot read image dir {}: {err}", dir.display());
            return Vec::new();
        }
    };
    entries
        .into_iter()
        .map(|path| {
            let name = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("sample");
            let thumb = slint::Image::load_from_path(&path).unwrap_or_default();
            DigitSample {
                name: name.into(),
                path: path.display().to_string().into(),
                thumb,
                label: model::digit_from_name(&path).map_or(-1, |d| d as i32),
            }
        })
        .collect()
}

/// Compact badge text for the header chip; the full description stays in
/// the latency line.
fn short_backend(desc: &str) -> String {
    let lower = desc.to_ascii_lowercase();
    if lower.contains("npu") || lower.contains("rknn") {
        if lower.contains("int8") {
            "NPU · RKNPU INT8".into()
        } else {
            "NPU · RKNPU".into()
        }
    } else if lower.contains("cpu") || lower.contains("flex") {
        "CPU · Burn Flex".into()
    } else if lower.contains("wgpu")
        || lower.contains("vulkan")
        || lower.contains("opengl")
        || lower.contains("gles")
        || lower.contains("gpu")
    {
        "GPU · Burn WGPU".into()
    } else {
        desc.into()
    }
}

/// Shared UI state: the sample list and the persistent probability model
/// (updated in place so the bars animate instead of being rebuilt).
struct App {
    samples: Vec<DigitSample>,
    probabilities: Rc<VecModel<DigitProbability>>,
}

impl App {
    fn index_of(&self, path: &str) -> Option<usize> {
        self.samples.iter().position(|s| s.path == path)
    }

    fn apply_prediction(&self, ui: &DigitRecognizer, path: &Path) {
        let t0 = Instant::now();
        let prediction = model::predict(path);
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
        for (digit, probability) in prediction.probabilities.iter().enumerate() {
            let row = DigitProbability {
                digit: digit as i32,
                probability: *probability,
            };
            if digit < self.probabilities.row_count() {
                self.probabilities.set_row_data(digit, row);
            } else {
                self.probabilities.push(row);
            }
        }
        ui.set_predicted(prediction.digit as i32);
        let confidence = prediction.probabilities[prediction.digit];
        ui.set_confidence(confidence);
        let position = self
            .index_of(path.to_string_lossy().as_ref())
            .map(|i| format!("第 {}/{} 张 · ", i + 1, self.samples.len()))
            .unwrap_or_default();
        if model::model_live() {
            let full_label = model::backend_label();
            let lower = full_label.to_ascii_lowercase();
            let card_backend = if lower.contains("npu") || lower.contains("rknn") {
                if lower.contains("int8") {
                    "RK3588 RKNPU (INT8)"
                } else {
                    "RK3588 RKNPU (FP16)"
                }
            } else {
                full_label
            };
            ui.set_latency(format!("{card_backend} · {latency_ms:.2} ms").into());
            ui.set_status(
                format!(
                    "{position}识别完成 · 推理 {latency_ms:.2} ms · 置信度 {:.1}%",
                    confidence * 100.0
                )
                .into(),
            );
        } else {
            ui.set_latency("演示输出 · 非真实推理".into());
            ui.set_status(format!("{position}演示输出(假模型)· 真实模型接入前仅供参考").into());
        }
    }

    fn select(&self, ui: &DigitRecognizer, path: &str) {
        let Some(sample) = self.samples.iter().find(|s| s.path == path) else {
            return;
        };
        let file = PathBuf::from(sample.path.as_str());
        if !sample_usable(sample) {
            ui.set_status(format!("无法加载 {}", file.display()).into());
            return;
        }
        // The sample image is already decoded at full resolution; reuse it.
        ui.set_current_image(sample.thumb.clone());
        ui.set_current_path(sample.path.clone());
        ui.set_current_name(sample.name.clone());
        ui.set_truth(sample.label);
        self.apply_prediction(ui, &file);
    }

    fn step(&self, ui: &DigitRecognizer, delta: i32) {
        let current = self.index_of(ui.get_current_path().as_str());
        let Some(next) = step_index(self.samples.len(), current, delta, |i| {
            sample_usable(&self.samples[i])
        }) else {
            ui.set_status("没有可用的样本图片".into());
            return;
        };
        let path = self.samples[next].path.clone();
        self.select(ui, path.as_str());
    }
}

fn sample_usable(sample: &DigitSample) -> bool {
    sample.thumb.size().width > 0
}

/// Index reached by moving `delta` samples (wrapping) from `current`,
/// skipping samples for which `usable` is false so one unreadable PNG
/// cannot trap prev/next/auto-play. Without a current sample, starts at
/// the first usable one (or the last one for a negative `delta`).
fn step_index(
    len: usize,
    current: Option<usize>,
    delta: i32,
    usable: impl Fn(usize) -> bool,
) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let dir: i64 = if delta < 0 { -1 } else { 1 };
    let len_i = len as i64;
    let start = match current {
        Some(cur) => (cur as i64 + i64::from(delta)).rem_euclid(len_i),
        None if dir < 0 => len_i - 1,
        None => 0,
    };
    (0..len_i)
        .map(|k| (start + k * dir).rem_euclid(len_i) as usize)
        .find(|&i| usable(i))
}

fn capture_png(window: &slint::Window, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let pixels = window.take_snapshot()?;
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut encoder = png::Encoder::new(file, pixels.width(), pixels.height());
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(pixels.as_bytes())?;
    writer.finish()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return Ok(());
    }
    if let Some(kind) = parse_backend_arg(&args)? {
        model::set_preferred_backend(kind);
    }
    let dir = image_dir(&args);
    if args
        .iter()
        .any(|arg| arg == "--benchmark" || arg == "--bench")
    {
        return run_headless_benchmark(&dir);
    }
    let capture_once = arg_value(&args, "--capture").cloned();
    slint::BackendSelector::new()
        .backend_name("linuxkms".into())
        .renderer_name("femtovg".into())
        .require_opengl_es_with_version(3, 0)
        .select()?;
    let quit = Arc::new(AtomicBool::new(false));
    for signal in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
        signal_hook::flag::register(signal, quit.clone())?;
    }
    let capture_request = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGUSR1, capture_request.clone())?;

    let ui = DigitRecognizer::new()?;
    let app = Rc::new(App {
        samples: load_samples(&dir),
        probabilities: Rc::new(VecModel::from(
            (0..10)
                .map(|digit| DigitProbability {
                    digit,
                    probability: 0.0,
                })
                .collect::<Vec<_>>(),
        )),
    });
    ui.set_samples(ModelRc::new(VecModel::from(app.samples.clone())));
    ui.set_probabilities(ModelRc::from(app.probabilities.clone()));
    let live = model::model_live();
    eprintln!(
        "[digit-recognizer] model_live={} backend=\"{}\" badge=\"{}\"",
        live,
        model::backend_label(),
        short_backend(model::backend_label())
    );
    ui.set_model_live(live);
    if live {
        ui.set_backend(short_backend(model::backend_label()).into());
    }
    if app.samples.is_empty() {
        ui.set_status(format!("目录 {} 中没有 PNG 图片", dir.display()).into());
    } else {
        app.step(&ui, 1);
    }

    let weak = ui.as_weak();
    let app_for_select = app.clone();
    ui.on_select_sample(move |path| {
        if let Some(ui) = weak.upgrade() {
            app_for_select.select(&ui, path.as_str());
        }
    });
    let weak = ui.as_weak();
    let app_for_step = app.clone();
    ui.on_step_sample(move |delta| {
        if let Some(ui) = weak.upgrade() {
            app_for_step.step(&ui, delta);
        }
    });
    let weak = ui.as_weak();
    let app_for_run = app.clone();
    ui.on_run_recognition(move || {
        let Some(ui) = weak.upgrade() else {
            return;
        };
        let path = ui.get_current_path();
        if path.is_empty() {
            ui.set_status("请先选择一张图片".into());
        } else {
            app_for_run.apply_prediction(&ui, &PathBuf::from(path.as_str()));
        }
    });

    // Screenshots are taken right after a real GPU frame so they show
    // exactly what is on the panel.
    let capture_path = capture_once.clone().unwrap_or_else(|| {
        std::env::var("DIGIT_RECOGNIZER_CAPTURE").unwrap_or_else(|_| DEFAULT_CAPTURE_PATH.into())
    });
    let render_capture = capture_request.clone();
    let quit_after_capture = capture_once.is_some();
    let weak = ui.as_weak();
    ui.window()
        .set_rendering_notifier(move |state, _api| {
            if !matches!(state, slint::RenderingState::AfterRendering)
                || !render_capture.swap(false, Ordering::Relaxed)
            {
                return;
            }
            let Some(ui) = weak.upgrade() else {
                return;
            };
            match capture_png(ui.window(), &capture_path) {
                Ok(()) => eprintln!("GPU_CAPTURE: {capture_path}"),
                Err(err) => eprintln!("GPU_CAPTURE_FAILED: {err}"),
            }
            if quit_after_capture {
                let _ = slint::quit_event_loop();
            }
        })
        .map_err(|err| format!("cannot install rendering notifier: {err:?}"))?;

    // `--capture`: give the bar animations time to settle before the shot.
    let capture_timer = slint::Timer::default();
    if capture_once.is_some() {
        let request = capture_request.clone();
        capture_timer.start(
            slint::TimerMode::SingleShot,
            Duration::from_millis(1500),
            move || request.store(true, Ordering::Relaxed),
        );
    }

    let weak = ui.as_weak();
    let quit_for_timer = quit.clone();
    let capture_for_timer = capture_request.clone();
    let quit_timer = slint::Timer::default();
    quit_timer.start(
        slint::TimerMode::Repeated,
        Duration::from_millis(250),
        move || {
            let Some(ui) = weak.upgrade() else {
                return;
            };
            if quit_for_timer.load(Ordering::Relaxed) {
                ui.hide().ok();
                slint::quit_event_loop().ok();
            } else if capture_for_timer.load(Ordering::Relaxed) {
                // An idle UI renders nothing; force a frame for the capture.
                ui.window().request_redraw();
            }
        },
    );
    ui.run()?;
    drop(quit_timer);
    drop(capture_timer);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_wraps_in_both_directions() {
        let all = |_: usize| true;
        assert_eq!(step_index(5, Some(4), 1, all), Some(0));
        assert_eq!(step_index(5, Some(0), -1, all), Some(4));
        assert_eq!(step_index(5, Some(2), 1, all), Some(3));
        assert_eq!(step_index(5, Some(2), -1, all), Some(1));
        assert_eq!(step_index(1, Some(0), 1, all), Some(0));
    }

    #[test]
    fn step_without_current_starts_at_an_end() {
        let all = |_: usize| true;
        assert_eq!(step_index(5, None, 1, all), Some(0));
        assert_eq!(step_index(5, None, -1, all), Some(4));
        assert_eq!(step_index(0, None, 1, all), None);
    }

    #[test]
    fn step_skips_unreadable_samples() {
        // Sample 1 and 2 failed to decode: next from 0 must reach 3 instead
        // of getting stuck on 1, and prev from 3 must reach 0.
        let usable = |i: usize| i != 1 && i != 2;
        assert_eq!(step_index(5, Some(0), 1, usable), Some(3));
        assert_eq!(step_index(5, Some(3), -1, usable), Some(0));
        assert_eq!(step_index(5, None, 1, |i| i >= 2), Some(2));
        assert_eq!(step_index(3, Some(0), 1, |_| false), None);
    }
}
