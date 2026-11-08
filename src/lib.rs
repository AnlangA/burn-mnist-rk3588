pub mod bench;
pub mod data;
pub mod model;
pub mod npu;

#[cfg(test)]
mod tests {
    use crate::bench::{digit_from_filename, percentile};
    use crate::data::{MNIST_MEAN, MNIST_STD, decode_png_bytes_as_mnist_28x28, normalize_pixel};
    #[cfg(any(feature = "cpu", feature = "gpu"))]
    use crate::data::{MnistSample, MnistSplit};
    use crate::model::BackendKind;
    #[cfg(any(feature = "cpu", feature = "gpu"))]
    use crate::model::{MnistCnn, resolve_device};
    use crate::npu::{
        RKNB_MAGIC, RknbProfileEntry, parse_rknb_bundle, prediction_from_probabilities,
        read_sysfs_u64, resolve_rknn_model_path, softmax_row_10,
    };
    use flate2::{Compression, write::ZlibEncoder};
    use std::io::Write;
    use std::path::Path;

    #[cfg(any(feature = "cpu", feature = "gpu"))]
    fn test_device() -> burn::tensor::Device {
        #[cfg(feature = "cpu")]
        {
            resolve_device(BackendKind::Cpu).unwrap().0
        }
        #[cfg(all(not(feature = "cpu"), feature = "gpu"))]
        {
            resolve_device(BackendKind::Gpu).unwrap().0
        }
    }

    fn make_grayscale_png(width: u32, height: u32, fill: u8) -> Vec<u8> {
        let mut out = vec![137, 80, 78, 71, 13, 10, 26, 10];
        let write_chunk = |out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]| {
            out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            out.extend_from_slice(tag);
            out.extend_from_slice(data);
            out.extend_from_slice(&0u32.to_be_bytes());
        };
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.push(8); // bit depth
        ihdr.push(0); // color type: grayscale
        ihdr.push(0);
        ihdr.push(0);
        ihdr.push(0);
        write_chunk(&mut out, b"IHDR", &ihdr);

        let mut raw = Vec::with_capacity((height * (width + 1)) as usize);
        for _ in 0..height {
            raw.push(0); // filter type 0
            raw.extend(std::iter::repeat_n(fill, width as usize));
        }
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&raw).unwrap();
        let idat = enc.finish().unwrap();
        write_chunk(&mut out, b"IDAT", &idat);
        write_chunk(&mut out, b"IEND", &[]);
        out
    }

    #[test]
    fn normalize_pixel_bounds() {
        let z = normalize_pixel(0);
        let o = normalize_pixel(255);
        assert!((z - (-MNIST_MEAN / MNIST_STD)).abs() < 1e-5);
        assert!((o - ((1.0 - MNIST_MEAN) / MNIST_STD)).abs() < 1e-5);
    }

    #[test]
    fn png_decoder_handles_valid_and_invalid_inputs() {
        assert!(decode_png_bytes_as_mnist_28x28(&[]).is_err());
        assert!(decode_png_bytes_as_mnist_28x28(b"not a png").is_err());

        let png = make_grayscale_png(224, 224, 255);
        let decoded = decode_png_bytes_as_mnist_28x28(&png).unwrap();
        assert_eq!(decoded.len(), 784);
        let expected = (1.0 - MNIST_MEAN) / MNIST_STD;
        for &v in &decoded {
            assert!((v - expected).abs() < 1e-4, "got {v}, expected {expected}");
        }
    }

    #[test]
    fn bench_helpers_handle_edge_cases() {
        assert_eq!(percentile(&[], 0.5), 0.0);
        assert_eq!(percentile(&[1.0, 2.0, 3.0], 0.5), 2.0);
        assert_eq!(
            digit_from_filename(Path::new("assets/images/d7_0000.png")),
            Some(7)
        );
        assert_eq!(digit_from_filename(Path::new("photo.png")), None);
    }

    #[cfg(any(feature = "cpu", feature = "gpu"))]
    #[test]
    fn batch_edge_cases_and_shapes() {
        let device = test_device();
        let split = MnistSplit {
            samples: vec![
                MnistSample {
                    pixels: [0.0; 784],
                    label: 3,
                },
                MnistSample {
                    pixels: [1.0; 784],
                    label: 7,
                },
            ],
        };
        assert!(split.batch(&[], &device).is_err());
        assert!(split.batch(&[0, 5], &device).is_err());
        let (imgs, lbls) = split.batch(&[0, 1], &device).unwrap();
        assert_eq!(imgs.dims(), [2, 1, 28, 28]);
        assert_eq!(lbls.dims(), [2]);
    }

    #[cfg(any(feature = "cpu", feature = "gpu"))]
    #[test]
    fn model_forward_and_burnpack_roundtrip() {
        let device = test_device();
        let model = MnistCnn::new(&device);
        let sample = [0.25f32; 784];
        let pred1 = model.predict_28x28(&sample, &device).unwrap();
        let sum: f32 = pred1.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        assert!(pred1.digit < 10);

        let tmp_path =
            std::env::temp_dir().join(format!("test_mnist_cnn_{}.bpk", std::process::id()));
        model.save_burnpack(&tmp_path).unwrap();
        let loaded = MnistCnn::load_burnpack_file(&tmp_path, &device).unwrap();
        let pred2 = loaded.predict_28x28(&sample, &device).unwrap();
        let _ = std::fs::remove_file(&tmp_path);

        assert_eq!(pred1.digit, pred2.digit);
        for (a, b) in pred1.probabilities.iter().zip(pred2.probabilities.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn panthor_fdinfo_deduplicates_drm_client_ids() {
        use crate::model::parse_panthor_fdinfo_dir;
        let dir = std::env::temp_dir().join(format!("test_panthor_fdinfo_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let fd_client_413 = "pos:\t0\ndrm-driver:\tpanthor\ndrm-client-id:\t413\ndrm-engine-panthor:\t250000000 ns\n";
        std::fs::write(dir.join("5"), fd_client_413).unwrap();
        std::fs::write(dir.join("6"), fd_client_413).unwrap();
        std::fs::write(dir.join("7"), fd_client_413).unwrap();

        let fd_client_414 = "pos:\t0\ndrm-driver:\tpanthor\ndrm-client-id:\t414\ndrm-engine-panthor:\t50000000 ns\n";
        std::fs::write(dir.join("8"), fd_client_414).unwrap();

        let fd_other = "pos:\t0\ndrm-driver:\trockchip\ndrm-client-id:\t12\ndrm-engine-panthor:\t999999999 ns\n";
        std::fs::write(dir.join("9"), fd_other).unwrap();

        let total_ns = parse_panthor_fdinfo_dir(&dir);
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(total_ns, 300_000_000);
    }

    #[test]
    fn npu_backend_parsing_and_bundle_roundtrip() {
        assert_eq!(BackendKind::parse("npu").unwrap(), BackendKind::Npu);
        assert_eq!(BackendKind::parse("rknn").unwrap(), BackendKind::Npu);
        assert_eq!(
            BackendKind::parse("npu-int8").unwrap(),
            BackendKind::NpuInt8
        );
        assert!(BackendKind::Npu.is_npu());
        assert!(!BackendKind::Npu.is_int8());
        assert!(BackendKind::NpuInt8.is_npu());
        assert!(BackendKind::NpuInt8.is_int8());

        // Too-small buffer rejected
        assert!(parse_rknb_bundle(b"short", false).is_err());

        // Standalone single-profile buffer fallback
        let standalone = vec![0x52u8; 64];
        let parsed_single = parse_rknb_bundle(&standalone, false).unwrap();
        assert_eq!(parsed_single.len(), 1);
        assert_eq!(parsed_single[0].batch_size, 1);
        assert_eq!(parsed_single[0].input_width, 28);
        assert!(!parsed_single[0].int8);

        // Multi-profile RKNB bundle
        let mut bundle = vec![0xAAu8; 128];
        let manifest_json = serde_json::to_vec(&serde_json::json!({
            "profiles": [
                {
                    "name": "fp16_b1",
                    "batch_size": 1,
                    "input_width": 28,
                    "int8": false,
                    "offset": 0,
                    "size": 64
                },
                {
                    "name": "int8_b6",
                    "batch_size": 6,
                    "input_width": 32,
                    "int8": true,
                    "offset": 64,
                    "size": 64
                }
            ]
        }))
        .unwrap();
        let manifest_offset = bundle.len() as u64;
        let manifest_len = manifest_json.len() as u64;
        bundle.extend_from_slice(&manifest_json);
        bundle.extend_from_slice(&manifest_offset.to_le_bytes());
        bundle.extend_from_slice(&manifest_len.to_le_bytes());
        bundle.extend_from_slice(RKNB_MAGIC);

        let profiles: Vec<RknbProfileEntry> = parse_rknb_bundle(&bundle, false).unwrap();
        assert_eq!(profiles.len(), 2);
        assert_eq!(profiles[0].name, "fp16_b1");
        assert_eq!(profiles[1].name, "int8_b6");
        assert_eq!(profiles[1].batch_size, 6);
        assert_eq!(profiles[1].input_width, 32);
        assert!(profiles[1].int8);

        // Corrupt manifest offset rejected
        let mut bad_bundle = bundle.clone();
        let len = bad_bundle.len();
        bad_bundle[len - 24..len - 16].copy_from_slice(&999999u64.to_le_bytes());
        assert!(parse_rknb_bundle(&bad_bundle, false).is_err());
    }

    #[test]
    fn npu_softmax_and_sysfs_helpers() {
        let logits = [0.0f32, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let probs = softmax_row_10(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        let pred = prediction_from_probabilities(probs);
        assert_eq!(pred.digit, 2);
        assert!(pred.confidence > 0.99);

        let nan_logits = [f32::NAN; 10];
        let fallback_probs = softmax_row_10(&nan_logits);
        assert!((fallback_probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);

        let tmp_file =
            std::env::temp_dir().join(format!("test_rknpu_sysfs_{}", std::process::id()));
        std::fs::write(&tmp_file, "800000000\n").unwrap();
        assert_eq!(read_sysfs_u64(&tmp_file), 800_000_000);
        let _ = std::fs::remove_file(&tmp_file);
        assert_eq!(read_sysfs_u64(&tmp_file), 0);

        let resolved = resolve_rknn_model_path(Path::new("artifacts/mnist_cnn.bpk"), false);
        assert!(resolved.to_string_lossy().ends_with(".rknn"));
    }

    #[test]
    fn rknn_artifact_bundle_integrity_check() {
        let rknn_path = Path::new("artifacts/mnist_cnn.rknn");
        if rknn_path.is_file() {
            let bytes = std::fs::read(rknn_path).unwrap();
            let profiles = parse_rknb_bundle(&bytes, false).unwrap();
            assert_eq!(profiles.len(), 8);
            for p in &profiles {
                assert_eq!(&bytes[p.offset..p.offset + 4], b"RKNN");
            }
        }
        let int8_path = Path::new("artifacts/mnist_cnn_int8.rknn");
        if int8_path.is_file() {
            let bytes = std::fs::read(int8_path).unwrap();
            let profiles = parse_rknb_bundle(&bytes, true).unwrap();
            assert_eq!(profiles.len(), 1);
            assert_eq!(profiles[0].batch_size, 1);
            assert!(profiles[0].int8);
            assert_eq!(&bytes[0..4], b"RKNN");
        }
    }
}
