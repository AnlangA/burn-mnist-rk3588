pub mod data;
pub mod model;

#[cfg(test)]
mod tests {
    use crate::data::{
        MNIST_MEAN, MNIST_STD, MnistSample, MnistSplit, decode_png_bytes_as_mnist_28x28,
        normalize_pixel,
    };
    use crate::model::{BackendKind, MnistCnn, resolve_device};
    use flate2::{Compression, write::ZlibEncoder};
    use std::io::Write;

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

        let mut raw = Vec::new();
        for _ in 0..height {
            raw.push(0); // filter type 0
            for _ in 0..width {
                raw.push(fill);
            }
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
    fn batch_edge_cases_and_shapes() {
        let (device, _) = resolve_device(BackendKind::Cpu).unwrap();
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

    #[test]
    fn model_forward_and_burnpack_roundtrip() {
        let (device, _) = resolve_device(BackendKind::Cpu).unwrap();
        let model = MnistCnn::new(&device);
        let sample = [0.25f32; 784];
        let pred1 = model.predict_28x28(&sample, &device).unwrap();
        let sum: f32 = pred1.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        assert!(pred1.digit < 10);

        let tmp_path = std::env::temp_dir().join(format!(
            "test_mnist_cnn_{}.bpk",
            std::process::id()
        ));
        model.save_burnpack(&tmp_path).unwrap();
        let loaded = MnistCnn::load_burnpack_file(&tmp_path, &device).unwrap();
        let pred2 = loaded.predict_28x28(&sample, &device).unwrap();
        let _ = std::fs::remove_file(&tmp_path);

        assert_eq!(pred1.digit, pred2.digit);
        for (a, b) in pred1.probabilities.iter().zip(pred2.probabilities.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }
}
