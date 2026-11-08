use burn::prelude::*;
use burn::tensor::{Device, Int, TensorData};
use flate2::read::{GzDecoder, ZlibDecoder};
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub const MNIST_MEAN: f32 = 0.1307;
pub const MNIST_STD: f32 = 0.3081;

#[derive(Clone, Debug)]
pub struct MnistSample {
    /// Normalized 28x28 grayscale pixels: `(p / 255.0 - 0.1307) / 0.3081`
    pub pixels: [f32; 784],
    pub label: u8,
}

#[derive(Clone, Debug)]
pub struct MnistSplit {
    pub samples: Vec<MnistSample>,
}

impl MnistSplit {
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Extract a mini-batch `[batch_size, 1, 28, 28]` of images and `[batch_size]` of labels.
    pub fn batch(
        &self,
        indices: &[usize],
        device: &Device,
    ) -> Result<(Tensor<4>, Tensor<1, Int>), String> {
        if indices.is_empty() {
            return Err("batch indices cannot be empty".into());
        }
        let batch_size = indices.len();
        let mut image_floats = Vec::with_capacity(batch_size * 784);
        let mut label_ints = Vec::with_capacity(batch_size);

        for &idx in indices {
            let sample = self.samples.get(idx).ok_or_else(|| {
                format!(
                    "sample index {idx} out of bounds (len={})",
                    self.samples.len()
                )
            })?;
            image_floats.extend_from_slice(&sample.pixels);
            label_ints.push(i64::from(sample.label));
        }

        let images = Tensor::<4>::from_data(
            TensorData::new(image_floats, [batch_size, 1, 28, 28]),
            device,
        );
        let targets =
            Tensor::<1, Int>::from_data(TensorData::new(label_ints, [batch_size]), device);
        Ok((images, targets))
    }
}

#[inline]
pub fn normalize_pixel(byte: u8) -> f32 {
    ((f32::from(byte) / 255.0) - MNIST_MEAN) / MNIST_STD
}

#[inline]
pub fn normalize_unit_float(val_0_to_1: f32) -> f32 {
    (val_0_to_1.clamp(0.0, 1.0) - MNIST_MEAN) / MNIST_STD
}

fn read_maybe_gz(path_gz: &Path, path_raw: &Path) -> Result<Vec<u8>, String> {
    if path_gz.is_file() {
        let file = File::open(path_gz)
            .map_err(|e| format!("failed to open {}: {e}", path_gz.display()))?;
        let mut decoder = GzDecoder::new(file);
        let mut buf = Vec::new();
        decoder
            .read_to_end(&mut buf)
            .map_err(|e| format!("failed to gunzip {}: {e}", path_gz.display()))?;
        Ok(buf)
    } else if path_raw.is_file() {
        std::fs::read(path_raw).map_err(|e| format!("failed to read {}: {e}", path_raw.display()))
    } else {
        Err(format!(
            "missing MNIST file: checked {} and {}",
            path_gz.display(),
            path_raw.display()
        ))
    }
}

fn read_be_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| format!("unexpected EOF reading u32 at offset {offset}"))?;
    Ok(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

pub fn load_idx_split(
    dir: &Path,
    images_stem: &str,
    labels_stem: &str,
    max_samples: Option<usize>,
) -> Result<MnistSplit, String> {
    let img_gz = dir.join(format!("{images_stem}.gz"));
    let img_raw = dir.join(images_stem);
    let lbl_gz = dir.join(format!("{labels_stem}.gz"));
    let lbl_raw = dir.join(labels_stem);

    let img_bytes = read_maybe_gz(&img_gz, &img_raw)?;
    let lbl_bytes = read_maybe_gz(&lbl_gz, &lbl_raw)?;

    if img_bytes.len() < 16 {
        return Err("image IDX file shorter than 16-byte header".into());
    }
    if lbl_bytes.len() < 8 {
        return Err("label IDX file shorter than 8-byte header".into());
    }

    let img_magic = read_be_u32(&img_bytes, 0)?;
    let img_count = read_be_u32(&img_bytes, 4)? as usize;
    let rows = read_be_u32(&img_bytes, 8)? as usize;
    let cols = read_be_u32(&img_bytes, 12)? as usize;

    if img_magic != 2051 {
        return Err(format!(
            "invalid MNIST image magic {img_magic}, expected 2051"
        ));
    }
    if rows != 28 || cols != 28 {
        return Err(format!(
            "unexpected MNIST dimensions {rows}x{cols}, expected 28x28"
        ));
    }

    let lbl_magic = read_be_u32(&lbl_bytes, 0)?;
    let lbl_count = read_be_u32(&lbl_bytes, 4)? as usize;
    if lbl_magic != 2049 {
        return Err(format!(
            "invalid MNIST label magic {lbl_magic}, expected 2049"
        ));
    }
    if img_count != lbl_count {
        return Err(format!(
            "MNIST count mismatch: {img_count} images vs {lbl_count} labels"
        ));
    }

    let expected_img_bytes = 16 + img_count * 784;
    let expected_lbl_bytes = 8 + lbl_count;
    if img_bytes.len() != expected_img_bytes {
        return Err(format!(
            "image payload size mismatch: got {}, expected {expected_img_bytes}",
            img_bytes.len()
        ));
    }
    if lbl_bytes.len() != expected_lbl_bytes {
        return Err(format!(
            "label payload size mismatch: got {}, expected {expected_lbl_bytes}",
            lbl_bytes.len()
        ));
    }

    let limit = max_samples.unwrap_or(img_count).min(img_count);
    let mut samples = Vec::with_capacity(limit);
    for i in 0..limit {
        let img_offset = 16 + i * 784;
        let raw_pixels = &img_bytes[img_offset..img_offset + 784];
        let mut pixels = [0.0f32; 784];
        for (dst, &b) in pixels.iter_mut().zip(raw_pixels.iter()) {
            *dst = normalize_pixel(b);
        }
        let label = lbl_bytes[8 + i];
        if label >= 10 {
            return Err(format!("invalid MNIST label {label} at index {i}"));
        }
        samples.push(MnistSample { pixels, label });
    }

    Ok(MnistSplit { samples })
}

pub fn load_train_split(dir: &Path, max_samples: Option<usize>) -> Result<MnistSplit, String> {
    load_idx_split(
        dir,
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        max_samples,
    )
}

pub fn load_test_split(dir: &Path, max_samples: Option<usize>) -> Result<MnistSplit, String> {
    load_idx_split(
        dir,
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
        max_samples,
    )
}

fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let a = i32::from(a);
    let b = i32::from(b);
    let c = i32::from(c);
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
    }
}

/// Decode an 8-bit PNG file (grayscale, RGB, gray+alpha, or RGBA) and downsample
/// it to a normalized 28x28 MNIST input array `[f32; 784]`.
pub fn load_png_as_mnist_28x28(path: &Path) -> Result<[f32; 784], String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("failed to read PNG {}: {e}", path.display()))?;
    decode_png_bytes_as_mnist_28x28(&bytes)
}

pub fn decode_png_bytes_as_mnist_28x28(bytes: &[u8]) -> Result<[f32; 784], String> {
    const PNG_SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
    if bytes.len() < 33 || bytes[..8] != PNG_SIG {
        return Err("invalid PNG signature".into());
    }

    let mut offset = 8usize;
    let mut width = 0usize;
    let mut height = 0usize;
    let mut bit_depth = 0u8;
    let mut color_type = 0u8;
    let mut idat = Vec::new();
    let mut seen_ihdr = false;
    let mut seen_iend = false;

    while offset + 8 <= bytes.len() {
        let length = read_be_u32(bytes, offset)? as usize;
        let chunk_type = &bytes[offset + 4..offset + 8];
        let data_start = offset + 8;
        let data_end = data_start
            .checked_add(length)
            .ok_or_else(|| "PNG chunk length overflow".to_string())?;
        let crc_end = data_end
            .checked_add(4)
            .ok_or_else(|| "PNG chunk CRC overflow".to_string())?;
        if crc_end > bytes.len() {
            return Err("truncated PNG chunk".into());
        }
        let chunk = &bytes[data_start..data_end];
        match chunk_type {
            b"IHDR" => {
                if chunk.len() < 13 {
                    return Err("IHDR chunk too short".into());
                }
                width = read_be_u32(chunk, 0)? as usize;
                height = read_be_u32(chunk, 4)? as usize;
                bit_depth = chunk[8];
                color_type = chunk[9];
                let interlace = chunk[12];
                if interlace != 0 {
                    return Err("interlaced PNG is not supported".into());
                }
                seen_ihdr = true;
            }
            b"IDAT" => {
                idat.extend_from_slice(chunk);
            }
            b"IEND" => {
                seen_iend = true;
                break;
            }
            _ => {}
        }
        offset = crc_end;
    }

    if !seen_ihdr || !seen_iend || width == 0 || height == 0 {
        return Err("incomplete PNG image".into());
    }
    if bit_depth != 8 {
        return Err(format!("unsupported PNG bit depth {bit_depth}, expected 8"));
    }
    let channels: usize = match color_type {
        0 => 1, // Grayscale
        2 => 3, // RGB
        4 => 2, // Grayscale + Alpha
        6 => 4, // RGBA
        other => return Err(format!("unsupported PNG color type {other}")),
    };

    let mut decoder = ZlibDecoder::new(idat.as_slice());
    let mut raw = Vec::new();
    decoder
        .read_to_end(&mut raw)
        .map_err(|e| format!("failed to inflate PNG IDAT: {e}"))?;

    let stride = width * channels;
    let expected_len = height * (1 + stride);
    if raw.len() != expected_len {
        return Err(format!(
            "decompressed PNG size mismatch: got {}, expected {expected_len}",
            raw.len()
        ));
    }

    let mut recon = vec![0u8; height * stride];
    for y in 0..height {
        let filter = raw[y * (1 + stride)];
        let src_row = &raw[y * (1 + stride) + 1..(y + 1) * (1 + stride)];
        let row_start = y * stride;
        for i in 0..stride {
            let x_byte = src_row[i];
            let a = if i >= channels {
                recon[row_start + i - channels]
            } else {
                0
            };
            let b = if y > 0 {
                recon[(y - 1) * stride + i]
            } else {
                0
            };
            let c = if y > 0 && i >= channels {
                recon[(y - 1) * stride + i - channels]
            } else {
                0
            };
            recon[row_start + i] = match filter {
                0 => x_byte,
                1 => x_byte.wrapping_add(a),
                2 => x_byte.wrapping_add(b),
                3 => x_byte.wrapping_add(((u16::from(a) + u16::from(b)) / 2) as u8),
                4 => x_byte.wrapping_add(paeth_predictor(a, b, c)),
                other => return Err(format!("invalid PNG row filter {other}")),
            };
        }
    }

    // Convert to 28x28 via area/box averaging over [0..width] x [0..height].
    let mut out = [0.0f32; 784];
    for out_y in 0..28 {
        let y0 = out_y * height / 28;
        let y1 = ((out_y + 1) * height / 28).max(y0 + 1).min(height);
        for out_x in 0..28 {
            let x0 = out_x * width / 28;
            let x1 = ((out_x + 1) * width / 28).max(x0 + 1).min(width);
            let mut sum = 0.0f32;
            let mut count = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = y * stride + x * channels;
                    let luma = match channels {
                        1 | 2 => f32::from(recon[idx]) / 255.0,
                        3 | 4 => {
                            let r = f32::from(recon[idx]);
                            let g = f32::from(recon[idx + 1]);
                            let b = f32::from(recon[idx + 2]);
                            (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                        }
                        _ => unreachable!(),
                    };
                    sum += luma;
                    count += 1;
                }
            }
            let avg = if count > 0 { sum / (count as f32) } else { 0.0 };
            out[out_y * 28 + out_x] = normalize_unit_float(avg);
        }
    }

    Ok(out)
}
