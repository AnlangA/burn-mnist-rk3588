// SPDX-License-Identifier: GPL-3.0-only
//! Parse the kernel's native-endian DRM IN_FORMATS blob without unaligned casts.

pub(crate) fn modifiers_for_format(blob: &[u8], format: u32) -> Option<Vec<u64>> {
    let u32_at = |offset: usize| {
        Some(u32::from_ne_bytes(
            blob.get(offset..offset.checked_add(4)?)?.try_into().ok()?,
        ))
    };
    let u64_at = |offset: usize| {
        Some(u64::from_ne_bytes(
            blob.get(offset..offset.checked_add(8)?)?.try_into().ok()?,
        ))
    };
    if u32_at(0)? != 1 || u32_at(4)? != 0 {
        return None;
    }
    let count = u32_at(8)? as usize;
    let formats = u32_at(12)? as usize;
    let modifier_count = u32_at(16)? as usize;
    let modifiers = u32_at(20)? as usize;
    blob.get(formats..formats.checked_add(count.checked_mul(4)?)?)?;
    blob.get(modifiers..modifiers.checked_add(modifier_count.checked_mul(24)?)?)?;
    let Some(index) = (0..count).find(|i| u32_at(formats + i * 4) == Some(format)) else {
        return Some(Vec::new());
    };
    let mut result = Vec::new();
    for i in 0..modifier_count {
        let record = modifiers + i * 24;
        let mask = u64_at(record)?;
        let offset = u32_at(record + 8)? as usize;
        if index >= offset && index - offset < 64 && mask & (1 << (index - offset)) != 0 {
            result.push(u64_at(record + 16)?);
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob() -> Vec<u8> {
        let mut b = Vec::new();
        for value in [1_u32, 0, 2, 24, 2, 32, 0x34325258, 0x36314752] {
            b.extend(value.to_ne_bytes());
        }
        for (mask, modifier) in [(1_u64, 0x0800000000000011_u64), (2, 0)] {
            b.extend(mask.to_ne_bytes());
            b.extend(0_u32.to_ne_bytes());
            b.extend(0_u32.to_ne_bytes());
            b.extend(modifier.to_ne_bytes());
        }
        b
    }

    #[test]
    fn does_not_assume_rgb_is_linear() {
        assert_eq!(
            modifiers_for_format(&blob(), 0x34325258),
            Some(vec![0x0800000000000011])
        );
        assert_eq!(modifiers_for_format(&blob(), 0x36314752), Some(vec![0]));
        assert_eq!(modifiers_for_format(&blob(), 0), Some(vec![]));
    }

    #[test]
    fn rejects_truncated_or_invalid_layouts() {
        let valid = blob();
        for length in 0..valid.len() {
            assert!(modifiers_for_format(&valid[..length], 0x34325258).is_none());
        }
        let mut invalid = valid;
        invalid[12..16].copy_from_slice(&u32::MAX.to_ne_bytes());
        assert!(modifiers_for_format(&invalid, 0x34325258).is_none());
    }
}
