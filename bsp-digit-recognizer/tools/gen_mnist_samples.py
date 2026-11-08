#!/usr/bin/env python3
"""Regenerate the MNIST sample PNGs used by the digit-recognizer app.

Reads the gzipped MNIST test set (idx format) from the directory named by
MNIST_SRC (default /tmp/mnist-data) and writes the first two test images
of each digit, upscaled 8x with nearest-neighbour to keep the pixel look,
to the output directory given as the first argument.

Download the source data first (kept out of the repository on purpose):

    curl -L -o MNIST_SRC/t10k-images-idx3-ubyte.gz \\
        https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
    curl -L -o MNIST_SRC/t10k-labels-idx1-ubyte.gz \\
        https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
"""
import gzip
import os
import pathlib
import struct
import sys

from PIL import Image

PER_DIGIT = 2
SCALE = 8  # 28 * 8 = 224 px

src = pathlib.Path(os.environ.get("MNIST_SRC", "/tmp/mnist-data"))

with gzip.open(src / "t10k-images-idx3-ubyte.gz") as f:
    magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
    assert magic == 2051, f"bad image magic {magic}"
    images = f.read()
with gzip.open(src / "t10k-labels-idx1-ubyte.gz") as f:
    magic, label_count = struct.unpack(">II", f.read(8))
    assert magic == 2049, f"bad label magic {magic}"
    labels = f.read()
assert count == label_count and len(images) == count * rows * cols

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "assets/images")
out.mkdir(parents=True, exist_ok=True)
for old in out.glob("d*_*.png"):
    old.unlink()

seen: dict[int, int] = {}
for i in range(count):
    label = labels[i]
    if seen.get(label, 0) >= PER_DIGIT:
        if len(seen) == 10 and all(v >= PER_DIGIT for v in seen.values()):
            break
        continue
    pixels = images[i * rows * cols : (i + 1) * rows * cols]
    img = Image.frombytes("L", (cols, rows), pixels)
    img = img.resize((cols * SCALE, rows * SCALE), Image.NEAREST)
    img.save(out / f"d{label}_{i:04d}.png")
    seen[label] = seen.get(label, 0) + 1

total = sum(seen.values())
print(f"saved {total} images to {out}")
assert total == 10 * PER_DIGIT
