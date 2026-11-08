#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Runtime script for Google Colab / Linux runtime:
# 1. Checks Google Drive cache ($GDRIVE_CACHE_DIR) for Rust + Burn environment and MNIST dataset.
# 2. If found in Drive, migrates directly to the runtime directory ($RUNTIME_DIR).
# 3. If missing in Drive, downloads them into the runtime and copies them back to Google Drive.
# 4. Automatically clones https://github.com/AnlangA/burn-mnist-rk3588.git in the runtime.
# 5. Runs Rust Burn MNIST training and exports artifacts for RK3588 CPU and GPU deployment.

set -euo pipefail

GDRIVE_CACHE_DIR="${GDRIVE_CACHE_DIR:-/content/drive/MyDrive/burn-mnist-rk3588-cache}"
RUNTIME_DIR="${RUNTIME_DIR:-/tmp/burn-mnist-runtime}"
REPO_URL="${REPO_URL:-https://github.com/AnlangA/burn-mnist-rk3588.git}"
LOCAL_REPO_OUT="${LOCAL_REPO_OUT:-}"
BACKEND="${BACKEND:-gpu}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.001}"

DRIVE_MNIST_DIR="${GDRIVE_CACHE_DIR}/mnist"
DRIVE_RUST_ENV_TAR="${GDRIVE_CACHE_DIR}/rust-burn-cargo-cache.tar.gz"
DRIVE_ARTIFACTS_DIR="${GDRIVE_CACHE_DIR}/artifacts"

RUNTIME_MNIST_DIR="${RUNTIME_DIR}/mnist"
RUNTIME_CARGO_HOME="${RUNTIME_DIR}/cargo-home"
RUNTIME_REPO_DIR="${RUNTIME_DIR}/burn-mnist-rk3588"

MNIST_FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

MNIST_BASE_URLS=(
    "https://ossci-datasets.s3.amazonaws.com/mnist"
    "https://storage.googleapis.com/cvdf-datasets/mnist"
)

log() {
    printf '[colab-runtime] %s\n' "$*"
}

mkdir -p "${GDRIVE_CACHE_DIR}" "${DRIVE_MNIST_DIR}" "${DRIVE_ARTIFACTS_DIR}"
mkdir -p "${RUNTIME_DIR}" "${RUNTIME_MNIST_DIR}"

# ---------------------------------------------------------------------------
# Step 1: Check Google Drive for MNIST dataset; migrate or download + cache
# ---------------------------------------------------------------------------
mnist_cached_in_drive=true
for f in "${MNIST_FILES[@]}"; do
    if [[ ! -s "${DRIVE_MNIST_DIR}/${f}" ]]; then
        mnist_cached_in_drive=false
        break
    fi
done

if [[ "${mnist_cached_in_drive}" == "true" ]]; then
    log "MNIST dataset found in Google Drive (${DRIVE_MNIST_DIR}). Migrating to runtime (${RUNTIME_MNIST_DIR})..."
    for f in "${MNIST_FILES[@]}"; do
        cp -f "${DRIVE_MNIST_DIR}/${f}" "${RUNTIME_MNIST_DIR}/${f}"
    done
else
    log "MNIST dataset not found in Google Drive. Downloading to runtime and caching to Google Drive..."
    for f in "${MNIST_FILES[@]}"; do
        if [[ ! -s "${RUNTIME_MNIST_DIR}/${f}" ]]; then
            downloaded=false
            for base_url in "${MNIST_BASE_URLS[@]}"; do
                log "Downloading ${base_url}/${f} ..."
                if curl -fL --retry 3 --connect-timeout 15 -o "${RUNTIME_MNIST_DIR}/${f}" "${base_url}/${f}"; then
                    downloaded=true
                    break
                fi
            done
            if [[ "${downloaded}" != "true" ]]; then
                echo "Error: Failed to download ${f}" >&2
                exit 1
            fi
        fi
        cp -f "${RUNTIME_MNIST_DIR}/${f}" "${DRIVE_MNIST_DIR}/${f}"
    done
    log "Cached MNIST dataset to Google Drive (${DRIVE_MNIST_DIR})."
fi

# ---------------------------------------------------------------------------
# Step 2: Check Google Drive for Rust + Burn environment; migrate or install
# ---------------------------------------------------------------------------
if ! command -v cargo >/dev/null 2>&1; then
    if [[ -x "${HOME}/.cargo/bin/cargo" ]]; then
        export PATH="${HOME}/.cargo/bin:${PATH}"
    fi
fi

if [[ -s "${DRIVE_RUST_ENV_TAR}" ]]; then
    log "Rust Burn environment cache found in Google Drive (${DRIVE_RUST_ENV_TAR}). Migrating to runtime..."
    mkdir -p "${RUNTIME_CARGO_HOME}"
    tar -xzf "${DRIVE_RUST_ENV_TAR}" -C "${RUNTIME_CARGO_HOME}"
    export CARGO_HOME="${RUNTIME_CARGO_HOME}"
    if [[ -d "${RUNTIME_CARGO_HOME}/bin" ]]; then
        export PATH="${RUNTIME_CARGO_HOME}/bin:${PATH}"
    fi
else
    log "Rust Burn environment not found in Google Drive. Setting up Rust toolchain and caching to Drive..."
    if ! command -v cargo >/dev/null 2>&1; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
        export PATH="${HOME}/.cargo/bin:${PATH}"
    fi
fi

log "Using rustc: $(rustc --version) | cargo: $(cargo --version)"

# ---------------------------------------------------------------------------
# Step 3: Automatically git clone the public GitHub training repository
# ---------------------------------------------------------------------------
if [[ -d "${RUNTIME_REPO_DIR}/.git" ]]; then
    log "Updating existing repository in runtime (${RUNTIME_REPO_DIR})..."
    git -C "${RUNTIME_REPO_DIR}" fetch origin
    git -C "${RUNTIME_REPO_DIR}" reset --hard origin/main
else
    log "Cloning ${REPO_URL} into ${RUNTIME_REPO_DIR}..."
    rm -rf "${RUNTIME_REPO_DIR}"
    git clone "${REPO_URL}" "${RUNTIME_REPO_DIR}"
fi

cd "${RUNTIME_REPO_DIR}"

# If Google Drive did not have the Burn cargo cache yet, fetch dependencies and archive to Drive
if [[ ! -s "${DRIVE_RUST_ENV_TAR}" ]]; then
    log "Fetching Rust Burn dependencies and archiving cargo cache to Google Drive..."
    cargo fetch
    C_HOME="${CARGO_HOME:-${HOME}/.cargo}"
    if [[ -d "${C_HOME}/registry" ]]; then
        tar -czf "${DRIVE_RUST_ENV_TAR}" -C "${C_HOME}" registry
        log "Saved Rust Burn cargo cache to Google Drive: ${DRIVE_RUST_ENV_TAR}"
    fi
fi

# ---------------------------------------------------------------------------
# Step 4: Run Rust Burn MNIST training
# ---------------------------------------------------------------------------
log "Starting Rust Burn MNIST training (backend=${BACKEND}, epochs=${EPOCHS}, batch_size=${BATCH_SIZE})..."
cargo run --release --bin train -- \
    --data-dir "${RUNTIME_MNIST_DIR}" \
    --artifact-dir "${RUNTIME_REPO_DIR}/artifacts" \
    --backend "${BACKEND}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}"

# Copy trained artifacts to Google Drive and optional local checkout
cp -f "${RUNTIME_REPO_DIR}/artifacts/"* "${DRIVE_ARTIFACTS_DIR}/"
log "Copied trained artifacts to Google Drive: ${DRIVE_ARTIFACTS_DIR}"

if [[ -n "${LOCAL_REPO_OUT}" ]]; then
    mkdir -p "${LOCAL_REPO_OUT}/artifacts"
    cp -f "${RUNTIME_REPO_DIR}/artifacts/"* "${LOCAL_REPO_OUT}/artifacts/"
    log "Downloaded trained artifacts to local repository: ${LOCAL_REPO_OUT}/artifacts"
fi

log "Runtime training pipeline completed successfully."
