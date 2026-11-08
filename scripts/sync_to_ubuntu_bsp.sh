#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Sends trained Burn MNIST artifacts to the Ubuntu server's
# /home/a/workspace/rk3588/last-rk3588/atk-dlrk3588-bsp repository,
# builds & deploys both CPU (Burn Flex) and GPU (Burn WGPU on Mali-G610 panthor)
# versions to the RK3588 board, and collects hardware benchmark results.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${REPO_ROOT}/.." && pwd)"
CONNECT_SH="${CONNECT_SH:-${WORKSPACE_DIR}/connect.sh}"
KEY_FILE="${KEY_FILE:-${WORKSPACE_DIR}/key}"
SSH_USER="${SSH_USER:-a}"
SSH_HOST="${SSH_HOST:-192.168.10.117}"
REMOTE_BSP_DIR="${REMOTE_BSP_DIR:-/home/a/workspace/rk3588/last-rk3588/atk-dlrk3588-bsp}"
REMOTE_APP_DIR="${REMOTE_BSP_DIR}/app/digit-recognizer"
REMOTE_MODEL_DIR="${REMOTE_APP_DIR}/assets/model"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/artifacts}"

if [[ ! -x "${CONNECT_SH}" ]]; then
    echo "Error: connect.sh not found or not executable at ${CONNECT_SH}" >&2
    exit 1
fi

for required in "mnist_cnn.bpk" "mnist_cnn.safetensors" "training_metrics.json"; do
    if [[ ! -s "${ARTIFACT_DIR}/${required}" ]]; then
        echo "Error: Missing trained artifact ${ARTIFACT_DIR}/${required}" >&2
        exit 1
    fi
done

log() {
    printf '[sync-bsp] %s\n' "$*"
}

upload_file() {
    local src="$1"
    local dst="$2"
    base64 < "${src}" | "${CONNECT_SH}" "base64 -d > '${dst}'"
}

log "Creating model directory on Ubuntu server: ${REMOTE_MODEL_DIR}"
"${CONNECT_SH}" "mkdir -p '${REMOTE_MODEL_DIR}' '${REMOTE_BSP_DIR}/app/burn-mnist-rk3588/src/bin'"

log "Uploading trained Burn MNIST model artifacts to ${SSH_USER}@${SSH_HOST}:${REMOTE_MODEL_DIR} ..."
upload_file "${ARTIFACT_DIR}/mnist_cnn.bpk" "${REMOTE_MODEL_DIR}/mnist_cnn.bpk"
upload_file "${ARTIFACT_DIR}/mnist_cnn.safetensors" "${REMOTE_MODEL_DIR}/mnist_cnn.safetensors"
upload_file "${ARTIFACT_DIR}/training_metrics.json" "${REMOTE_MODEL_DIR}/training_metrics.json"

log "Uploading burn-mnist-rk3588 source crate to ${REMOTE_BSP_DIR}/app/burn-mnist-rk3588 ..."
upload_file "${REPO_ROOT}/Cargo.toml" "${REMOTE_BSP_DIR}/app/burn-mnist-rk3588/Cargo.toml"
upload_file "${REPO_ROOT}/Cargo.lock" "${REMOTE_BSP_DIR}/app/burn-mnist-rk3588/Cargo.lock"
upload_file "${REPO_ROOT}/src/lib.rs" "${REMOTE_BSP_DIR}/app/burn-mnist-rk3588/src/lib.rs"
upload_file "${REPO_ROOT}/src/data.rs" "${REMOTE_BSP_DIR}/app/burn-mnist-rk3588/src/data.rs"
upload_file "${REPO_ROOT}/src/model.rs" "${REMOTE_BSP_DIR}/app/burn-mnist-rk3588/src/model.rs"
upload_file "${REPO_ROOT}/src/bin/train.rs" "${REMOTE_BSP_DIR}/app/burn-mnist-rk3588/src/bin/train.rs"
upload_file "${REPO_ROOT}/src/bin/bench.rs" "${REMOTE_BSP_DIR}/app/burn-mnist-rk3588/src/bin/bench.rs"

log "Verifying uploaded files on Ubuntu server..."
"${CONNECT_SH}" "ls -lh '${REMOTE_MODEL_DIR}' '${REMOTE_BSP_DIR}/app/burn-mnist-rk3588/src'"

log "Artifacts and crate synced successfully to ${REMOTE_BSP_DIR}."
