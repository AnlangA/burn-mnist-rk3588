#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Sends trained Burn MNIST artifacts and crate sources to the Ubuntu server's
# /home/a/workspace/rk3588/last-rk3588/atk-dlrk3588-bsp repository using
# expect + scp (avoiding piping large files over ssh/expect).
# Optionally (--deploy-and-bench) builds & deploys both CPU (Burn Flex) and
# GPU (Burn WGPU on Mali-G610 panthor) versions to the RK3588 board and
# collects hardware benchmark results.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${REPO_ROOT}/.." && pwd)"
KEY_FILE="${KEY_FILE:-${WORKSPACE_DIR}/key}"
SSH_USER="${SSH_USER:-a}"
SSH_HOST="${SSH_HOST:-192.168.10.117}"
REMOTE_BSP_DIR="${REMOTE_BSP_DIR:-/home/a/workspace/rk3588/last-rk3588/atk-dlrk3588-bsp}"
REMOTE_APP_DIR="${REMOTE_BSP_DIR}/app/digit-recognizer"
REMOTE_MODEL_DIR="${REMOTE_APP_DIR}/assets/model"
REMOTE_CRATE_DIR="${REMOTE_BSP_DIR}/app/burn-mnist-rk3588"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/artifacts}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/mnist}"

if [[ ! -f "${KEY_FILE}" ]]; then
    echo "Error: Password file not found at ${KEY_FILE}" >&2
    exit 1
fi

export SSH_USER SSH_HOST
export SSH_PASS
SSH_PASS="$(tr -d '\r\n' < "${KEY_FILE}")"

for required in "mnist_cnn.bpk" "mnist_cnn.safetensors" "training_metrics.json"; do
    if [[ ! -s "${ARTIFACT_DIR}/${required}" ]]; then
        echo "Error: Missing trained artifact ${ARTIFACT_DIR}/${required}" >&2
        exit 1
    fi
done

if command -v shasum >/dev/null 2>&1; then
    LOCAL_BPK_SHA1="$(shasum -a 1 "${ARTIFACT_DIR}/mnist_cnn.bpk" | awk '{print $1}')"
    LOCAL_SAFE_SHA1="$(shasum -a 1 "${ARTIFACT_DIR}/mnist_cnn.safetensors" | awk '{print $1}')"
else
    LOCAL_BPK_SHA1="$(sha1sum "${ARTIFACT_DIR}/mnist_cnn.bpk" | awk '{print $1}')"
    LOCAL_SAFE_SHA1="$(sha1sum "${ARTIFACT_DIR}/mnist_cnn.safetensors" | awk '{print $1}')"
fi

log() {
    printf '[sync-bsp] %s\n' "$*"
}

log "Local artifact SHA1: mnist_cnn.bpk=${LOCAL_BPK_SHA1}, mnist_cnn.safetensors=${LOCAL_SAFE_SHA1}"

remote_ssh() {
    export REMOTE_CMD="$*"
    /usr/bin/expect -c '
        log_user 0
        set timeout 1800
        spawn ssh -o StrictHostKeyChecking=accept-new "$env(SSH_USER)@$env(SSH_HOST)" "$env(REMOTE_CMD)"
        expect {
            "yes/no" { send "yes\r"; exp_continue }
            -re "(?i)password:" { send "$env(SSH_PASS)\r" }
            timeout { puts stderr "SSH connection timed out"; exit 1 }
            eof {
                if {[info exists expect_out(buffer)]} { puts -nonewline $expect_out(buffer) }
                catch wait result
                exit [lindex $result 3]
            }
        }
        expect -re "\r\n|\n"
        log_user 1
        expect eof
        catch wait result
        exit [lindex $result 3]
    '
}

scp_upload() {
    local dst="$1"
    shift
    export SCP_DST="${SSH_USER}@${SSH_HOST}:${dst}"
    /usr/bin/expect -f - "$@" <<'EOF'
        set timeout 600
        spawn scp -o StrictHostKeyChecking=accept-new {*}$argv $env(SCP_DST)
        expect {
            "yes/no" { send "yes\r"; exp_continue }
            -re "(?i)password:" { send "$env(SSH_PASS)\r"; exp_continue }
            timeout { puts stderr "SCP upload timed out"; exit 1 }
            eof
        }
        catch wait result
        exit [lindex $result 3]
EOF
}

scp_download() {
    local remote_src="$1"
    local local_dst="$2"
    export SCP_SRC="${SSH_USER}@${SSH_HOST}:${remote_src}"
    export SCP_LOCAL="${local_dst}"
    /usr/bin/expect -c '
        set timeout 600
        spawn scp -o StrictHostKeyChecking=accept-new "$env(SCP_SRC)" "$env(SCP_LOCAL)"
        expect {
            "yes/no" { send "yes\r"; exp_continue }
            -re "(?i)password:" { send "$env(SSH_PASS)\r"; exp_continue }
            timeout { puts stderr "SCP download timed out"; exit 1 }
            eof
        }
        catch wait result
        exit [lindex $result 3]
    '
}

DEPLOY_AND_BENCH=0
STOP_DISPLAY_FOR_BENCH="${STOP_DISPLAY_FOR_BENCH:-0}"
for arg in "$@"; do
    case "$arg" in
        --deploy-and-bench)
            DEPLOY_AND_BENCH=1
            ;;
        --stop-display-for-bench)
            STOP_DISPLAY_FOR_BENCH=1
            ;;
        --help|-h)
            echo "Usage: $0 [--deploy-and-bench] [--stop-display-for-bench]"
            echo "  Syncs trained Burn MNIST artifacts and crate sources to ${SSH_USER}@${SSH_HOST}:${REMOTE_BSP_DIR}."
            echo "  With --deploy-and-bench, cross-compiles CPU/GPU binaries first, stages files while display stays on,"
            echo "  atomically updates digit-recognizer with rollback protection, runs benchmarks, and verifies display rendering."
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg (use --help)" >&2
            exit 1
            ;;
    esac
done

BOARD_GUARD_INSTALLED=0
DISPLAY_VERIFIED_OK=0

ensure_board_display_active() {
    log "Ensuring digit-recognizer.service is active and rendering on RK3588 board..."
    remote_ssh "cd '${REMOTE_BSP_DIR}' && source scripts/lib.sh && read -r -a ssh_opts <<< \"\$BOARD_SSH_OPTS\" && \
        ssh \"\${ssh_opts[@]}\" \"\$BOARD_HOST\" \"sudo -S -p '' bash -c '
            if [[ -x /opt/burn-mnist-rk3588/board-service-guard.sh ]]; then
                /opt/burn-mnist-rk3588/board-service-guard.sh ensure-active
            else
                systemctl reset-failed digit-recognizer.service 2>/dev/null || true
                systemctl start digit-recognizer.service 2>/dev/null || true
                systemctl is-active digit-recognizer.service
            fi
        '\" < \"\$BOARD_SUDO_PASSWORD_FILE\""
}

on_exit_cleanup() {
    local exit_code="$1"
    trap - EXIT INT TERM
    if [[ "${DEPLOY_AND_BENCH}" -eq 1 ]] && { [[ "${exit_code}" -ne 0 ]] || [[ "${DISPLAY_VERIFIED_OK}" -eq 0 ]]; }; then
        log "Trap triggered (exit_code=${exit_code}); recovering digit-recognizer.service on RK3588 board so screen does not stay black..."
        ensure_board_display_active || log "WARNING: Failed to verify digit-recognizer.service during cleanup trap"
    fi
    exit "${exit_code}"
}

trap 'on_exit_cleanup $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

log "Creating target directories on Ubuntu server (${SSH_USER}@${SSH_HOST})..."
remote_ssh "mkdir -p '${REMOTE_MODEL_DIR}' '${REMOTE_CRATE_DIR}/src/bin' '${REMOTE_CRATE_DIR}/artifacts' '${REMOTE_CRATE_DIR}/data/mnist'"

log "Uploading trained Burn MNIST model artifacts to ${REMOTE_MODEL_DIR} and ${REMOTE_CRATE_DIR}/artifacts ..."
scp_upload "${REMOTE_MODEL_DIR}/" \
    "${ARTIFACT_DIR}/mnist_cnn.bpk" \
    "${ARTIFACT_DIR}/mnist_cnn.safetensors" \
    "${ARTIFACT_DIR}/training_metrics.json"

remote_ssh "cp -f '${REMOTE_MODEL_DIR}/mnist_cnn.bpk' '${REMOTE_MODEL_DIR}/mnist_cnn.safetensors' '${REMOTE_MODEL_DIR}/training_metrics.json' '${REMOTE_CRATE_DIR}/artifacts/'"

log "Uploading burn-mnist-rk3588 crate sources to ${REMOTE_CRATE_DIR} ..."
scp_upload "${REMOTE_CRATE_DIR}/" \
    "${REPO_ROOT}/Cargo.toml" \
    "${REPO_ROOT}/Cargo.lock" \
    "${REPO_ROOT}/README.md"

scp_upload "${REMOTE_CRATE_DIR}/src/" \
    "${REPO_ROOT}/src/lib.rs" \
    "${REPO_ROOT}/src/data.rs" \
    "${REPO_ROOT}/src/model.rs" \
    "${REPO_ROOT}/src/bench.rs"

scp_upload "${REMOTE_CRATE_DIR}/src/bin/" \
    "${REPO_ROOT}/src/bin/train.rs" \
    "${REPO_ROOT}/src/bin/bench.rs" \
    "${REPO_ROOT}/src/bin/mnist-cpu.rs" \
    "${REPO_ROOT}/src/bin/mnist-gpu.rs"

if [[ -f "${DATA_DIR}/t10k-images-idx3-ubyte.gz" && -f "${DATA_DIR}/t10k-labels-idx1-ubyte.gz" ]]; then
    log "Uploading MNIST test dataset (t10k) to ${REMOTE_CRATE_DIR}/data/mnist ..."
    scp_upload "${REMOTE_CRATE_DIR}/data/mnist/" \
        "${DATA_DIR}/t10k-images-idx3-ubyte.gz" \
        "${DATA_DIR}/t10k-labels-idx1-ubyte.gz"
fi

log "Verifying uploaded files and SHA1 checksums on Ubuntu server..."
remote_ssh "ls -lh '${REMOTE_MODEL_DIR}' '${REMOTE_CRATE_DIR}/src' '${REMOTE_CRATE_DIR}/src/bin' && \
    sha1sum '${REMOTE_MODEL_DIR}/mnist_cnn.bpk' '${REMOTE_CRATE_DIR}/artifacts/mnist_cnn.bpk' '${REMOTE_MODEL_DIR}/mnist_cnn.safetensors' && \
    test \"\$(sha1sum '${REMOTE_MODEL_DIR}/mnist_cnn.bpk' | awk '{print \$1}')\" = '${LOCAL_BPK_SHA1}' && \
    test \"\$(sha1sum '${REMOTE_CRATE_DIR}/artifacts/mnist_cnn.bpk' | awk '{print \$1}')\" = '${LOCAL_BPK_SHA1}'"

if [[ "${DEPLOY_AND_BENCH}" -eq 1 ]]; then
    log "Phase 1/4: Cross-compiling CPU & GPU versions on Ubuntu server (board display stays active)..."
    remote_ssh "cd '${REMOTE_BSP_DIR}' && source scripts/lib.sh && \
        export CARGO_TARGET_DIR=/tmp/burn-probe/target && \
        export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=\"\$AARCH64_CC\" && \
        cargo build --manifest-path '${REMOTE_CRATE_DIR}/Cargo.toml' --release --target aarch64-unknown-linux-gnu --no-default-features --features cpu --bin mnist-cpu && \
        cargo build --manifest-path '${REMOTE_CRATE_DIR}/Cargo.toml' --release --target aarch64-unknown-linux-gnu --no-default-features --features gpu --bin mnist-gpu && \
        cargo build --manifest-path '${REMOTE_CRATE_DIR}/Cargo.toml' --release --target aarch64-unknown-linux-gnu --features cpu,gpu --bin bench && \
        export DISPLAY_SYSROOT=\${DISPLAY_SYSROOT:-\$BSP_BUILD/slint-sysroot} && \
        export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=\$BSP_ROOT/scripts/display-linker.sh && \
        export PKG_CONFIG_SYSROOT_DIR_aarch64_unknown_linux_gnu=\$DISPLAY_SYSROOT && \
        export PKG_CONFIG_LIBDIR_aarch64_unknown_linux_gnu=\$DISPLAY_SYSROOT/usr/lib/aarch64-linux-gnu/pkgconfig:\$DISPLAY_SYSROOT/usr/share/pkgconfig && \
        export PKG_CONFIG_PATH_aarch64_unknown_linux_gnu= && \
        cargo build --manifest-path '${REMOTE_APP_DIR}/Cargo.toml' --release --target aarch64-unknown-linux-gnu --no-default-features --features cpu && \
        cp -f /tmp/burn-probe/target/aarch64-unknown-linux-gnu/release/digit-recognizer /tmp/burn-probe/target/aarch64-unknown-linux-gnu/release/digit-recognizer-cpu && \
        cargo build --manifest-path '${REMOTE_APP_DIR}/Cargo.toml' --release --target aarch64-unknown-linux-gnu --no-default-features --features gpu && \
        cp -f /tmp/burn-probe/target/aarch64-unknown-linux-gnu/release/digit-recognizer /tmp/burn-probe/target/aarch64-unknown-linux-gnu/release/digit-recognizer-gpu && \
        ./scripts/build-display.sh"

    log "Phase 2/4: Staging binaries, models, and service guard script to RK3588 board (board display still active)..."
    LOCAL_GUARD_SCRIPT="$(mktemp "${TMPDIR:-/tmp}/board-service-guard.XXXXXX")"
    cat <<'BOARD_GUARD_EOF' > "${LOCAL_GUARD_SCRIPT}"
#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR="/var/backups/digit-recognizer-last"
STAGING_DIR="/opt/burn-mnist-rk3588/staging"

log_guard() {
    printf '[board-guard] %s\n' "$*"
}

verify_rendering() {
    local verbose="${1:-0}"
    systemctl is-active --quiet digit-recognizer.service || return 1
    local pid
    pid="$(systemctl show -p MainPID --value digit-recognizer.service 2>/dev/null || echo 0)"
    [[ -n "${pid}" && "${pid}" -gt 0 ]] || return 1
    local logs
    logs="$(journalctl -u digit-recognizer.service _PID="${pid}" --no-pager 2>/dev/null || true)"
    grep -q "Rendering at " <<<"${logs}" || return 1
    grep -q "GPU_SCANOUT:" <<<"${logs}" || return 1
    if [[ "${verbose}" -eq 1 ]]; then
        log_guard "Service status: $(systemctl is-active digit-recognizer.service) (MainPID=${pid})"
        grep -E "Connected DRM output|Using FemtoVG|Rendering at |GPU_SCANOUT:" <<<"${logs}" | tail -n 4 || true
    fi
    return 0
}

wait_until_rendering() {
    local max_iters="${1:-24}"
    local i substate result
    for ((i = 1; i <= max_iters; i++)); do
        if verify_rendering 0; then
            verify_rendering 1
            return 0
        fi
        substate="$(systemctl show -p SubState --value digit-recognizer.service 2>/dev/null || true)"
        result="$(systemctl show -p Result --value digit-recognizer.service 2>/dev/null || true)"
        if [[ "${substate}" == "auto-restart" || "${substate}" == "failed" ]] || { [[ -n "${result}" && "${result}" != "success" ]]; }; then
            break
        fi
        sleep 0.5
    done
    return 1
}

backup_current() {
    mkdir -p "${BACKUP_DIR}"
    # Only overwrite existing backup if the current service is healthy or no backup exists yet.
    if [[ ! -x "${BACKUP_DIR}/digit-recognizer" ]] || verify_rendering 0; then
        if [[ -x /usr/local/bin/digit-recognizer ]]; then
            cp -af /usr/local/bin/digit-recognizer "${BACKUP_DIR}/digit-recognizer"
            log_guard "Backed up /usr/local/bin/digit-recognizer to ${BACKUP_DIR}/digit-recognizer"
        fi
        if [[ -x /opt/burn-mnist-rk3588/cpu/digit-recognizer-cpu ]]; then
            cp -af /opt/burn-mnist-rk3588/cpu/digit-recognizer-cpu "${BACKUP_DIR}/digit-recognizer-cpu"
        fi
        if [[ -x /opt/burn-mnist-rk3588/gpu/digit-recognizer-gpu ]]; then
            cp -af /opt/burn-mnist-rk3588/gpu/digit-recognizer-gpu "${BACKUP_DIR}/digit-recognizer-gpu"
        fi
        if [[ -f /usr/share/digit-recognizer/model/mnist_cnn.bpk ]]; then
            cp -af /usr/share/digit-recognizer/model/mnist_cnn.bpk "${BACKUP_DIR}/mnist_cnn.bpk"
        fi
    else
        log_guard "Preserving existing known-good backup in ${BACKUP_DIR}"
    fi
}

rollback_to_backup() {
    log_guard "Rolling back digit-recognizer binary and model from ${BACKUP_DIR} ..."
    systemctl stop digit-recognizer.service 2>/dev/null || true
    if [[ -x "${BACKUP_DIR}/digit-recognizer" ]]; then
        install -m 0755 "${BACKUP_DIR}/digit-recognizer" /usr/local/bin/digit-recognizer.rollback
        mv -f /usr/local/bin/digit-recognizer.rollback /usr/local/bin/digit-recognizer
    fi
    if [[ -x "${BACKUP_DIR}/digit-recognizer-cpu" ]]; then
        install -m 0755 "${BACKUP_DIR}/digit-recognizer-cpu" /opt/burn-mnist-rk3588/cpu/digit-recognizer-cpu.rollback
        mv -f /opt/burn-mnist-rk3588/cpu/digit-recognizer-cpu.rollback /opt/burn-mnist-rk3588/cpu/digit-recognizer-cpu
    fi
    if [[ -x "${BACKUP_DIR}/digit-recognizer-gpu" ]]; then
        install -m 0755 "${BACKUP_DIR}/digit-recognizer-gpu" /opt/burn-mnist-rk3588/gpu/digit-recognizer-gpu.rollback
        mv -f /opt/burn-mnist-rk3588/gpu/digit-recognizer-gpu.rollback /opt/burn-mnist-rk3588/gpu/digit-recognizer-gpu
    fi
    if [[ -f "${BACKUP_DIR}/mnist_cnn.bpk" ]]; then
        install -m 0644 "${BACKUP_DIR}/mnist_cnn.bpk" /usr/share/digit-recognizer/model/mnist_cnn.bpk.rollback
        mv -f /usr/share/digit-recognizer/model/mnist_cnn.bpk.rollback /usr/share/digit-recognizer/model/mnist_cnn.bpk
        cp -f "${BACKUP_DIR}/mnist_cnn.bpk" /opt/burn-mnist-rk3588/model/mnist_cnn.bpk
    fi
    systemctl reset-failed digit-recognizer.service 2>/dev/null || true
    systemctl start seatd.service 2>/dev/null || true
    systemctl start digit-recognizer.service
    if wait_until_rendering 24; then
        log_guard "Rollback succeeded: digit-recognizer.service is active and rendering."
        return 0
    fi
    log_guard "ERROR: Rollback failed to restore rendering!"
    journalctl -u digit-recognizer.service -n 20 --no-pager >&2 || true
    return 1
}

ensure_active() {
    if verify_rendering 1; then
        return 0
    fi
    log_guard "digit-recognizer.service is not active/rendering; starting service..."
    systemctl reset-failed digit-recognizer.service 2>/dev/null || true
    systemctl start seatd.service 2>/dev/null || true
    systemctl start digit-recognizer.service 2>/dev/null || true
    if wait_until_rendering 24; then
        return 0
    fi
    log_guard "Service failed to start or render; triggering rollback..."
    rollback_to_backup
}

deploy_staged() {
    local staged_bin="${1:-${STAGING_DIR}/digit-recognizer}"
    backup_current

    # Stage all non-running or atomically-replaceable files BEFORE stopping the display service
    if [[ -f "${STAGING_DIR}/mnist-cpu" ]]; then
        install -m 0755 "${STAGING_DIR}/mnist-cpu" /opt/burn-mnist-rk3588/cpu/mnist-cpu.next
        mv -f /opt/burn-mnist-rk3588/cpu/mnist-cpu.next /opt/burn-mnist-rk3588/cpu/mnist-cpu
        ln -sf /opt/burn-mnist-rk3588/cpu/mnist-cpu /usr/local/bin/mnist-cpu
    fi
    if [[ -f "${STAGING_DIR}/mnist-gpu" ]]; then
        install -m 0755 "${STAGING_DIR}/mnist-gpu" /opt/burn-mnist-rk3588/gpu/mnist-gpu.next
        mv -f /opt/burn-mnist-rk3588/gpu/mnist-gpu.next /opt/burn-mnist-rk3588/gpu/mnist-gpu
        ln -sf /opt/burn-mnist-rk3588/gpu/mnist-gpu /usr/local/bin/mnist-gpu
    fi
    if [[ -f "${STAGING_DIR}/bench" ]]; then
        install -m 0755 "${STAGING_DIR}/bench" /opt/burn-mnist-rk3588/bench.next
        mv -f /opt/burn-mnist-rk3588/bench.next /opt/burn-mnist-rk3588/bench
    fi
    if [[ -f "${STAGING_DIR}/digit-recognizer-cpu" ]]; then
        install -m 0755 "${STAGING_DIR}/digit-recognizer-cpu" /opt/burn-mnist-rk3588/cpu/digit-recognizer-cpu.next
        mv -f /opt/burn-mnist-rk3588/cpu/digit-recognizer-cpu.next /opt/burn-mnist-rk3588/cpu/digit-recognizer-cpu
        ln -sf /opt/burn-mnist-rk3588/cpu/digit-recognizer-cpu /usr/local/bin/digit-recognizer-cpu
    fi
    if [[ -f "${STAGING_DIR}/digit-recognizer-gpu" ]]; then
        install -m 0755 "${STAGING_DIR}/digit-recognizer-gpu" /opt/burn-mnist-rk3588/gpu/digit-recognizer-gpu.next
        mv -f /opt/burn-mnist-rk3588/gpu/digit-recognizer-gpu.next /opt/burn-mnist-rk3588/gpu/digit-recognizer-gpu
        ln -sf /opt/burn-mnist-rk3588/gpu/digit-recognizer-gpu /usr/local/bin/digit-recognizer-gpu
    fi
    cp -f /usr/share/digit-recognizer/images/*.png /opt/burn-mnist-rk3588/images/ 2>/dev/null || true
    if [[ -w /sys/bus/platform/devices/fb000000.gpu/profiling ]]; then
        echo 3 > /sys/bus/platform/devices/fb000000.gpu/profiling
    fi

    install -m 0755 "${staged_bin}" /usr/local/bin/digit-recognizer.next
    install -m 0644 /opt/burn-mnist-rk3588/model/mnist_cnn.bpk /usr/share/digit-recognizer/model/mnist_cnn.bpk.next

    trap 'ensure_active >/dev/null 2>&1 || true' EXIT INT TERM HUP

    local t0_ms t1_ms
    t0_ms="$(awk '{printf "%.0f\n", $1 * 1000}' /proc/uptime)"
    log_guard "Stopping digit-recognizer.service right before atomic binary/model swap..."
    systemctl stop digit-recognizer.service 2>/dev/null || true
    mv -f /usr/share/digit-recognizer/model/mnist_cnn.bpk.next /usr/share/digit-recognizer/model/mnist_cnn.bpk
    mv -f /usr/local/bin/digit-recognizer.next /usr/local/bin/digit-recognizer
    systemctl reset-failed digit-recognizer.service 2>/dev/null || true
    systemctl start seatd.service 2>/dev/null || true
    systemctl start digit-recognizer.service

    if wait_until_rendering 24; then
        t1_ms="$(awk '{printf "%.0f\n", $1 * 1000}' /proc/uptime)"
        trap - EXIT INT TERM HUP
        log_guard "Deployed and verified new digit-recognizer (display transition took $((t1_ms - t0_ms)) ms)."
        return 0
    fi

    trap - EXIT INT TERM HUP
    log_guard "ERROR: Newly deployed digit-recognizer failed to start or render!"
    journalctl -u digit-recognizer.service -n 20 --no-pager >&2 || true
    rollback_to_backup
    return 1
}

verify_sha1() {
    local expected_bpk="$1"
    local expected_safe="$2"
    sha1sum /opt/burn-mnist-rk3588/model/mnist_cnn.bpk /usr/share/digit-recognizer/model/mnist_cnn.bpk /opt/burn-mnist-rk3588/model/mnist_cnn.safetensors
    local bpk1 bpk2 safe1
    bpk1="$(sha1sum /opt/burn-mnist-rk3588/model/mnist_cnn.bpk | awk '{print $1}')"
    bpk2="$(sha1sum /usr/share/digit-recognizer/model/mnist_cnn.bpk | awk '{print $1}')"
    safe1="$(sha1sum /opt/burn-mnist-rk3588/model/mnist_cnn.safetensors | awk '{print $1}')"
    [[ "${bpk1}" == "${expected_bpk}" ]] || { log_guard "SHA1 mismatch on /opt/burn-mnist-rk3588/model/mnist_cnn.bpk: ${bpk1} != ${expected_bpk}"; return 1; }
    [[ "${bpk2}" == "${expected_bpk}" ]] || { log_guard "SHA1 mismatch on /usr/share/digit-recognizer/model/mnist_cnn.bpk: ${bpk2} != ${expected_bpk}"; return 1; }
    [[ "${safe1}" == "${expected_safe}" ]] || { log_guard "SHA1 mismatch on /opt/burn-mnist-rk3588/model/mnist_cnn.safetensors: ${safe1} != ${expected_safe}"; return 1; }
    log_guard "Verified board model SHA1: mnist_cnn.bpk=${bpk1}, mnist_cnn.safetensors=${safe1}"
}

case "${1:-ensure-active}" in
    verify)
        verify_rendering 1
        ;;
    backup)
        backup_current
        ;;
    rollback)
        rollback_to_backup
        ;;
    ensure-active)
        ensure_active
        ;;
    deploy)
        deploy_staged "${2:-${STAGING_DIR}/digit-recognizer}"
        ;;
    verify-sha1)
        verify_sha1 "$2" "$3"
        ;;
    *)
        echo "Usage: $0 {verify|backup|rollback|ensure-active|deploy [binary_path]|verify-sha1 <bpk_sha1> <safe_sha1>}" >&2
        exit 2
        ;;
esac
BOARD_GUARD_EOF
    scp_upload "/tmp/burn-probe/board-service-guard.sh" "${LOCAL_GUARD_SCRIPT}"
    rm -f "${LOCAL_GUARD_SCRIPT}"
    remote_ssh "chmod 0755 /tmp/burn-probe/board-service-guard.sh && \
cd '${REMOTE_BSP_DIR}' && source scripts/lib.sh && read -r -a ssh_opts <<< \"\$BOARD_SSH_OPTS\" && \
ssh \"\${ssh_opts[@]}\" \"\$BOARD_HOST\" \"sudo -S -p '' bash -c 'mkdir -p /opt/burn-mnist-rk3588/{cpu,gpu,model,data/mnist,images,results,staging} /usr/share/digit-recognizer/model && chown -R ubuntu:ubuntu /opt/burn-mnist-rk3588'\" < \"\$BOARD_SUDO_PASSWORD_FILE\" && \
scp \"\${ssh_opts[@]}\" /tmp/burn-probe/board-service-guard.sh \"\$BOARD_HOST:/opt/burn-mnist-rk3588/board-service-guard.sh\" && \
ssh \"\${ssh_opts[@]}\" \"\$BOARD_HOST\" \"chmod 0755 /opt/burn-mnist-rk3588/board-service-guard.sh\" && \
ssh \"\${ssh_opts[@]}\" \"\$BOARD_HOST\" \"sudo -S -p '' /opt/burn-mnist-rk3588/board-service-guard.sh backup\" < \"\$BOARD_SUDO_PASSWORD_FILE\" && \
scp \"\${ssh_opts[@]}\" '${REMOTE_MODEL_DIR}/mnist_cnn.bpk' '${REMOTE_MODEL_DIR}/mnist_cnn.safetensors' '${REMOTE_MODEL_DIR}/training_metrics.json' \"\$BOARD_HOST:/opt/burn-mnist-rk3588/model/\" && \
if [[ -f '${REMOTE_CRATE_DIR}/data/mnist/t10k-images-idx3-ubyte.gz' ]]; then \
    scp \"\${ssh_opts[@]}\" '${REMOTE_CRATE_DIR}/data/mnist/t10k-images-idx3-ubyte.gz' '${REMOTE_CRATE_DIR}/data/mnist/t10k-labels-idx1-ubyte.gz' \"\$BOARD_HOST:/opt/burn-mnist-rk3588/data/mnist/\"; \
fi && \
scp \"\${ssh_opts[@]}\" \
    /tmp/burn-probe/target/aarch64-unknown-linux-gnu/release/mnist-cpu \
    /tmp/burn-probe/target/aarch64-unknown-linux-gnu/release/mnist-gpu \
    /tmp/burn-probe/target/aarch64-unknown-linux-gnu/release/bench \
    /tmp/burn-probe/target/aarch64-unknown-linux-gnu/release/digit-recognizer-cpu \
    /tmp/burn-probe/target/aarch64-unknown-linux-gnu/release/digit-recognizer-gpu \
    \"\${DISPLAY_OUT:-\$BSP_BUILD/slint}/aarch64-unknown-linux-gnu/release/digit-recognizer\" \
    \"\$BOARD_HOST:/opt/burn-mnist-rk3588/staging/\""

    BOARD_GUARD_INSTALLED=1

    log "Phase 3/4: Atomically replacing digit-recognizer binary & model and verifying display service..."
    remote_ssh "cd '${REMOTE_BSP_DIR}' && source scripts/lib.sh && read -r -a ssh_opts <<< \"\$BOARD_SSH_OPTS\" && \
        trap 'ssh \"\${ssh_opts[@]}\" \"\$BOARD_HOST\" \"sudo -S -p \\\"\\\" /opt/burn-mnist-rk3588/board-service-guard.sh ensure-active\" < \"\$BOARD_SUDO_PASSWORD_FILE\" || true' EXIT INT TERM HUP && \
        ssh \"\${ssh_opts[@]}\" \"\$BOARD_HOST\" \"sudo -S -p '' /opt/burn-mnist-rk3588/board-service-guard.sh deploy\" < \"\$BOARD_SUDO_PASSWORD_FILE\" && \
        ssh \"\${ssh_opts[@]}\" \"\$BOARD_HOST\" \"/opt/burn-mnist-rk3588/board-service-guard.sh verify-sha1 '${LOCAL_BPK_SHA1}' '${LOCAL_SAFE_SHA1}'\" && \
        trap - EXIT INT TERM HUP"

    log "Phase 4/4: Running RK3588 CPU & GPU benchmarks and verifying digit-recognizer.service rendering..."
    remote_ssh "cd '${REMOTE_BSP_DIR}' && source scripts/lib.sh && read -r -a ssh_opts <<< \"\$BOARD_SSH_OPTS\" && \
        trap 'ssh \"\${ssh_opts[@]}\" \"\$BOARD_HOST\" \"sudo -S -p \\\"\\\" /opt/burn-mnist-rk3588/board-service-guard.sh ensure-active\" < \"\$BOARD_SUDO_PASSWORD_FILE\" || true' EXIT INT TERM HUP && \
        if [[ '${STOP_DISPLAY_FOR_BENCH}' -eq 1 ]]; then \
            ssh \"\${ssh_opts[@]}\" \"\$BOARD_HOST\" \"sudo -S -p '' systemctl stop digit-recognizer.service\" < \"\$BOARD_SUDO_PASSWORD_FILE\"; \
        fi && \
        ssh \"\${ssh_opts[@]}\" \"\$BOARD_HOST\" '/opt/burn-mnist-rk3588/cpu/mnist-cpu --model /opt/burn-mnist-rk3588/model/mnist_cnn.bpk --data-dir /opt/burn-mnist-rk3588/data/mnist --images-dir /opt/burn-mnist-rk3588/images --iterations 100 --output-json /opt/burn-mnist-rk3588/results/rk3588_cpu_benchmark.json && \
            /opt/burn-mnist-rk3588/gpu/mnist-gpu --backend gpu-all --model /opt/burn-mnist-rk3588/model/mnist_cnn.bpk --data-dir /opt/burn-mnist-rk3588/data/mnist --images-dir /opt/burn-mnist-rk3588/images --iterations 100 --output-json /opt/burn-mnist-rk3588/results/rk3588_gpu_benchmark.json && \
            /opt/burn-mnist-rk3588/bench --backend all --model /opt/burn-mnist-rk3588/model/mnist_cnn.bpk --data-dir /opt/burn-mnist-rk3588/data/mnist --images-dir /opt/burn-mnist-rk3588/images --iterations 100 --output-json /opt/burn-mnist-rk3588/results/rk3588_benchmark_results.json' && \
        ssh \"\${ssh_opts[@]}\" \"\$BOARD_HOST\" \"sudo -S -p '' /opt/burn-mnist-rk3588/board-service-guard.sh ensure-active\" < \"\$BOARD_SUDO_PASSWORD_FILE\" && \
        trap - EXIT INT TERM HUP && \
        scp \"\${ssh_opts[@]}\" \"\$BOARD_HOST:/opt/burn-mnist-rk3588/results/rk3588_cpu_benchmark.json\" '${REMOTE_CRATE_DIR}/artifacts/rk3588_cpu_benchmark.json' && \
        scp \"\${ssh_opts[@]}\" \"\$BOARD_HOST:/opt/burn-mnist-rk3588/results/rk3588_gpu_benchmark.json\" '${REMOTE_CRATE_DIR}/artifacts/rk3588_gpu_benchmark.json' && \
        scp \"\${ssh_opts[@]}\" \"\$BOARD_HOST:/opt/burn-mnist-rk3588/results/rk3588_benchmark_results.json\" '${REMOTE_CRATE_DIR}/artifacts/rk3588_benchmark_results.json'"

    DISPLAY_VERIFIED_OK=1

    log "Downloading RK3588 benchmark JSON artifacts to ${ARTIFACT_DIR} ..."
    scp_download "${REMOTE_CRATE_DIR}/artifacts/rk3588_cpu_benchmark.json" "${ARTIFACT_DIR}/rk3588_cpu_benchmark.json"
    scp_download "${REMOTE_CRATE_DIR}/artifacts/rk3588_gpu_benchmark.json" "${ARTIFACT_DIR}/rk3588_gpu_benchmark.json"
    scp_download "${REMOTE_CRATE_DIR}/artifacts/rk3588_benchmark_results.json" "${ARTIFACT_DIR}/rk3588_benchmark_results.json"
fi

log "Artifacts and crate synced successfully to ${REMOTE_BSP_DIR}."
