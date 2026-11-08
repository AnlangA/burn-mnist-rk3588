#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Colab / Jupyter Notebook Script for Rust Burn MNIST Training with Google Drive Cache.

Workflow:
1. Mounts Google Drive (if running in Google Colab) or resolves GDRIVE_CACHE_DIR.
2. Checks whether the Rust Burn cargo environment and MNIST handwritten digit dataset
   already exist in Google Drive.
3. If present in Google Drive, migrates them directly to the active runtime.
   If absent, downloads them into the runtime and copies them back to Google Drive.
4. Clones https://github.com/AnlangA/burn-mnist-rk3588.git in the runtime.
5. Runs Rust Burn MNIST CNN training and saves the trained weights (.bpk & .safetensors)
   for RK3588 CPU and GPU deployment.
"""

# %% [markdown]
# # Rust Burn MNIST Training Runtime (Google Drive Cache + RK3588 CPU/GPU Export)

# %%
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request

REPO_URL = os.environ.get(
    "REPO_URL", "https://github.com/AnlangA/burn-mnist-rk3588.git"
)
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]
MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist",
    "https://storage.googleapis.com/cvdf-datasets/mnist",
]


def mount_google_drive() -> pathlib.Path:
    """Mount Google Drive in Colab, or fall back to GDRIVE_CACHE_DIR."""
    if "GDRIVE_CACHE_DIR" in os.environ:
        cache_dir = pathlib.Path(os.environ["GDRIVE_CACHE_DIR"]).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[notebook] Using Google Drive cache directory: {cache_dir}")
        return cache_dir

    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
        cache_dir = pathlib.Path("/content/drive/MyDrive/burn-mnist-rk3588-cache")
    except ImportError:
        cache_dir = pathlib.Path.home() / ".cache" / "google-drive-burn-mnist"

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[notebook] Mounted/initialized Drive cache at: {cache_dir}")
    return cache_dir


# %%
def ensure_mnist_dataset(drive_cache: pathlib.Path, runtime_dir: pathlib.Path) -> pathlib.Path:
    """Check Google Drive for MNIST data; migrate if present, else download & copy to Drive."""
    drive_mnist = drive_cache / "mnist"
    runtime_mnist = runtime_dir / "mnist"
    drive_mnist.mkdir(parents=True, exist_ok=True)
    runtime_mnist.mkdir(parents=True, exist_ok=True)

    all_in_drive = all((drive_mnist / f).is_file() and (drive_mnist / f).stat().st_size > 0 for f in MNIST_FILES)

    if all_in_drive:
        print(f"[notebook] Found MNIST dataset in Google Drive ({drive_mnist}). Migrating to runtime...")
        for fname in MNIST_FILES:
            shutil.copy2(drive_mnist / fname, runtime_mnist / fname)
    else:
        print("[notebook] MNIST dataset missing in Google Drive. Downloading and caching to Google Drive...")
        for fname in MNIST_FILES:
            dest = runtime_mnist / fname
            if not (dest.is_file() and dest.stat().st_size > 0):
                downloaded = False
                for base in MNIST_MIRRORS:
                    url = f"{base}/{fname}"
                    try:
                        print(f"  Downloading {url} -> {dest}")
                        urllib.request.urlretrieve(url, dest)
                        downloaded = True
                        break
                    except Exception as err:  # noqa: BLE001
                        print(f"  Mirror failed ({url}): {err}")
                if not downloaded:
                    raise RuntimeError(f"Failed to download {fname} from all mirrors")
            shutil.copy2(dest, drive_mnist / fname)
        print(f"[notebook] Cached MNIST dataset to Google Drive: {drive_mnist}")

    return runtime_mnist


# %%
def ensure_rust_burn_env(drive_cache: pathlib.Path, runtime_dir: pathlib.Path) -> dict[str, str]:
    """Check Google Drive for Rust + Burn environment; migrate if present, else install & cache."""
    env = os.environ.copy()
    cargo_bin = pathlib.Path.home() / ".cargo" / "bin"
    if cargo_bin.is_dir():
        env["PATH"] = f"{cargo_bin}:{env.get('PATH', '')}"

    drive_env_tar = drive_cache / "rust-burn-cargo-cache.tar.gz"
    runtime_cargo_home = runtime_dir / "cargo-home"

    if drive_env_tar.is_file() and drive_env_tar.stat().st_size > 0:
        print(f"[notebook] Found Rust Burn environment in Google Drive ({drive_env_tar}). Migrating to runtime...")
        runtime_cargo_home.mkdir(parents=True, exist_ok=True)
        with tarfile.open(drive_env_tar, "r:gz") as tar:
            tar.extractall(runtime_cargo_home)
        env["CARGO_HOME"] = str(runtime_cargo_home)
    else:
        print("[notebook] Rust Burn cache not yet in Google Drive. Checking Rust toolchain...")
        if shutil.which("cargo", path=env.get("PATH")) is None:
            print("[notebook] Installing Rust toolchain via rustup...")
            subprocess.run(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
                shell=True,
                check=True,
                env=env,
            )
            env["PATH"] = f"{cargo_bin}:{env.get('PATH', '')}"

    return env


# %%
def clone_and_train(
    drive_cache: pathlib.Path,
    runtime_dir: pathlib.Path,
    runtime_mnist: pathlib.Path,
    env: dict[str, str],
) -> pathlib.Path:
    """Clone https://github.com/AnlangA/burn-mnist-rk3588.git in runtime, cache Burn deps, and train."""
    repo_dir = runtime_dir / "burn-mnist-rk3588"
    if (repo_dir / ".git").is_dir():
        print(f"[notebook] Updating existing repo clone at {repo_dir}...")
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "origin"], check=True, env=env)
        subprocess.run(["git", "-C", str(repo_dir), "reset", "--hard", "origin/main"], check=True, env=env)
    else:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        print(f"[notebook] Cloning {REPO_URL} into {repo_dir}...")
        subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True, env=env)

    drive_env_tar = drive_cache / "rust-burn-cargo-cache.tar.gz"
    if not (drive_env_tar.is_file() and drive_env_tar.stat().st_size > 0):
        print("[notebook] Fetching Burn crate graph and archiving to Google Drive...")
        subprocess.run(["cargo", "fetch"], cwd=repo_dir, check=True, env=env)
        cargo_home = pathlib.Path(env.get("CARGO_HOME", pathlib.Path.home() / ".cargo"))
        registry_dir = cargo_home / "registry"
        if registry_dir.is_dir():
            with tarfile.open(drive_env_tar, "w:gz") as tar:
                tar.add(registry_dir, arcname="registry")
            print(f"[notebook] Archived Rust Burn environment to Google Drive: {drive_env_tar}")

    backend = env.get("BACKEND", "gpu")
    epochs = env.get("EPOCHS", "5")
    batch_size = env.get("BATCH_SIZE", "128")
    lr = env.get("LR", "0.001")
    artifact_dir = repo_dir / "artifacts"

    print(f"[notebook] Running Rust Burn MNIST training (backend={backend}, epochs={epochs})...")
    subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "train",
            "--",
            "--data-dir",
            str(runtime_mnist),
            "--artifact-dir",
            str(artifact_dir),
            "--backend",
            backend,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--lr",
            str(lr),
        ],
        cwd=repo_dir,
        check=True,
        env=env,
    )

    drive_artifacts = drive_cache / "artifacts"
    drive_artifacts.mkdir(parents=True, exist_ok=True)
    for item in artifact_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, drive_artifacts / item.name)
    print(f"[notebook] Saved trained artifacts to Google Drive: {drive_artifacts}")

    local_repo_out = env.get("LOCAL_REPO_OUT")
    if local_repo_out:
        out_dir = pathlib.Path(local_repo_out) / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        for item in artifact_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, out_dir / item.name)
        print(f"[notebook] Downloaded trained artifacts to local repository: {out_dir}")

    return artifact_dir


def main() -> int:
    runtime_dir = pathlib.Path(
        os.environ.get("RUNTIME_DIR", "/tmp/burn-mnist-colab-runtime")
    ).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)

    drive_cache = mount_google_drive()
    runtime_mnist = ensure_mnist_dataset(drive_cache, runtime_dir)
    env = ensure_rust_burn_env(drive_cache, runtime_dir)
    artifact_dir = clone_and_train(drive_cache, runtime_dir, runtime_mnist, env)
    print(f"[notebook] Complete! Artifacts ready at: {artifact_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
