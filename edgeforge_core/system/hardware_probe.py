from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "system" / "system_info.json"


def package_version(package_name: str) -> str | None:
    """Return the installed package version, or None if it is missing."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def run_command(
    command: list[str],
    working_directory: Path | None = None,
) -> str | None:
    """Run a command and return its output when successful."""
    try:
        result = subprocess.run(
            command,
            cwd=working_directory,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    output = result.stdout.strip()
    return output or None


def get_cpu_name() -> str:
    """Return a readable CPU name."""
    if platform.system() == "Windows":
        command = [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_Processor | "
                "Select-Object -First 1 -ExpandProperty Name"
            ),
        ]

        cpu_name = run_command(command)

        if cpu_name:
            return cpu_name.strip()

    return platform.processor() or platform.machine() or "Unknown CPU"


def decode_nvml_text(value: str | bytes) -> str:
    """Convert NVIDIA library text into a normal string."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return value


def collect_nvidia_information() -> dict[str, Any]:
    """Collect NVIDIA GPU and driver information."""
    try:
        from pynvml import (  # type: ignore[import-untyped]
            NVMLError,
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName,
            nvmlInit,
            nvmlShutdown,
            nvmlSystemGetDriverVersion,
        )
    except ImportError as error:
        return {
            "available": False,
            "driver_version": None,
            "devices": [],
            "error": f"NVML package unavailable: {error}",
        }

    nvml_started = False

    try:
        nvmlInit()
        nvml_started = True

        device_count = int(nvmlDeviceGetCount())

        driver_version = decode_nvml_text(nvmlSystemGetDriverVersion())

        devices: list[dict[str, Any]] = []

        for device_index in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(device_index)
            memory_information = nvmlDeviceGetMemoryInfo(handle)

            total_vram_bytes = int(memory_information.total)

            device_name = decode_nvml_text(nvmlDeviceGetName(handle))

            devices.append(
                {
                    "index": device_index,
                    "name": device_name,
                    "total_vram_gb": round(
                        total_vram_bytes / (1024**3),
                        2,
                    ),
                }
            )

        return {
            "available": device_count > 0,
            "driver_version": driver_version,
            "devices": devices,
            "error": None,
        }

    except (NVMLError, OSError, TypeError, ValueError) as error:
        return {
            "available": False,
            "driver_version": None,
            "devices": [],
            "error": str(error),
        }

    finally:
        if nvml_started:
            try:
                nvmlShutdown()
            except (NVMLError, OSError):
                pass


def collect_pytorch_information() -> dict[str, Any]:
    """Collect PyTorch and CUDA information."""
    try:
        import torch
    except ImportError as error:
        return {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "cuda_runtime_version": None,
            "cudnn_version": None,
            "device_count": 0,
            "devices": [],
            "error": str(error),
        }

    cuda_available = torch.cuda.is_available()
    cuda_devices: list[dict[str, Any]] = []

    if cuda_available:
        device_count = int(torch.cuda.device_count())

        for device_index in range(device_count):
            properties = torch.cuda.get_device_properties(device_index)

            total_memory_bytes = int(properties.total_memory)

            cuda_devices.append(
                {
                    "index": device_index,
                    "name": torch.cuda.get_device_name(device_index),
                    "total_memory_gb": round(
                        total_memory_bytes / (1024**3),
                        2,
                    ),
                    "compute_capability": (f"{properties.major}.{properties.minor}"),
                }
            )

    cudnn_version: int | None = None

    if torch.backends.cudnn.is_available():
        cudnn_version = torch.backends.cudnn.version()

    return {
        "installed": True,
        "version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "device_count": int(torch.cuda.device_count()),
        "devices": cuda_devices,
        "error": None,
    }


def collect_onnx_runtime_information() -> dict[str, Any]:
    """Collect ONNX Runtime version and execution providers."""
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError as error:
        return {
            "installed": False,
            "version": None,
            "available_providers": [],
            "dll_preload_error": None,
            "error": str(error),
        }

    dll_preload_error: str | None = None

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except (RuntimeError, OSError) as error:
            dll_preload_error = str(error)

    return {
        "installed": True,
        "version": str(ort.__version__),
        "available_providers": list(ort.get_available_providers()),
        "dll_preload_error": dll_preload_error,
        "error": None,
    }


def collect_openvino_information() -> dict[str, Any]:
    """Collect OpenVINO version and available devices."""
    try:
        import openvino as ov  # type: ignore[import-untyped]
    except ImportError as error:
        return {
            "installed": False,
            "version": None,
            "available_devices": [],
            "error": str(error),
        }

    try:
        core = ov.Core()

        return {
            "installed": True,
            "version": package_version("openvino"),
            "available_devices": list(core.available_devices),
            "error": None,
        }

    except (RuntimeError, OSError, TypeError, ValueError) as error:
        return {
            "installed": True,
            "version": package_version("openvino"),
            "available_devices": [],
            "error": str(error),
        }


def collect_git_information() -> dict[str, Any]:
    """Collect the current Git branch and commit."""
    branch = run_command(
        ["git", "branch", "--show-current"],
        working_directory=PROJECT_ROOT,
    )

    commit = run_command(
        ["git", "rev-parse", "--short", "HEAD"],
        working_directory=PROJECT_ROOT,
    )

    status = run_command(
        ["git", "status", "--porcelain"],
        working_directory=PROJECT_ROOT,
    )

    return {
        "branch": branch,
        "commit": commit,
        "has_uncommitted_changes": bool(status),
    }


def collect_system_snapshot() -> dict[str, Any]:
    """Collect one complete EdgeForge system snapshot."""
    memory_information = psutil.virtual_memory()
    total_ram_bytes = int(memory_information.total)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": {
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "operating_system_version": platform.version(),
            "architecture": platform.machine(),
            "cpu_name": get_cpu_name(),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "ram_gb": round(
                total_ram_bytes / (1024**3),
                2,
            ),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": {
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "onnx": package_version("onnx"),
            "onnxruntime_gpu": package_version("onnxruntime-gpu"),
            "openvino": package_version("openvino"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "psutil": package_version("psutil"),
            "nvidia_ml_py": package_version("nvidia-ml-py"),
        },
        "nvidia": collect_nvidia_information(),
        "pytorch": collect_pytorch_information(),
        "onnx_runtime": (collect_onnx_runtime_information()),
        "openvino": collect_openvino_information(),
        "git": collect_git_information(),
    }


def save_system_snapshot(
    snapshot: dict[str, Any],
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Save the system snapshot as a JSON file."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            snapshot,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return output_path


def print_summary(snapshot: dict[str, Any]) -> None:
    """Print the most important system details."""
    system = snapshot["system"]
    pytorch = snapshot["pytorch"]
    nvidia = snapshot["nvidia"]
    onnx_runtime = snapshot["onnx_runtime"]
    openvino = snapshot["openvino"]
    git = snapshot["git"]

    print("=" * 72)
    print("EDGEFORGE LOCAL - SYSTEM HARDWARE SNAPSHOT")
    print("=" * 72)

    print(
        f"Operating system: {system['operating_system']} "
        f"{system['operating_system_release']}"
    )

    print(f"CPU:              {system['cpu_name']}")

    print(
        "CPU cores:        "
        f"{system['physical_cpu_cores']} physical, "
        f"{system['logical_cpu_cores']} logical"
    )

    print(f"RAM:              {system['ram_gb']} GB")

    if nvidia["devices"]:
        first_gpu = nvidia["devices"][0]

        print(f"GPU:              {first_gpu['name']}")

        print(f"GPU memory:       {first_gpu['total_vram_gb']} GB")

        print(f"NVIDIA driver:    {nvidia['driver_version']}")
    else:
        print("GPU:              No NVIDIA GPU detected")

        if nvidia["error"]:
            print(f"GPU error:        {nvidia['error']}")

    print(f"Python:           {snapshot['python']['version']}")
    print(f"PyTorch:          {pytorch.get('version')}")

    print(f"CUDA available:   {pytorch.get('cuda_available')}")

    print(f"ONNX providers:   {onnx_runtime.get('available_providers', [])}")

    print(f"OpenVINO devices: {openvino.get('available_devices', [])}")

    print(f"Git branch:       {git['branch']}")
    print(f"Git commit:       {git['commit']}")

    print(f"Uncommitted work: {git['has_uncommitted_changes']}")


def main() -> int:
    """Run the hardware probe and save the result."""
    snapshot = collect_system_snapshot()
    output_path = save_system_snapshot(snapshot)

    print_summary(snapshot)

    print("-" * 72)
    print(f"Saved JSON file:  {output_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
