from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def git_commit(repo: Optional[str] = None) -> Optional[str]:
    """Return the current git commit hash, or None when unavailable."""
    cwd = str(repo) if repo else None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def collect_env(repo: Optional[str] = None) -> Dict[str, object]:
    """Collect reproducibility metadata for experiment logging."""
    env: Dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(repo) or git_commit(Path(__file__).resolve().parents[2]),
    }

    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu"] = props.name
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)
            env["gpu_capability"] = f"{props.major}.{props.minor}"
    except ImportError:
        env["torch"] = None

    try:
        import numpy

        env["numpy"] = numpy.__version__
    except ImportError:
        env["numpy"] = None

    try:
        import scipy

        env["scipy"] = scipy.__version__
    except ImportError:
        env["scipy"] = None

    return env


def format_env_report(env: Dict[str, object]) -> str:
    lines = ["environment:"]
    for key, value in env.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def peak_gpu_memory_gb(device: Optional[int] = None) -> float:
    """Peak allocated CUDA memory in GB (0.0 when CUDA is unavailable)."""
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
    except ImportError:
        pass
    return 0.0
