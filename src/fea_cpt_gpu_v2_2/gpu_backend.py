"""Optional GPU helpers for fea_cpt_gpu."""

from __future__ import annotations

import os

import numpy as np

# Keep torch genuinely lazy. On Windows, ProcessPoolExecutor starts fresh
# interpreters; eager import can make every worker probe/initialize CUDA before
# the worker initializer has a chance to disable GPU use.
_torch = None
_torch_checked = False


def _gpu_enabled_by_env() -> bool:
    flag = os.getenv("FEA_CPT_USE_GPU", "1").strip().lower()
    return flag not in {"0", "false", "off", "no"}


def _get_torch():
    global _torch, _torch_checked
    if not _gpu_enabled_by_env():
        return None
    if not _torch_checked:
        try:
            import torch as torch_module
        except Exception:  # pragma: no cover - optional dependency
            torch_module = None
        _torch = torch_module
        _torch_checked = True
    return _torch


def gpu_device() -> str:
    torch = _get_torch()
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def gpu_backend_info() -> str:
    torch = _get_torch()
    if torch is None:
        if not _gpu_enabled_by_env():
            return "[GPU] 已通过环境变量 FEA_CPT_USE_GPU=0 禁用，使用 CPU"
        return "[GPU] torch 未安装，使用 CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return f"[GPU] 使用 CUDA: {device_name}"
    return "[GPU] 未检测到 CUDA，使用 CPU"


def rank2_hankel_quality(residual: np.ndarray, eps: float, max_cols: int = 128) -> float:
    """
    Compute Q_MP-like rank-2 reconstruction quality.

    Uses GPU SVD when CUDA is available; otherwise falls back to NumPy SVD.
    """
    values = np.asarray(residual, dtype=float)
    hankel_cols = max(4, min(len(values) // 3, max_cols))
    hankel_rows = len(values) - hankel_cols + 1
    if hankel_rows < 4:
        return 0.0

    hankel = np.column_stack([values[i : i + hankel_rows] for i in range(hankel_cols)])

    torch = _get_torch()
    if torch is not None and gpu_device() == "cuda":
        try:
            tensor = torch.as_tensor(hankel, dtype=torch.float32, device="cuda")
            u, s, vh = torch.linalg.svd(tensor, full_matrices=False)
            rank = min(2, int(s.shape[0]))
            hankel_hat = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
            num = torch.linalg.norm(tensor - hankel_hat)
            den = torch.linalg.norm(tensor) + float(eps)
            return float((1.0 - num / den).item())
        except Exception:
            pass

    u, s, vh = np.linalg.svd(hankel, full_matrices=False)
    rank = min(2, len(s))
    hankel_hat = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
    return float(1.0 - np.linalg.norm(hankel - hankel_hat) / (np.linalg.norm(hankel) + eps))
