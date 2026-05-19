"""拟合模块：抛物线/双曲线、最小二乘、鲁棒损失、RANSAC。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def tdoa_hyperbola(x: np.ndarray, xs: float, z: float, v: float, xr: float) -> np.ndarray:
    """TDOA 双曲线模型。"""
    return (np.sqrt((x - xs) ** 2 + z**2) - np.sqrt((xr - xs) ** 2 + z**2)) / v


def fit_parabola(x: np.ndarray, t: np.ndarray) -> dict:
    """抛物线拟合（作为对照模型）。"""
    coef = np.polyfit(x, t, deg=2)
    pred = np.polyval(coef, x)
    rmse = float(np.sqrt(np.mean((pred - t) ** 2)))
    return {"coef": coef, "pred": pred, "rmse": rmse, "model": "parabola"}


def coarse_grid_init(
    x: np.ndarray,
    t: np.ndarray,
    xr: float,
    v: float,
    x_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    nx: int = 120,
    nz: int = 80,
) -> tuple[float, float]:
    """网格粗搜索初始化，提升非线性优化稳定性。"""
    xs_grid = np.linspace(x_bounds[0], x_bounds[1], nx)
    z_grid = np.linspace(z_bounds[0], z_bounds[1], nz)
    best = (float((x_bounds[0] + x_bounds[1]) * 0.5), float((z_bounds[0] + z_bounds[1]) * 0.5))
    best_err = np.inf
    for xs in xs_grid:
        for z in z_grid:
            pred = tdoa_hyperbola(x, xs, z, v, xr)
            err = float(np.mean((pred - t) ** 2))
            if err < best_err:
                best_err = err
                best = (float(xs), float(z))
    return best


def fit_hyperbola_ls(
    x: np.ndarray,
    t: np.ndarray,
    xr: float,
    v: float,
    x_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    robust_loss: str = "linear",
    init: tuple[float, float] | None = None,
) -> dict:
    """双曲线受约束最小二乘拟合，支持鲁棒损失。"""
    if init is None:
        init = coarse_grid_init(x, t, xr, v, x_bounds, z_bounds)

    def residual(p: np.ndarray) -> np.ndarray:
        xs, z = float(p[0]), float(p[1])
        return tdoa_hyperbola(x, xs, z, v, xr) - t

    result = least_squares(
        residual,
        x0=np.asarray(init, dtype=np.float64),
        bounds=([x_bounds[0], z_bounds[0]], [x_bounds[1], z_bounds[1]]),
        loss=robust_loss,
    )
    xs, z = float(result.x[0]), float(result.x[1])
    pred = tdoa_hyperbola(x, xs, z, v, xr)
    rmse = float(np.sqrt(np.mean((pred - t) ** 2)))
    return {
        "model": "hyperbola",
        "xs": xs,
        "z": z,
        "pred": pred,
        "rmse": rmse,
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "loss": robust_loss,
    }


def fit_hyperbola_ransac(
    x: np.ndarray,
    t: np.ndarray,
    xr: float,
    v: float,
    x_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    iters: int = 200,
    resid_th: float = 8e-4,
    robust_loss: str = "linear",
) -> dict:
    """RANSAC + 最小二乘精修。"""
    n = len(x)
    if n < 4:
        return fit_hyperbola_ls(x, t, xr, v, x_bounds, z_bounds, robust_loss=robust_loss)

    best_inliers = None
    best_count = -1
    rng = np.random.default_rng(20260518)

    for _ in range(iters):
        idx = np.sort(rng.choice(n, size=min(4, n), replace=False))
        try:
            base = fit_hyperbola_ls(
                x[idx],
                t[idx],
                xr,
                v,
                x_bounds,
                z_bounds,
                robust_loss="linear",
            )
        except Exception:
            continue
        pred_all = tdoa_hyperbola(x, base["xs"], base["z"], v, xr)
        resid = np.abs(pred_all - t)
        inliers = resid <= resid_th
        count = int(np.sum(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or np.sum(best_inliers) < 4:
        final = fit_hyperbola_ls(x, t, xr, v, x_bounds, z_bounds, robust_loss=robust_loss)
        final["used_ransac"] = False
        final["inlier_count"] = int(n)
        return final

    final = fit_hyperbola_ls(
        x[best_inliers],
        t[best_inliers],
        xr,
        v,
        x_bounds,
        z_bounds,
        robust_loss=robust_loss,
    )
    final["used_ransac"] = True
    final["inlier_count"] = int(np.sum(best_inliers))
    final["inlier_mask"] = best_inliers
    return final
