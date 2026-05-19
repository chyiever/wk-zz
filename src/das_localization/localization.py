"""定位主流程：串联预处理、TDOA、拟合与状态判决。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fitting import fit_hyperbola_ls, fit_hyperbola_ransac, fit_parabola
from .preprocess import bandpass_filter, highpass_filter, normalize_channels, remove_dc
from .tdoa import apply_physical_tau_window, estimate_delays, select_reference_channel


@dataclass
class LocalizeConfig:
    fs: float
    dx: float
    velocity: float = 343.0
    ref_mode: str = "best_snr"
    ref_ch: int = 0
    delay_method: str = "gcc_phat"
    max_tau_sec: float = 0.02
    psr_th: float = 6.0
    n_valid_min: int = 8
    fit_model: str = "hyperbola"
    fit_solver: str = "least_squares"
    robust_loss: str = "linear"
    use_ransac: bool = False
    ransac_iters: int = 200
    ransac_resid_th: float = 8e-4
    x_bounds: tuple[float, float] = (-20.0, 120.0)
    z_bounds: tuple[float, float] = (0.1, 5.0)
    do_demean: bool = True
    do_highpass: bool = True
    hp_cutoff: float = 100.0
    hp_order: int = 2
    do_bandpass: bool = False
    bp_lowcut: float = 120.0
    bp_highcut: float = 800.0
    bp_order: int = 4
    normalize: bool = True
    rmse_th: float = 0.001
    conf_th: float = 0.6


def _confidence_from_quality(psr: np.ndarray, rmse: float, valid_ratio: float) -> float:
    psr_score = float(np.clip(np.nanmean(psr) / 12.0, 0.0, 1.0))
    rmse_score = float(np.exp(-rmse / 0.001))
    conf = 0.45 * psr_score + 0.35 * rmse_score + 0.20 * valid_ratio
    return float(np.clip(conf, 0.0, 1.0))


def _judge_status(conf: float, rmse: float, n_valid: int, n_valid_min: int, rmse_th: float, conf_th: float) -> tuple[str, str]:
    if n_valid < n_valid_min:
        return "rejected", "insufficient_valid_channels"
    if rmse > rmse_th * 2:
        return "rejected", "high_fit_rmse"
    if conf < conf_th:
        return "low_confidence", "low_confidence_score"
    return "ok", "ok"


def preprocess_data(data_fc: np.ndarray, cfg: LocalizeConfig) -> np.ndarray:
    """预处理入口，输入输出 shape 都是 (frames, channels)。"""
    x = np.asarray(data_fc, dtype=np.float64)
    if cfg.do_demean:
        x = remove_dc(x, axis=0)
    if cfg.do_highpass:
        x = highpass_filter(x, fs=cfg.fs, cutoff_hz=cfg.hp_cutoff, order=cfg.hp_order, axis=0)
    if cfg.do_bandpass:
        x = bandpass_filter(
            x,
            fs=cfg.fs,
            lowcut_hz=cfg.bp_lowcut,
            highcut_hz=cfg.bp_highcut,
            order=cfg.bp_order,
            axis=0,
        )
    if cfg.normalize:
        x = normalize_channels(x, axis=0)
    return x


def localize_single_event(data_fc: np.ndarray, cfg: LocalizeConfig) -> dict:
    """单事件定位主函数。"""
    proc = preprocess_data(data_fc, cfg)
    ref_idx = select_reference_channel(proc, mode=cfg.ref_mode, fixed_idx=cfg.ref_ch)
    delay_res = estimate_delays(
        proc,
        fs=cfg.fs,
        ref_idx=ref_idx,
        method=cfg.delay_method,
        max_tau_sec=cfg.max_tau_sec,
        psr_th=cfg.psr_th,
    )

    physical_mask = apply_physical_tau_window(delay_res.delays, cfg.max_tau_sec)
    valid = delay_res.valid_mask & physical_mask
    valid[ref_idx] = True

    x_pos = np.arange(proc.shape[1], dtype=np.float64) * cfg.dx
    xr = float(x_pos[ref_idx])

    use_idx = np.where(valid)[0]
    n_valid = int(use_idx.size)
    if n_valid < 3:
        return {
            "source_x": None,
            "source_z": None,
            "tdoa": delay_res.delays,
            "valid_channels": n_valid,
            "rmse_tdoa": np.inf,
            "confidence": 0.0,
            "status": "rejected",
            "reason": "too_few_points_for_fit",
            "ref_idx": ref_idx,
            "psr": delay_res.psr,
        }

    xv = x_pos[use_idx]
    tv = delay_res.delays[use_idx]

    if cfg.fit_model == "parabola":
        fit = fit_parabola(xv, tv)
        x_peak = float(-fit["coef"][1] / (2.0 * fit["coef"][0])) if abs(fit["coef"][0]) > 1e-12 else float(np.nan)
        rmse = float(fit["rmse"])
        conf = _confidence_from_quality(delay_res.psr[use_idx], rmse, n_valid / proc.shape[1])
        status, reason = _judge_status(conf, rmse, n_valid, cfg.n_valid_min, cfg.rmse_th, cfg.conf_th)
        return {
            "source_x": x_peak,
            "source_z": None,
            "tdoa": delay_res.delays,
            "valid_channels": n_valid,
            "rmse_tdoa": rmse,
            "confidence": conf,
            "status": status,
            "reason": reason,
            "ref_idx": ref_idx,
            "psr": delay_res.psr,
            "fit": fit,
            "fit_model": "parabola",
        }

    robust_loss = cfg.robust_loss if cfg.fit_solver == "robust" else "linear"
    if cfg.use_ransac:
        fit = fit_hyperbola_ransac(
            xv,
            tv,
            xr,
            cfg.velocity,
            cfg.x_bounds,
            cfg.z_bounds,
            iters=cfg.ransac_iters,
            resid_th=cfg.ransac_resid_th,
            robust_loss=robust_loss,
        )
    else:
        fit = fit_hyperbola_ls(
            xv,
            tv,
            xr,
            cfg.velocity,
            cfg.x_bounds,
            cfg.z_bounds,
            robust_loss=robust_loss,
        )

    rmse = float(fit["rmse"])
    conf = _confidence_from_quality(delay_res.psr[use_idx], rmse, n_valid / proc.shape[1])
    status, reason = _judge_status(conf, rmse, n_valid, cfg.n_valid_min, cfg.rmse_th, cfg.conf_th)

    return {
        "source_x": float(fit["xs"]),
        "source_z": float(fit["z"]),
        "tdoa": delay_res.delays,
        "valid_channels": n_valid,
        "rmse_tdoa": rmse,
        "confidence": conf,
        "status": status,
        "reason": reason,
        "ref_idx": ref_idx,
        "psr": delay_res.psr,
        "fit": fit,
        "fit_model": "hyperbola",
    }
