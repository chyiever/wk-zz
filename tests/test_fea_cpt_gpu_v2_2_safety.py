from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fea_cpt_gpu_v2_2 import features
from fea_cpt_gpu_v2_2 import signal_ops
from fea_cpt_gpu_v2_2 import sliding_window as sw


def _signal(fs: float, duration_s: float = 0.02) -> np.ndarray:
    t = np.arange(int(fs * duration_s), dtype=float) / fs
    return (
        np.sin(2.0 * np.pi * 6_000.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 12_000.0 * t + 0.7)
    ) * np.exp(-np.maximum(t - 0.004, 0.0) / 0.006)


class FeaturePipelineSafetyTests(unittest.TestCase):
    def test_active_backend_stft_istft_are_paired(self) -> None:
        fs = 200_000.0
        values = _signal(fs)
        freqs, times, spectrum, power = signal_ops.compute_stft_power(
            values, fs, 0.64, 0.875, None
        )
        backend = signal_ops.active_stft_backend()
        restored = signal_ops.inverse_stft(
            spectrum, fs, 0.64, 0.875, backend=backend, length=len(values)
        )
        self.assertAlmostEqual(float(times[0]), 0.0)
        self.assertEqual(power.dtype, np.float64)
        tolerance = 1e-5 if backend == "torch" else 1e-10
        self.assertLess(np.linalg.norm(values - restored) / np.linalg.norm(values), tolerance)
        self.assertEqual(freqs.shape[0], spectrum.shape[-2])
        if backend == "torch":
            restored_cpu = signal_ops.inverse_stft(
                spectrum, fs, 0.64, 0.875, backend="torch_cpu", length=len(values)
            )
            self.assertLess(
                np.linalg.norm(values - restored_cpu) / np.linalg.norm(values), tolerance
            )

    def test_harmonic_and_residual_energy_share_one_range(self) -> None:
        fs = 200_000.0
        values = _signal(fs)
        params = sw.build_params_for_band((100.0, 60_000.0), fs)
        record = signal_ops.FeatureRecord(
            sample_id="synthetic", sample_name="synthetic", sample_type="test",
            sample_type_code=0, path=Path("synthetic"), signal=values,
            sample_rate=fs, metadata={},
        )
        context = signal_ops.build_context(record, params)
        residual_energy = float(np.sum(context.residual_power, dtype=np.float64))
        self.assertTrue(
            np.isclose(
                context.total_energy_tf,
                context.harmonic_energy + residual_energy,
                rtol=1e-12,
                atol=1e-12,
            )
        )

    def test_feature_policy_emits_harmonics_only_from_explicit_context(self) -> None:
        fs = 200_000.0
        params = {
            "b_100_60k": sw.build_params_for_band((100.0, 60_000.0), fs),
            "b_5k_15k": sw.build_params_for_band((5_000.0, 15_000.0), fs),
        }
        requests, harmonic_name = sw.build_feature_request_map(
            params, harmonic_band_name="b_100_60k"
        )
        self.assertEqual(harmonic_name, "b_100_60k")
        self.assertIn("H2_ratio", requests["b_100_60k"])
        self.assertNotIn("H2_ratio", requests["b_5k_15k"])

    def test_classic_entropy_and_mfcc_primitives_are_well_defined(self) -> None:
        fs = 200_000.0
        values = _signal(fs)
        self.assertGreaterEqual(features._permutation_entropy(values), 0.0)
        self.assertLessEqual(features._permutation_entropy(values), 1.0)
        self.assertGreaterEqual(features._singular_spectrum_entropy(values, 1e-12), 0.0)
        freqs, _times, _spec, power = signal_ops.compute_stft_power(values, fs, 0.64, 0.875, None)
        mfcc = features._mfcc_features(freqs, power, (1_000.0, 60_000.0), 1e-12)
        self.assertEqual(set(mfcc), {f"MFCC_{index:02d}" for index in range(1, 14)})
        self.assertTrue(np.all(np.isfinite(list(mfcc.values()))))

    def test_safe_batch_matches_single_window_reference(self) -> None:
        fs = 200_000.0
        values = _signal(fs)
        bands = [
            ("b_100_60k", (100.0, 60_000.0)),
            ("b_5k_15k", (5_000.0, 15_000.0)),
        ]
        params = {name: sw.build_params_for_band(band, fs) for name, band in bands}
        families = {
            "b_100_60k": ("time", "spectral", "ridge", "harmonic", "background"),
            "b_5k_15k": ("time", "spectral", "ridge", "background"),
        }
        requests, _ = sw.build_feature_request_map(params, families, "b_100_60k")
        band_signals = sw.build_canonical_band_signals(values, fs, params)
        windows = sw.list_window_ranges(len(values), fs, 0.02, 0.0)
        shared_arrays = {"signal_pre": values}
        shared_arrays.update(
            {f"band_{index}": band_signals[name] for index, name in enumerate(params)}
        )
        signal_pack = sw._SharedArrayPack(shared_arrays)
        stft_pack = sw._compute_one_stft_chunk(
            values, band_signals, windows, fs, params, 0, 1
        )
        try:
            batch_row, batch_log = sw._worker_process_window(
                signal_pack.get_info(), stft_pack.get_info(), 0, 0, len(values),
                len(values), len(values), 0, fs, params, {}, None, bands, True, requests,
            )
            reference_row, reference_log = sw._worker_process_window(
                signal_pack.get_info(), None, 0, 0, len(values), len(values), len(values),
                0, fs, params, {}, None, bands, False, requests,
            )
        finally:
            stft_pack.cleanup()
            signal_pack.cleanup()

        self.assertEqual(batch_log["missing_selected_features"], "")
        self.assertEqual(reference_log["missing_selected_features"], "")
        feature_columns = [name for name in batch_row if "__" in name]
        self.assertEqual(
            set(feature_columns), {name for name in reference_row if "__" in name}
        )
        for name in feature_columns:
            left = float(batch_row[name])
            right = float(reference_row[name])
            if np.isnan(left) and np.isnan(right):
                continue
            self.assertTrue(
                np.isclose(left, right, rtol=1e-8, atol=1e-10),
                msg=f"{name}: batch={left}, reference={right}",
            )

    def test_wavelet_packet_uses_frequency_order_and_band_scaled_rate(self) -> None:
        fs = 1_000_000.0
        values = _signal(fs, 0.01)
        energies, wavelet_fs = signal_ops.compute_wavelet_node_energies(
            values, "db4", 4, fs, (100.0, 1_000.0), return_sample_rate=True
        )
        self.assertEqual(len(energies), 16)
        self.assertTrue(np.isclose(wavelet_fs, 4_000.0, rtol=1e-3))
        centers = [
            features._node_center_hz(i, len(energies), wavelet_fs) for i in range(16)
        ]
        self.assertTrue(np.all(np.diff(centers) > 0.0))
        self.assertLess(centers[0], 1_000.0)

    def test_damped_match_is_phase_and_position_tolerant(self) -> None:
        fs = 100_000.0
        n = 3_000
        scores = []
        for start, phase in ((200, 0.0), (1_500, np.pi / 2.0)):
            t = np.maximum(np.arange(n) - start, 0) / fs
            causal = np.arange(n) >= start
            values = causal * np.exp(-t / 0.002) * np.cos(
                2 * np.pi * 8_000 * t + phase
            )
            score, atom = features._damped_atom_match(
                values, fs, (8_000.0,), (2.0,), 1e-12, start_indices=(start,)
            )
            scores.append(score)
            self.assertTrue(np.isclose(np.linalg.norm(atom), 1.0, rtol=1e-8))
        self.assertGreater(min(scores), 0.999)
        self.assertLess(abs(scores[0] - scores[1]), 1e-6)

    def test_processed_log_gate_rejects_missing_feature_rows(self) -> None:
        windows = [(0, 0, 100, 100, 100)]
        requests = {"b": frozenset({"SNR_band_db", "E_excess"})}
        row = {"window_id": 0, "b__SNR_band_db": 4.0}
        log = {"window_id": 0, "missing_selected_features": ""}
        errors = sw.validate_completed_file([row], [log], windows, requests, 4)
        self.assertTrue(errors)
        self.assertIn("missing_features", errors[0])


if __name__ == "__main__":
    unittest.main()
