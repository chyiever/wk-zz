import json
from pathlib import Path


def lines(text: str) -> list[str]:
    return [ln + "\n" for ln in text.strip("\n").split("\n")]


def main() -> None:
    nb_path = Path("notebooks/2026-05-19-realdata_feature_dataset_build-3.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    helper_cell = """
# =========================
# Helpers
# =========================

def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger('realdata_feature_dataset_v3')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _scalar_text(value: object) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore').strip()
    if isinstance(value, np.generic):
        value = value.item()
    if hasattr(value, 'tolist') and not isinstance(value, str):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _scalar_text(value[0])
    return str(value).strip()


def _first_property(props: dict[str, object], names: tuple[str, ...]) -> object | None:
    normalized = {str(k).lower(): v for k, v in props.items()}
    for name in names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return None


def _coerce_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return float(arr.item())
        if arr.size == 1:
            return float(arr.reshape(()).item())
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _infer_sample_rate_from_filename(path: Path) -> float | None:
    import re
    m = re.search(r'(?<!\\d)(\\d+(?:\\.\\d+)?)\\s*([kKmM])(?![a-zA-Z])', path.stem)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit == 'k':
        return val * 1_000.0
    if unit == 'm':
        return val * 1_000_000.0
    return None


def parse_starttime(starttime_raw: str) -> datetime | None:
    if not starttime_raw:
        return None
    fmts = [
        '%Y%m%dT%H%M%S.%f', '%Y%m%dT%H%M%S',
        '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(starttime_raw, fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(starttime_raw.replace('Z', '+00:00'))
    except Exception:
        return None


def make_windows_matrix(signal_values: np.ndarray, sample_rate: float, window_duration_s: float, overlap: float):
    n = len(signal_values)
    win = int(round(window_duration_s * sample_rate))
    if win <= 0:
        raise ValueError('window_samples must be positive')
    if n < win:
        return np.zeros((0, win), dtype=float), np.zeros(0, dtype=int), win, 0

    step = max(1, int(round(win * (1.0 - overlap))))
    nwin = 1 + (n - win) // step
    starts = np.arange(nwin, dtype=int) * step
    idx = starts[:, None] + np.arange(win, dtype=int)[None, :]
    windows = signal_values[idx]
    return windows, starts, win, step


def _get_array_module(use_gpu: bool):
    if use_gpu and (cp is not None):
        return cp
    return np


def _stft_power_batch(windows_batch: np.ndarray, sample_rate: float, window_ms: float = 0.64, overlap: float = 0.875):
    xp = _get_array_module(USE_CUPY_GPU)
    x = xp.asarray(windows_batch, dtype=xp.float64)
    bsz, n = x.shape

    nperseg = max(16, int(round(sample_rate * window_ms / 1_000.0)))
    noverlap = min(nperseg - 1, int(round(nperseg * overlap)))
    hop = max(1, nperseg - noverlap)
    nfft = int(2 ** np.ceil(np.log2(nperseg)))

    # mimic scipy.signal.stft(..., boundary='zeros', padded=True)
    left = nperseg // 2
    right = nperseg // 2
    xpad = xp.pad(x, ((0, 0), (left, right)), mode='constant')
    npad = xpad.shape[1]
    n_time = 1 + int(np.ceil((npad - nperseg) / hop))
    target_len = (n_time - 1) * hop + nperseg
    if target_len > npad:
        xpad = xp.pad(xpad, ((0, 0), (0, target_len - npad)), mode='constant')

    starts = xp.arange(n_time, dtype=xp.int64) * hop
    frame_idx = starts[:, None] + xp.arange(nperseg, dtype=xp.int64)[None, :]
    frames = xpad[:, frame_idx]  # [B, T, L]
    win = xp.hanning(nperseg).astype(xp.float64)
    frames = frames * win[None, None, :]

    spec = xp.fft.rfft(frames, n=nfft, axis=2)  # [B, T, F]
    power = xp.abs(spec) ** 2
    power = xp.transpose(power, (0, 2, 1))  # [B, F, T]

    freqs = xp.asarray(np.fft.rfftfreq(nfft, d=1.0 / sample_rate), dtype=xp.float64)
    times = xp.asarray((xp.asnumpy(starts) - left) / sample_rate, dtype=xp.float64) if xp is cp else (starts - left) / sample_rate
    return freqs, times, power


def _dynamic_ridge_one(power_ft: np.ndarray, freqs: np.ndarray, search_hz: tuple[float, float], jump_penalty_hz: float, prior_hz: np.ndarray | None = None) -> np.ndarray:
    # CPU DP for one window on [F, T] power
    mask = (freqs >= search_hz[0]) & (freqs <= search_hz[1])
    fs = freqs[mask]
    pw = power_ft[mask, :]
    n_freq, n_time = pw.shape
    if n_freq == 0 or n_time == 0:
        return np.zeros(n_time, dtype=float)

    score = np.log(pw + 1e-12)
    if prior_hz is not None and len(prior_hz) == n_time:
        score = score - np.abs(fs[:, None] - prior_hz[None, :]) / max(jump_penalty_hz, 1.0)

    dp = np.empty_like(score)
    back = np.zeros_like(score, dtype=int)
    dp[:, 0] = score[:, 0]
    gap = np.abs(fs[:, None] - fs[None, :]) / max(jump_penalty_hz, 1.0)
    for t in range(1, n_time):
        cand = dp[:, t - 1][None, :] - gap
        back[:, t] = np.argmax(cand, axis=1)
        dp[:, t] = score[:, t] + cand[np.arange(n_freq), back[:, t]]

    ridge_local = np.zeros(n_time, dtype=int)
    ridge_local[-1] = int(np.argmax(dp[:, -1]))
    for t in range(n_time - 1, 0, -1):
        ridge_local[t - 1] = back[ridge_local[t], t]
    return fs[ridge_local]


def _compute_sc_batch(power_bft: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    mask = (freqs >= band[0]) & (freqs <= band[1])
    pb = power_bft[:, mask, :]
    fb = freqs[mask]
    if pb.shape[1] == 0:
        return np.zeros(pb.shape[0], dtype=float)
    num = np.sum(pb * fb[None, :, None], axis=1)
    den = np.sum(pb, axis=1) + 1e-12
    sc_t = num / den
    return np.mean(sc_t, axis=1)


def compute_5_features_batch_gpu(windows_batch: np.ndarray, sample_rate: float) -> dict[str, np.ndarray]:
    xp = _get_array_module(USE_CUPY_GPU)
    freqs_xp, _times_xp, power_xp = _stft_power_batch(
        windows_batch=windows_batch,
        sample_rate=sample_rate,
        window_ms=0.64,
        overlap=0.875,
    )

    # move to CPU once for deterministic post-processing and CSV output
    if xp is cp:
        freqs = cp.asnumpy(freqs_xp)
        power_bft = cp.asnumpy(power_xp)
    else:
        freqs = np.asarray(freqs_xp)
        power_bft = np.asarray(power_xp)

    out = {}
    out['b_1k_10k__SC_mean'] = _compute_sc_batch(power_bft, freqs, BANDS['b_1k_10k'])
    out['b_1k_100k__SC_mean'] = _compute_sc_batch(power_bft, freqs, BANDS['b_1k_100k'])
    out['b_40k_60k__SC_mean'] = _compute_sc_batch(power_bft, freqs, BANDS['b_40k_60k'])

    c_f = np.zeros(power_bft.shape[0], dtype=float)
    c_h = np.zeros(power_bft.shape[0], dtype=float)
    for i in range(power_bft.shape[0]):
        pft = power_bft[i]
        ridge_f1 = _dynamic_ridge_one(pft, freqs, (1_000.0, 10_000.0), jump_penalty_hz=2_000.0)
        ridge_f2 = _dynamic_ridge_one(pft, freqs, (2_000.0, 20_000.0), jump_penalty_hz=2_000.0, prior_hz=2.0 * ridge_f1 if ridge_f1.size else None)

        if ridge_f1.size > 2:
            t = np.arange(ridge_f1.size, dtype=float)
            eps = 1e-12
            d1 = np.gradient(ridge_f1, t + eps)
            d2 = np.gradient(d1, t + eps)
            c_f[i] = float(np.mean(np.abs(d2) / (np.mean(np.abs(ridge_f1)) + eps)))
        else:
            c_f[i] = 0.0

        active = ridge_f1 > 0.0
        if np.any(active):
            c_h[i] = float(np.mean(np.abs(ridge_f2[active] - 2.0 * ridge_f1[active])))
        else:
            c_h[i] = 0.0

    out['b_1k_10k__C_f'] = c_f
    out['b_1k_10k__C_h'] = c_h
    return out


def _load_npz_source(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        signal_values = np.asarray(data['phase_data'], dtype=float)
        sample_rate = float(np.asarray(data['sample_rate']).item())
        starttime_raw = _scalar_text(data.get('starttime', '')) if 'starttime' in data else ''
        arrival_time_raw = _scalar_text(data.get('arrival_time', '')) if 'arrival_time' in data else ''
        sample_type = _scalar_text(data.get('type', path.parent.name)) if 'type' in data else path.parent.name
    return {
        'source_format': 'npz',
        'signal_values': signal_values,
        'sample_rate': sample_rate,
        'starttime_raw': starttime_raw,
        'arrival_time_raw': arrival_time_raw,
        'sample_type': sample_type,
        'source_group_name': '',
        'source_channel_name': '',
        'source_detail': '',
    }


def _select_tdms_channel(tdms_file):
    if TDMS_GROUP_NAME and TDMS_CHANNEL_NAME:
        for g in tdms_file.groups():
            if str(g.name).lower() == TDMS_GROUP_NAME.lower():
                for c in g.channels():
                    if str(c.name).lower() == TDMS_CHANNEL_NAME.lower():
                        return g, c
        raise ValueError(f'Cannot find TDMS group/channel: {TDMS_GROUP_NAME}/{TDMS_CHANNEL_NAME}')

    if TDMS_CHANNEL_NAME:
        for g in tdms_file.groups():
            for c in g.channels():
                if str(c.name).lower() == TDMS_CHANNEL_NAME.lower():
                    return g, c

    pref = {'phase_data', 'signal', 'data', 'values', 'channel0', 'ch0'}
    for g in tdms_file.groups():
        for c in g.channels():
            if str(c.name).lower() in pref:
                return g, c

    best = None
    best_len = -1
    for g in tdms_file.groups():
        for c in g.channels():
            try:
                arr = np.asarray(c[:])
                if arr.size == 0:
                    continue
                if not np.issubdtype(arr.dtype, np.number):
                    arr = arr.astype(float)
            except Exception:
                continue
            if arr.size > best_len:
                best_len = arr.size
                best = (g, c)
    if best is None:
        raise ValueError('No usable numeric channel found in TDMS')
    return best


def _load_tdms_source(path: Path) -> dict[str, object]:
    if TdmsFile is None:
        raise ImportError('nptdms is required for .tdms files. Install with: pip install nptdms')

    td = TdmsFile.read(path)
    g, c = _select_tdms_channel(td)

    signal_values = np.asarray(c[:], dtype=float)
    props = {}
    props.update(getattr(td, 'properties', {}) or {})
    props.update(getattr(g, 'properties', {}) or {})
    props.update(getattr(c, 'properties', {}) or {})

    sample_rate = _coerce_float(_first_property(props, ('sample_rate', 'sample_rate_hz', 'sampling_rate', 'sampling_rate_hz')))
    if sample_rate is None:
        wf_inc = _coerce_float(_first_property(props, ('wf_increment',)))
        if wf_inc and wf_inc > 0:
            sample_rate = 1.0 / wf_inc
    if sample_rate is None or sample_rate <= 0:
        sample_rate = _infer_sample_rate_from_filename(path)
    if (sample_rate is None or sample_rate <= 0) and TDMS_FALLBACK_SAMPLE_RATE_HZ is not None:
        sample_rate = float(TDMS_FALLBACK_SAMPLE_RATE_HZ)
    if sample_rate is None or sample_rate <= 0:
        raise ValueError(
            f'Cannot infer sample rate from TDMS file: {path}. '
            'Provide TDMS_FALLBACK_SAMPLE_RATE_HZ or include rate text like 500K in filename.'
        )

    starttime_raw = _scalar_text(_first_property(props, ('starttime', 'start_time', 'wf_start_time', 'wf_starttime')))
    arrival_time_raw = _scalar_text(_first_property(props, ('arrival_time', 'arrivaltime', 'arrival_time_text')))
    sample_type = _scalar_text(_first_property(props, ('type', 'sample_type', 'sampletype'))) or path.parent.name

    return {
        'source_format': 'tdms',
        'signal_values': signal_values,
        'sample_rate': float(sample_rate),
        'starttime_raw': starttime_raw,
        'arrival_time_raw': arrival_time_raw,
        'sample_type': sample_type,
        'source_group_name': str(g.name),
        'source_channel_name': str(c.name),
        'source_detail': f'{g.name}/{c.name}',
    }


def load_source_file(path: Path) -> dict[str, object]:
    suf = path.suffix.lower()
    if suf == '.npz':
        return _load_npz_source(path)
    if suf == '.tdms':
        return _load_tdms_source(path)
    raise ValueError(f'Unsupported file type: {path.suffix}')
"""

    main_cell = """
# =========================
# Main pipeline
# =========================
runtime_log_path = OUTPUT_ROOT / RUNTIME_LOG_NAME
processed_list_path = OUTPUT_ROOT / PROCESSED_LIST_NAME
logger = build_logger(runtime_log_path)

source_files = sorted(
    p for p in RAW_DATA_ROOT.rglob('*')
    if p.is_file() and p.suffix.lower() in {'.npz', '.tdms'}
)
if MAX_FILES is not None:
    source_files = source_files[:MAX_FILES]
if not source_files:
    raise FileNotFoundError(f'No npz/tdms files found under: {RAW_DATA_ROOT}')

processed_set: set[str] = set()
if processed_list_path.exists():
    processed_set = {ln.strip() for ln in processed_list_path.read_text(encoding='utf-8').splitlines() if ln.strip()}

logger.info('Found %d source files', len(source_files))
logger.info('Already processed: %d', len(processed_set))
logger.info('Window config: duration=%.6fs overlap=%.2f', WINDOW_DURATION_S, WINDOW_OVERLAP)
logger.info('Selected features: %s', ', '.join(SELECTED_FEATURES))
logger.info('NPZ_PER_CSV = %d', NPZ_PER_CSV)
logger.info('WINDOW_BATCH_SIZE = %d, USE_CUPY_GPU=%s', WINDOW_BATCH_SIZE, USE_CUPY_GPU)

processed_now = 0
window_total = 0


def chunk_paths(chunk_index: int) -> tuple[Path, Path]:
    suffix = f'part_{chunk_index:04d}.csv'
    return (
        OUTPUT_ROOT / f'{FEATURE_CSV_PREFIX}_{suffix}',
        OUTPUT_ROOT / f'{LOG_CSV_PREFIX}_{suffix}',
    )


for file_idx, fp in enumerate(tqdm(source_files, desc='Files'), start=1):
    fp_str = str(fp)
    if fp_str in processed_set:
        continue

    chunk_index = (file_idx - 1) // NPZ_PER_CSV + 1
    feature_csv_path, log_csv_path = chunk_paths(chunk_index)

    src = load_source_file(fp)
    raw_signal = np.asarray(src['signal_values'], dtype=float)
    sample_rate = float(src['sample_rate'])

    centered = raw_signal - float(np.mean(raw_signal))
    signal_pre = butter_filter(centered, sample_rate=sample_rate, band_hz=PREPROC_BAND, order=4)

    starttime_raw = str(src['starttime_raw'])
    arrival_time_raw = str(src['arrival_time_raw'])
    sample_type = str(src['sample_type'])
    source_format = str(src['source_format'])
    source_group_name = str(src.get('source_group_name', ''))
    source_channel_name = str(src.get('source_channel_name', ''))
    source_detail = str(src.get('source_detail', ''))

    start_dt = parse_starttime(starttime_raw)
    n_samples = len(signal_pre)
    duration_s = n_samples / sample_rate if sample_rate > 0 else np.nan

    windows, starts, win_len, step_len = make_windows_matrix(signal_pre, sample_rate, WINDOW_DURATION_S, WINDOW_OVERLAP)
    rows_features: list[dict[str, object]] = []
    rows_log: list[dict[str, object]] = []

    for b0 in range(0, len(windows), WINDOW_BATCH_SIZE):
        wb = windows[b0:b0 + WINDOW_BATCH_SIZE]
        sb = starts[b0:b0 + WINDOW_BATCH_SIZE]
        batch_feat = compute_5_features_batch_gpu(wb, sample_rate)

        for j in range(len(wb)):
            win_id = int(b0 + j)
            i0 = int(sb[j])
            i1 = int(i0 + win_len)

            base = {
                'source_file_name': fp.name,
                'source_file_path': fp_str,
                'source_format': source_format,
                'source_group_name': source_group_name,
                'source_channel_name': source_channel_name,
                'source_detail': source_detail,
                'window_id': win_id,
                'window_start_index': i0,
                'window_end_index': i1,
                'window_length_samples': int(win_len),
                'window_step_samples': int(step_len),
                'window_duration_s': float(win_len / sample_rate),
                'window_start_offset_s': float(i0 / sample_rate),
                'sample_rate_hz': float(sample_rate),
                'source_n_samples': int(n_samples),
                'source_duration_s': float(duration_s),
                'starttime_raw': starttime_raw,
                'arrival_time_raw': arrival_time_raw,
                'sample_type': sample_type,
                'csv_chunk_index': int(chunk_index),
            }
            if start_dt is not None:
                base['window_start_datetime'] = (start_dt + timedelta(seconds=float(i0 / sample_rate))).strftime('%Y-%m-%d %H:%M:%S.%f')
            else:
                base['window_start_datetime'] = ''

            feat_row = dict(base)
            for fn in SELECTED_FEATURES:
                feat_row[fn] = float(batch_feat[fn][j])
            rows_features.append(feat_row)

            log_row = dict(base)
            log_row['missing_selected_features'] = ''
            rows_log.append(log_row)

    df_features = pd.DataFrame(rows_features)
    df_log = pd.DataFrame(rows_log)

    feature_header = (not feature_csv_path.exists()) or (feature_csv_path.stat().st_size == 0)
    log_header = (not log_csv_path.exists()) or (log_csv_path.stat().st_size == 0)
    df_features.to_csv(feature_csv_path, mode='a', header=feature_header, index=False, encoding='utf-8-sig')
    df_log.to_csv(log_csv_path, mode='a', header=log_header, index=False, encoding='utf-8-sig')

    with processed_list_path.open('a', encoding='utf-8') as f:
        f.write(fp_str + '\\n')
    processed_set.add(fp_str)

    processed_now += 1
    window_total += len(df_features)
    logger.info(
        'Processed file=%s, format=%s, sample_rate=%.1fHz, n_samples=%d, windows=%d, chunk=%d',
        fp.name, source_format, sample_rate, n_samples, len(df_features), chunk_index
    )

logger.info('Run finished. Newly processed files=%d, total windows in this run=%d', processed_now, window_total)
logger.info('Processed list: %s', processed_list_path)
print('Done')
print(f'newly_processed_files={processed_now}')
print(f'total_windows_this_run={window_total}')
"""

    nb["cells"][3]["source"] = lines(helper_cell)
    nb["cells"][4]["source"] = lines(main_cell)

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Patched: {nb_path}")


if __name__ == "__main__":
    main()
