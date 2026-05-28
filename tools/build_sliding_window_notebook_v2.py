"""生成 2026-05-28-realdata_continuous_feature_batch_extract_v2.ipynb 笔记本。

v2 优化版：ProcessPoolExecutor + GPU STFT + 自动检测核心数
使用 Python 以 encoding='utf-8' 创建 .ipynb 文件，确保中文编码正确。
"""

import json
from pathlib import Path


def make_cell(cell_type: str, source: str, cell_id: str = '') -> dict:
    """创建一个 notebook cell。"""
    cell = {
        'cell_type': cell_type,
        'source': source.split('\n'),
        'metadata': {},
    }
    if cell_id:
        cell['metadata']['id'] = cell_id
    if cell_type == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
    return cell


def build_notebook() -> dict:
    """构建完整的 notebook 结构（v2 优化版）。"""
    cells = []

    # Cell 0: H1 标题
    cells.append(make_cell('markdown', '''# 真实连续数据全量特征批量提取（滑窗版 v2）

本 notebook 是 v1 的**性能优化版本**，用于对连续真实数据（npz/tdms 文件）进行**滑窗全量特征提取**。

## v2 优化内容（对比 v1）

| 优化项 | v1 | v2 | 预期收益 |
|--------|----|----|----------|
| 并行模式 | ThreadPoolExecutor | **ProcessPoolExecutor** | 突破 GIL 限制，CPU +20~30% |
| Workers 数量 | 固定 8 | **自动检测**（CPU 核心数 - 2） | 跑满物理核，CPU +10~15% |
| STFT 实现 | scipy.signal.stft (CPU) | **torch.stft (GPU)** | STFT 迁移到 GPU，总体 +15~20% |
| GPU 利用率 | ~45% | **~70~85%** | STFT + SVD 都在 GPU |
| CPU 利用率 | ~55% | **~75~90%** | ProcessPool + 更多 workers |
| 预期总体加速 | 基准（127s/文件） | **60~85s/文件** | **1.5~2.0x** |

## 核心特性

- **滑窗处理**：对连续信号按固定时长窗口滑动切分，窗口间可配置重叠率
- **全量特征**：每个窗口计算所有频带的全量特征（~80 特征/频带），非选定子集
- **统一采样率**：200kHz 数据自动升采样到 500kHz，确保特征跨文件可比
- **高效并行**：ProcessPoolExecutor 窗级并行 + torch.stft GPU 加速
- **断点续跑**：支持中断后从已处理位置继续，避免重复计算
- **分块输出**：每 N 个文件输出一个 CSV，降低单文件过大风险
''', 'cell-title'))

    # Cell 1: 环境导入
    cells.append(make_cell('code', '''from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# 启用 GPU
os.environ.setdefault('FEA_CPT_USE_GPU', '1')

# 确保 src 在 sys.path 中
workspace = Path.cwd()
if not (workspace / 'src').exists():
    workspace = workspace.parent
if str(workspace / 'src') not in sys.path:
    sys.path.insert(0, str(workspace / 'src'))

# v2: 使用 fea_cpt_gpu_v2 模块
from fea_cpt_gpu_v2.sliding_window import (
    SlidingWindowConfig,
    build_sliding_window_dataset,
    discover_source_files,
    gpu_backend_info,
    list_window_ranges,
    upsample_to_target,
    process_source_file,
    _auto_detect_workers,
)
from fea_cpt_gpu_v2.params import DEFAULT_FEATURE_PARAMS

print(f'workspace = {workspace}')
print(gpu_backend_info())
print(f'Python = {sys.version}')
print(f'CPU 核心数: {os.cpu_count()}')
print(f'推荐 workers: {_auto_detect_workers()}')
print('v2 所有模块加载成功')
''', 'cell-import'))

    # Cell 2: 全局配置
    cells.append(make_cell('code', '''# =========================
# 全局配置（在此处修改所有参数）
# =========================

# ========== 数据路径 ==========
# 支持单个路径或路径列表，会自动递归发现 .npz / .tdms 文件
RAW_DATA_ROOTS = [
    Path(r'G:\\20260323_ZZ_pccp\\FIP\\24-900-1800\\fip-24上午'),
    # 可添加更多路径：
    # Path(r'G:\\20260323_ZZ_pccp\\FIP\\24-900-1800\\fip-24下午'),
]

# ========== 滑窗参数 ==========
WINDOW_DURATION_S = 0.02   # 窗口时长（秒），默认 20ms
WINDOW_OVERLAP = 0.50      # 窗口重叠比例，默认 50%
assert 0.0 <= WINDOW_OVERLAP < 1.0, "重叠比例必须在 [0, 1) 范围内"

# ========== 升采样 ==========
# 低于此采样率的信号将自动升采样到此值（使用 scipy.signal.resample_poly）
# 200kHz -> 500kHz: 升采样 2.5 倍
# 500kHz -> 500kHz: 不处理
TARGET_SAMPLE_RATE = 500_000.0  # Hz

# ========== 预处理 ==========
# 整条信号去均值 + 带通滤波（在滑窗前执行一次）
PREPROC_BAND = (1_000.0, 95_000.0)  # Hz

# ========== 频带配置 ==========
# 每个频带会独立调用 build_context + compute_all_features，输出 ~80 个特征
# 频带越多，计算量越大，但特征越丰富
BANDS = [
    ('b_1k_100k',  (1_000.0,  100_000.0)),
    ('b_1k_10k',   (1_000.0,  10_000.0)),
    ('b_10k_20k',  (10_000.0, 20_000.0)),
    ('b_20k_40k',  (20_000.0, 40_000.0)),
    ('b_40k_60k',  (40_000.0, 60_000.0)),
    ('b_60k_100k', (60_000.0, 100_000.0)),
]

# ========== 并行设置 ==========
# v2: None 表示自动检测 CPU 核心数（推荐）
# 也可手动指定，如 WINDOW_WORKERS = 12
WINDOW_WORKERS = None
WINDOW_BATCH_SIZE = 256  # 每批处理的窗口数

# ========== 输出设置 ==========
OUTPUT_ROOT = workspace / 'outputs' / 'realdata_continuous_features_20260528_v2'
NPZ_PER_CSV = 100        # 每个 CSV 包含的文件数（分块输出）
MAX_FILES: int | None = None  # None = 处理全部文件，整数 = 仅处理前 N 个

# ========== TDMS 可选配置 ==========
TDMS_FALLBACK_SAMPLE_RATE_HZ: float | None = None  # TDMS 文件无法推断采样率时的备用值

# 创建输出目录
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

import os
auto_workers = _auto_detect_workers() if WINDOW_WORKERS is None else WINDOW_WORKERS
print(f'数据路径: {len(RAW_DATA_ROOTS)} 个根目录')
print(f'滑窗: {WINDOW_DURATION_S*1000:.0f}ms, 重叠 {WINDOW_OVERLAP*100:.0f}%')
print(f'目标采样率: {TARGET_SAMPLE_RATE/1000:.0f} kHz')
print(f'预处理频带: {PREPROC_BAND[0]}-{PREPROC_BAND[1]} Hz')
print(f'频带数: {len(BANDS)}')
print(f'并行: {auto_workers} workers (自动检测), batch={WINDOW_BATCH_SIZE}')
print(f'输出目录: {OUTPUT_ROOT}')
print(f'每 CSV 文件数: {NPZ_PER_CSV}')
print(f'最大文件数: {MAX_FILES if MAX_FILES else "全部"}')

# 构建配置对象
config = SlidingWindowConfig(
    bands=BANDS,
    preproc_band=PREPROC_BAND,
    window_duration_s=WINDOW_DURATION_S,
    window_overlap=WINDOW_OVERLAP,
    target_sample_rate=TARGET_SAMPLE_RATE,
    tdms_fallback_sample_rate=TDMS_FALLBACK_SAMPLE_RATE_HZ,
    window_workers=WINDOW_WORKERS,
    window_batch_size=WINDOW_BATCH_SIZE,
)
print('\\n配置对象创建成功')
''', 'cell-config'))

    # Cell 3: 数据文件发现
    cells.append(make_cell('code', '''# =========================
# 数据文件发现
# =========================

source_files = discover_source_files(RAW_DATA_ROOTS, max_files=MAX_FILES)

if not source_files:
    raise FileNotFoundError(f'未找到任何 .npz/.tdms 文件，请检查路径: {RAW_DATA_ROOTS}')

# 按父文件夹统计
from collections import Counter
folder_counts = Counter(f.parent.name for f in source_files)

print(f'发现 {len(source_files)} 个源文件')
print(f'\\n各文件夹文件数:')
for folder, count in sorted(folder_counts.items()):
    print(f'  {folder}: {count}')

# 检查格式分布
npz_count = sum(1 for f in source_files if f.suffix.lower() == '.npz')
tdms_count = sum(1 for f in source_files if f.suffix.lower() == '.tdms')
print(f'\\n文件格式: {npz_count} npz, {tdms_count} tdms')
''', 'cell-discovery'))

    # Cell 4: 单文件测试
    cells.append(make_cell('code', '''# =========================
# 单文件处理测试（可选）
# =========================
# 在批量处理前，先用第一个文件测试流程是否正常

import time

test_file = source_files[0]
print(f'测试文件: {test_file.name}')
print(f'  路径: {test_file}')

# 加载并检查基本信息
from fea_cpt_gpu_v2.sliding_window import load_source_file, upsample_to_target

src = load_source_file(test_file, config.tdms_fallback_sample_rate)
raw = np.asarray(src['signal_values'], dtype=float)
orig_rate = float(src['sample_rate'])
print(f'  原始采样率: {orig_rate/1000:.0f} kHz')
print(f'  原始样本数: {len(raw):,}')
print(f'  原始时长: {len(raw)/orig_rate:.3f} s')

# 测试升采样
sig_up, eff_rate = upsample_to_target(raw, orig_rate, config.target_sample_rate)
print(f'  升采样后采样率: {eff_rate/1000:.0f} kHz')
print(f'  升采样后样本数: {len(sig_up):,}')
print(f'  升采样倍数: {len(sig_up)/len(raw):.2f}x')

# 测试滑窗
from fea_cpt_gpu_v2.sliding_window import list_window_ranges
windows = list_window_ranges(len(sig_up), eff_rate, config.window_duration_s, config.window_overlap)
print(f'  窗口数: {len(windows)}')

# 测试完整流程
print('\\n开始单文件特征计算...')
t0 = time.time()
df_feat, df_log = process_source_file(test_file, config)
elapsed = time.time() - t0

print(f'  耗时: {elapsed:.1f} s')
print(f'  特征表形状: {df_feat.shape}')
print(f'  窗口数: {len(df_feat)}')
print(f'  特征列数: {len(df_feat.columns) - len([c for c in df_feat.columns if c.startswith("source") or c.startswith("window") or c.startswith("sample") or c.startswith("starttime") or c.startswith("arrival")])}')
print(f'  v1 基准耗时: 127.0 s')
print(f'  加速比: {127.0/elapsed:.2f}x')
print('\\n单文件测试通过！')
''', 'cell-test'))

    # Cell 5: 批量处理主流程
    cells.append(make_cell('code', '''# =========================
# 批量处理主流程
# =========================

import time
from datetime import datetime

processed_list_path = OUTPUT_ROOT / 'processed_source_files.txt'

print(f'{"="*60}')
print(f'批量处理开始 (v2 优化版)')
print(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'文件数: {len(source_files)}')
print(f'输出目录: {OUTPUT_ROOT}')
print(f'{"="*60}')

t_start = time.time()

stats = build_sliding_window_dataset(
    source_paths=source_files,
    config=config,
    output_dir=OUTPUT_ROOT,
    processed_list_path=processed_list_path,
    npz_per_csv=NPZ_PER_CSV,
    show_progress=True,
)

t_elapsed = time.time() - t_start

print(f'\\n{"="*60}')
print(f'批量处理完成')
print(f'耗时: {t_elapsed:.1f} s ({t_elapsed/60:.1f} min)')
print(f'新处理文件数: {stats["processed"]}')
print(f'跳过文件数: {stats["skipped"]}')
print(f'失败文件数: {stats["failed"]}')
print(f'总窗口数: {stats["windows"]}')
if stats["processed"] > 0:
    print(f'单文件平均耗时: {t_elapsed/stats["processed"]:.1f} s')
    print(f'单窗口平均耗时: {t_elapsed/stats["windows"]:.3f} s')
    print(f'v1 基准单文件耗时: 127.0 s')
    print(f'加速比: {127.0/(t_elapsed/stats["processed"]):.2f}x')
print(f'{"="*60}')
''', 'cell-main'))

    # Cell 6: 结果汇总
    cells.append(make_cell('code', '''# =========================
# 处理结果汇总
# =========================

# 读取输出的 CSV 文件
feature_chunks = sorted(OUTPUT_ROOT.glob('features_part_*.csv'))
log_chunks = sorted(OUTPUT_ROOT.glob('log_part_*.csv'))

print(f'特征 CSV 文件数: {len(feature_chunks)}')
print(f'日志 CSV 文件数: {len(log_chunks)}')

if feature_chunks:
    # 读取第一个 chunk 查看结构
    df_sample = pd.read_csv(feature_chunks[0], nrows=5)
    print(f'\\n特征 CSV 列数: {len(df_sample.columns)}')
    print(f'\\n前 3 列: {list(df_sample.columns[:3])}')
    print(f'后 3 列: {list(df_sample.columns[-3:])}')

    # 统计总窗口数
    total_windows = 0
    total_files_set = set()
    for chunk in feature_chunks:
        df = pd.read_csv(chunk, usecols=['source_file_name', 'window_id'])
        total_windows += len(df)
        total_files_set.update(df['source_file_name'].unique())
    print(f'\\n总窗口数: {total_windows:,}')
    print(f'总文件数: {len(total_files_set)}')

    # 采样率分布
    sr_col = 'sample_rate_hz'
    if sr_col in df_sample.columns:
        df_all_sr = pd.concat([pd.read_csv(c, usecols=[sr_col]) for c in feature_chunks], ignore_index=True)
        print(f'\\n采样率分布:')
        print(df_all_sr[sr_col].value_counts().sort_index())

    # 各文件窗口数统计
    print(f'\\n各文件窗口数 (前10):')
    df_all = pd.concat([pd.read_csv(c, usecols=['source_file_name', 'window_id']) for c in feature_chunks], ignore_index=True)
    win_counts = df_all.groupby('source_file_name')['window_id'].count().sort_values(ascending=False)
    for fname, count in win_counts.head(10).items():
        print(f'  {fname}: {count} windows')
''', 'cell-summary'))

    # Cell 7: 特征质量检查
    cells.append(make_cell('code', '''# =========================
# 特征质量检查
# =========================

if feature_chunks:
    # 随机抽样检查
    df_check = pd.read_csv(feature_chunks[0])

    # 找出特征列（非元数据列）
    meta_cols = {
        'source_file_name', 'source_file_path', 'source_format',
        'source_group_name', 'source_channel_name', 'source_detail',
        'window_id', 'window_start_index', 'window_end_index',
        'window_length_samples', 'window_step_samples',
        'window_duration_s', 'window_start_offset_s',
        'sample_rate_hz', 'original_sample_rate_hz',
        'source_n_samples', 'source_duration_s',
        'starttime_raw', 'arrival_time_raw', 'sample_type',
        'window_start_datetime',
    }
    feature_cols = [c for c in df_check.columns if c not in meta_cols]

    print(f'特征列数: {len(feature_cols)}')

    # 检查 NaN 比例
    nan_stats = []
    for col in feature_cols[:20]:  # 检查前20个特征
        series = pd.to_numeric(df_check[col], errors='coerce')
        nan_ratio = series.isna().sum() / len(series)
        nan_stats.append({
            'feature': col,
            'nan_ratio': nan_ratio,
            'min': series.min(),
            'max': series.max(),
            'mean': series.mean(),
        })

    df_nan = pd.DataFrame(nan_stats)
    print('\\n前20个特征的质量统计:')
    print(df_nan.to_string(index=False))

    # 检查是否有全 NaN 的特征
    full_nan = [c for c in feature_cols if pd.to_numeric(df_check[c], errors='coerce').isna().all()]
    if full_nan:
        print(f'\\n警告: {len(full_nan)} 个特征全为 NaN')
    else:
        print('\\n无全 NaN 特征，数据质量良好')
''', 'cell-quality'))

    # Cell 8: 运行日志
    cells.append(make_cell('code', '''# =========================
# 运行日志
# =========================

processed_log = OUTPUT_ROOT / 'processed_source_files.txt'
failed_log = OUTPUT_ROOT / 'failed_samples.log'

if processed_log.exists():
    lines = processed_log.read_text(encoding='utf-8').strip().split('\\n')
    print(f'已处理文件记录: {len(lines)} 条')
    print('\\n最近处理的 5 个文件:')
    for line in lines[-5:]:
        print(f'  {Path(line).name}')

if failed_log.exists():
    content = failed_log.read_text(encoding='utf-8')
    print(f'\\n失败样本日志:')
    print(content)
else:
    print('\\n无失败样本')
''', 'cell-log'))

    # Cell 9: 特征预览
    cells.append(make_cell('code', '''# =========================
# 特征预览
# =========================

if feature_chunks:
    df_preview = pd.read_csv(feature_chunks[0])
    print(f'数据形状: {df_preview.shape}')
    print(f'\\n元数据列示例:')
    meta_cols_preview = ['source_file_name', 'window_id', 'sample_rate_hz', 'window_start_offset_s']
    print(df_preview[meta_cols_preview].head(3).to_string(index=False))

    # 找到第一个特征列
    feature_like_cols = [c for c in df_preview.columns if '__' in c]
    if feature_like_cols:
        print(f'\\n特征列示例 (前5个):')
        print(df_preview[feature_like_cols[:5]].head(3).to_string(index=False))
''', 'cell-preview'))

    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {
                'name': 'python',
                'version': '3.12.0',
            },
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def main():
    """生成 notebook 文件。"""
    nb = build_notebook()

    output_path = Path(__file__).parent.parent / 'notebooks' / '2026-05-28-realdata_continuous_feature_batch_extract_v2.ipynb'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    # 中文自检
    content = output_path.read_text(encoding='utf-8')
    chinese_count = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
    replacement_count = content.count('\ufffd')
    print(f'Chinese character count: {chinese_count}')
    print(f'Replacement character count: {replacement_count}')
    if replacement_count > 0:
        print('警告: 发现替换字符，可能存在编码问题')
    else:
        print('中文编码自检通过')

    print(f'Notebook 已生成: {output_path}')
    print(f'Cell 数量: {len(nb["cells"])}')
    for i, cell in enumerate(nb['cells']):
        src_preview = cell['source'][0][:60] if cell['source'] else '(empty)'
        print(f'  Cell {i} [{cell["cell_type"]}]: {src_preview}...')


if __name__ == '__main__':
    main()
