#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成真实数据标签预测 notebook（UTF-8）。

说明：
- 仅使用 Python 标准库写入 .ipynb（json），避免终端重定向直接修改 notebook。
- notebook 内显式保证特征顺序：严格按 selected_features.csv 的顺序取列后再预测。
- 支持优先读取训练阶段导出的阈值（每个实验组单独阈值）。
"""

from __future__ import annotations

import json
from pathlib import Path


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


def main() -> None:
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb["cells"].append(
        md_cell(
            "# 真实数据标签预测（跨工况模型）\n\n"
            "本 notebook 用于：\n"
            "- 加载 `outputs/model_test_cross_condition_260518-2` 下 7 组实验模型；\n"
            "- 对真实数据特征目录做断丝标签预测；\n"
            "- 打印正负样本数量、总样本数量、正样本时间；\n"
            "- 绘制断丝疑似概率（%）随时间变化，并按时间不连续分段绘图。"
        )
    )

    nb["cells"].append(
        code_cell(
            "from __future__ import annotations\n\n"
            "from pathlib import Path\n"
            "import re\n"
            "import json\n\n"
            "import joblib\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "plt.rcParams['figure.dpi'] = 120\n"
            "plt.rcParams['axes.unicode_minus'] = False\n"
        )
    )

    nb["cells"].append(
        code_cell(
            "# ====== 用户配置区 ======\n"
            "PROJECT_ROOT = Path(r\"E:\\\\codes\\\\ZZ-BK\")\n"
            "MODEL_ROOT = PROJECT_ROOT / r\"outputs\\\\model_test_cross_condition_260518-2\"\n\n"
            "# 可配置多个真实特征目录\n"
            "FEATURE_DIRS = [\n"
            "    PROJECT_ROOT / r\"outputs\\\\realdata_feature_dataset_20260523\",\n"
            "    PROJECT_ROOT / r\"outputs\\\\realdata_feature_dataset_20260519_v3\",\n"
            "]\n\n"
            "# 模型类型：'logistic_regression' | 'rbf_svm' | 'random_forest'\n"
            "MODEL_TYPE = 'logistic_regression'\n\n"
            "# 指定实验组列表，例如 ['exp_01_train_F130', 'exp_07_train_F130_F130A_F130C']\n"
            "# 单实验也可直接写字符串，例如 'exp_07_train_F130_F130A_F130C'\n"
            "# 设为 None 时自动扫描全部 exp_*\n"
            "EXPERIMENTS = None\n\n"
            "# 是否优先使用训练阶段导出的阈值（推荐 True）\n"
            "USE_TRAINED_THRESHOLD = True\n\n"
            "# 手动阈值（当 USE_TRAINED_THRESHOLD=False 或未找到训练阈值时使用）\n"
            "# 二分类阈值：概率 >= 阈值 判为正样本（断丝）\n"
            "PROB_THRESHOLD = 0.5\n\n"
            "# 连续时间分段阈值（秒）\n"
            "CONTINUITY_GAP_SECONDS = 5.0\n\n"
            "OUTPUT_DIR = PROJECT_ROOT / r\"outputs\\\\realdata_prediction_cross_condition\"\n"
            "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n\n"
            "print('MODEL_ROOT =', MODEL_ROOT)\n"
            "print('MODEL_TYPE =', MODEL_TYPE)\n"
            "print('USE_TRAINED_THRESHOLD =', USE_TRAINED_THRESHOLD)\n"
            "print('FEATURE_DIRS =')\n"
            "for p in FEATURE_DIRS:\n"
            "    print('  -', p)\n"
        )
    )

    nb["cells"].append(
        code_cell(
            "def list_experiments(model_root: Path, experiments: list[str] | str | None) -> list[str]:\n"
            "    # 兼容三种输入：None / 单个字符串 / 字符串列表\n"
            "    if experiments is None:\n"
            "        return sorted([p.name for p in model_root.iterdir() if p.is_dir() and p.name.startswith('exp_')])\n"
            "    if isinstance(experiments, str):\n"
            "        exp = experiments.strip()\n"
            "        if not exp:\n"
            "            return sorted([p.name for p in model_root.iterdir() if p.is_dir() and p.name.startswith('exp_')])\n"
            "        return [exp]\n"
            "    if isinstance(experiments, (list, tuple)):\n"
            "        out = [str(x).strip() for x in experiments if str(x).strip()]\n"
            "        return out\n"
            "    raise TypeError('EXPERIMENTS must be None, str, list[str], or tuple[str, ...]')\n"
            "\n"
            "def load_selected_features(exp_dir: Path) -> list[str]:\n"
            "    sf = exp_dir / 'selected_features.csv'\n"
            "    if not sf.exists():\n"
            "        raise FileNotFoundError(f'missing selected_features.csv: {sf}')\n"
            "    df = pd.read_csv(sf)\n"
            "    if 'feature' not in df.columns:\n"
            "        raise ValueError(f\"selected_features.csv missing 'feature' column: {sf}\")\n"
            "    features = [str(x) for x in df['feature'].dropna().tolist()]\n"
            "    if len(features) != 5:\n"
            "        print(f'[WARN] {exp_dir.name} selected feature count = {len(features)} (expected 5)')\n"
            "    return features\n\n"
            "def load_model(exp_dir: Path, model_type: str):\n"
            "    model_path = exp_dir / 'models' / f'{model_type}.joblib'\n"
            "    if not model_path.exists():\n"
            "        raise FileNotFoundError(f'missing model file: {model_path}')\n"
            "    model = joblib.load(model_path)\n"
            "    if not hasattr(model, 'predict_proba'):\n"
            "        raise TypeError(f'model has no predict_proba: {model_path}')\n"
            "    return model, model_path\n\n"
            "def load_trained_threshold(exp_dir: Path, model_type: str) -> float | None:\n"
            "    # 优先从 summary.csv 按模型名读取；若缺失则回退到 metrics_*.json\n"
            "    summary_path = exp_dir / 'summary.csv'\n"
            "    if summary_path.exists():\n"
            "        sdf = pd.read_csv(summary_path)\n"
            "        if {'model_name', 'threshold'}.issubset(set(sdf.columns)):\n"
            "            sub = sdf[sdf['model_name'].astype(str) == str(model_type)]\n"
            "            if not sub.empty:\n"
            "                vals = pd.to_numeric(sub['threshold'], errors='coerce').dropna()\n"
            "                if not vals.empty:\n"
            "                    return float(vals.iloc[0])\n"
            "    for p in sorted(exp_dir.glob(f'metrics_{model_type}_*.json')):\n"
            "        try:\n"
            "            obj = json.loads(p.read_text(encoding='utf-8'))\n"
            "            if obj.get('threshold', None) is not None:\n"
            "                return float(obj['threshold'])\n"
            "        except Exception:\n"
            "            continue\n"
            "    return None\n\n"
            "def sorted_feature_files(feature_dir: Path) -> list[Path]:\n"
            "    return sorted(feature_dir.glob('*window_features*.csv'))\n\n"
            "def infer_log_path(feature_csv_path: Path) -> Path | None:\n"
            "    # 例如：window_features_v3_part_0001.csv -> window_log_v3_part_0001.csv\n"
            "    cand = feature_csv_path.with_name(feature_csv_path.name.replace('window_features', 'window_log'))\n"
            "    return cand if cand.exists() else None\n\n"
            "def load_feature_with_time(feature_csv_path: Path) -> pd.DataFrame:\n"
            "    df = pd.read_csv(feature_csv_path)\n"
            "    log_path = infer_log_path(feature_csv_path)\n"
            "    if log_path is not None:\n"
            "        log_df = pd.read_csv(log_path)\n"
            "        if 'window_start_datetime' in log_df.columns:\n"
            "            if 'window_start_datetime' not in df.columns:\n"
            "                df['window_start_datetime'] = log_df['window_start_datetime']\n"
            "            else:\n"
            "                # 优先保留特征表已有值，缺失时用 log 表补齐\n"
            "                df['window_start_datetime'] = df['window_start_datetime'].where(\n"
            "                    df['window_start_datetime'].notna(), log_df['window_start_datetime']\n"
            "                )\n"
            "    return df\n\n"
            "def build_time_series(df: pd.DataFrame) -> pd.Series:\n"
            "    # 优先级：window_start_datetime > starttime_raw > arrival_time_raw > offset秒\n"
            "    for col in ['window_start_datetime', 'starttime_raw', 'arrival_time_raw']:\n"
            "        if col in df.columns:\n"
            "            t = pd.to_datetime(df[col], errors='coerce')\n"
            "            if t.notna().any():\n"
            "                return t\n"
            "    offset_s = pd.to_numeric(df.get('window_start_offset_s', np.nan), errors='coerce')\n"
            "    if np.isfinite(offset_s).any():\n"
            "        return pd.Timestamp('1970-01-01') + pd.to_timedelta(offset_s.fillna(0.0), unit='s')\n"
            "    return pd.to_datetime(pd.RangeIndex(len(df)), unit='s', origin='unix', errors='coerce')\n\n"
            "def split_continuous_segments(ts: pd.Series, gap_seconds: float) -> pd.Series:\n"
            "    # 相邻时间差大于阈值则切分新段\n"
            "    dt = ts.sort_values().diff().dt.total_seconds()\n"
            "    cuts = dt.isna() | (dt > gap_seconds)\n"
            "    return (cuts.cumsum() - 1).rename('segment_id')\n"
        )
    )

    nb["cells"].append(
        code_cell(
            "experiments = list_experiments(MODEL_ROOT, EXPERIMENTS)\n"
            "print('experiments =', experiments)\n\n"
            "all_pred_rows = []\n"
            "summary_rows = []\n\n"
            "for exp_name in experiments:\n"
            "    exp_dir = MODEL_ROOT / exp_name\n"
            "    selected_features = load_selected_features(exp_dir)\n"
            "    model, model_path = load_model(exp_dir, MODEL_TYPE)\n"
            "    trained_threshold = load_trained_threshold(exp_dir, MODEL_TYPE)\n"
            "    if USE_TRAINED_THRESHOLD and trained_threshold is not None:\n"
            "        threshold_in_use = float(trained_threshold)\n"
            "        threshold_source = 'trained'\n"
            "    else:\n"
            "        threshold_in_use = float(PROB_THRESHOLD)\n"
            "        threshold_source = 'manual'\n\n"
            "    print('\\n' + '=' * 88)\n"
            "    print(f'Experiment: {exp_name}')\n"
            "    print(f'Model     : {MODEL_TYPE} ({model_path.name})')\n"
            "    print(f'Features  : {selected_features}')\n"
            "    print(f'Threshold : {threshold_in_use:.6f} (source={threshold_source})')\n\n"
            "    for feature_dir in FEATURE_DIRS:\n"
            "        files = sorted_feature_files(feature_dir)\n"
            "        if not files:\n"
            "            print(f'[WARN] no feature files in {feature_dir}')\n"
            "            continue\n"
            "        print(f'  FeatureDir: {feature_dir.name} files={len(files)}')\n\n"
            "        for feature_csv in files:\n"
            "            df = load_feature_with_time(feature_csv)\n"
            "            missing = [c for c in selected_features if c not in df.columns]\n"
            "            if missing:\n"
            "                print(f'    [SKIP] {feature_csv.name}: missing features -> {missing}')\n"
            "                continue\n\n"
            "            # 核心约束：严格按 selected_features 顺序取列，保证模型输入向量顺序一致\n"
            "            x = df.loc[:, selected_features].apply(pd.to_numeric, errors='coerce')\n"
            "            prob_pos = model.predict_proba(x)[:, 1]\n"
            "            pred = (prob_pos >= threshold_in_use).astype(int)\n"
            "            ts = build_time_series(df)\n\n"
            "            out = pd.DataFrame({\n"
            "                'experiment': exp_name,\n"
            "                'model_type': MODEL_TYPE,\n"
            "                'feature_dir': feature_dir.name,\n"
            "                'feature_file': feature_csv.name,\n"
            "                'source_file_name': df.get('source_file_name', ''),\n"
            "                'window_id': df.get('window_id', np.arange(len(df))),\n"
            "                'window_start_offset_s': pd.to_numeric(df.get('window_start_offset_s', np.nan), errors='coerce'),\n"
            "                'window_start_datetime': ts,\n"
            "                'prob_pos': prob_pos,\n"
            "                'prob_pos_pct': prob_pos * 100.0,\n"
            "                'pred_label': pred,\n"
            "                'threshold_in_use': threshold_in_use,\n"
            "                'threshold_source': threshold_source,\n"
            "            })\n\n"
            "            n_total = int(len(out))\n"
            "            n_pos = int((out['pred_label'] == 1).sum())\n"
            "            n_neg = n_total - n_pos\n"
            "            print(f'    {feature_csv.name}: pos={n_pos}, neg={n_neg}, total={n_total}')\n\n"
            "            summary_rows.append({\n"
            "                'experiment': exp_name,\n"
            "                'model_type': MODEL_TYPE,\n"
            "                'feature_dir': feature_dir.name,\n"
            "                'feature_file': feature_csv.name,\n"
            "                'positive_count': n_pos,\n"
            "                'negative_count': n_neg,\n"
            "                'total_count': n_total,\n"
            "                'threshold': threshold_in_use,\n"
            "                'threshold_source': threshold_source,\n"
            "            })\n"
            "            all_pred_rows.append(out)\n\n"
            "if not all_pred_rows:\n"
            "    raise RuntimeError('No predictions generated. Please check paths and model settings.')\n\n"
            "pred_df = pd.concat(all_pred_rows, ignore_index=True)\n"
            "summary_df = pd.DataFrame(summary_rows)\n"
            "print('\\nPrediction dataframe shape =', pred_df.shape)\n"
            "print('Summary dataframe shape    =', summary_df.shape)\n"
        )
    )

    nb["cells"].append(
        code_cell(
            "pred_out_csv = OUTPUT_DIR / f'predictions_{MODEL_TYPE}.csv'\n"
            "summary_out_csv = OUTPUT_DIR / f'summary_{MODEL_TYPE}.csv'\n"
            "pred_df.to_csv(pred_out_csv, index=False, encoding='utf-8-sig')\n"
            "summary_df.to_csv(summary_out_csv, index=False, encoding='utf-8-sig')\n"
            "print('saved:', pred_out_csv)\n"
            "print('saved:', summary_out_csv)\n"
            "summary_df.head(20)\n"
        )
    )

    nb["cells"].append(
        code_cell(
            "# 全局统计\n"
            "global_total = int(len(pred_df))\n"
            "global_pos = int((pred_df['pred_label'] == 1).sum())\n"
            "global_neg = global_total - global_pos\n"
            "print(f'[GLOBAL] pos={global_pos}, neg={global_neg}, total={global_total}')\n"
            "print('threshold mode =', 'trained' if USE_TRAINED_THRESHOLD else 'manual')\n\n"
            "# 分组统计\n"
            "agg = (\n"
            "    pred_df.groupby(['experiment', 'feature_dir', 'feature_file'], as_index=False)['pred_label']\n"
            "    .agg(positive_count=lambda s: int((s == 1).sum()), total_count='count')\n"
            ")\n"
            "agg['negative_count'] = agg['total_count'] - agg['positive_count']\n"
            "agg = agg[['experiment', 'feature_dir', 'feature_file', 'positive_count', 'negative_count', 'total_count']]\n"
            "agg.sort_values(['experiment', 'feature_dir', 'feature_file']).head(100)\n"
        )
    )

    nb["cells"].append(
        code_cell(
            "# 打印正样本时间（断丝样本）\n"
            "pos_df = pred_df[pred_df['pred_label'] == 1].copy()\n"
            "pos_df = pos_df.sort_values(['experiment', 'feature_dir', 'feature_file', 'window_start_datetime', 'window_id'])\n\n"
            "if pos_df.empty:\n"
            "    print('当前配置下没有预测为正样本的数据。')\n"
            "else:\n"
            "    cols = [\n"
            "        'experiment', 'feature_dir', 'feature_file', 'source_file_name',\n"
            "        'window_id', 'window_start_datetime', 'window_start_offset_s', 'prob_pos_pct'\n"
            "    ]\n"
            "    print('Positive sample rows =', len(pos_df))\n"
            "    display(pos_df[cols].head(500))\n"
        )
    )

    nb["cells"].append(
        code_cell(
            "# 按连续时间段绘制概率曲线\n"
            "plot_root = OUTPUT_DIR / f'plots_{MODEL_TYPE}'\n"
            "plot_root.mkdir(parents=True, exist_ok=True)\n"
            "plot_count = 0\n\n"
            "for (exp_name, feature_dir_name, feature_file), sub in pred_df.groupby(['experiment', 'feature_dir', 'feature_file']):\n"
            "    sub = sub.copy().sort_values(['window_start_datetime', 'window_id']).reset_index(drop=True)\n"
            "    ts = pd.to_datetime(sub['window_start_datetime'], errors='coerce')\n"
            "    valid = ts.notna()\n"
            "    if not valid.any():\n"
            "        print(f'[SKIP PLOT] no valid datetime: {exp_name} | {feature_dir_name} | {feature_file}')\n"
            "        continue\n\n"
            "    sub = sub.loc[valid].copy()\n"
            "    ts = pd.to_datetime(sub['window_start_datetime'])\n"
            "    sub['segment_id'] = split_continuous_segments(ts, gap_seconds=CONTINUITY_GAP_SECONDS).values\n\n"
            "    for segment_id, g in sub.groupby('segment_id'):\n"
            "        g = g.sort_values('window_start_datetime')\n"
            "        if len(g) < 2:\n"
            "            continue\n\n"
            "        fig, ax = plt.subplots(figsize=(11, 4))\n"
            "        x = pd.to_datetime(g['window_start_datetime'])\n"
            "        y = g['prob_pos_pct']\n"
            "        th = float(g['threshold_in_use'].iloc[0])\n"
            "        ax.plot(x, y, lw=1.4)\n"
            "        ax.axhline(th * 100.0, color='r', ls='--', lw=1.0, label=f'threshold={th:.3f}')\n"
            "        ax.set_ylim(0, 100)\n"
            "        ax.set_xlabel('时间 (HH:MM:SS)')\n"
            "        ax.set_ylabel('断丝疑似概率 (%)')\n"
            "        ax.set_title(f'{exp_name} | {MODEL_TYPE} | {feature_dir_name} | {feature_file} | seg={int(segment_id)}')\n"
            "        ax.grid(alpha=0.3)\n"
            "        ax.legend(loc='upper right')\n"
            "        fig.autofmt_xdate()\n\n"
            "        safe_file = re.sub(r'[^0-9A-Za-z._-]+', '_', feature_file)\n"
            "        out_png = plot_root / f'{exp_name}__{feature_dir_name}__{safe_file}__seg{int(segment_id):03d}.png'\n"
            "        fig.savefig(out_png, bbox_inches='tight')\n"
            "        plt.show()\n"
            "        plt.close(fig)\n"
            "        plot_count += 1\n\n"
            "print('saved plots =', plot_count)\n"
            "print('plot dir    =', plot_root)\n"
        )
    )

    nb["cells"].append(
        md_cell(
            "## 使用说明\n\n"
            "1. 在“用户配置区”修改 `FEATURE_DIRS`、`MODEL_TYPE`、`EXPERIMENTS`、`USE_TRAINED_THRESHOLD`、`PROB_THRESHOLD`。\n"
            "2. 顺序执行全部单元格。\n"
            "3. 输出内容：\n"
            "- `outputs/realdata_prediction_cross_condition/predictions_<model_type>.csv`\n"
            "- `outputs/realdata_prediction_cross_condition/summary_<model_type>.csv`\n"
            "- `outputs/realdata_prediction_cross_condition/plots_<model_type>/*.png`\n"
        )
    )

    nb["cells"].append(
        md_cell(
            "## 附加预测：`dataset_build_260518_features_gpu`\n\n"
            "本节使用当前已选实验组与模型，对\n"
            "`E:\\\\codes\\\\ZZ-BK\\\\outputs\\\\dataset_build_260518_features_gpu`\n"
            "中的 `*_features.csv` 样本执行标签预测。"
        )
    )

    nb["cells"].append(
        code_cell(
            "# 附加预测目录（可按需修改）\n"
            "EXTRA_FEATURE_ROOT = PROJECT_ROOT / r\"outputs\\\\dataset_build_260518_features_gpu\"\n"
            "extra_files = sorted(EXTRA_FEATURE_ROOT.glob('*_features.csv'))\n"
            "print('EXTRA_FEATURE_ROOT =', EXTRA_FEATURE_ROOT)\n"
            "print('extra feature files =', len(extra_files))\n"
            "for p in extra_files[:20]:\n"
            "    print('  -', p.name)\n"
        )
    )

    nb["cells"].append(
        code_cell(
            "extra_rows = []\n"
            "extra_summary_rows = []\n\n"
            "for exp_name in experiments:\n"
            "    exp_dir = MODEL_ROOT / exp_name\n"
            "    selected_features = load_selected_features(exp_dir)\n"
            "    model, _ = load_model(exp_dir, MODEL_TYPE)\n"
            "    trained_threshold = load_trained_threshold(exp_dir, MODEL_TYPE)\n"
            "    if USE_TRAINED_THRESHOLD and trained_threshold is not None:\n"
            "        threshold_in_use = float(trained_threshold)\n"
            "        threshold_source = 'trained'\n"
            "    else:\n"
            "        threshold_in_use = float(PROB_THRESHOLD)\n"
            "        threshold_source = 'manual'\n\n"
            "    print('\\n' + '-' * 88)\n"
            "    print(f'[EXTRA] Experiment={exp_name}, threshold={threshold_in_use:.6f} ({threshold_source})')\n\n"
            "    for feature_csv in extra_files:\n"
            "        df = pd.read_csv(feature_csv)\n"
            "        missing = [c for c in selected_features if c not in df.columns]\n"
            "        if missing:\n"
            "            print(f'  [SKIP] {feature_csv.name}: missing features -> {missing}')\n"
            "            continue\n\n"
            "        # 保持特征顺序与训练完全一致\n"
            "        x = df.loc[:, selected_features].apply(pd.to_numeric, errors='coerce')\n"
            "        prob_pos = model.predict_proba(x)[:, 1]\n"
            "        pred = (prob_pos >= threshold_in_use).astype(int)\n\n"
            "        ts = build_time_series(df)\n"
            "        out = pd.DataFrame({\n"
            "            'experiment': exp_name,\n"
            "            'model_type': MODEL_TYPE,\n"
            "            'feature_file': feature_csv.name,\n"
            "            'source_file_name': df.get('source_file_name', ''),\n"
            "            'window_id': df.get('window_id', np.arange(len(df))),\n"
            "            'window_start_datetime': ts,\n"
            "            'prob_pos': prob_pos,\n"
            "            'prob_pos_pct': prob_pos * 100.0,\n"
            "            'pred_label': pred,\n"
            "            'threshold_in_use': threshold_in_use,\n"
            "            'threshold_source': threshold_source,\n"
            "        })\n\n"
            "        n_total = int(len(out))\n"
            "        n_pos = int((out['pred_label'] == 1).sum())\n"
            "        n_neg = n_total - n_pos\n"
            "        print(f'  {feature_csv.name}: pos={n_pos}, neg={n_neg}, total={n_total}')\n\n"
            "        extra_rows.append(out)\n"
            "        extra_summary_rows.append({\n"
            "            'experiment': exp_name,\n"
            "            'model_type': MODEL_TYPE,\n"
            "            'feature_file': feature_csv.name,\n"
            "            'positive_count': n_pos,\n"
            "            'negative_count': n_neg,\n"
            "            'total_count': n_total,\n"
            "            'threshold': threshold_in_use,\n"
            "            'threshold_source': threshold_source,\n"
            "        })\n\n"
            "if not extra_rows:\n"
            "    print('[EXTRA] 没有生成预测结果，请检查输入目录或特征列一致性。')\n"
            "else:\n"
            "    extra_pred_df = pd.concat(extra_rows, ignore_index=True)\n"
            "    extra_summary_df = pd.DataFrame(extra_summary_rows)\n"
            "    extra_out_dir = OUTPUT_DIR / 'extra_dataset_build_260518_features_gpu'\n"
            "    extra_out_dir.mkdir(parents=True, exist_ok=True)\n"
            "    extra_pred_csv = extra_out_dir / f'predictions_{MODEL_TYPE}.csv'\n"
            "    extra_summary_csv = extra_out_dir / f'summary_{MODEL_TYPE}.csv'\n"
            "    extra_pred_df.to_csv(extra_pred_csv, index=False, encoding='utf-8-sig')\n"
            "    extra_summary_df.to_csv(extra_summary_csv, index=False, encoding='utf-8-sig')\n"
            "    print('saved:', extra_pred_csv)\n"
            "    print('saved:', extra_summary_csv)\n"
            "    display(extra_summary_df.head(50))\n"
        )
    )

    out_path = Path(r"E:\codes\ZZ-BK\notebooks\2026-05-25-realdata_label_prediction_cross_condition.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

    print(str(out_path))


if __name__ == "__main__":
    main()
