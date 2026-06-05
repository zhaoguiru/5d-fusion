#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LHD 物理验证（修正版：四象限基于 B 和 R）
==========================================

修正：论文中的四象限分类基于 B（边界）和 R（储备），
不是 B 和 S（结构）。

作者: 赵国瑞
日期: 2026-06-05
"""

import os
import sys
import re
import csv
import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib 未安装，仅输出 CSV 和文本报告")

# ============================================================
# 诊断类型推断
# ============================================================

DIAG_TYPE_PATTERNS = {
    'Bolometer': {
        'pattern': r'(?i)bolometer|bolo|rad',
        'label': 'Bolometer (Radiation)',
        'phys': 'P_rad',
    },
    'Divertor-Interferometer': {
        'pattern': r'(?i)divertor.*interferometer|div.*interf|interferometer',
        'label': 'Divertor Interferometer (Density)',
        'phys': 'n_e',
    },
    'SXfluc': {
        'pattern': r'(?i)sxfluc|sx.*fluc|soft.*x.*fluc',
        'label': 'SX Fluctuation (Core T_e fluctuation)',
        'phys': 'Te_fluc',
    },
    'SXmp': {
        'pattern': r'(?i)sxmp|sx.*mp|soft.*x.*mp',
        'label': 'SX Multi-Pinhole (Core T_e profile)',
        'phys': 'Te_prof',
    },
    'ECE': {
        'pattern': r'(?i)ece|electron.*cyclotron',
        'label': 'ECE (Electron Temperature)',
        'phys': 'Te_ece',
    },
    'MP': {
        'pattern': r'(?i)mp$|magnetic.*probe|mirnov',
        'label': 'Magnetic Probe (B-fluctuation)',
        'phys': 'B_fluc',
    },
}


def infer_diag_type(shot_name):
    for diag_type, info in DIAG_TYPE_PATTERNS.items():
        if re.search(info['pattern'], shot_name):
            return diag_type
    return 'Unknown'


# ============================================================
# 读取 summary.csv
# ============================================================

def load_summary(csv_path):
    if not os.path.exists(csv_path):
        print(f"[ERROR] 文件不存在: {csv_path}")
        return None

    records = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('status') != 'ok':
                continue
            numeric_fields = ['n_samples', 'n_windows', 'duration_ms',
                            'k_in_mean', 'k_in_std', 'k_in_max', 'k_in_min',
                            'n_events', 'event_severity_max',
                            'B_mean', 'S_mean', 'R_mean', 'D_mean', 'I_mean']
            for field in numeric_fields:
                val = row.get(field, '')
                if val == '':
                    row[field] = np.nan
                else:
                    try:
                        row[field] = float(val)
                    except ValueError:
                        row[field] = np.nan
            records.append(row)

    df = pd.DataFrame(records)
    df['diag_type'] = df['shot'].apply(infer_diag_type)

    print(f"[INFO] 读取 {len(df)} 条成功记录")
    print(f"[INFO] 诊断类型分布:")
    print(df['diag_type'].value_counts().to_string())

    return df


# ============================================================
# 四象限分类（修正：基于 B 和 R，与论文一致）
# ============================================================

def classify_quadrant_br(df):
    """
    基于 B_mean 和 R_mean 的四象限分类，与论文一致。

    Q1: 高 B, 高 R — 理想型（大型、高储备）
    Q2: 低 B, 高 R — 宝藏型（紧凑但高储备）
    Q3: 低 B, 低 R — 淘汰型（双低）
    Q4: 高 B, 低 R — 陷阱型（高边界但低储备）
    """
    b_med = df['B_mean'].median()
    r_med = df['R_mean'].median()

    def get_quadrant(row):
        b = row['B_mean']
        r = row['R_mean']
        if b >= b_med and r >= r_med:
            return 'Q1'
        elif b < b_med and r >= r_med:
            return 'Q2'
        elif b < b_med and r < r_med:
            return 'Q3'
        else:
            return 'Q4'

    df['quadrant'] = df.apply(get_quadrant, axis=1)

    print(f"\n[INFO] 四象限分类（B median={b_med:.4f}, R median={r_med:.4f}）:")
    quad_counts = df['quadrant'].value_counts().sort_index()
    print(quad_counts.to_string())

    # 各象限 k_in 统计
    print(f"\n=== 各象限 k_in 统计 ===")
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        sub = df[df['quadrant'] == q]['k_in_mean'].dropna()
        if len(sub) > 0:
            print(f"  {q}: n={len(sub)}, mean={sub.mean():.4f}, std={sub.std():.4f}, "
                  f"median={sub.median():.4f}, min={sub.min():.4f}, max={sub.max():.4f}")

    return df


# ============================================================
# 信号代理量验证
# ============================================================

def validate_with_proxies(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    proxy_vars = [
        ('k_in_std', 'k_in Standard Deviation', 'Multi-channel coupling strength'),
        ('n_events', 'Number of Events', 'Disruption/instability frequency'),
        ('event_severity_max', 'Max Event Severity', 'Peak fluctuation amplitude'),
        ('duration_ms', 'Duration (ms)', 'Discharge length'),
        ('n_windows', 'Number of Windows', 'Temporal segmentation count'),
    ]

    stats_records = []

    for col, label, phys_meaning in proxy_vars:
        if col not in df.columns:
            continue

        valid = df[[col, 'k_in_mean']].dropna()
        if len(valid) < 10:
            continue

        pr, pp = stats.pearsonr(valid[col], valid['k_in_mean'])
        sr, sp = stats.spearmanr(valid[col], valid['k_in_mean'])

        stats_records.append({
            'proxy': col,
            'label': label,
            'physical_meaning': phys_meaning,
            'n': len(valid),
            'pearson_r': pr,
            'pearson_p': pp,
            'spearman_r': sr,
            'spearman_p': sp,
        })

        print(f"\n{label} ({col}):")
        print(f"  n={len(valid)}, Pearson r={pr:.4f} (p={pp:.2e}), Spearman rho={sr:.4f} (p={sp:.2e})")
        print(f"  物理含义: {phys_meaning}")

        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(valid[col], valid['k_in_mean'], c='steelblue', alpha=0.4, s=20, edgecolors='none')
            z = np.polyfit(valid[col], valid['k_in_mean'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid[col].min(), valid[col].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', lw=2, label='Linear fit')
            text = (f"n = {len(valid)}\n"
                    f"Pearson r = {pr:.4f}\n"
                    f"Spearman rho = {sr:.4f}\n"
                    f"p = {pp:.2e}")
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel('$k_{in}$ (mean)', fontsize=12)
            ax.set_title(f'$k_{{in}}$ vs {label}', fontsize=14)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'kin_vs_{col}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  [OK] 散点图: {output_dir}/kin_vs_{col}.png")

    if stats_records:
        stats_df = pd.DataFrame(stats_records)
        stats_df.to_csv(os.path.join(output_dir, 'proxy_correlation_stats.csv'), index=False)
        print(f"\n[OK] {output_dir}/proxy_correlation_stats.csv")
        return stats_df

    return None


# ============================================================
# 按诊断类型分析
# ============================================================

def analyze_by_diagnosis(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    stats = df.groupby('diag_type').agg({
        'k_in_mean': ['count', 'mean', 'std', 'min', 'max', 'median']
    }).reset_index()
    stats.columns = ['diag_type', 'count', 'mean', 'std', 'min', 'max', 'median']

    stats['label'] = stats['diag_type'].apply(
        lambda x: DIAG_TYPE_PATTERNS.get(x, {}).get('label', x)
    )
    stats['phys'] = stats['diag_type'].apply(
        lambda x: DIAG_TYPE_PATTERNS.get(x, {}).get('phys', x)
    )

    stats.to_csv(os.path.join(output_dir, 'kin_stats_by_diagnosis.csv'), index=False)

    print("\n=== k_in 按诊断类型分布 ===")
    print(stats.to_string(index=False))

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(12, 7))
        diag_types = [dt for dt in DIAG_TYPE_PATTERNS.keys() if dt in df['diag_type'].values]
        data = [df[df['diag_type'] == dt]['k_in_mean'].dropna() for dt in diag_types]
        labels = [DIAG_TYPE_PATTERNS[dt]['label'] for dt in diag_types]

        if data and any(len(d) > 0 for d in data):
            bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel('$k_{in}$ (mean)', fontsize=14)
            ax.set_title('$k_{in}$ Distribution Across Diagnostic Types', fontsize=16)
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=20, ha='right', fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'kin_by_diagnosis_boxplot.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OK] {output_dir}/kin_by_diagnosis_boxplot.png")

    return stats


# ============================================================
# 交叉分析
# ============================================================

def cross_analysis(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if 'quadrant' not in df.columns:
        print("[WARN] 无 quadrant 列")
        return None

    cross_mean = pd.crosstab(df['quadrant'], df['diag_type'], values=df['k_in_mean'], aggfunc='mean')
    cross_count = pd.crosstab(df['quadrant'], df['diag_type'])

    cross_mean.to_csv(os.path.join(output_dir, 'cross_mean_kin.csv'))
    cross_count.to_csv(os.path.join(output_dir, 'cross_count.csv'))

    print("\n=== 四象限 × 诊断类型: 平均 k_in ===")
    print(cross_mean.to_string())
    print("\n=== 样本数 ===")
    print(cross_count.to_string())

    # Q3 检验
    q3_kin = df[df['quadrant'] == 'Q3']['k_in_mean'].dropna()
    other_kin = df[df['quadrant'] != 'Q3']['k_in_mean'].dropna()

    if len(q3_kin) > 5 and len(other_kin) > 5:
        t_stat, t_p = stats.ttest_ind(q3_kin, other_kin)
        print(f"\n=== Q3 象限 vs 其他象限 t 检验 ===")
        print(f"  Q3: n={len(q3_kin)}, mean={q3_kin.mean():.4f}, std={q3_kin.std():.4f}")
        print(f"  Other: n={len(other_kin)}, mean={other_kin.mean():.4f}, std={other_kin.std():.4f}")
        print(f"  t-statistic={t_stat:.4f}, p-value={t_p:.2e}")
        if t_p < 0.001:
            print(f"  *** Q3 象限 k_in 显著高于其他象限 (p < 0.001)")
        elif t_p < 0.05:
            print(f"  ** Q3 象限 k_in 显著高于其他象限 (p < 0.05)")
        else:
            print(f"  Q3 象限 k_in 与其他象限无显著差异 (p = {t_p:.3f})")
    elif len(q3_kin) > 0:
        print(f"\n=== Q3 象限 ===")
        print(f"  Q3: n={len(q3_kin)}, mean={q3_kin.mean():.4f}, std={q3_kin.std():.4f}")
        print(f"  [WARN] Q3 样本数不足，无法进行 t 检验")

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(cross_mean.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=cross_mean.values.max())
        ax.set_xticks(range(len(cross_mean.columns)))
        ax.set_xticklabels(cross_mean.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(cross_mean.index)))
        ax.set_yticklabels(cross_mean.index)
        for i in range(len(cross_mean.index)):
            for j in range(len(cross_mean.columns)):
                if not np.isnan(cross_mean.values[i, j]):
                    ax.text(j, i, f'{cross_mean.values[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=10)
        plt.colorbar(im, ax=ax, label='Mean $k_{in}$')
        ax.set_title('$k_{in}$ Heatmap: Quadrant × Diagnostic Type', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quadrant_diag_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] {output_dir}/quadrant_diag_heatmap.png")

    return cross_mean


# ============================================================
# 主程序
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("用法: python validate_v6e_fixed.py summary.csv [output_dir]")
        sys.exit(1)

    summary_csv = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './validation_output_fixed'

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("LHD 物理验证（修正版：四象限基于 B 和 R）")
    print("=" * 70)

    print("\n[Step 1] 读取 summary.csv...")
    df = load_summary(summary_csv)
    if df is None or len(df) == 0:
        print("[ERROR] 无有效数据")
        sys.exit(1)

    print("\n[Step 2] 四象限分类（基于 B 和 R）...")
    df = classify_quadrant_br(df)

    print("\n[Step 3] 按诊断类型分析...")
    analyze_by_diagnosis(df, output_dir)

    print("\n[Step 4] 信号代理量验证...")
    validate_with_proxies(df, output_dir)

    print("\n[Step 5] 交叉分析...")
    cross_analysis(df, output_dir)

    df.to_csv(os.path.join(output_dir, 'validated_data_fixed.csv'), index=False)

    print("\n" + "=" * 70)
    print("验证完成！")
    print("=" * 70)
    print(f"\n输出目录: {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    main()
