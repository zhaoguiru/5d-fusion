#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LHD 物理验证：从 v6e summary.csv 直接分析
==========================================

无需外部数据库，直接利用 v6e 已有的输出做物理验证：
1. 从 shot 名称推断诊断类型
2. 四象限分类（基于 B_mean / S_mean）
3. 信号代理量与 k_in 的相关性
4. 按诊断类型的 k_in 分布
5. 四象限 × 诊断类型交叉分析

作者: 赵国瑞
日期: 2026-06-05
"""

import os
import sys
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# 配置
# ============================================================

# 诊断类型推断：从 shot 名称或目录结构推断
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
    """从 shot 名称推断诊断类型。"""
    for diag_type, info in DIAG_TYPE_PATTERNS.items():
        if re.search(info['pattern'], shot_name):
            return diag_type
    return 'Unknown'


# ============================================================
# 读取 summary.csv
# ============================================================

def load_summary(csv_path):
    """读取 v6e 的 summary.csv，返回 DataFrame。"""
    if not os.path.exists(csv_path):
        print(f"[ERROR] 文件不存在: {csv_path}")
        return None

    records = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('status') != 'ok':
                continue
            # 转换数值字段
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

    # 推断诊断类型
    df['diag_type'] = df['shot'].apply(infer_diag_type)

    print(f"[INFO] 读取 {len(df)} 条成功记录")
    print(f"[INFO] 诊断类型分布:")
    print(df['diag_type'].value_counts().to_string())

    return df


# ============================================================
# 四象限分类
# ============================================================

def classify_quadrant(df):
    """基于 B_mean 和 S_mean 的中位数进行四象限分类。"""
    b_med = df['B_mean'].median()
    s_med = df['S_mean'].median()

    def get_quadrant(row):
        b = row['B_mean']
        s = row['S_mean']
        if b >= b_med and s >= s_med:
            return 'Q1'
        elif b < b_med and s >= s_med:
            return 'Q2'
        elif b < b_med and s < s_med:
            return 'Q3'
        else:
            return 'Q4'

    df['quadrant'] = df.apply(get_quadrant, axis=1)

    print(f"\n[INFO] 四象限分类（B median={b_med:.4f}, S median={s_med:.4f}）:")
    print(df['quadrant'].value_counts().sort_index().to_string())

    return df


# ============================================================
# 信号代理量验证
# ============================================================

def validate_with_proxies(df, output_dir):
    """用信号代理量验证 k_in。"""
    os.makedirs(output_dir, exist_ok=True)

    # 定义代理量
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

        # 散点图
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
        print(f"\n[OK] 统计汇总: {output_dir}/proxy_correlation_stats.csv")
        return stats_df

    return None


# ============================================================
# 按诊断类型分析
# ============================================================

def analyze_by_diagnosis(df, output_dir):
    """按诊断类型分析 k_in 分布。"""
    os.makedirs(output_dir, exist_ok=True)

    # 统计
    stats = df.groupby('diag_type').agg({
        'k_in_mean': ['count', 'mean', 'std', 'min', 'max', 'median']
    }).reset_index()
    stats.columns = ['diag_type', 'count', 'mean', 'std', 'min', 'max', 'median']

    # 添加标签
    stats['label'] = stats['diag_type'].apply(
        lambda x: DIAG_TYPE_PATTERNS.get(x, {}).get('label', x)
    )
    stats['phys'] = stats['diag_type'].apply(
        lambda x: DIAG_TYPE_PATTERNS.get(x, {}).get('phys', x)
    )

    stats.to_csv(os.path.join(output_dir, 'kin_stats_by_diagnosis.csv'), index=False)

    print("\n=== k_in 按诊断类型分布 ===")
    print(stats.to_string(index=False))

    # 箱线图
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
        ax.set_title('$k_{in}$ Distribution Across Diagnostic Types\\n(Physical Validation)', fontsize=16)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=20, ha='right', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kin_by_diagnosis_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] {output_dir}/kin_by_diagnosis_boxplot.png")

    # 小提琴图
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_data = [d for d in data if len(d) > 0]
    plot_labels = [l for l, d in zip(labels, data) if len(d) > 0]
    if plot_data:
        parts = ax.violinplot(plot_data, positions=range(1, len(plot_data)+1),
                              showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(plot_labels)+1))
        ax.set_xticklabels(plot_labels, rotation=20, ha='right', fontsize=10)
        ax.set_ylabel('$k_{in}$ (mean)', fontsize=14)
        ax.set_title('$k_{in}$ Probability Density by Diagnostic Type', fontsize=16)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kin_by_diagnosis_violin.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] {output_dir}/kin_by_diagnosis_violin.png")

    return stats


# ============================================================
# 四象限 × 诊断类型交叉分析
# ============================================================

def cross_analysis(df, output_dir):
    """交叉分析四象限和诊断类型。"""
    os.makedirs(output_dir, exist_ok=True)

    if 'quadrant' not in df.columns:
        print("[WARN] 无 quadrant 列，跳过交叉分析")
        return None

    # 交叉表：平均 k_in
    cross_mean = pd.crosstab(df['quadrant'], df['diag_type'], values=df['k_in_mean'], aggfunc='mean')
    cross_count = pd.crosstab(df['quadrant'], df['diag_type'])

    cross_mean.to_csv(os.path.join(output_dir, 'cross_mean_kin.csv'))
    cross_count.to_csv(os.path.join(output_dir, 'cross_count.csv'))

    print("\n=== 四象限 × 诊断类型: 平均 k_in ===")
    print(cross_mean.to_string())
    print("\n=== 样本数 ===")
    print(cross_count.to_string())

    # 热力图
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

    # 关键发现：Q3 象限的 k_in 是否显著高于其他象限？
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

    return cross_mean


# ============================================================
# 五维特征分布
# ============================================================

def plot_5d_distributions(df, output_dir):
    """绘制五维特征的分布。"""
    os.makedirs(output_dir, exist_ok=True)

    dims = ['B_mean', 'S_mean', 'R_mean', 'D_mean', 'I_mean']
    labels = ['Boundary (B)', 'Structure (S)', 'Reserve (R)', 'Direction (D)', 'Intensity (I)']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (dim, label) in enumerate(zip(dims, labels)):
        ax = axes[i]
        data = df[dim].dropna()
        ax.hist(data, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={data.mean():.3f}')
        ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median={data.median():.3f}')
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{label} Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # k_in 分布
    ax = axes[5]
    data = df['k_in_mean'].dropna()
    ax.hist(data, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={data.mean():.3f}')
    ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median={data.median():.3f}')
    ax.set_xlabel('$k_{in}$', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('$k_{in}$ Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5d_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {output_dir}/5d_distributions.png")


# ============================================================
# 主程序
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("用法: python validate_v6e_summary.py summary.csv [output_dir]")
        print("  summary.csv: v6e 生成的 summary.csv")
        print("  output_dir: 输出目录（默认: ./validation_output）")
        sys.exit(1)

    summary_csv = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './validation_output'

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("LHD 物理验证（从 v6e summary.csv）")
    print("=" * 70)

    # 1. 读取数据
    print("\n[Step 1] 读取 summary.csv...")
    df = load_summary(summary_csv)
    if df is None or len(df) == 0:
        print("[ERROR] 无有效数据")
        sys.exit(1)

    # 2. 四象限分类
    print("\n[Step 2] 四象限分类...")
    df = classify_quadrant(df)

    # 3. 五维分布
    print("\n[Step 3] 五维特征分布...")
    plot_5d_distributions(df, output_dir)

    # 4. 按诊断类型分析
    print("\n[Step 4] 按诊断类型分析...")
    analyze_by_diagnosis(df, output_dir)

    # 5. 信号代理量验证
    print("\n[Step 5] 信号代理量验证...")
    validate_with_proxies(df, output_dir)

    # 6. 交叉分析
    print("\n[Step 6] 四象限 × 诊断类型交叉分析...")
    cross_analysis(df, output_dir)

    # 7. 保存完整数据
    df.to_csv(os.path.join(output_dir, 'validated_data.csv'), index=False)

    print("\n" + "=" * 70)
    print("验证完成！")
    print("=" * 70)
    print(f"\n输出目录: {os.path.abspath(output_dir)}")
    print(f"\n论文可用图表:")
    print(f"  {output_dir}/5d_distributions.png          - 五维 + k_in 分布")
    print(f"  {output_dir}/kin_by_diagnosis_boxplot.png    - 诊断类型箱线图")
    print(f"  {output_dir}/kin_by_diagnosis_violin.png     - 诊断类型小提琴图")
    print(f"  {output_dir}/quadrant_diag_heatmap.png       - 四象限 × 诊断类型热力图")
    print(f"  {output_dir}/kin_vs_*.png                    - 代理量散点图")
    print(f"\n论文可用数据:")
    print(f"  {output_dir}/kin_stats_by_diagnosis.csv      - 诊断类型统计")
    print(f"  {output_dir}/proxy_correlation_stats.csv       - 代理量相关性")
    print(f"  {output_dir}/cross_mean_kin.csv              - 交叉分析表")
    print(f"  {output_dir}/validated_data.csv              - 完整数据（含 quadrant）")

    print("\n论文写作素材:")
    print("  - 如果不同诊断类型的 k_in 均值接近 → '五维映射的诊断无关性'")
    print("  - 如果 Q3 象限 k_in 显著高于其他 → '启动瞬态的多维耦合'")
    print("  - 如果代理量与 k_in 相关性弱 → 'k_in 是独立维度'")


if __name__ == '__main__':
    main()
