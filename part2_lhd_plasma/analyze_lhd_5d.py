#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LHD 5D Data Analysis & Visualization Suite
五维系统论核聚变数据分析与论文级画图脚本
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
FUSION_DIR = "/mnt/d/vmare/zgr/AI/fusion"
INPUT_CSV = os.path.join(FUSION_DIR, "lhd_5d_results_v6e/summary.csv")
OUTPUT_DIR = os.path.join(FUSION_DIR, "lhd_5d_figures")

DIM_LABELS = ["B", "S", "R", "D", "I"]
DIM_FULL_NAMES = {
    "B": "Boundary (B)",
    "S": "Structure (S)",
    "R": "Reserve (R)",
    "D": "Direction (D)",
    "I": "Intensity (I)"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 论文级字体设置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# ==================== 数据加载 ====================

def load_data(csv_path):
    """加载 CSV 并返回 DataFrame（仅成功样本）"""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                rows.append(row)

    df = pd.DataFrame(rows)

    # 转换数值列
    numeric_cols = ["n_samples", "n_windows", "duration_ms", "k_in_mean", "k_in_std",
                    "k_in_max", "k_in_min", "n_events", "event_severity_max",
                    "B_mean", "S_mean", "R_mean", "D_mean", "I_mean"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 提取 shot 编号（用于时间序列）
    df['shot_num'] = df['shot'].str.extract(r'(\d{5,})').astype(float)

    return df

# ==================== 图 1: 五维分布直方图 ====================

def plot_5d_histograms(df, outdir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

    for idx, dim in enumerate(DIM_LABELS):
        ax = axes[idx]
        vals = df[f"{dim}_mean"].dropna()

        ax.hist(vals, bins=80, color=colors[idx], edgecolor='white', alpha=0.85, density=True)

        # 拟合 KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals)
        x_range = np.linspace(vals.min(), vals.max(), 500)
        ax.plot(x_range, kde(x_range), color='black', lw=2, label='KDE')

        # 均值线
        mean_val = vals.mean()
        ax.axvline(mean_val, color='red', linestyle='--', lw=2, label=f'Mean={mean_val:.3f}')

        ax.set_xlabel(DIM_FULL_NAMES[dim])
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Distribution of {DIM_FULL_NAMES[dim]}')
        ax.legend(loc='upper right')
        ax.set_xlim(0, 1)

    # k_in 放在第6个子图
    ax = axes[5]
    vals = df["k_in_mean"].dropna()
    ax.hist(vals, bins=80, color='#6A4C93', edgecolor='white', alpha=0.85, density=True)

    kde = gaussian_kde(vals)
    x_range = np.linspace(vals.min(), vals.max(), 500)
    ax.plot(x_range, kde(x_range), color='black', lw=2, label='KDE')

    mean_val = vals.mean()
    ax.axvline(mean_val, color='red', linestyle='--', lw=2, label=f'Mean={mean_val:.3f}')

    ax.set_xlabel(r'Internal Synergy Coefficient $k_{in}$')
    ax.set_ylabel('Probability Density')
    ax.set_title(r'Distribution of $k_{in}$')
    ax.legend(loc='upper right')

    plt.suptitle('Five-Dimensional Distributions of LHD Plasma Discharges (N=9712)', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig1_5d_distributions.png'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig1_5d_distributions.pdf'))
    plt.close()
    print("[OK] fig1_5d_distributions saved")

# ==================== 图 2: 五维箱线图 ====================

def plot_5d_boxplots(df, outdir):
    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = [df[f"{d}_mean"].dropna() for d in DIM_LABELS]
    bp = ax.boxplot(data_to_plot, labels=[DIM_FULL_NAMES[d] for d in DIM_LABELS],
                    patch_artist=True, notch=True, showfliers=False)

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Normalized Value')
    ax.set_title('Five-Dimensional Boxplots of LHD Plasma Discharges (N=9712)')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig2_5d_boxplots.png'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig2_5d_boxplots.pdf'))
    plt.close()
    print("[OK] fig2_5d_boxplots saved")

# ==================== 图 3: 相关性热力图 ====================

def plot_correlation_heatmap(df, outdir):
    dim_cols = [f"{d}_mean" for d in DIM_LABELS] + ["k_in_mean"]
    corr = df[dim_cols].corr()

    # 重命名
    corr_labels = [DIM_FULL_NAMES[d] for d in DIM_LABELS] + [r'$k_{in}$']

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # 添加数值
    for i in range(len(corr_labels)):
        for j in range(len(corr_labels)):
            text = ax.text(j, i, f'{corr.values[i, j]:.2f}',
                          ha="center", va="center", color="black" if abs(corr.values[i,j]) < 0.5 else "white",
                          fontsize=11)

    ax.set_xticks(range(len(corr_labels)))
    ax.set_yticks(range(len(corr_labels)))
    ax.set_xticklabels(corr_labels, rotation=45, ha='right')
    ax.set_yticklabels(corr_labels)
    ax.set_title('Pearson Correlation Matrix of Five Dimensions (N=9712)')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson r')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig3_correlation_heatmap.png'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig3_correlation_heatmap.pdf'))
    plt.close()
    print("[OK] fig3_correlation_heatmap saved")

# ==================== 图 4: 散点矩阵 (Pair Plot) ====================

def plot_pair_scatter(df, outdir):
    dim_cols = [f"{d}_mean" for d in DIM_LABELS]
    n = len(dim_cols)

    fig, axes = plt.subplots(n, n, figsize=(14, 14))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

    for i, dim_i in enumerate(dim_cols):
        for j, dim_j in enumerate(dim_cols):
            ax = axes[i, j]

            if i == j:
                # 对角线：直方图
                vals = df[dim_i].dropna()
                ax.hist(vals, bins=60, color=colors[i], edgecolor='white', alpha=0.8)
                ax.set_title(DIM_FULL_NAMES[DIM_LABELS[i]], fontsize=11)
            else:
                # 非对角线：散点图（采样避免过密）
                sample = df.sample(min(2000, len(df))) if len(df) > 2000 else df
                x = sample[dim_j]
                y = sample[dim_i]
                ax.scatter(x, y, c=colors[i], alpha=0.3, s=8, edgecolors='none')

                # 拟合线
                z = np.polyfit(x.dropna(), y.dropna(), 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), 'k--', lw=1.5, alpha=0.7)

            if i == n - 1:
                ax.set_xlabel(DIM_FULL_NAMES[DIM_LABELS[j]], fontsize=10)
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(DIM_FULL_NAMES[DIM_LABELS[i]], fontsize=10)
            else:
                ax.set_yticklabels([])

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    plt.suptitle('Pairwise Scatter Matrix of Five Dimensions (N=9712)', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig4_pair_scatter.png'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig4_pair_scatter.pdf'))
    plt.close()
    print("[OK] fig4_pair_scatter saved")

# ==================== 图 5: k_in vs 各维度散点 ====================

def plot_kin_vs_dims(df, outdir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

    for idx, dim in enumerate(DIM_LABELS):
        ax = axes[idx]
        x = df[f"{dim}_mean"]
        y = df["k_in_mean"]

        # 采样
        mask = ~(x.isna() | y.isna())
        x_clean = x[mask]
        y_clean = y[mask]
        if len(x_clean) > 3000:
            idx_sample = np.random.choice(len(x_clean), 3000, replace=False)
            x_clean = x_clean.iloc[idx_sample]
            y_clean = y_clean.iloc[idx_sample]

        ax.scatter(x_clean, y_clean, c=colors[idx], alpha=0.4, s=10, edgecolors='none')

        # 拟合
        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax.plot(x_line, p(x_line), 'k--', lw=2)

        # 相关系数
        r, _ = stats.pearsonr(x_clean, y_clean)
        ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(DIM_FULL_NAMES[dim])
        ax.set_ylabel(r'$k_{in}$')
        ax.set_title(f'{DIM_FULL_NAMES[dim]} vs $k_{{in}}$')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, df["k_in_mean"].max() * 1.05)

    # 第6个：k_in vs event_severity
    ax = axes[5]
    x = df["event_severity_max"]
    y = df["k_in_mean"]
    mask = ~(x.isna() | y.isna()) & (x > 0)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) > 3000:
        idx_sample = np.random.choice(len(x_clean), 3000, replace=False)
        x_clean = x_clean.iloc[idx_sample]
        y_clean = y_clean.iloc[idx_sample]

    ax.scatter(x_clean, y_clean, c='#6A4C93', alpha=0.4, s=10, edgecolors='none')

    if len(x_clean) > 10:
        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax.plot(x_line, p(x_line), 'k--', lw=2)
        r, _ = stats.pearsonr(x_clean, y_clean)
        ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Event Severity Max')
    ax.set_ylabel(r'$k_{in}$')
    ax.set_title(r'Event Severity vs $k_{in}$')

    plt.suptitle(r'Internal Synergy Coefficient $k_{in}$ vs. Five Dimensions (N=9712)', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig5_kin_vs_dims.png'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig5_kin_vs_dims.pdf'))
    plt.close()
    print("[OK] fig5_kin_vs_dims saved")

# ==================== 图 6: 四象限划分 (B-R 平面) ====================

def plot_quadrant_BR(df, outdir):
    fig, ax = plt.subplots(figsize=(10, 8))

    B = df["B_mean"]
    R = df["R_mean"]
    kin = df["k_in_mean"]

    mask = ~(B.isna() | R.isna() | kin.isna())
    B, R, kin = B[mask], R[mask], kin[mask]

    # 中位数划分
    B_med = B.median()
    R_med = R.median()

    # 四象限着色
    q1 = (B >= B_med) & (R >= R_med)   # High B, High R
    q2 = (B < B_med) & (R >= R_med)    # Low B, High R
    q3 = (B < B_med) & (R < R_med)     # Low B, Low R
    q4 = (B >= B_med) & (R < R_med)    # High B, Low R

    scatter = ax.scatter(B, R, c=kin, cmap='viridis', alpha=0.6, s=15, edgecolors='none')

    ax.axvline(B_med, color='red', linestyle='--', lw=2, label=f'B median = {B_med:.3f}')
    ax.axhline(R_med, color='red', linestyle='--', lw=2, label=f'R median = {R_med:.3f}')

    # 象限标注
    ax.text(0.95, 0.95, f'Q1: High B, High R\n(n={q1.sum()}, k_in={kin[q1].mean():.3f})',
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(0.05, 0.95, f'Q2: Low B, High R\n(n={q2.sum()}, k_in={kin[q2].mean():.3f})',
            transform=ax.transAxes, ha='left', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.text(0.05, 0.05, f'Q3: Low B, Low R\n(n={q3.sum()}, k_in={kin[q3].mean():.3f})',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    ax.text(0.95, 0.05, f'Q4: High B, Low R\n(n={q4.sum()}, k_in={kin[q4].mean():.3f})',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(r'$k_{in}$')

    ax.set_xlabel('Boundary (B)')
    ax.set_ylabel('Reserve (R)')
    ax.set_title(r'Quadrant Classification on B-R Plane Colored by $k_{in}$ (N=9712)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig6_quadrant_BR.png'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig6_quadrant_BR.pdf'))
    plt.close()
    print("[OK] fig6_quadrant_BR saved")

# ==================== 图 7: 时间序列 (按 shot 编号) ====================

def plot_timeseries(df, outdir):
    # 按 shot 编号排序
    df_sorted = df.sort_values('shot_num').dropna(subset=['shot_num'])

    if len(df_sorted) < 100:
        print("[SKIP] fig7_timeseries: too few shots with numeric IDs")
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#6A4C93']

    # 采样显示（避免过多点）
    if len(df_sorted) > 5000:
        df_plot = df_sorted.iloc[::len(df_sorted)//5000]
    else:
        df_plot = df_sorted

    for idx, dim in enumerate(DIM_LABELS + ["k_in"]):
        ax = axes[idx]
        col = f"{dim}_mean" if dim != "k_in" else "k_in_mean"
        y = df_plot[col]
        x = df_plot['shot_num']

        ax.scatter(x, y, c=colors[idx], alpha=0.4, s=8, edgecolors='none')

        # 滚动均值
        window = max(len(df_sorted) // 100, 50)
        rolling = df_sorted[col].rolling(window=window, min_periods=1).mean()
        ax.plot(df_sorted['shot_num'], rolling, color='black', lw=2, label=f'MA({window})')

        label = DIM_FULL_NAMES[dim] if dim in DIM_FULL_NAMES else r'$k_{in}$'
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs Shot Number')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1 if dim != "k_in" else df["k_in_mean"].max() * 1.05)

    axes[-2].set_xlabel('Shot Number')
    axes[-1].set_xlabel('Shot Number')

    plt.suptitle('Temporal Evolution of Five Dimensions Across LHD Shots (N=9712)', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig7_timeseries.png'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig7_timeseries.pdf'))
    plt.close()
    print("[OK] fig7_timeseries saved")

# ==================== 图 8: 五维雷达图 (均值) ====================

def plot_radar_mean(df, outdir):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    means = [df[f"{d}_mean"].mean() for d in DIM_LABELS]
    stds = [df[f"{d}_mean"].std() for d in DIM_LABELS]

    angles = np.linspace(0, 2 * np.pi, len(DIM_LABELS), endpoint=False).tolist()
    means += means[:1]
    stds_upper = [m + s for m, s in zip(means[:-1], stds)] + [means[0] + stds[0]]
    stds_lower = [max(0, m - s) for m, s in zip(means[:-1], stds)] + [max(0, means[0] - stds[0])]
    angles += angles[:1]

    ax.plot(angles, means, 'o-', linewidth=2, color='#2E86AB', label='Mean')
    ax.fill(angles, means, alpha=0.25, color='#2E86AB')
    ax.fill_between(angles, stds_lower, stds_upper, alpha=0.15, color='gray', label='±1σ')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([DIM_FULL_NAMES[d] for d in DIM_LABELS])
    ax.set_ylim(0, 1)
    ax.set_title('Mean Five-Dimensional Profile of LHD Plasma (N=9712)', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig8_radar_mean.png'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig8_radar_mean.pdf'))
    plt.close()
    print("[OK] fig8_radar_mean saved")

# ==================== 图 9: 事件严重度分布 ====================

def plot_event_analysis(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：事件数量分布
    ax1 = axes[0]
    events = df["n_events"].dropna()
    ax1.hist(events, bins=50, color='#C73E1D', edgecolor='white', alpha=0.8)
    ax1.set_xlabel('Number of Events per Shot')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Event Count Distribution (Mean={events.mean():.1f})')
    ax1.grid(axis='y', alpha=0.3)

    # 右：事件严重度分布（log scale）
    ax2 = axes[1]
    severity = df["event_severity_max"].dropna()
    severity = severity[severity > 0]  # 只取有事件的
    ax2.hist(severity, bins=50, color='#F18F01', edgecolor='white', alpha=0.8)
    ax2.set_xlabel('Max Event Severity')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Severity Distribution (N={len(severity)})')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Event Analysis of LHD Plasma Discharges', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig9_event_analysis.png'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig9_event_analysis.pdf'))
    plt.close()
    print("[OK] fig9_event_analysis saved")

# ==================== 图 10: 五维聚类树状图 ====================

def plot_clustering(df, outdir):
    # 采样做聚类（太多点算不动）
    sample_size = min(500, len(df))
    df_sample = df.sample(sample_size, random_state=42)

    dim_cols = [f"{d}_mean" for d in DIM_LABELS]
    X = df_sample[dim_cols].dropna().values

    if len(X) < 10:
        print("[SKIP] fig10_clustering: too few valid samples")
        return

    # 层次聚类
    Z = linkage(X, method='ward')

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=Z[-3, 2])
    ax.set_xlabel('Shot Index')
    ax.set_ylabel('Ward Distance')
    ax.set_title(f'Hierarchical Clustering of LHD Shots (Sample N={len(X)})')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig10_clustering.png'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig10_clustering.pdf'))
    plt.close()
    print("[OK] fig10_clustering saved")

# ==================== 统计分析报告 ====================

def generate_report(df, outdir):
    report_path = os.path.join(outdir, 'statistical_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("LHD Five-Dimensional Statistical Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total valid shots: {len(df)}\n")
        f.write(f"Source breakdown: {df['reason'].value_counts().to_dict()}\n\n")

        f.write("--- Descriptive Statistics ---\n")
        desc = df[[f"{d}_mean" for d in DIM_LABELS] + ["k_in_mean"]].describe()
        f.write(desc.to_string())
        f.write("\n\n")

        f.write("--- Correlation Matrix ---\n")
        dim_cols = [f"{d}_mean" for d in DIM_LABELS] + ["k_in_mean"]
        corr = df[dim_cols].corr()
        f.write(corr.to_string())
        f.write("\n\n")

        f.write("--- Quadrant Analysis (B-R Plane) ---\n")
        B = df["B_mean"]
        R = df["R_mean"]
        B_med = B.median()
        R_med = R.median()

        q1 = ((B >= B_med) & (R >= R_med)).sum()
        q2 = ((B < B_med) & (R >= R_med)).sum()
        q3 = ((B < B_med) & (R < R_med)).sum()
        q4 = ((B >= B_med) & (R < R_med)).sum()

        f.write(f"B median: {B_med:.4f}, R median: {R_med:.4f}\n")
        f.write(f"Q1 (High B, High R): {q1} shots\n")
        f.write(f"Q2 (Low B, High R):  {q2} shots\n")
        f.write(f"Q3 (Low B, Low R):   {q3} shots\n")
        f.write(f"Q4 (High B, Low R):  {q4} shots\n\n")

        f.write("--- k_in by Quadrant ---\n")
        for q_name, mask in [("Q1", (B >= B_med) & (R >= R_med)),
                              ("Q2", (B < B_med) & (R >= R_med)),
                              ("Q3", (B < B_med) & (R < R_med)),
                              ("Q4", (B >= B_med) & (R < R_med))]:
            kin_q = df.loc[mask, "k_in_mean"]
            f.write(f"{q_name}: mean={kin_q.mean():.6f}, std={kin_q.std():.6f}, min={kin_q.min():.6f}, max={kin_q.max():.6f}\n")

        f.write("\n--- Normality Tests (Shapiro-Wilk on sample n=5000) ---\n")
        sample = df.sample(min(5000, len(df)), random_state=42)
        for dim in DIM_LABELS + ["k_in"]:
            col = f"{dim}_mean" if dim != "k_in" else "k_in_mean"
            stat, p = stats.shapiro(sample[col].dropna())
            f.write(f"{dim}: W={stat:.4f}, p={p:.2e}\n")

    print(f"[OK] statistical_report.txt saved")

# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("LHD 5D Analysis & Visualization Suite")
    print("=" * 60)

    print(f"\n[1/3] Loading data from {INPUT_CSV}...")
    df = load_data(INPUT_CSV)
    print(f"       Loaded {len(df)} valid shots")

    print(f"\n[2/3] Generating figures to {OUTPUT_DIR}...")
    plot_5d_histograms(df, OUTPUT_DIR)
    plot_5d_boxplots(df, OUTPUT_DIR)
    plot_correlation_heatmap(df, OUTPUT_DIR)
    plot_pair_scatter(df, OUTPUT_DIR)
    plot_kin_vs_dims(df, OUTPUT_DIR)
    plot_quadrant_BR(df, OUTPUT_DIR)
    plot_timeseries(df, OUTPUT_DIR)
    plot_radar_mean(df, OUTPUT_DIR)
    plot_event_analysis(df, OUTPUT_DIR)
    plot_clustering(df, OUTPUT_DIR)

    print(f"\n[3/3] Generating statistical report...")
    generate_report(df, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("All done. Outputs:")
    print(f"  Figures: {OUTPUT_DIR}")
    print(f"  Report:  {os.path.join(OUTPUT_DIR, 'statistical_report.txt')}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
