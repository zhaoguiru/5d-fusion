#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LHD 核聚变 — 物理参数五维 BSRDI 自动探测与映射
自动读取CSV列名，匹配B/S/R/D/I，计算k_in
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("./fusion_5d_data")
FIG_DIR = OUTPUT_DIR / "lhd_figures"
FIG_DIR.mkdir(exist_ok=True)

# 列名关键词映射（自动匹配）
COLUMN_MAP = {
    'B': ['R_m', 'major_radius', 'a_m', 'minor_radius', 'radius', 'R0', 'R', 'Rmaj', 'Rgeo'],
    'S': ['tauE', 'tau_E', 'confinement', 'tau', 'tau_e', 'energy_confinement', 'H98', 'H_mode', 'confinement_time'],
    'R': ['Pfus', 'P_fus', 'fusion_power', 'W_MJ', 'stored_energy', 'W_tot', 'power', 'Pheat', 'P_heat', 'heating'],
    'D': ['Ip', 'I_p', 'plasma_current', 'current', 'q95', 'q_95', 'safety_factor', 'q0', 'q', 'rotational_transform'],
    'I': ['Bt', 'B_t', 'toroidal_field', 'B0', 'magnetic_field', 'field', 'beta_N', 'beta', 'B_T', 'Bfield']
}

def find_column(df, keywords):
    """根据关键词查找匹配列名"""
    for kw in keywords:
        matches = [c for c in df.columns if kw.lower() in c.lower()]
        if matches:
            return matches[0]
    return None

def normalize_ratio(s):
    """比值法归一化: val / max(val) ∈ (0,1]"""
    s = pd.to_numeric(s, errors='coerce').dropna()
    if len(s) == 0:
        return None
    mx = s.max()
    if mx <= 0:
        return None
    return (s / mx).clip(1e-9, 1.0)

def compute_kin(features_df):
    """k_in = ∏_{i<j} min(xi,xj)/max(xi,xj)"""
    dims = ['B', 'S', 'R', 'D', 'I']
    k_in = np.ones(len(features_df))
    for i in range(5):
        for j in range(i+1, 5):
            di, dj = dims[i], dims[j]
            gamma = np.minimum(features_df[di], features_df[dj]) / np.maximum(features_df[di], features_df[dj])
            k_in *= gamma
    return k_in

def main():
    print("=" * 60)
    print("LHD 核聚变 — 物理参数五维 BSRDI 自动探测")
    print("=" * 60)
    
    # 1. 探测可用 CSV
    all_csv = sorted(OUTPUT_DIR.glob("*.csv"))
    print(f"\n发现 {len(all_csv)} 个 CSV 文件:")
    for i, f in enumerate(all_csv, 1):
        print(f"  {i}. {f.name}")
    
    # 2. 优先尝试已知文件
    candidates = [
        'fusion_5d_ready.csv',
        'lhd_5d_ready.csv',
        'fusion_devices.csv',
        'fusion_devices_expanded.csv',
        'lhd_5d_selective.csv',
        'lhd_5d_synergy.csv'
    ]
    
    df = None
    used_file = None
    for fname in candidates:
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            print(f"\n{'='*60}")
            print(f"加载: {fname}")
            df = pd.read_csv(fpath)
            used_file = fname
            break
    
    if df is None:
        print("\n未找到候选文件，请从以下文件中选择:")
        for i, f in enumerate(all_csv, 1):
            print(f"  {i}. {f.name}")
        choice = input("\n输入文件名（或编号）: ").strip()
        try:
            idx = int(choice) - 1
            fpath = all_csv[idx]
        except:
            fpath = OUTPUT_DIR / choice
        df = pd.read_csv(fpath)
        used_file = fpath.name
    
    print(f"\n数据: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"列名: {list(df.columns)}")
    
    # 3. 自动映射
    mapped = {}
    print("\n{'='*60}")
    print("自动五维映射:")
    for dim, keywords in COLUMN_MAP.items():
        col = find_column(df, keywords)
        if col:
            mapped[dim] = col
            print(f"  ✓ [{dim}] → {col}")
        else:
            print(f"  ✗ [{dim}] → 未找到匹配")
    
    # 4. 手动补全缺失维度
    missing = [d for d in ['B','S','R','D','I'] if d not in mapped]
    if missing:
        print(f"\n⚠️ 缺失维度: {missing}")
        print(f"可用列名:")
        for i, c in enumerate(df.columns, 1):
            print(f"  {i}. {c}")
        for dim in missing:
            col_input = input(f"\n请为 [{dim}] 输入列名（或编号）: ").strip()
            try:
                idx = int(col_input) - 1
                mapped[dim] = df.columns[idx]
            except:
                mapped[dim] = col_input
            print(f"  [{dim}] → {mapped[dim]}")
    
    # 5. 提取五维并归一化
    features = pd.DataFrame(index=df.index)
    print(f"\n{'='*60}")
    print("五维归一化结果:")
    for dim, col in mapped.items():
        norm = normalize_ratio(df[col])
        if norm is not None:
            features[dim] = norm
            print(f"  [{dim}] {col}: [{norm.min():.4f}, {norm.max():.4f}]")
        else:
            print(f"  错误: {col} 归一化失败")
            return
    
    # 6. 计算 k_in
    features['k_in'] = compute_kin(features)
    
    # 合并
    result = pd.concat([df, features], axis=1)
    
    # 保存
    out_csv = OUTPUT_DIR / "lhd_5d_physical_bsrdi.csv"
    result.to_csv(out_csv, index=False)
    print(f"\n✓ 结果保存: {out_csv}")
    
    # 7. 统计
    print(f"\n{'='*60}")
    print("五维统计:")
    for dim in ['B', 'S', 'R', 'D', 'I']:
        print(f"  {dim}: {result[dim].mean():.4f} ± {result[dim].std():.4f}")
    print(f"\n  k_in 均值:    {result['k_in'].mean():.6f}")
    print(f"  k_in 中位数:  {result['k_in'].median():.6f}")
    print(f"  k_in 最大值:  {result['k_in'].max():.6f}")
    print(f"  k_in > 0 比例: {(result['k_in'] > 1e-6).mean():.2%}")
    
    # 8. 可视化
    print(f"\n{'='*60}")
    print("生成可视化...")
    
    # 五维分布
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, dim in enumerate(['B', 'S', 'R', 'D', 'I', 'k_in']):
        ax = axes[idx]
        ax.hist(result[dim], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(result[dim].mean(), color='red', linestyle='--',
                   label=f'μ={result[dim].mean():.4f}')
        ax.set_title(f'{dim} Distribution')
        ax.set_xlabel(dim); ax.set_ylabel('Count')
        ax.legend()
    plt.suptitle(f'Five-Dimensional Physical Parameters ({used_file})', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig20_physical_5d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 五维分布: {FIG_DIR / 'fig20_physical_5d.png'}")
    
    # 雷达图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    cats = ['B\nBoundary', 'S\nStructure', 'R\nReserve', 'D\nDirection', 'I\nIntensity']
    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals = [result['B'].mean(), result['S'].mean(), result['R'].mean(),
            result['D'].mean(), result['I'].mean()]
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', linewidth=2, color='darkblue', markersize=8)
    ax.fill(angles, vals, alpha=0.25, color='darkblue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, size=11)
    ax.set_ylim(0, 1)
    ax.set_title('Plasma Five-Dimensional Profile (Mean)', size=14, pad=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig21_physical_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 雷达图: {FIG_DIR / 'fig21_physical_radar.png'}")
    
    # 四象限 B vs S
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(result['B'], result['S'], c='steelblue', alpha=0.5, s=20)
    ax.axhline(result['S'].median(), color='black', linestyle='--', alpha=0.4)
    ax.axvline(result['B'].median(), color='black', linestyle='--', alpha=0.4)
    ax.text(0.75, 0.75, 'Q1: High B, High S', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(0.25, 0.75, 'Q2: Low B, High S', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    ax.text(0.25, 0.25, 'Q3: Low B, Low S', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    ax.text(0.75, 0.25, 'Q4: High B, Low S', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.3))
    ax.set_xlabel('B: Boundary (Normalized)', fontsize=12)
    ax.set_ylabel('S: Structure (Normalized)', fontsize=12)
    ax.set_title('Plasma Quadrant: Boundary vs Structure', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig22_physical_quadrant.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 四象限: {FIG_DIR / 'fig22_physical_quadrant.png'}")
    
    print(f"\n{'='*60}")
    print("物理参数五维 BSRDI 映射完成。")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
