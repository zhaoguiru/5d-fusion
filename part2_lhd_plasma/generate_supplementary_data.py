#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成补充数据 S1 和 S2
========================

从已有数据文件提取并生成标准格式的补充数据，随论文投稿。

S1: 1,028 个聚变设计记录的五维特征和 k_in
S2: 9,712 个 LHD 放电的五维特征、k_in、诊断类型、四象限

作者: 赵国瑞
日期: 2026-06-05
"""

import os
import pandas as pd
import numpy as np
import re

# ============================================================
# 配置路径
# ============================================================

# S1 源文件：设计记录
S1_SOURCE = "fusion_5d_data/fusion_5d_ready.csv"

# S2 源文件：v6e summary.csv
S2_SOURCE = "lhd_5d_results_v6e/summary.csv"

# 输出文件名
S1_OUTPUT = "Supplementary_Data_S1.csv"
S2_OUTPUT = "Supplementary_Data_S2.csv"

# ============================================================
# 诊断类型推断（与论文一致）
# ============================================================

DIAG_PATTERNS = {
    'Bolometer': r'(?i)bolometer|bolo|rad',
    'Divertor-Interferometer': r'(?i)divertor.*interferometer|div.*interf|interferometer',
    'SXfluc': r'(?i)sxfluc|sx.*fluc|soft.*x.*fluc',
    'SXmp': r'(?i)sxmp|sx.*mp|soft.*x.*mp',
    'ECE': r'(?i)ece|electron.*cyclotron',
    'MP': r'(?i)mp$|magnetic.*probe|mirnov',
}


def infer_diag_type(shot_name):
    for diag_type, pattern in DIAG_PATTERNS.items():
        if re.search(pattern, str(shot_name)):
            return diag_type
    return 'Unknown'


# ============================================================
# 四象限分类（基于 B 和 R 中位数，与论文一致）
# ============================================================

def classify_quadrant(df):
    b_med = df['B_mean'].median()
    r_med = df['R_mean'].median()

    def get_q(row):
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

    return df.apply(get_q, axis=1)


# ============================================================
# 生成 S1
# ============================================================

def generate_s1():
    print("=" * 60)
    print("生成 S1: 设计记录补充数据")
    print("=" * 60)

    if not os.path.exists(S1_SOURCE):
        print(f"[ERROR] 源文件不存在: {S1_SOURCE}")
        return False

    df = pd.read_csv(S1_SOURCE)
    print(f"[INFO] 读取 {len(df)} 条设计记录")

    # 检查必要列
    required = ['R', 'tau_E', 'P_fus', 'I_p', 'B_t']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] 缺少列: {missing}")
        print(f"[INFO] 可用列: {list(df.columns)}")
        # 尝试找到替代列名
        col_map = {}
        for col in df.columns:
            if 'radius' in col.lower() or 'r' == col.lower():
                col_map['R'] = col
            elif 'tau' in col.lower() or 'confinement' in col.lower():
                col_map['tau_E'] = col
            elif 'power' in col.lower() or 'pfus' in col.lower():
                col_map['P_fus'] = col
            elif 'current' in col.lower() or 'ip' in col.lower():
                col_map['I_p'] = col
            elif 'field' in col.lower() or 'bt' in col.lower():
                col_map['B_t'] = col

        if col_map:
            print(f"[INFO] 列名映射: {col_map}")
            df = df.rename(columns={v: k for k, v in col_map.items()})

    # 选择输出列
    output_cols = ['shot', 'device', 'mode', 'R', 'tau_E', 'P_fus', 'I_p', 'B_t']
    if 'Q' in df.columns:
        output_cols.append('Q')
    if 'k_in' in df.columns:
        output_cols.append('k_in')

    # 如果源文件没有 k_in，计算它
    if 'k_in' not in df.columns:
        print("[INFO] 源文件无 k_in，计算中...")
        # 归一化到 (0,1]
        for col in ['R', 'tau_E', 'P_fus', 'I_p', 'B_t']:
            if col in df.columns:
                cmin = df[col].min()
                cmax = df[col].max()
                if cmax > cmin:
                    df[f'{col}_norm'] = (df[col] - cmin) / (cmax - cmin)
                    df[f'{col}_norm'] = df[f'{col}_norm'].clip(1e-12, 1.0)

        # 计算 k_in
        dims = ['R_norm', 'tau_E_norm', 'P_fus_norm', 'I_p_norm', 'B_t_norm']
        if all(d in df.columns for d in dims):
            gammas = []
            for d in dims:
                x = df[d]
                gamma = np.minimum(x, 0.5) / np.maximum(x, 0.5)
                gamma = gamma.clip(1e-12, 1.0)
                gammas.append(gamma)

            log_gammas = [np.log(g) for g in gammas]
            df['k_in'] = np.exp(np.mean(log_gammas, axis=0))
            output_cols.append('k_in')
            print("[INFO] k_in 计算完成")

    # 选择存在的列
    final_cols = [c for c in output_cols if c in df.columns]
    s1_df = df[final_cols].copy()

    # 保存
    s1_df.to_csv(S1_OUTPUT, index=False)
    print(f"[OK] S1 已保存: {S1_OUTPUT} ({len(s1_df)} 条)")
    print(f"[INFO] 列: {list(s1_df.columns)}")

    return True


# ============================================================
# 生成 S2
# ============================================================

def generate_s2():
    print("\n" + "=" * 60)
    print("生成 S2: LHD 放电汇总补充数据")
    print("=" * 60)

    if not os.path.exists(S2_SOURCE):
        print(f"[ERROR] 源文件不存在: {S2_SOURCE}")
        return False

    df = pd.read_csv(S2_SOURCE)
    print(f"[INFO] 读取 {len(df)} 条记录")

    # 只保留成功记录
    if 'status' in df.columns:
        df = df[df['status'] == 'ok'].copy()
        print(f"[INFO] 筛选成功记录: {len(df)} 条")

    # 转换数值列
    numeric_cols = ['k_in_mean', 'B_mean', 'S_mean', 'R_mean', 'D_mean', 'I_mean',
                    'k_in_std', 'k_in_max', 'k_in_min', 'n_events', 'event_severity_max',
                    'n_samples', 'n_windows', 'duration_ms']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 推断诊断类型
    if 'shot' in df.columns:
        df['diag_type'] = df['shot'].apply(infer_diag_type)
        print(f"[INFO] 诊断类型分布:")
        print(df['diag_type'].value_counts().to_string())

    # 四象限分类
    if all(c in df.columns for c in ['B_mean', 'R_mean']):
        df['quadrant'] = classify_quadrant(df)
        print(f"[INFO] 四象限分布:")
        print(df['quadrant'].value_counts().sort_index().to_string())

    # 选择输出列
    output_cols = ['shot', 'diag_type', 'quadrant', 'k_in_mean', 'k_in_std',
                   'B_mean', 'S_mean', 'R_mean', 'D_mean', 'I_mean',
                   'n_events', 'event_severity_max', 'n_samples', 'duration_ms']
    final_cols = [c for c in output_cols if c in df.columns]
    s2_df = df[final_cols].copy()

    # 重命名列，更清晰
    rename_map = {
        'shot': 'shot_no',
        'k_in_mean': 'k_in',
        'k_in_std': 'k_in_std',
        'B_mean': 'B',
        'S_mean': 'S',
        'R_mean': 'R',
        'D_mean': 'D',
        'I_mean': 'I',
        'n_events': 'num_events',
        'event_severity_max': 'max_event_severity',
        'n_samples': 'num_samples',
        'duration_ms': 'duration_ms',
    }
    s2_df = s2_df.rename(columns={k: v for k, v in rename_map.items() if k in s2_df.columns})

    # 保存
    s2_df.to_csv(S2_OUTPUT, index=False)
    print(f"[OK] S2 已保存: {S2_OUTPUT} ({len(s2_df)} 条)")
    print(f"[INFO] 列: {list(s2_df.columns)}")

    return True


# ============================================================
# 主程序
# ============================================================

def main():
    print("补充数据生成脚本")
    print("=" * 60)

    ok1 = generate_s1()
    ok2 = generate_s2()

    print("\n" + "=" * 60)
    print("生成完成！")
    print("=" * 60)

    if ok1:
        print(f"  S1: {S1_OUTPUT} ({os.path.getsize(S1_OUTPUT):,} bytes)")
    if ok2:
        print(f"  S2: {S2_OUTPUT} ({os.path.getsize(S2_OUTPUT):,} bytes)")

    print("\n投稿时请将这两个 CSV 文件作为补充数据上传。")


if __name__ == '__main__':
    main()
