#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LHD Physics Validation P0: W_th / tau_E vs k_in Correlation
============================================================
Part 2 Supplement for PPFC Submission

Author: Guiru Zhao
Date: 2026-06-05

修复：支持 .shot 文件格式（LHD 参数文件），shot 号提取匹配 Bolometer-XXXXXX-1.shot
"""

import os
import sys
import re
import json
import csv
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# =============================================================================
# 0. 全局配置
# =============================================================================

LHD_MAJOR_RADIUS = 3.9
LHD_MINOR_RADIUS = 0.5
LHD_VOLUME_EST = 2 * np.pi**2 * 3.9 * 0.5**2

WTH_ALIASES = [
    'wmhd', 'wtot', 'wth', 'stored_energy', 'storedenergy',
    'thermal_energy', 'diamagnetic_flux', 'diamagnetic_flux_(wb)',
    'diamag_flux', 'w_dia', 'w_diamag', 'energy_content',
    'w_th_e', 'wth_e', 'thermalenergy', 'w_mhd', 'wmhd_j', 'wmhd_(j)',
    'stored_energy_(j)', 'thermal_energy_(j)'
]

TAU_E_ALIASES = [
    'tau_e', 'taue', 'tau_e0', 'tau_e_0', 'energy_confinement_time',
    'confinement_time', 'confinementtime', 'tau_thermal', 'tau_th',
    'tau_e_th', 'tau_e_thermal', 'tau_e_star', 'tau_e_st', 'tau_e98',
    'tau_e_(s)', 'taue_(s)', 'energy_confinement_time_(s)'
]

P_HEAT_ALIASES = [
    'p_heat', 'pheat', 'p_total', 'ptot', 'p_in', 'pin',
    'p_nbi', 'pnbi', 'p_ech', 'pech', 'p_ich', 'pich',
    'p_lh', 'plh', 'p_nb', 'pnb', 'p_ec', 'pec',
    'heating_power', 'heatingpower', 'absorbed_power', 'absorbedpower',
    'p_heat_(mw)', 'p_total_(mw)', 'p_in_(mw)'
]

N_E_ALIASES = [
    'n_e', 'ne', 'n_e0', 'ne0', 'n_e_avg', 'ne_avg',
    'line_density', 'linedensity', 'n_bar', 'nbar', 'n_e_bar', 'nebar',
    'n_e_(10^19_m^-3)', 'ne_(10^19_m^-3)', 'n_e_19'
]

T_E_ALIASES = [
    't_e', 'te', 't_e0', 'te0', 't_e_avg', 'te_avg',
    'electron_temperature', 'electrontemperature', 'temp_e', 'tempe',
    't_e_core', 'te_core', 't_e_max', 'te_max',
    't_e_(kev)', 'te_(kev)', 't_e_avg_(kev)'
]

B_T_ALIASES = ['b_t', 'bt', 'toroidal_field', 'toroidalfield', 'b_tor', 'btor', 'bf',
               'b_t_(t)', 'bt_(t)', 'toroidal_field_(t)']
I_P_ALIASES = ['i_p', 'ip', 'plasma_current', 'plasmacurrent', 'current', 'tor_current',
               'i_p_(ma)', 'ip_(ma)', 'plasma_current_(ma)']
R_MAJOR_ALIASES = ['r_major', 'rmajor', 'major_radius', 'majorradius', 'r0', 'r_0', 'r',
                   'r_major_(m)', 'major_radius_(m)']
A_MINOR_ALIASES = ['a_minor', 'aminor', 'minor_radius', 'minorradius', 'a', 'a_0', 'a0',
                   'a_minor_(m)', 'minor_radius_(m)']

PARAM_ALIASES = {
    'wth': WTH_ALIASES,
    'tau_e': TAU_E_ALIASES,
    'p_heat': P_HEAT_ALIASES,
    'n_e': N_E_ALIASES,
    't_e': T_E_ALIASES,
    'b_t': B_T_ALIASES,
    'i_p': I_P_ALIASES,
    'r_major': R_MAJOR_ALIASES,
    'a_minor': A_MINOR_ALIASES,
}

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

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('p0_validation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. 读取 summary.csv
# =============================================================================

def load_summary(csv_path: Union[str, Path]) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error(f"文件不存在: {csv_path}")
        return pd.DataFrame()

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

    logger.info(f"读取 {len(df)} 条成功记录")
    logger.info(f"诊断类型分布:\n{df['diag_type'].value_counts().to_string()}")
    return df


def infer_diag_type(shot_name: str) -> str:
    for diag_type, info in DIAG_TYPE_PATTERNS.items():
        if re.search(info['pattern'], shot_name):
            return diag_type
    return 'Unknown'


# =============================================================================
# 2. 四象限分类（基于 B 和 R）
# =============================================================================

def classify_quadrant_br(df: pd.DataFrame) -> pd.DataFrame:
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

    logger.info(f"四象限分类（B median={b_med:.4f}, R median={r_med:.4f}）:")
    quad_counts = df['quadrant'].value_counts().sort_index()
    logger.info(f"\n{quad_counts.to_string()}")

    logger.info("=== 各象限 k_in 统计 ===")
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        sub = df[df['quadrant'] == q]['k_in_mean'].dropna()
        if len(sub) > 0:
            logger.info(f"  {q}: n={len(sub)}, mean={sub.mean():.4f}, std={sub.std():.4f}, "
                        f"median={sub.median():.4f}")
    return df


# =============================================================================
# 3. 参数文件解析器（支持 .prm 和 .shot）
# =============================================================================

class PRMParser:
    @staticmethod
    def parse_file(file_path: Union[str, Path]) -> Dict[str, float]:
        file_path = Path(file_path)
        if not file_path.exists():
            return {}

        params = {}
        encodings = ['utf-8', 'shift-jis', 'cp932', 'euc-jp', 'latin-1', 'ascii']
        lines = None

        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc, errors='ignore') as f:
                    lines = f.readlines()
                break
            except Exception:
                continue

        if lines is None:
            logger.warning(f"无法读取: {file_path}")
            return {}

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('!') or line.startswith('//'):
                continue

            match = None
            for sep in ['=', ':']:
                if sep in line:
                    parts = line.split(sep, 1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower().replace(' ', '_')
                        val_str = parts[1].strip()
                        match = (key, val_str)
                        break

            if match is None:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        float(parts[-1])
                        key = parts[0].lower().replace(' ', '_')
                        val_str = parts[-1]
                        match = (key, val_str)
                    except ValueError:
                        pass

            if match:
                key, val_str = match
                val_str = re.sub(r'[\s,;\[\]\(\)\{\}]', '', val_str.split()[0] if val_str.split() else val_str)
                val_str = re.sub(r'[a-zA-Z/°\^\-]+$|^[a-zA-Z/°]+$', '', val_str)
                try:
                    val = float(val_str)
                    if np.isfinite(val) and not np.isnan(val):
                        params[key] = val
                except ValueError:
                    pass

        return params

    @staticmethod
    def extract_physical_params(file_path: Union[str, Path]) -> Dict[str, Optional[float]]:
        raw = PRMParser.parse_file(file_path)
        result = {k: None for k in PARAM_ALIASES.keys()}

        for std_name, aliases in PARAM_ALIASES.items():
            for alias in aliases:
                if alias in raw:
                    result[std_name] = raw[alias]
                    break
                for k, v in raw.items():
                    if alias in k or k in alias:
                        result[std_name] = v
                        break
                if result[std_name] is not None:
                    break

        return result


# =============================================================================
# 4. W_th / tau_E 估算器
# =============================================================================

class PhysicsEstimator:
    @staticmethod
    def estimate_wth_from_diagnostics(
        signal_mean: float,
        signal_rms: float,
        diagnostic_type: str = 'bolometer',
        ne: Optional[float] = None,
        te: Optional[float] = None,
        r_major: float = LHD_MAJOR_RADIUS,
        a_minor: float = LHD_MINOR_RADIUS
    ) -> Optional[float]:
        volume = 2 * np.pi**2 * r_major * a_minor**2

        if ne is not None and te is not None:
            wth = 1.5 * ne * 1e19 * te * 1e3 * 1.602e-19 * volume / 1e6
            return wth

        if diagnostic_type.lower() in ['bolometer', 'radiation']:
            proxy = signal_rms * 1.0
        elif diagnostic_type.lower() in ['sx', 'sxfluc', 'sxmp', 'soft_xray']:
            proxy = signal_rms * 2.0
        elif diagnostic_type.lower() in ['ece', 'ece_radiometer']:
            proxy = signal_mean * 5.0 if signal_mean > 0 else signal_rms * 5.0
        else:
            proxy = signal_rms

        wth = proxy * 0.01
        return wth if wth > 0 else None

    @staticmethod
    def estimate_tau_e(
        wth: Optional[float],
        p_heat: Optional[float],
        signal_structure: float,
        method: str = 'direct'
    ) -> Optional[float]:
        if method == 'direct' and wth is not None and p_heat is not None:
            if p_heat > 0:
                return wth / p_heat

        tau_proxy = 0.01 + signal_structure * 0.1
        return tau_proxy

    @staticmethod
    def iss04_scaling(
        r_major: float,
        a_minor: float,
        n_e: float,
        b_t: float,
        p_heat: float,
        iota: float = 0.5
    ) -> float:
        tau = 0.134 * (a_minor ** 2.28) * (r_major ** 0.64) * \
              (p_heat ** -0.61) * (n_e ** 0.54) * (b_t ** 0.84) * (iota ** 0.41)
        return tau


# =============================================================================
# 5. PRM/SHOT 索引构建（核心：支持 .shot 文件格式）
# =============================================================================

def extract_shot_from_filename(filename: str) -> Optional[str]:
    """
    从文件名提取 shot 号。
    支持格式：
      Bolometer-1002593-1.shot  -> 1002593
      Bolometer-1002593-1.prm   -> 1002593
      SXfluc-75660-1.shot       -> 75660
      12345.prm                 -> 12345
      shot_12345.prm            -> 12345
    """
    stem = Path(filename).stem  # 不含扩展名

    # 模式1: Diagnostic-XXXXXX-N.shot/prm（LHD 标准格式）
    m = re.search(r'-(\d{3,})-\d+$', stem)
    if m:
        return m.group(1)

    # 模式2: 纯数字
    if re.match(r'^\d+$', stem):
        return stem

    # 模式3: shot_ 前缀
    m = re.search(r'shot[_\-]?(\d+)', stem, re.IGNORECASE)
    if m:
        return m.group(1)

    # 模式4: 末尾数字
    m = re.search(r'(\d{3,})$', stem)
    if m:
        return m.group(1)

    # 模式5: 中间数字段
    m = re.search(r'_(\d{5,})_', stem)
    if m:
        return m.group(1)

    return None


def build_prm_index(data_root: Union[str, Path]) -> Dict[str, Path]:
    """
    预先扫描整个 data_root 目录树，建立 {shot: file_path} 索引。
    支持 .prm 和 .shot 文件。
    """
    data_root = Path(data_root)
    if not data_root.exists():
        logger.warning(f"data_root 不存在: {data_root}")
        return {}

    logger.info(f"正在扫描 {data_root} 下的所有参数文件（.prm / .shot）...")
    start_time = datetime.now()

    prm_index = {}
    count = 0
    matched = 0

    # 搜索 .prm 和 .shot 文件
    for ext in ['*.prm', '*.PRM', '*.shot', '*.SHOT']:
        for file_path in data_root.rglob(ext):
            count += 1
            shot = extract_shot_from_filename(file_path.name)
            if shot:
                matched += 1
                # 如果同一个 shot 对应多个文件，保留路径较短的
                if shot not in prm_index or len(str(file_path)) < len(str(prm_index[shot])):
                    prm_index[shot] = file_path

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"索引构建完成: {count} 个文件扫描，{matched} 个成功匹配 shot，"
                f"{len(prm_index)} 个唯一 shot 映射，耗时 {elapsed:.1f}s")

    # 保存索引
    index_save = {k: str(v) for k, v in prm_index.items()}
    return prm_index


# =============================================================================
# 6. 核心处理函数（多进程）
# =============================================================================

def process_single_shot(
    shot_info: Dict,
    prm_index: Dict[str, str],
    use_estimator: bool = True
) -> Dict:
    """处理单个放电：从索引字典查找参数文件，提取物理参数。"""
    shot = shot_info.get('shot', None)
    result = {
        'shot': shot,
        'kin': shot_info.get('k_in_mean', np.nan),
        'diag_type': shot_info.get('diag_type', 'unknown'),
        'quadrant': shot_info.get('quadrant', 'unknown'),
        'B_mean': shot_info.get('B_mean', np.nan),
        'S_mean': shot_info.get('S_mean', np.nan),
        'R_mean': shot_info.get('R_mean', np.nan),
        'D_mean': shot_info.get('D_mean', np.nan),
        'I_mean': shot_info.get('I_mean', np.nan),
        'duration_ms': shot_info.get('duration_ms', np.nan),
        'n_events': shot_info.get('n_events', np.nan),
        'event_severity_max': shot_info.get('event_severity_max', np.nan),
        'wth': np.nan,
        'tau_e': np.nan,
        'p_heat': np.nan,
        'n_e': np.nan,
        't_e': np.nan,
        'b_t': np.nan,
        'r_major': np.nan,
        'a_minor': np.nan,
        'wth_source': 'missing',
        'tau_e_source': 'missing',
        'iss04_tau_e': np.nan,
    }

    if shot is None:
        return result

    try:
        # 从索引字典查找参数文件
        shot_str = str(shot).strip()
        prm_path_str = prm_index.get(shot_str)
        prm_path = Path(prm_path_str) if prm_path_str else None

        phys = {}
        if prm_path and prm_path.exists():
            phys = PRMParser.extract_physical_params(prm_path)
            result['wth_source'] = 'prm_direct' if phys.get('wth') else 'missing'
            result['tau_e_source'] = 'prm_direct' if phys.get('tau_e') else 'missing'

        # 填充从参数文件得到的值
        for key in ['wth', 'tau_e', 'p_heat', 'n_e', 't_e', 'b_t', 'r_major', 'a_minor']:
            if phys.get(key) is not None:
                result[key] = phys[key]

        # 如果缺失，尝试估算
        if use_estimator and (np.isnan(result['wth']) or np.isnan(result['tau_e'])):
            signal_mean = shot_info.get('I_mean', 0)
            signal_rms = shot_info.get('k_in_std', 0)
            signal_structure = shot_info.get('S_mean', 0)

            if np.isnan(result['wth']):
                est_wth = PhysicsEstimator.estimate_wth_from_diagnostics(
                    signal_mean=signal_mean,
                    signal_rms=signal_rms,
                    diagnostic_type=result['diag_type'],
                    ne=result['n_e'] if not np.isnan(result['n_e']) else None,
                    te=result['t_e'] if not np.isnan(result['t_e']) else None,
                    r_major=result['r_major'] if not np.isnan(result['r_major']) else LHD_MAJOR_RADIUS,
                    a_minor=result['a_minor'] if not np.isnan(result['a_minor']) else LHD_MINOR_RADIUS,
                )
                if est_wth is not None:
                    result['wth'] = est_wth
                    result['wth_source'] = 'estimated'

            if np.isnan(result['tau_e']):
                est_tau = PhysicsEstimator.estimate_tau_e(
                    wth=result['wth'] if not np.isnan(result['wth']) else None,
                    p_heat=result['p_heat'] if not np.isnan(result['p_heat']) else None,
                    signal_structure=signal_structure,
                    method='direct' if not np.isnan(result['p_heat']) else 'proxy'
                )
                if est_tau is not None:
                    result['tau_e'] = est_tau
                    result['tau_e_source'] = 'estimated'

        # ISS04 标度律
        if (not np.isnan(result['r_major']) and not np.isnan(result['a_minor']) and
            not np.isnan(result['n_e']) and not np.isnan(result['b_t']) and
            not np.isnan(result['p_heat'])):
            result['iss04_tau_e'] = PhysicsEstimator.iss04_scaling(
                r_major=result['r_major'],
                a_minor=result['a_minor'],
                n_e=result['n_e'],
                b_t=result['b_t'],
                p_heat=result['p_heat']
            )

    except Exception as e:
        logger.error(f"Shot {shot} 处理失败: {e}")
        traceback.print_exc()

    return result


# =============================================================================
# 7. 主分析类
# =============================================================================

class P0Validator:
    def __init__(
        self,
        summary_csv: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = './p0_output',
        workers: int = 4,
        use_estimator: bool = True,
        checkpoint_file: Optional[str] = 'p0_checkpoint.json'
    ):
        self.summary_csv = Path(summary_csv)
        self.data_root = Path(data_root) if data_root else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = max(1, min(workers, cpu_count()))
        self.use_estimator = use_estimator
        self.checkpoint_file = self.output_dir / checkpoint_file if checkpoint_file else None

        self.df_summary: Optional[pd.DataFrame] = None
        self.df_physics: Optional[pd.DataFrame] = None
        self.df_merged: Optional[pd.DataFrame] = None
        self.prm_index: Dict[str, str] = {}

        logger.info(f"P0Validator 初始化: workers={self.workers}, output={self.output_dir}")

    def load_summary_data(self) -> pd.DataFrame:
        logger.info(f"加载 summary.csv: {self.summary_csv}")
        df = load_summary(self.summary_csv)
        if df.empty:
            raise ValueError("summary.csv 为空或不存在")
        df = classify_quadrant_br(df)
        self.df_summary = df
        logger.info(f"加载完成: {len(df)} 条记录")
        return df

    def build_index(self):
        if self.data_root and self.data_root.exists():
            idx = build_prm_index(self.data_root)
            self.prm_index = {k: str(v) for k, v in idx.items()}
            index_path = self.output_dir / 'prm_index.json'
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(self.prm_index, f, indent=2, ensure_ascii=False)
            logger.info(f"参数索引已保存: {index_path}")
        else:
            logger.warning("未提供 data_root，跳过索引构建")

    def run_extraction(self) -> pd.DataFrame:
        if self.df_summary is None:
            self.load_summary_data()

        if not self.prm_index and self.data_root:
            self.build_index()

        df = self.df_summary
        shots = df.to_dict('records')
        total = len(shots)

        completed = set()
        if self.checkpoint_file and self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    chk = json.load(f)
                completed = set(chk.get('completed_shots', []))
                logger.info(f"断点续传: 已跳过 {len(completed)} 个已完成 shot")
            except Exception:
                pass

        pending = [s for s in shots if s.get('shot') not in completed]
        logger.info(f"待处理: {len(pending)} / {total}")

        if not pending:
            logger.info("所有 shot 已处理，直接加载缓存")
            return self._load_cached_physics()

        results = []
        process_fn = partial(
            process_single_shot,
            prm_index=self.prm_index,
            use_estimator=self.use_estimator
        )

        temp_results = []
        batch_size = max(1, min(100, len(pending) // (self.workers * 4)))
        if batch_size < 10:
            batch_size = 10

        try:
            with Pool(processes=self.workers) as pool:
                for i, res in enumerate(pool.imap_unordered(process_fn, pending)):
                    temp_results.append(res)
                    shot_no = res.get('shot')
                    if shot_no:
                        completed.add(shot_no)

                    if (i + 1) % batch_size == 0 or (i + 1) == len(pending):
                        self._save_checkpoint(list(completed))
                        logger.info(f"进度: {i+1}/{len(pending)} ({(i+1)/len(pending)*100:.1f}%)")
        except Exception as e:
            logger.error(f"多进程处理中断: {e}")
            if temp_results:
                self._save_checkpoint(list(completed))
            raise

        if completed and len(completed) > len(temp_results):
            old_df = self._load_cached_physics()
            if old_df is not None:
                new_df = pd.DataFrame(temp_results)
                df_physics = pd.concat([old_df, new_df], ignore_index=True)
            else:
                df_physics = pd.DataFrame(temp_results)
        else:
            df_physics = pd.DataFrame(temp_results)

        df_physics = df_physics.drop_duplicates(subset='shot', keep='last')

        physics_cache = self.output_dir / 'physics_extracted.csv'
        df_physics.to_csv(physics_cache, index=False)
        logger.info(f"物理参数已保存: {physics_cache} ({len(df_physics)} 条)")

        self.df_physics = df_physics
        return df_physics

    def _save_checkpoint(self, completed_shots: List):
        if not self.checkpoint_file:
            return
        tmp_file = self.checkpoint_file.with_suffix('.tmp')
        try:
            with open(tmp_file, 'w') as f:
                json.dump({
                    'completed_shots': completed_shots,
                    'timestamp': datetime.now().isoformat(),
                    'total': len(completed_shots)
                }, f)
            os.replace(tmp_file, self.checkpoint_file)
        except Exception as e:
            logger.warning(f"Checkpoint 保存失败: {e}")

    def _load_cached_physics(self) -> Optional[pd.DataFrame]:
        cache = self.output_dir / 'physics_extracted.csv'
        if cache.exists():
            return pd.read_csv(cache)
        return None

    def merge_and_analyze(self) -> pd.DataFrame:
        if self.df_physics is None:
            self.df_physics = self._load_cached_physics()
        if self.df_physics is None:
            raise ValueError("没有可用的物理参数数据，请先运行 run_extraction()")

        df_summary = self.df_summary.copy()
        df_phy = self.df_physics.copy()

        df_merged = pd.merge(df_summary, df_phy, on='shot', how='inner', suffixes=('', '_phy'))
        self.df_merged = df_merged

        merged_path = self.output_dir / 'merged_kin_physics.csv'
        df_merged.to_csv(merged_path, index=False)
        logger.info(f"合并数据已保存: {merged_path} ({len(df_merged)} 条)")

        self._generate_statistics(df_merged)
        return df_merged

    def _generate_statistics(self, df: pd.DataFrame):
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("P0 PHYSICS VALIDATION REPORT")
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("=" * 70)
        report_lines.append("")

        report_lines.append(f"总样本数: {len(df)}")
        for col in ['wth', 'tau_e', 'p_heat', 'n_e', 't_e']:
            if col in df.columns:
                valid = df[col].notna().sum()
                report_lines.append(f"  {col}: {valid} 有效 ({valid/len(df)*100:.1f}%)")

        report_lines.append("")

        def corr_report(x_col: str, y_col: str, label: str):
            if x_col not in df.columns or y_col not in df.columns:
                return
            valid_df = df[[x_col, y_col]].dropna()
            if len(valid_df) < 10:
                return

            pearson_r, pearson_p = stats.pearsonr(valid_df[x_col], valid_df[y_col])
            spearman_r, spearman_p = stats.spearmanr(valid_df[x_col], valid_df[y_col])

            report_lines.append(f"{label} (n={len(valid_df)})")
            report_lines.append(f"  Pearson  r = {pearson_r:.4f}, p = {pearson_p:.2e}")
            report_lines.append(f"  Spearman ρ = {spearman_r:.4f}, p = {spearman_p:.2e}")
            report_lines.append("")

        corr_report('k_in_mean', 'wth', 'k_in vs W_th')
        corr_report('k_in_mean', 'tau_e', 'k_in vs tau_E')
        corr_report('wth', 'tau_e', 'W_th vs tau_E')
        corr_report('k_in_mean', 'iss04_tau_e', 'k_in vs ISS04 tau_E')
        corr_report('tau_e', 'iss04_tau_e', 'tau_E (exp) vs ISS04')
        corr_report('k_in_mean', 'p_heat', 'k_in vs P_heat')
        corr_report('k_in_mean', 'n_e', 'k_in vs n_e')
        corr_report('k_in_mean', 't_e', 'k_in vs T_e')
        corr_report('k_in_mean', 'duration_ms', 'k_in vs Duration')
        corr_report('k_in_mean', 'event_severity_max', 'k_in vs Event Severity')

        if 'wth_source' in df.columns:
            report_lines.append("W_th 数据来源:")
            report_lines.append(str(df['wth_source'].value_counts().to_dict()))
        if 'tau_e_source' in df.columns:
            report_lines.append("tau_E 数据来源:")
            report_lines.append(str(df['tau_e_source'].value_counts().to_dict()))

        if 'quadrant' in df.columns:
            report_lines.append("")
            report_lines.append("=== 各象限 W_th / tau_E 统计 ===")
            for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                sub = df[df['quadrant'] == q]
                if len(sub) > 0:
                    wth_mean = sub['wth'].mean() if 'wth' in sub.columns else np.nan
                    tau_mean = sub['tau_e'].mean() if 'tau_e' in sub.columns else np.nan
                    report_lines.append(f"  {q}: n={len(sub)}, W_th_mean={wth_mean:.4f}, tau_E_mean={tau_mean:.4f}")

        report_text = "\n".join(report_lines)
        report_path = self.output_dir / 'statistics_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info(f"统计报告已保存: {report_path}")
        print("\n" + report_text)

    def plot_publication_figures(self):
        if self.df_merged is None:
            raise ValueError("请先运行 merge_and_analyze()")

        df = self.df_merged

        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.02,
        })

        color_main = '#2E5AAC'
        color_fit = '#E67E22'

        def scatter_with_fit(ax, x, y, xlabel, ylabel, title, color=color_main):
            valid = pd.DataFrame({'x': x, 'y': y}).dropna()
            if len(valid) < 10:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
                return None, None

            x_arr = valid['x'].values
            y_arr = valid['y'].values

            ax.scatter(x_arr, y_arr, c=color, alpha=0.4, s=20, edgecolors='none', label='Data')

            slope, intercept, r_value, p_value, std_err = stats.linregress(x_arr, y_arr)
            x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, '--', color=color_fit, linewidth=2,
                    label=f'Linear fit: r={r_value:.3f}, p={p_value:.2e}')

            spearman_r, spearman_p = stats.spearmanr(x_arr, y_arr)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)

            textstr = f'Pearson r={r_value:.3f}\nSpearman ρ={spearman_r:.3f}\nn={len(valid)}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)

            return r_value, spearman_r

        # Figure 1: k_in vs W_th
        fig, ax = plt.subplots(figsize=(6, 5))
        if 'wth' in df.columns and 'k_in_mean' in df.columns:
            scatter_with_fit(ax, df['k_in_mean'], df['wth'],
                           xlabel=r'$k_{in}$ (Internal Synergy)',
                           ylabel=r'$W_{th}$ (MJ)',
                           title='Internal Synergy vs Thermal Stored Energy')
        fig.savefig(self.output_dir / 'fig01_kin_vs_wth.png')
        fig.savefig(self.output_dir / 'fig01_kin_vs_wth.pdf')
        plt.close(fig)

        # Figure 2: k_in vs tau_E
        fig, ax = plt.subplots(figsize=(6, 5))
        if 'tau_e' in df.columns and 'k_in_mean' in df.columns:
            scatter_with_fit(ax, df['k_in_mean'], df['tau_e'],
                           xlabel=r'$k_{in}$ (Internal Synergy)',
                           ylabel=r'$\tau_E$ (s)',
                           title='Internal Synergy vs Energy Confinement Time')
        fig.savefig(self.output_dir / 'fig02_kin_vs_tau_e.png')
        fig.savefig(self.output_dir / 'fig02_kin_vs_tau_e.pdf')
        plt.close(fig)

        # Figure 3: tau_E exp vs ISS04
        fig, ax = plt.subplots(figsize=(6, 5))
        if 'tau_e' in df.columns and 'iss04_tau_e' in df.columns:
            valid = df[['tau_e', 'iss04_tau_e']].dropna()
            if len(valid) > 10:
                ax.scatter(valid['iss04_tau_e'], valid['tau_e'],
                         c=color_main, alpha=0.4, s=20, edgecolors='none')
                lim = [min(valid.min().min(), 0), valid.max().max()]
                ax.plot(lim, lim, 'k--', linewidth=1, label='1:1 line')
                ax.set_xlabel(r'$\tau_E^{ISS04}$ (s)')
                ax.set_ylabel(r'$\tau_E^{exp}$ (s)')
                ax.set_title('Experimental vs ISS04 Scaling')
                ax.legend()
                ax.grid(True, alpha=0.3)
                r, p = stats.pearsonr(valid['iss04_tau_e'], valid['tau_e'])
                textstr = f'Pearson r={r:.3f}\nn={len(valid)}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', bbox=props)
        fig.savefig(self.output_dir / 'fig03_tau_exp_vs_iss04.png')
        fig.savefig(self.output_dir / 'fig03_tau_exp_vs_iss04.pdf')
        plt.close(fig)

        # Figure 4: 2x2 综合面板
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        if 'wth' in df.columns and 'k_in_mean' in df.columns:
            scatter_with_fit(axes[0,0], df['k_in_mean'], df['wth'],
                           xlabel=r'$k_{in}$', ylabel=r'$W_{th}$ (MJ)',
                           title='(a) Synergy vs Stored Energy')
        if 'tau_e' in df.columns and 'k_in_mean' in df.columns:
            scatter_with_fit(axes[0,1], df['k_in_mean'], df['tau_e'],
                           xlabel=r'$k_{in}$', ylabel=r'$\tau_E$ (s)',
                           title='(b) Synergy vs Confinement Time')
        if 'wth' in df.columns and 'tau_e' in df.columns:
            scatter_with_fit(axes[1,0], df['wth'], df['tau_e'],
                           xlabel=r'$W_{th}$ (MJ)', ylabel=r'$\tau_E$ (s)',
                           title='(c) Stored Energy vs Confinement Time')
        if 'tau_e' in df.columns and 'p_heat' in df.columns:
            scatter_with_fit(axes[1,1], df['p_heat'], df['tau_e'],
                           xlabel=r'$P_{heat}$ (MW)', ylabel=r'$\tau_E$ (s)',
                           title='(d) Heating Power vs Confinement Time')
        plt.tight_layout()
        fig.savefig(self.output_dir / 'fig04_composite_panel.png')
        fig.savefig(self.output_dir / 'fig04_composite_panel.pdf')
        plt.close(fig)

        # Figure 5: 数据来源质量
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if 'wth_source' in df.columns:
            source_counts = df['wth_source'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(source_counts)))
            axes[0].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%',
                       colors=colors, startangle=90)
            axes[0].set_title('W_th Data Source')
        if 'tau_e_source' in df.columns:
            source_counts = df['tau_e_source'].value_counts()
            colors = plt.cm.Set2(np.linspace(0, 1, len(source_counts)))
            axes[1].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%',
                       colors=colors, startangle=90)
            axes[1].set_title(r'$\tau_E$ Data Source')
        plt.tight_layout()
        fig.savefig(self.output_dir / 'fig05_data_source_quality.png')
        fig.savefig(self.output_dir / 'fig05_data_source_quality.pdf')
        plt.close(fig)

        # Figure 6: 四象限分色
        fig, ax = plt.subplots(figsize=(8, 6))
        quadrant_colors = {'Q1': '#2ECC71', 'Q2': '#3498DB', 'Q3': '#E74C3C', 'Q4': '#F39C12'}
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            sub = df[df['quadrant'] == q]
            if 'wth' in sub.columns and 'k_in_mean' in sub.columns:
                valid = sub[['k_in_mean', 'wth']].dropna()
                if len(valid) > 0:
                    ax.scatter(valid['k_in_mean'], valid['wth'],
                             c=quadrant_colors.get(q, 'gray'), alpha=0.5, s=20,
                             label=f'{q} (n={len(valid)})', edgecolors='none')
        ax.set_xlabel(r'$k_{in}$ (Internal Synergy)')
        ax.set_ylabel(r'$W_{th}$ (MJ)')
        ax.set_title('Internal Synergy vs Stored Energy by Quadrant')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.output_dir / 'fig06_kin_vs_wth_by_quadrant.png')
        fig.savefig(self.output_dir / 'fig06_kin_vs_wth_by_quadrant.pdf')
        plt.close(fig)

        logger.info("所有图表已保存")

    def run_full_pipeline(self):
        logger.info("=" * 70)
        logger.info("P0 PHYSICS VALIDATION PIPELINE START")
        logger.info("=" * 70)

        try:
            self.load_summary_data()
            self.run_extraction()
            self.merge_and_analyze()
            self.plot_publication_figures()

            logger.info("=" * 70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"所有输出保存在: {self.output_dir.absolute()}")
            logger.info("=" * 70)
        except Exception as e:
            logger.error(f"PIPELINE FAILED: {e}")
            traceback.print_exc()
            raise


# =============================================================================
# 8. 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='P0 Physics Validation: Extract W_th / tau_E from LHD .shot files and correlate with k_in'
    )
    parser.add_argument('--summary-csv', required=True, help='Path to v6e summary.csv')
    parser.add_argument('--data-root', default=None,
                        help='Root directory containing scattered .shot/.prm files')
    parser.add_argument('--output', default='./p0_output', help='Output directory')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--no-estimator', action='store_true',
                        help='Disable physics estimation when PRM parameters are missing')
    parser.add_argument('--checkpoint', default='p0_checkpoint.json', help='Checkpoint file name')

    args = parser.parse_args()

    validator = P0Validator(
        summary_csv=args.summary_csv,
        data_root=args.data_root,
        output_dir=args.output,
        workers=args.workers,
        use_estimator=not args.no_estimator,
        checkpoint_file=args.checkpoint
    )

    validator.run_full_pipeline()


if __name__ == '__main__':
    main()
