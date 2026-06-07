#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LHD 5D Analyzer v6e - Fixed Decompress + Real-Time Stream Parser
修复：ZLIB 解压失败时 fallback 原始数据；正确处理 offset_binary；实时写入；断点续传
"""

import os
import sys
import csv
import glob
import struct
import zlib
import gzip
import bz2
import re
import numpy as np
from collections import defaultdict

# ==================== 配置 ====================
FUSION_DIR = "/mnt/d/vmare/zgr/AI/fusion"
RAW_DIR = os.path.join(FUSION_DIR, "fusion_5d_data/lhd_raw")
EXTRACT_DIR = os.path.join(FUSION_DIR, "fusion_5d_data/lhd_extracted")
OUTPUT_DIR = os.path.join(FUSION_DIR, "lhd_5d_results_v6e")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary.csv")

DIM_LABELS = ["B", "S", "R", "D", "I"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 字段定义 ====================
fieldnames = ["shot", "status", "reason", "n_samples", "n_windows", "duration_ms",
              "k_in_mean", "k_in_std", "k_in_max", "k_in_min",
              "n_events", "event_severity_max",
              "B_mean", "S_mean", "R_mean", "D_mean", "I_mean"]

# ==================== 断点续传 ====================
def load_processed_shots(csv_path):
    processed = set()
    if not os.path.exists(csv_path):
        return processed
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "ok" and row.get("shot"):
                    processed.add(row["shot"])
    except Exception as e:
        print(f"[WARN] 读取已有 CSV 失败: {e}")
    return processed

# ==================== Shot 名称提取 ====================
def extract_shot_name(prm_path, base_dir):
    rel_path = os.path.relpath(prm_path, base_dir)
    parts = rel_path.split(os.sep)
    for p in parts:
        if "Shot" in p or "shot" in p.lower():
            return p
    fname = os.path.basename(prm_path).replace(".prm", "")
    m = re.search(r'(\d{5,})', fname)
    return m.group(1) if m else fname

# ==================== PRM 解析 ====================
def parse_prm(prm_path):
    params = {}
    try:
        with open(prm_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 3:
                    key = parts[1].strip()
                    val = parts[2].strip()
                    params[key] = val
    except Exception as e:
        return None
    return params

def find_dat_for_prm(prm_path):
    dat_path = prm_path.replace(".prm", ".dat")
    if os.path.exists(dat_path):
        return dat_path
    dir_path = os.path.dirname(prm_path)
    base_name = os.path.basename(prm_path).replace(".prm", "")
    candidates = glob.glob(os.path.join(dir_path, f"{base_name}*.dat"))
    if candidates:
        return candidates[0]
    return None

# ==================== 修复：解压 + fallback ====================

def decompress_data(raw_bytes, method, comp_length=None, data_length=None):
    """
    尝试按指定方法解压。如果失败，且原始数据长度合理，fallback 到原始数据。
    修复 v6c 的 bug：PRM 标记 ZLIB 但 .dat 实际未压缩时，直接返回原始数据。
    """
    method = (method or "NONE").upper()

    # 如果标记为 NONE 或空，直接返回
    if method == "NONE" or method == "":
        return raw_bytes, True, "raw"

    # 如果 CompLength == DataLength，大概率未压缩
    if comp_length and data_length:
        try:
            if int(comp_length) == int(data_length):
                return raw_bytes, True, "raw_uncompressed"
        except:
            pass

    # 尝试指定方法解压
    if method == "ZLIB":
        try:
            return zlib.decompress(raw_bytes), True, "zlib"
        except:
            for pos in [0, 16, 32, 64, 128, 256]:
                try:
                    return zlib.decompress(raw_bytes[pos:]), True, f"zlib_skip{pos}"
                except:
                    continue
    elif method == "GZIP":
        try:
            return gzip.decompress(raw_bytes), True, "gzip"
        except:
            pass
    elif method == "BZ2":
        try:
            return bz2.decompress(raw_bytes), True, "bz2"
        except:
            pass

    # 尝试所有方法
    for func, name in [(zlib.decompress, "zlib"), (gzip.decompress, "gzip"), (bz2.decompress, "bz2")]:
        try:
            result = func(raw_bytes)
            if len(result) > 100:
                return result, True, name
        except:
            pass

    # 最终 fallback：如果原始数据长度合理，直接返回原始数据
    if len(raw_bytes) >= 100:
        return raw_bytes, True, "fallback_raw"

    return None, False, "fail"

# ==================== 修复：decode_binary ====================

def decode_binary(data_bytes, image_type, resolution, binary_coding):
    image_type = (image_type or "INT16").upper()
    resolution = int(resolution) if resolution else 16
    binary_coding = (binary_coding or "shifted_2's_complementary").lower()

    dtype_map = {
        "INT8": np.int8, "UINT8": np.uint8,
        "INT16": np.int16, "UINT16": np.uint16,
        "INT32": np.int32, "UINT32": np.uint32,
        "FLOAT32": np.float32, "FLOAT64": np.float64,
    }
    dtype = dtype_map.get(image_type, np.int16)

    itemsize = np.dtype(dtype).itemsize
    if len(data_bytes) % itemsize != 0:
        data_bytes = data_bytes[:-(len(data_bytes) % itemsize)]

    if len(data_bytes) < itemsize:
        return None

    arr = np.frombuffer(data_bytes, dtype=dtype).astype(np.float64)

    # 修复：offset_binary 处理
    if "offset" in binary_coding:
        offset = 2 ** (resolution - 1)
        arr = arr - offset
    elif "shifted" in binary_coding and "complementary" in binary_coding:
        if resolution < 16 and dtype == np.int16:
            shift = 16 - resolution
            arr = np.right_shift(arr.astype(np.int32), shift)

    return arr

def parse_dat_with_prm(dat_path, params):
    if not os.path.exists(dat_path):
        return None

    with open(dat_path, "rb") as f:
        raw = f.read()

    image_type = params.get("ImageType", "INT16")
    comp_method = params.get("CompressionMethod", "NONE")
    binary_coding = params.get("BinaryCoding", "shifted_2's_complementary")
    resolution = params.get("Resolution(bit)", "16")
    comp_length = params.get("CompLength(byte)")
    data_length = params.get("DataLength(byte)")

    # 截取 CompLength
    if comp_length:
        try:
            raw = raw[:int(comp_length)]
        except:
            pass

    # 解压（带 fallback）
    data_bytes, ok, decomp_info = decompress_data(raw, comp_method, comp_length, data_length)
    if not ok or data_bytes is None:
        return None

    # 截取 DataLength
    if data_length:
        try:
            data_bytes = data_bytes[:int(data_length)]
        except:
            pass

    arr = decode_binary(data_bytes, image_type, resolution, binary_coding)

    # 如果 decode 后全是常数或空，尝试不截取 DataLength 重新 decode（有些数据 DataLength 是解压后长度）
    if arr is not None and (len(arr) == 0 or np.all(arr == arr[0])):
        arr2 = decode_binary(data_bytes, image_type, resolution, binary_coding)
        if arr2 is not None and len(arr2) > 0 and not np.all(arr2 == arr2[0]):
            arr = arr2

    return arr

# ==================== 独立 .dat 解析（盲解，与 v6c 相同）====================

def split_concatenated_file(filepath):
    with open(filepath, "rb") as f:
        raw = f.read()
    pattern = re.compile(rb'==>\s*(.+?)\s*<==\r?\n')
    matches = list(pattern.finditer(raw))
    if not matches:
        return [(os.path.basename(filepath).replace(".dat", ""), raw)]
    segments = []
    for i, m in enumerate(matches):
        shot_name = m.group(1).decode("utf-8", errors="ignore").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        segments.append((shot_name, raw[start:end]))
    return segments

ZLIB_MAGICS = [b'\x78\x9c', b'\x78\x01', b'\x78\xda', b'\x78\x5e', b'\x78\x9e', b'\x08\x1d', b'\x78\xa7', b'\x78\x0b']

def find_zlib_streams(data):
    positions = []
    for magic in ZLIB_MAGICS:
        pos = 0
        while True:
            idx = data.find(magic, pos)
            if idx == -1:
                break
            positions.append(idx)
            pos = idx + 1
    return sorted(set(positions))

def try_decompress(data, offset=0):
    payload = data[offset:]
    for func in [zlib.decompress, gzip.decompress, bz2.decompress]:
        try:
            result = func(payload)
            if len(result) > 100:
                return result, True
        except:
            pass
    try:
        dco = zlib.decompressobj()
        result = dco.decompress(payload)
        if len(result) > 100:
            return result, True
    except:
        pass
    return None, False

def parse_binary_blind(data_bytes):
    if len(data_bytes) < 100:
        return None

    candidates = [
        (np.int16, "int16"), (np.int32, "int32"), (np.float32, "float32"),
        (np.float64, "float64"), (np.uint16, "uint16"),
    ]

    best_arr = None
    best_score = -1

    for dtype, name in candidates:
        itemsize = np.dtype(dtype).itemsize
        if len(data_bytes) % itemsize != 0:
            trimmed = data_bytes[:-(len(data_bytes) % itemsize)]
        else:
            trimmed = data_bytes

        if len(trimmed) < itemsize * 10:
            continue

        try:
            arr = np.frombuffer(trimmed, dtype=dtype).astype(np.float64)
            n = len(arr)
            if n < 100 or np.all(arr == arr[0]):
                continue
            if np.issubdtype(dtype, np.floating) and np.any(~np.isfinite(arr)):
                continue

            score = min(n / 100000, 1.0) * 50
            score += min(np.std(arr) / (np.abs(np.mean(arr)) + 1e-12), 10) * 10
            score += min(len(np.unique(arr[:1000])) / 1000, 1.0) * 20

            if dtype == np.int16 and np.min(arr) >= -40000 and np.max(arr) <= 40000:
                score += 20

            if score > best_score:
                best_score = score
                best_arr = arr
        except:
            continue

    return best_arr

def parse_segment_blind(shot_name, data_bytes):
    if len(data_bytes) < 100:
        return None, "too_short"

    arr = parse_binary_blind(data_bytes)
    if arr is not None and len(arr) >= 1000:
        return arr, "ok_uncompressed"

    zlib_positions = find_zlib_streams(data_bytes)
    if not zlib_positions or zlib_positions[0] > 0:
        zlib_positions = [0] + zlib_positions

    for pos in zlib_positions[:20]:
        decompressed, ok = try_decompress(data_bytes, pos)
        if ok and decompressed and len(decompressed) > 100:
            arr = parse_binary_blind(decompressed)
            if arr is not None and len(arr) >= 1000:
                return arr, f"ok_zlib_offset{pos}"

    for skip in [16, 32, 48, 64, 128, 256, 512]:
        if skip >= len(data_bytes):
            continue
        decompressed, ok = try_decompress(data_bytes, skip)
        if ok and decompressed and len(decompressed) > 100:
            arr = parse_binary_blind(decompressed)
            if arr is not None and len(arr) >= 1000:
                return arr, f"ok_zlib_skip{skip}"

    for dtype in [np.float32, np.int16, np.int32]:
        itemsize = np.dtype(dtype).itemsize
        if len(data_bytes) >= itemsize * 1000:
            try:
                arr = np.frombuffer(data_bytes[:len(data_bytes)//itemsize*itemsize], dtype=dtype).astype(np.float64)
                if not np.all(arr == arr[0]) and len(arr) >= 1000:
                    return arr, f"ok_raw_{dtype.__name__}"
            except:
                pass

    return None, "fail"

# ==================== BSRDI 五维计算 ====================

def compute_5d(arr):
    if arr is None or len(arr) == 0:
        return None

    n = len(arr)
    arr_f = arr.astype(np.float64)

    arr_min = np.min(arr_f)
    arr_max = np.max(arr_f)
    arr_mean = np.mean(arr_f)
    arr_std = np.std(arr_f)
    arr_rms = np.sqrt(np.mean(arr_f**2))
    max_abs = max(abs(arr_max), abs(arr_min), 1e-12)

    B = np.clip(ptp := (arr_max - arr_min) / (2 * max_abs), 0, 1)

    if n >= 100:
        arr_norm = (arr_f - arr_mean) / (arr_std + 1e-12)
        ac = np.correlate(arr_norm[:min(n, 10000)], arr_norm[:min(n, 10000)], mode='full')
        ac = ac[ac.size // 2:]
        ac = ac / (ac[0] + 1e-12)
        half_idx = np.where(ac < 0.5)[0]
        half_life = half_idx[0] if len(half_idx) > 0 else len(ac)
        S = np.tanh(half_life / 100.0)
    else:
        diff_std = np.std(np.diff(arr_f))
        S = 1.0 - np.tanh(diff_std / (arr_std + 1e-12))
    S = np.clip(S, 0, 1)

    R = np.clip(1.0 - np.tanh(abs(arr_mean) / max_abs), 0, 1)

    if n >= 2:
        D = np.clip(abs(np.mean(np.sign(np.diff(arr_f)))), 0, 1)
    else:
        D = 0.5

    I = np.clip(np.tanh(arr_rms / (max_abs + 1e-12)), 0, 1)

    return {"B": float(B), "S": float(S), "R": float(R), "D": float(D), "I": float(I)}

def compute_kin(dims):
    gammas = []
    for k in DIM_LABELS:
        x = dims[k]
        if x <= 0:
            gamma = 1e-12
        else:
            gamma = min(x, 0.5) / max(x, 0.5)
            gamma = max(gamma, 1e-12)
        gammas.append(gamma)

    log_gammas = [np.log(g) for g in gammas]
    return float(np.exp(np.mean(log_gammas)))

# ==================== 处理单个 shot ====================

def process_shot(shot_name, channels, source_type):
    all_dims = []
    all_kin = []
    total_samples = 0

    for ch in channels:
        arr = ch["arr"]
        dims = compute_5d(arr)
        if dims:
            k_in = compute_kin(dims)
            all_dims.append(dims)
            all_kin.append(k_in)
            total_samples += ch["n_samples"]

    if not all_dims:
        return None

    avg_dims = {k: np.mean([d[k] for d in all_dims]) for k in DIM_LABELS}
    avg_kin = np.mean(all_kin)

    first_arr = channels[0]["arr"]
    n = len(first_arr)
    window = min(1000, max(100, n // 10))
    n_windows = n // window
    duration_ms = n / 1000.0

    if n >= window * 2:
        arr_2d = first_arr[:n_windows * window].reshape(n_windows, window)
        window_stds = np.std(arr_2d, axis=1)
        threshold = np.mean(window_stds) + 2 * np.std(window_stds)
        events = window_stds > threshold
        n_events = int(np.sum(events))
        event_severity = float(np.max(window_stds[events]) if n_events > 0 else 0)
    else:
        n_events = 0
        event_severity = 0

    return {
        "shot": shot_name,
        "status": "ok",
        "reason": f"{len(channels)}ch_{source_type}",
        "n_samples": total_samples,
        "n_windows": n_windows,
        "duration_ms": round(duration_ms, 3),
        "k_in_mean": round(avg_kin, 6),
        "k_in_std": round(np.std(all_kin), 6) if len(all_kin) > 1 else 0.0,
        "k_in_max": round(np.max(all_kin), 6),
        "k_in_min": round(np.min(all_kin), 6),
        "n_events": n_events,
        "event_severity_max": event_severity,
        "B_mean": round(avg_dims["B"], 6),
        "S_mean": round(avg_dims["S"], 6),
        "R_mean": round(avg_dims["R"], 6),
        "D_mean": round(avg_dims["D"], 6),
        "I_mean": round(avg_dims["I"], 6),
    }

# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("LHD 5D Analyzer v6e - Fixed Decompress + Real-Time Parser")
    print("修复：ZLIB fallback | offset_binary | 实时写入 | 断点续传")
    print("=" * 60)

    # 加载已处理 shot（断点续传）
    processed = load_processed_shots(SUMMARY_CSV)
    if processed:
        print(f"[Resume] 发现 {len(processed)} 个已处理 shot，自动跳过")

    # 打开 CSV（追加模式）
    file_exists = os.path.exists(SUMMARY_CSV) and os.path.getsize(SUMMARY_CSV) > 0
    csv_file = open(SUMMARY_CSV, "a", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        csv_writer.writeheader()

    def write_row(row):
        csv_writer.writerow(row)
        csv_file.flush()

    total_shots = 0
    success_count = 0
    skip_count = 0

    # ========== Source 1: .prm + .dat（按 shot 实时处理）==========
    print("\n[Source 1] Scanning .prm files...")
    prm_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, "**/*.prm"), recursive=True))
    print(f"Found {len(prm_files)} .prm files")

    # 按 shot 分组
    shot_files = defaultdict(list)
    for prm_path in prm_files:
        shot_name = extract_shot_name(prm_path, EXTRACT_DIR)
        shot_files[shot_name].append(prm_path)

    total_shots_s1 = len(shot_files)
    print(f"Grouped into {total_shots_s1} shots")
    print(f"Processing {total_shots_s1} shots from Source 1...")

    for idx, (shot_name, prm_list) in enumerate(sorted(shot_files.items())):
        if shot_name in processed:
            skip_count += 1
            continue

        if (idx + 1) % 100 == 0 or (idx + 1) == total_shots_s1:
            print(f"  Source1 shot {idx+1}/{total_shots_s1} | 成功: {success_count} | 跳过: {skip_count}")

        channels = []
        for prm_path in prm_list:
            params = parse_prm(prm_path)
            if not params:
                continue
            dat_path = find_dat_for_prm(prm_path)
            if not dat_path:
                continue
            arr = parse_dat_with_prm(dat_path, params)
            if arr is not None and len(arr) > 0:
                channels.append({"arr": arr, "n_samples": len(arr)})

        if channels:
            row = process_shot(shot_name, channels, "prm")
            if row:
                write_row(row)
                success_count += 1
                processed.add(shot_name)
        else:
            write_row({
                "shot": shot_name,
                "status": "fail",
                "reason": "no_valid_channels",
                "n_samples": 0, "n_windows": 0, "duration_ms": 0,
                "k_in_mean": "", "k_in_std": "", "k_in_max": "", "k_in_min": "",
                "n_events": 0, "event_severity_max": 0,
                "B_mean": "", "S_mean": "", "R_mean": "", "D_mean": "", "I_mean": "",
            })
        total_shots += 1

    # 释放内存
    shot_files.clear()
    import gc
    gc.collect()

    # ========== Source 2: standalone .dat ==========
    print("\n[Source 2] Processing standalone .dat files...")
    dat_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.dat")))
    print(f"Found {len(dat_files)} .dat files")

    for i, filepath in enumerate(dat_files):
        if i % 500 == 0:
            print(f"  Processing .dat {i+1}/{len(dat_files)} ...")

        segments = split_concatenated_file(filepath)

        for shot_name, data_bytes in segments:
            if shot_name in processed:
                continue

            arr, status = parse_segment_blind(shot_name, data_bytes)
            if arr is None or status.startswith("fail"):
                write_row({
                    "shot": shot_name,
                    "status": "fail",
                    "reason": status,
                    "n_samples": 0, "n_windows": 0, "duration_ms": 0,
                    "k_in_mean": "", "k_in_std": "", "k_in_max": "", "k_in_min": "",
                    "n_events": 0, "event_severity_max": 0,
                    "B_mean": "", "S_mean": "", "R_mean": "", "D_mean": "", "I_mean": "",
                })
                total_shots += 1
                continue

            row = process_shot(shot_name, [{"arr": arr, "n_samples": len(arr)}], "raw")
            if row:
                write_row(row)
                success_count += 1
                processed.add(shot_name)
            total_shots += 1

    # 关闭 CSV
    csv_file.close()

    print(f"\n{'='*60}")
    print(f"Done. Successful: {success_count}/{total_shots} shots")
    print(f"Summary saved to {SUMMARY_CSV}")
    print(f"{'='*60}")

    # 重新读取统计
    print("\nComputing statistics...")
    ok_results = []
    with open(SUMMARY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "ok":
                ok_results.append(row)

    print(f"\n五维统计（成功样本 {len(ok_results)} shots）:")
    for dim in DIM_LABELS:
        vals = [float(r[f"{dim}_mean"]) for r in ok_results if r[f"{dim}_mean"] != ""]
        if vals:
            print(f"  {dim}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, min={np.min(vals):.4f}, max={np.max(vals):.4f}")

    k_vals = [float(r["k_in_mean"]) for r in ok_results if r["k_in_mean"] != ""]
    if k_vals:
        print(f"\n  k_in: mean={np.mean(k_vals):.6f}, std={np.std(k_vals):.6f}, min={np.min(k_vals):.6f}, max={np.max(k_vals):.6f}")

if __name__ == "__main__":
    main()
