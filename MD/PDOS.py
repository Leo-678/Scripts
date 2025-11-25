#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 LAMMPS 速度 dump 计算 VACF & PDOS（频率单位：THz）

示例：
    python pdos_vacf_thz.py dump.lammpstrj
    python pdos_vacf_thz.py dump.lammpstrj --output-prefix AlN_300K \
        --ninitial 30 --corlength-steps 5000 --ngap-steps 200 \
        --tfreq 10 --dt 0.001 --omaga-max 25 --max-omega-points 1500
"""

import sys
import time
import math
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ------- 默认参数（可通过命令行覆盖） ------- #
DEFAULT_NINITIAL         = 20       # 多时间起点数目 Ninitial
DEFAULT_CORLENGTH_STEPS  = 1000     # 相关长度（MD 原始步数）Corlength
DEFAULT_NGAP_STEPS       = 100      # 初始时间间隔（MD 原始步数）Ngap
DEFAULT_TFREQ            = 5        # 每 TFREQ 步保存一次 dump
DEFAULT_DT               = 0.001    # MD 时间步长 (ps)
DEFAULT_OMAGA_MAX        = 20.0     # 最大频率 (THz)
DEFAULT_MAX_OMEGA_POINTS = 1000     # 频率点数


def progress_bar(current, total, start_time, prefix=""):
    """
    在终端打印进度条和 ETA。
    current: 当前进度（0 ~ total）
    total  : 总数
    """
    frac = current / total if total > 0 else 1.0
    percent = frac * 100.0
    elapsed = time.time() - start_time
    eta = elapsed * (1.0 - frac) / frac if frac > 0 else 0.0

    bar_len = 40
    filled = int(bar_len * frac)
    bar = "█" * filled + "-" * (bar_len - filled)

    msg = (f"\r{prefix} |{bar}| {percent:6.2f}%  "
           f"Elapsed: {elapsed:6.1f}s  ETA: {eta:6.1f}s")
    print(msg, end="", flush=True)
    if current == total:
        print()  # 换行


# ============================================================
# 1. 读取 LAMMPS dump（速度）
# ============================================================

def read_lammps_dump_velocities(filename):
    """
    读取 LAMMPS dump 文件中所有帧的 (id, type, vx, vy, vz)。

    要求：
      ITEM: ATOMS ... id ... type ... vx vy vz
    返回：
      velocities: shape (n_frames, n_atoms, 3)
      types     : shape (n_atoms,)  （按 id 排序后的 type）
      ids       : shape (n_atoms,)
    """
    print(f"Reading LAMMPS dump from: {filename}")
    frames_vel = []
    ids_ref = None
    types_ref = None

    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            # TIMESTEP
            step_line = f.readline()
            if not step_line:
                break

            # NUMBER OF ATOMS
            line = f.readline()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError("Expected 'ITEM: NUMBER OF ATOMS'")
            natoms = int(f.readline().strip())

            # BOX BOUNDS
            line = f.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError("Expected 'ITEM: BOX BOUNDS'")
            for _ in range(3):
                f.readline()  # 跳过 3 行 box

            # ATOMS header
            line = f.readline()
            if not line.startswith("ITEM: ATOMS"):
                raise RuntimeError("Expected 'ITEM: ATOMS'")
            header_parts = line.strip().split()[2:]  # 去掉 "ITEM:" "ATOMS"

            def find_col(name):
                if name not in header_parts:
                    raise RuntimeError(f"Dump does not contain '{name}' column")
                return header_parts.index(name)

            id_col   = find_col("id")
            type_col = find_col("type")
            vx_col   = find_col("vx")
            vy_col   = find_col("vy")
            vz_col   = find_col("vz")

            ids   = np.zeros(natoms, dtype=int)
            types = np.zeros(natoms, dtype=int)
            vels  = np.zeros((natoms, 3), dtype=float)

            # 读原子数据
            for i in range(natoms):
                parts = f.readline().split()
                ids[i]        = int(parts[id_col])
                types[i]      = int(parts[type_col])
                vels[i, 0]    = float(parts[vx_col])
                vels[i, 1]    = float(parts[vy_col])
                vels[i, 2]    = float(parts[vz_col])

            # 以 id 排序，保证每帧原子顺序一致
            sort_idx     = np.argsort(ids)
            ids_sorted   = ids[sort_idx]
            types_sorted = types[sort_idx]
            vels_sorted  = vels[sort_idx, :]

            if ids_ref is None:
                ids_ref   = ids_sorted.copy()
                types_ref = types_sorted.copy()
            else:
                if not np.array_equal(ids_sorted, ids_ref):
                    raise RuntimeError("Atom IDs differ between frames; "
                                       "cannot align velocities.")

            frames_vel.append(vels_sorted)

    velocities = np.stack(frames_vel, axis=0)  # (n_frames, n_atoms, 3)
    print(f"  Total frames read: {velocities.shape[0]}, atoms: {velocities.shape[1]}")
    return velocities, types_ref, ids_ref


# ============================================================
# 2. 多时间起点 VACF 计算（标量 v·v0）
# ============================================================

def compute_vacf_multi_origin(vels, n_initial, corlength_steps, ngap_steps,
                              dT, tfreq, label=""):
    """
    计算指定原子集合的多时间起点 VACF。
    """
    n_frames, n_atoms, _ = vels.shape
    dt_save = dT * tfreq

    # 换算为帧单位
    gap_frames  = ngap_steps      // tfreq
    corr_frames = corlength_steps // tfreq
    if gap_frames <= 0:
        raise ValueError("NGAP_STEPS / TFREQ must be >= 1")
    if corr_frames <= 0:
        raise ValueError("CORLENGTH_STEPS / TFREQ must be >= 1")

    # 选择初始帧索引：0, gap_frames, 2*gap_frames, ...
    init_indices = []
    for k in range(n_initial):
        idx = k * gap_frames
        if idx >= n_frames:
            break
        init_indices.append(idx)
    if not init_indices:
        raise RuntimeError("No valid initial conditions; check NINITIAL / NGAP / frames.")

    actual_ninit = len(init_indices)
    print(f"{label} Using {actual_ninit} initial conditions "
          f"(gap = {gap_frames} frames, corr = {corr_frames} frames)")

    # 检查是否有足够的帧覆盖全部相关长度
    last_init = init_indices[-1]
    if last_init + corr_frames > n_frames:
        raise RuntimeError(
            f"Not enough frames: last_init({last_init}) + corr_frames({corr_frames}) "
            f"> n_frames({n_frames}).\n"
            f"Please increase trajectory length or adjust CORLENGTH_STEPS/NGAP_STEPS."
        )

    vacf_matrix = np.zeros((actual_ninit, corr_frames), dtype=float)

    start_time = time.time()
    for idx_k, init_idx in enumerate(init_indices):
        v0 = vels[init_idx]             # (n_atoms, 3)
        norm0 = np.sum(v0 * v0)
        if norm0 == 0.0:
            continue

        for lag in range(corr_frames):
            frame_idx = init_idx + lag
            v = vels[frame_idx]
            dot = np.sum(v * v0)        # sum_{i} v(t)·v(0)
            vacf_matrix[idx_k, lag] = dot / norm0

        progress_bar(idx_k + 1, actual_ninit, start_time,
                     prefix=f"[VACF {label}]")

    vacf = np.mean(vacf_matrix, axis=0)
    t    = np.arange(corr_frames, dtype=float) * dt_save
    return t, vacf


# ============================================================
# 3. 从 VACF 计算 DOS（频率单位：THz）
# ============================================================

def compute_dos_from_vacf(vcorr, n_local_atoms, dT, corlength_steps,
                          tfreq, omaga_max, maxT):
    """
    根据规范化 VACF 计算 PDOS，频率单位为 THz。

    与 C 代码类似：
      domaga = omagaval / maxT
      Tdelta = dT * Corlength * 0.5
      dtw   = dT * jj * TFREQ
    """
    r_Corlength = corlength_steps // tfreq  # VACF 点数
    if len(vcorr) < r_Corlength:
        r_Corlength = len(vcorr)

    domaga = omaga_max / float(maxT)        # THz 间隔
    Tdelta = dT * corlength_steps * 0.5     # 高斯窗宽度（ps）

    freq   = np.zeros(maxT, dtype=float)    # THz
    dosval = np.zeros(maxT, dtype=float)

    print(f"Computing DOS: max freq = {omaga_max} THz, points = {maxT}")
    start_time = time.time()
    for ii in range(maxT):
        dw = ii * domaga          # THz
        dstate = 0.0
        for jj in range(r_Corlength):
            dtw = dT * jj * tfreq                    # 物理时间 (ps)
            w_t = math.cos(2.0 * math.pi * dw * dtw) \
                  * math.exp(- (dtw / Tdelta) ** 2)  # 窗函数 * cos
            dstate += dT * tfreq * vcorr[jj] * w_t

        # 横坐标直接用频率 THz
        freq[ii]   = dw
        # 纵坐标归一化因子保留原形式（只是整体尺度问题）
        dosval[ii] = 6.0 * n_local_atoms * dstate * 0.31847

        if (ii + 1) % max(1, maxT // 50) == 0:
            progress_bar(ii + 1, maxT, start_time, prefix="[DOS]")

    return freq, dosval


# ============================================================
# 4. 主流程：读 dump -> 各类型 VACF+PDOS -> 总 VACF+DOS -> 画图
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute VACF and PDOS (THz) from LAMMPS dump (pure Python)."
    )
    parser.add_argument("dump_file", help="LAMMPS dump file with vx,vy,vz")
    parser.add_argument("--output-prefix", default="pdos",
                        help="Prefix for output text & figure files (default: pdos)")

    # 关键参数全部开放到命令行
    parser.add_argument("--ninitial", type=int, default=DEFAULT_NINITIAL,
                        help=f"Number of initial times (NINITIAL), default {DEFAULT_NINITIAL}")
    parser.add_argument("--corlength-steps", type=int, default=DEFAULT_CORLENGTH_STEPS,
                        help=f"Correlation length in MD steps (CORLENGTH_STEPS), default {DEFAULT_CORLENGTH_STEPS}")
    parser.add_argument("--ngap-steps", type=int, default=DEFAULT_NGAP_STEPS,
                        help=f"Gap between initial times in MD steps (NGAP_STEPS), default {DEFAULT_NGAP_STEPS}")
    parser.add_argument("--tfreq", type=int, default=DEFAULT_TFREQ,
                        help=f"Dump saving interval in MD steps (TFREQ), default {DEFAULT_TFREQ}")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT,
                        help=f"MD time step in ps (DT), default {DEFAULT_DT}")
    parser.add_argument("--omaga-max", type=float, default=DEFAULT_OMAGA_MAX,
                        help=f"Max frequency in THz for DOS (OMAGA_MAX), default {DEFAULT_OMAGA_MAX}")
    parser.add_argument("--max-omega-points", type=int, default=DEFAULT_MAX_OMEGA_POINTS,
                        help=f"Number of frequency points for DOS (MAX_OMEGA_POINTS), default {DEFAULT_MAX_OMEGA_POINTS}")

    args = parser.parse_args()

    dump_file = args.dump_file
    prefix    = args.output_prefix

    # 读取 dump
    velocities, types, ids = read_lammps_dump_velocities(dump_file)
    n_frames, n_atoms, _   = velocities.shape

    unique_types = np.unique(types)
    print(f"Detected {len(unique_types)} atom types: {unique_types.tolist()}")

    type_to_indices = {t: np.where(types == t)[0] for t in unique_types}

    vacf_results = {}
    dos_results  = {}

    # ------- 每种原子类型的 VACF/PDOS ------- #
    for t in unique_types:
        idx   = type_to_indices[t]
        v_sub = velocities[:, idx, :]   # (n_frames, n_atoms_type, 3)
        label = f"type {t}"
        print(f"\n=== Processing atom type {t} (natoms = {len(idx)}) ===")
        t_arr, vacf = compute_vacf_multi_origin(
            v_sub,
            n_initial       = args.ninitial,
            corlength_steps = args.corlength_steps,
            ngap_steps      = args.ngap_steps,
            dT              = args.dt,
            tfreq           = args.tfreq,
            label           = label
        )
        freq, dosval = compute_dos_from_vacf(
            vacf,
            n_local_atoms   = len(idx),
            dT              = args.dt,
            corlength_steps = args.corlength_steps,
            tfreq           = args.tfreq,
            omaga_max       = args.omaga_max,
            maxT            = args.max_omega_points
        )

        vacf_results[t] = (t_arr, vacf)
        dos_results[t]  = (freq, dosval)

        # 写出单独文本文件（freq 为 THz）
        np.savetxt(f"{prefix}_VACF_type{t}.txt",
                   np.column_stack([t_arr, vacf]),
                   header="t(ps) VACF(t)")
        np.savetxt(f"{prefix}_PDOS_type{t}.txt",
                   np.column_stack([freq, dosval]),
                   header="freq_THz PDOS")

    # ------- 全部原子总 VACF / 总 DOS ------- #
    print("\n=== Processing ALL atoms (total VACF & DOS) ===")
    t_arr_total, vacf_total = compute_vacf_multi_origin(
        velocities,
        n_initial       = args.ninitial,
        corlength_steps = args.corlength_steps,
        ngap_steps      = args.ngap_steps,
        dT              = args.dt,
        tfreq           = args.tfreq,
        label           = "TOTAL"
    )
    freq_total, dos_total = compute_dos_from_vacf(
        vacf_total,
        n_local_atoms   = n_atoms,
        dT              = args.dt,
        corlength_steps = args.corlength_steps,
        tfreq           = args.tfreq,
        omaga_max       = args.omaga_max,
        maxT            = args.max_omega_points
    )

    np.savetxt(f"{prefix}_VACF_total.txt",
               np.column_stack([t_arr_total, vacf_total]),
               header="t(ps) VACF_total(t)")
    np.savetxt(f"{prefix}_DOS_total.txt",
               np.column_stack([freq_total, dos_total]),
               header="freq_THz DOS_total")

    # ====================================================
    # 画组图：上 VACF，下 PDOS（总 + 各类型）
    # ====================================================
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), dpi=200)
    ax_vacf, ax_dos = axes

    # VACF
    ax_vacf.plot(t_arr_total, vacf_total, label="total", linewidth=2.5)
    for t in unique_types:
        t_arr, vacf = vacf_results[t]
        ax_vacf.plot(t_arr, vacf, label=f"type {t}", linewidth=1.5)
    ax_vacf.set_xlabel("t (ps)")
    ax_vacf.set_ylabel("VACF(t)")
    ax_vacf.set_title("Velocity Autocorrelation Function")
    ax_vacf.set_xlim(left=0)
    ax_vacf.legend()

    # PDOS（THz）
    ax_dos.plot(freq_total, dos_total, label="total", linewidth=2.5)
    for t in unique_types:
        freq, dosval = dos_results[t]
        ax_dos.plot(freq, dosval, label=f"type {t}", linewidth=1.5)
    ax_dos.set_xlabel("Frequency (THz)")
    ax_dos.set_ylabel("DOS (arb. units)")
    ax_dos.set_title("Partial DOS and Total DOS")
    ax_dos.set_xlim(left=0)
    ax_dos.legend()

    plt.tight_layout()
    figfile = f"{prefix}_VACF_PDOS.png"
    plt.savefig(figfile, dpi=300)
    plt.close()
    print(f"\nAll done. Figure saved to: {figfile}")


if __name__ == "__main__":
    main()
