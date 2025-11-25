#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 LAMMPS log.lammps 里解析多轮 run 的 thermo_style custom 输出，
并为每一轮生成一张 4 子图的大图：

1) 温度 & 体积 vs step（双 y 轴）
2) 总压 Press & 分压 Pxx/Pyy/Pzz vs step
3) 势能 PotEng、动能 KinEng、Enthalpy、TotEng vs step
4) 晶格常数 cella/cellb/cellc & 晶格角 cellalpha/beta/gamma vs step（双 y 轴）

适配命令：
thermo_style custom step temp press pxx pyy pzz pe ke enthalpy etotal vol \
                          cella cellb cellc cellalpha cellbeta cellgamma

用法：
    python plt.py log.lammps
    python plt.py log.lammps --prefix myrun
"""

import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ------------------ 工具函数：判断一个字符串是不是数字 ------------------ #

_num_pattern = re.compile(r'^[-+]?\d+(\.\d*)?([eEdD][-+]?\d+)?$')

def is_number(s: str) -> bool:
    """判断字符串 s 是否是类似 1, -2.3, 4.5e-3, 1.0D+02 这样的数字。"""
    return bool(_num_pattern.match(s))


# ------------------ 解析 log.lammps 中的 thermo 块 ------------------ #

def parse_log_thermo_blocks(logfile):
    """
    解析 LAMMPS log 文件中的 thermo 输出块。

    返回：
        runs: list[(header, data)]
            header: list[str]  数组列名，例如 ["Step","Temp","Press",...]
            data  : np.ndarray shape (n_steps, n_cols)

    对不完整 block（列数不统一 / 最后一行没写完）会自动忽略，
    所以可以在 MD 还在运行时安全调用。
    """
    with open(logfile, "r") as f:
        lines = f.read().splitlines()

    runs = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        # thermo block 一般以 "Step ..." 开头
        if line.startswith("Step"):
            header = line.split()
            ncols = len(header)
            i += 1  # 移到数据行的第一行

            data_rows = []

            while i < n:
                s = lines[i].strip()

                # 空行：认为当前 thermo block 结束
                if not s:
                    i += 1
                    break

                parts = s.split()

                # 碰到新的 Step / Loop time / 非数字开头的行：结束当前 block
                if parts[0] in ("Step", "Loop"):
                    break
                if not is_number(parts[0]):
                    break

                # 列数不对：通常是最后一行没写完，结束当前 block（不再继续读）
                if len(parts) != ncols:
                    # print(f"[warning] skip incomplete thermo line: {s}")
                    break

                # 尝试转成 float
                try:
                    row = [float(x.replace('D', 'E').replace('d', 'e')) for x in parts]
                except ValueError:
                    # 有非数字内容，结束本 block
                    break

                data_rows.append(row)
                i += 1

            # 如果这一轮有有效数据，就存下来；完全不完整的 block 丢弃
            if data_rows:
                try:
                    data = np.array(data_rows, dtype=float)
                except ValueError:
                    data = None

                if data is not None:
                    runs.append((header, data))
        else:
            i += 1

    return runs


# ------------------ 从 header 中找到某个物理量的列索引 ------------------ #

def build_col_finder(header):
    """
    传入 Thermo header，例如:
        ["Step","Temp","Press","Pxx","Pyy","Pzz",
         "PotEng","KinEng","Enthalpy","TotEng","Volume",
         "Cella","Cellb","Cellc","CellAlpha","CellBeta","CellGamma"]

    返回一个函数 find_col(name_list)：
        给定一个“候选名字列表”，返回第一个匹配到的列号。
    """
    header_lc = [h.lower() for h in header]

    def find_col(candidates, required=True):
        """
        candidates: list[str] ，优先匹配顺序
        required  : 若 True，找不到则抛错；若 False，找不到返回 None
        """
        for cand in candidates:
            cand_lc = cand.lower()
            if cand_lc in header_lc:
                return header_lc.index(cand_lc)
        if required:
            raise KeyError(f"None of {candidates} found in thermo header: {header}")
        else:
            return None

    return find_col


# ------------------ 为每一轮 run 画图 ------------------ #

def plot_one_run(run_index, header, data, prefix="thermo"):
    """
    对单轮 run 的 thermo 数据画图并保存。

    参数：
        run_index: 从 1 开始的序号
        header   : list[str]
        data     : np.ndarray (n_steps, n_cols)
        prefix   : 输出文件名前缀
    """
    if data.shape[0] < 2:
        print(f"[warning] run {run_index} has less than 2 rows, skip plotting.")
        return

    find_col = build_col_finder(header)

    # 提取需要的列索引（大小写不敏感，兼容 LAMMPS 默认名字）
    idx_step     = find_col(["step"])
    idx_temp     = find_col(["temp"])
    idx_press    = find_col(["press"])
    idx_pxx      = find_col(["pxx"])
    idx_pyy      = find_col(["pyy"])
    idx_pzz      = find_col(["pzz"])
    idx_pe       = find_col(["pe", "poteng"])
    idx_ke       = find_col(["ke", "kineng"])
    idx_enthalpy = find_col(["enthalpy"])
    idx_etotal   = find_col(["etotal", "toteng"])
    idx_vol      = find_col(["vol", "volume"])
    idx_cella    = find_col(["cella"])
    idx_cellb    = find_col(["cellb"])
    idx_cellc    = find_col(["cellc"])
    idx_ca       = find_col(["cellalpha"])
    idx_cb       = find_col(["cellbeta"])
    idx_cg       = find_col(["cellgamma"])

    step = data[:, idx_step]

    temp     = data[:, idx_temp]
    press    = data[:, idx_press]
    pxx      = data[:, idx_pxx]
    pyy      = data[:, idx_pyy]
    pzz      = data[:, idx_pzz]
    pe       = data[:, idx_pe]
    ke       = data[:, idx_ke]
    enthalpy = data[:, idx_enthalpy]
    etotal   = data[:, idx_etotal]
    vol      = data[:, idx_vol]
    cella    = data[:, idx_cella]
    cellb    = data[:, idx_cellb]
    cellc    = data[:, idx_cellc]
    ca       = data[:, idx_ca]
    cb       = data[:, idx_cb]
    cg       = data[:, idx_cg]

    # ---- 画 2x2 子图 ---- #
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=200)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # 1) 温度 + 体积（双 y 轴）
    ax1.plot(step, temp, label="Temp (K)", color="C0")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Temp (K)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax1b = ax1.twinx()
    ax1b.plot(step, vol, label="Volume", color="C1")
    ax1b.set_ylabel("Volume", color="C1")
    ax1b.tick_params(axis="y", labelcolor="C1")

    ax1.set_title("Temperature & Volume vs Step")

    # 2) 总压 + 分压
    ax2.plot(step, press, label="Press", linewidth=2.0, color="C0")
    ax2.plot(step, pxx,   label="Pxx",  linestyle="--", color="C1")
    ax2.plot(step, pyy,   label="Pyy",  linestyle="--", color="C2")
    ax2.plot(step, pzz,   label="Pzz",  linestyle="--", color="C3")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Pressure")
    ax2.set_title("Pressure & Stress Components vs Step")
    ax2.legend(fontsize=8)

    # 3) 势能 / 动能 / 焓 / 总能
    ax3.plot(step, pe,       label="PotEng",   color="C0")
    ax3.plot(step, ke,       label="KinEng",   color="C1")
    ax3.plot(step, enthalpy, label="Enthalpy", color="C2")
    ax3.plot(step, etotal,   label="TotEng",   color="C3", linestyle="--")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Energy")
    ax3.set_title("Energies vs Step")
    ax3.legend(fontsize=8)

    # 4) 晶格常数 + 晶格角（双 y 轴）
    ax4.plot(step, cella, label="cella", color="C0")
    ax4.plot(step, cellb, label="cellb", color="C1")
    ax4.plot(step, cellc, label="cellc", color="C2")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Lattice constants (Å)", color="C0")
    ax4.tick_params(axis="y", labelcolor="C0")

    ax4b = ax4.twinx()
    ax4b.plot(step, ca, label="alpha", linestyle="--", color="C3")
    ax4b.plot(step, cb, label="beta",  linestyle="--", color="C4")
    ax4b.plot(step, cg, label="gamma", linestyle="--", color="C5")
    ax4b.set_ylabel("Lattice angles (deg)", color="C3")
    ax4b.tick_params(axis="y", labelcolor="C3")

    # 合并左右 y 轴 legend
    lines_left,  labels_left  = ax4.get_legend_handles_labels()
    lines_right, labels_right = ax4b.get_legend_handles_labels()
    ax4.legend(lines_left + lines_right, labels_left + labels_right,
               fontsize=8, loc="best")
    ax4.set_title("Lattice constants & angles vs Step")

    fig.suptitle(f"Thermo summary of run {run_index}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    outname = f"{prefix}_run{run_index}.png"
    plt.savefig(outname, dpi=300)
    plt.close(fig)
    print(f"Saved figure for run {run_index}: {outname}")


# ------------------ 主程序 ------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Plot multiple LAMMPS thermo runs from log.lammps"
    )
    parser.add_argument("logfile", help="LAMMPS log file, e.g. log.lammps")
    parser.add_argument("--prefix", default="thermo",
                        help="Prefix for output figures (default: thermo)")
    args = parser.parse_args()

    runs = parse_log_thermo_blocks(args.logfile)
    if not runs:
        print("No complete thermo blocks (Step ... ) found in log file.")
        sys.exit(0)

    print(f"Detected {len(runs)} thermo run(s).")

    for i, (header, data) in enumerate(runs, start=1):
        print(f"Processing run {i} with {data.shape[0]} thermo lines ...")
        plot_one_run(i, header, data, prefix=args.prefix)


if __name__ == "__main__":
    main()
