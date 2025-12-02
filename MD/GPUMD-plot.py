#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 GPUMD thermo.out 中读取数据，并绘制 4 个子图：

列约定（无表头，无步长列）：
column   1  2  3  4   5   6   7    8    9    10 11 12 13 14 15 16 17 18
quantity T  K  U  Pxx Pyy Pzz Pyz  Pxz  Pxy  ax ay az bx by bz cx cy cz

1) T
2) K, U
3) Pxx, Pyy, Pzz
4) ax ay az bx by bz cx cy cz

横坐标：行号（index）
使用方法：
    python plot_thermo.py        # 默认读 thermo.out
    python plot_thermo.py foo    # 也可以指定其它文件名 foo
"""

import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")  # 无图形界面后端
import matplotlib.pyplot as plt


def load_thermo_fixed(filename):
    """
    简单读取 thermo.out，为固定 18 列格式：
    T K U Pxx Pyy Pzz Pyz Pxz Pxy ax ay az bx by bz cx cy cz
    """
    data = np.loadtxt(filename, comments="#")

    # 确保二维
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 18:
        raise RuntimeError(
            f"期望 {filename} 至少有 18 列，实际只有 {data.shape[1]} 列，请检查文件格式。"
        )
    return data


def main():
    # --------- 处理命令行参数：无参数 -> thermo.out ---------- #
    if len(sys.argv) < 2:
        infile = "thermo.out"
        print("未指定文件名，默认读取 thermo.out")
    else:
        infile = sys.argv[1]
        print(f"读取指定文件: {infile}")

    # --------- 读数据 ---------- #
    data = load_thermo_fixed(infile)
    nrows = data.shape[0]
    print(f"读取 {infile} 共 {nrows} 行。")

    # --------- 横坐标：行号 ---------- #
    x = np.arange(nrows)  # 0,1,2,...,n-1
    x_label = "index"
    # 如果你更喜欢从 1 开始，改成：
    # x = np.arange(1, nrows + 1)
    # x_label = "index (1-based)"

    # --------- 各列赋名（按你给的固定顺序） ---------- #
    T   = data[:, 0]   # col 1
    K   = data[:, 1]   # col 2
    U   = data[:, 2]   # col 3

    Pxx = data[:, 3]   # col 4
    Pyy = data[:, 4]   # col 5
    Pzz = data[:, 5]   # col 6
    Pyz = data[:, 6]   # col 7
    Pxz = data[:, 7]   # col 8
    Pxy = data[:, 8]   # col 9

    ax = data[:, 9]    # col 10
    ay = data[:, 10]   # col 11
    az = data[:, 11]   # col 12
    bx = data[:, 12]   # col 13
    by = data[:, 13]   # col 14
    bz = data[:, 14]   # col 15
    cx = data[:, 15]   # col 16
    cy = data[:, 16]   # col 17
    cz = data[:, 17]   # col 18

    # --------- 画 4 个子图 ---------- #
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax11, ax12, ax21, ax22 = axes.reshape(-1)

    # 子图1：T
    ax11.plot(x, T, linewidth=1)
    ax11.set_xlabel(x_label)
    ax11.set_ylabel("T (K)")
    ax11.set_title("Temperature")

    # 子图2：K & U
    ax12.plot(x, K, label="K", linewidth=1)
    ax12.plot(x, U, label="U", linewidth=1)
    ax12.set_xlabel(x_label)
    ax12.set_ylabel("Energy (eV)")
    ax12.set_title("Kinetic & Potential Energy")
    ax12.legend()

    # 子图3：Pxx, Pyy, Pzz
    ax21.plot(x, Pxx, label="Pxx", linewidth=1)
    ax21.plot(x, Pyy, label="Pyy", linewidth=1)
    ax21.plot(x, Pzz, label="Pzz", linewidth=1)
    ax21.set_xlabel(x_label)
    ax21.set_ylabel("P (thermo units)")
    ax21.set_title("Diagonal Pressure/Stress")
    ax21.legend()

    # （如果想看剪切分量，可以解开下面几行）
    # ax21.plot(x, Pyz, label="Pyz", linewidth=1, linestyle="--")
    # ax21.plot(x, Pxz, label="Pxz", linewidth=1, linestyle="--")
    # ax21.plot(x, Pxy, label="Pxy", linewidth=1, linestyle="--")

    # 子图4：晶格矢量分量
    ax22.plot(x, ax, label="ax", linewidth=1)
    ax22.plot(x, ay, label="ay", linewidth=1)
    ax22.plot(x, az, label="az", linewidth=1)
    ax22.plot(x, bx, label="bx", linewidth=1)
    ax22.plot(x, by, label="by", linewidth=1)
    ax22.plot(x, bz, label="bz", linewidth=1)
    ax22.plot(x, cx, label="cx", linewidth=1)
    ax22.plot(x, cy, label="cy", linewidth=1)
    ax22.plot(x, cz, label="cz", linewidth=1)
    ax22.set_xlabel(x_label)
    ax22.set_ylabel("Lattice component")
    ax22.set_title("Lattice Vectors (components)")
    ax22.legend(fontsize=8, ncol=3)

    fig.suptitle(f"{infile} summary", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("thermo_plots.png", dpi=300)
    plt.close(fig)

    print("已保存图像：thermo_plots.png")


if __name__ == "__main__":
    main()
