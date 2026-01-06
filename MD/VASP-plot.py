#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Headless (no-GUI) OSZICAR MD plotter
Extract:
- Temperature (K)
- Energy E (eV)

Plot:
- T vs MD step (or time)
- E vs MD step (or time)

Safe for SSH / HPC / batch jobs
"""

# ===== 强制无界面后端（必须放在最前面） =====
import matplotlib
matplotlib.use("Agg")

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator

# ================= 用户参数 =================
OSZICAR = "OSZICAR"

# 若想用真实时间轴（ps），填 POTIM（fs），否则用 MD step
POTIM_FS = None    # e.g. 2.0

OUTFIG = "oszicar_md_TE.png"
# ===========================================


# ---- 正则：匹配 MD 行 ----
# 示例：
#  1 T=   900. E= -.11415073E+04 F= ...
re_md = re.compile(
    r"^\s*(\d+)\s+T=\s*([0-9.+-Ee]+)\s+E=\s*([0-9.+-Ee]+)",
    re.I
)


def parse_oszicar(fname):
    steps, temps, energy = [], [], []

    with open(fname, "r", errors="ignore") as f:
        for line in f:
            m = re_md.match(line)
            if m:
                steps.append(int(m.group(1)))
                temps.append(float(m.group(2)))
                energy.append(float(m.group(3)))

    if not steps:
        raise RuntimeError("No MD information found in OSZICAR")

    return np.array(steps), np.array(temps), np.array(energy)


def nice_axis(ax):
    """统一美化坐标轴（不使用科学计数法）"""
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    fmt = ScalarFormatter(useMathText=False)
    fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    ax.grid(True, alpha=0.25, linestyle="--")


def plot_md(steps, T, E):
    # ---- x 轴 ----
    if POTIM_FS is not None:
        x = steps * POTIM_FS / 1000.0
        xlabel = "Time (ps)"
    else:
        x = steps
        xlabel = "MD step"

    fig, axs = plt.subplots(
        2, 1,
        figsize=(8.0, 6.5),
        sharex=True
    )

    # ---- Temperature ----
    axs[0].plot(x, T, lw=1.8, color="#1f77b4")
    axs[0].axhline(T.mean(), ls="--", lw=1.0, color="k", alpha=0.6)
    axs[0].set_ylabel("Temperature (K)", fontsize=11)
    nice_axis(axs[0])

    # ---- Energy ----
    axs[1].plot(x, E, lw=1.8, color="#d62728")
    axs[1].axhline(E.mean(), ls="--", lw=1.0, color="k", alpha=0.6)
    axs[1].set_ylabel("Energy E (eV)", fontsize=11)
    axs[1].set_xlabel(xlabel, fontsize=11)
    nice_axis(axs[1])

    plt.tight_layout()
    plt.savefig(OUTFIG, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    steps, T, E = parse_oszicar(OSZICAR)
    print(f"[INFO] Parsed {len(steps)} MD steps from {OSZICAR}")
    plot_md(steps, T, E)
    print(f"[INFO] Figure written to {OUTFIG}")
