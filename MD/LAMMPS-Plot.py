#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot LAMMPS thermo output from log.lammps.

Supported thermo_style:
thermo_style custom step temp press pxx pyy pzz pe ke enthalpy etotal vol \
                    cella cellb cellc cellalpha cellbeta cellgamma

Usage:
    python plt.py log.lammps
    python plt.py log.lammps --prefix thermo
"""

import re
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# Global plot style
# =========================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "axes.linewidth": 1.0,
})


# =========================================================
# Utilities
# =========================================================
_num_pattern = re.compile(r'^[-+]?\d+(\.\d*)?([eEdD][-+]?\d+)?$')


def is_number(s: str) -> bool:
    return bool(_num_pattern.match(s))


def smooth_if_needed(y, window=1):
    """
    当前默认不平滑。
    如果以后想弱平滑，可以把 window 改成 3/5。
    """
    if window <= 1:
        return y

    y = np.asarray(y)
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def set_common_grid(ax):
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(direction="in", top=True, right=True)


def merge_legend(ax, ax_right=None, loc="best"):
    lines, labels = ax.get_legend_handles_labels()

    if ax_right is not None:
        lines_r, labels_r = ax_right.get_legend_handles_labels()
        lines += lines_r
        labels += labels_r

    ax.legend(
        lines,
        labels,
        loc=loc,
        frameon=True,
        framealpha=0.88,
        borderpad=0.6,
        handlelength=2.6,
    )


# =========================================================
# Parse LAMMPS log
# =========================================================
def parse_log_thermo_blocks(logfile):
    with open(logfile, "r") as f:
        lines = f.read().splitlines()

    runs = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        if line.startswith("Step"):
            header = line.split()
            ncols = len(header)
            i += 1

            data_rows = []

            while i < n:
                s = lines[i].strip()

                if not s:
                    i += 1
                    break

                parts = s.split()

                if not parts:
                    i += 1
                    break

                if parts[0] in ("Step", "Loop"):
                    break

                if not is_number(parts[0]):
                    break

                if len(parts) != ncols:
                    break

                try:
                    row = [
                        float(x.replace("D", "E").replace("d", "e"))
                        for x in parts
                    ]
                except ValueError:
                    break

                data_rows.append(row)
                i += 1

            if data_rows:
                runs.append((header, np.array(data_rows, dtype=float)))

        else:
            i += 1

    return runs


def build_col_finder(header):
    header_lc = [h.lower() for h in header]

    def find_col(candidates, required=True):
        for cand in candidates:
            cand_lc = cand.lower()
            if cand_lc in header_lc:
                return header_lc.index(cand_lc)

        if required:
            raise KeyError(f"None of {candidates} found in thermo header: {header}")

        return None

    return find_col


# =========================================================
# Plot one run
# =========================================================
def plot_one_run(run_index, header, data, prefix="thermo"):
    if data.shape[0] < 5:
        print(f"[warning] run {run_index} has only {data.shape[0]} thermo lines. Skip.")
        return

    find_col = build_col_finder(header)

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

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=220)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # =====================================================
    # 1) Temperature + Volume
    # =====================================================
    ax1.plot(step, temp, label="Temp", color="tab:blue", lw=1.2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Temperature (K)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title("Temperature & Volume")
    set_common_grid(ax1)

    ax1b = ax1.twinx()
    ax1b.plot(step, vol, label="Volume", color="tab:orange", lw=1.8)
    ax1b.set_ylabel("Volume (Å³)", color="tab:orange")
    ax1b.tick_params(axis="y", labelcolor="tab:orange")
    ax1b.tick_params(direction="in")

    merge_legend(ax1, ax1b, loc="best")

    # =====================================================
    # 2) Pressure tensor
    # =====================================================
    ax2.plot(step, press, label="Press", color="tab:blue", lw=2.0)
    ax2.plot(step, pxx, label="Pxx", color="tab:orange", lw=1.2, ls="--")
    ax2.plot(step, pyy, label="Pyy", color="tab:green", lw=1.2, ls="-.")
    ax2.plot(step, pzz, label="Pzz", color="tab:red", lw=1.2, ls=":")

    ax2.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Pressure (bar)")
    ax2.set_title("Pressure & Stress Components")
    set_common_grid(ax2)
    ax2.legend(loc="best", framealpha=0.88)

    # =====================================================
    # 3) Energies
    # =====================================================
    ax3_left = ax3
    ax3_right = ax3_left.twinx()

    ax3_left.plot(step, ke, label="KinEng", color="tab:blue", lw=1.2)
    ax3_left.set_xlabel("Step")
    ax3_left.set_ylabel("KinEng", color="tab:blue")
    ax3_left.tick_params(axis="y", labelcolor="tab:blue")
    ax3_left.set_title("Energies")
    set_common_grid(ax3_left)

    ax3_right.plot(step, pe, label="PotEng", color="tab:red", lw=1.1)
    ax3_right.plot(step, enthalpy, label="Enthalpy", color="tab:orange", lw=1.1)
    ax3_right.plot(step, etotal, label="TotEng", color="tab:green", lw=1.1, ls="--")
    ax3_right.set_ylabel("PotEng / Enthalpy / TotEng", color="tab:red")
    ax3_right.tick_params(axis="y", labelcolor="tab:red")
    ax3_right.tick_params(direction="in")

    merge_legend(ax3_left, ax3_right, loc="best")

    # =====================================================
    # 4) Lattice constants + angles
    # =====================================================
    # cella/cellb/cellc 使用不同线型，避免重合时看不清
    ax4.plot(
        step, cella,
        label="cella",
        color="tab:blue",
        lw=2.0,
        ls="-",
        zorder=3,
    )

    ax4.plot(
        step, cellb,
        label="cellb",
        color="tab:orange",
        lw=2.0,
        ls="--",
        zorder=4,
    )

    ax4.plot(
        step, cellc,
        label="cellc",
        color="tab:green",
        lw=2.5,
        ls=":",
        zorder=5,
    )

    ax4.set_xlabel("Step")
    ax4.set_ylabel("Lattice constants (Å)", color="tab:blue")
    ax4.tick_params(axis="y", labelcolor="tab:blue")
    ax4.set_title("Lattice Constants & Angles")
    set_common_grid(ax4)

    ax4b = ax4.twinx()

    ax4b.plot(
        step, ca,
        label="alpha",
        color="tab:red",
        lw=1.4,
        ls="-.",
        alpha=0.72,
        zorder=1,
    )

    ax4b.plot(
        step, cb,
        label="beta",
        color="tab:purple",
        lw=1.4,
        ls="--",
        alpha=0.72,
        zorder=1,
    )

    ax4b.plot(
        step, cg,
        label="gamma",
        color="tab:brown",
        lw=1.6,
        ls=":",
        alpha=0.72,
        zorder=1,
    )

    ax4b.set_ylabel("Lattice angles (deg)", color="tab:red")
    ax4b.tick_params(axis="y", labelcolor="tab:red")
    ax4b.tick_params(direction="in")

    merge_legend(ax4, ax4b, loc="upper left")

    fig.suptitle(f"Thermo summary of run {run_index}", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    outname = f"{prefix}_run{run_index}.png"
    fig.savefig(outname, dpi=300)
    plt.close(fig)

    print(f"Saved figure for run {run_index}: {outname}")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Plot multiple LAMMPS thermo runs from log.lammps"
    )
    parser.add_argument("logfile", help="LAMMPS log file, e.g. log.lammps")
    parser.add_argument(
        "--prefix",
        default="thermo",
        help="Prefix for output figures. Default: thermo"
    )

    args = parser.parse_args()

    runs = parse_log_thermo_blocks(args.logfile)

    if not runs:
        print("No complete thermo blocks found.")
        sys.exit(0)

    print(f"Detected {len(runs)} thermo run(s).")

    for i, (header, data) in enumerate(runs, start=1):
        print(f"Processing run {i} with {data.shape[0]} thermo lines ...")
        plot_one_run(i, header, data, prefix=args.prefix)


if __name__ == "__main__":
    main()