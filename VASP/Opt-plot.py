#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import numpy as np

import matplotlib
matplotlib.use("Agg")  # no GUI backend

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


# =========================================================
# 1. Basic parsers
# =========================================================
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def read_incar(incar_path="INCAR"):
    """
    Parse simple INCAR key=value pairs.
    """
    incar = {}
    if not os.path.isfile(incar_path):
        return incar

    with open(incar_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.split("!")[0].split("#")[0].strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            incar[key.strip().upper()] = val.strip()
    return incar


# =========================================================
# 2. Parse ionic energies from OSZICAR
# =========================================================
def parse_oszicar(oszicar_path="OSZICAR"):
    """
    Read ionic-step energies from OSZICAR.

    Typical ionic-step lines in OSZICAR look like:
        1 F= -.70234892E+03 E0= -.70234891E+03  d E = ...
        2 F= -.71968432E+03 E0= -.71968431E+03  d E = ...

    We use F= as the plotted energy since it is the common free energy
    reported per ionic step for optimization progress.
    """
    if not os.path.isfile(oszicar_path):
        raise FileNotFoundError(f"{oszicar_path} not found.")

    ionic_steps = []
    energies = []

    pattern = re.compile(
        r"^\s*(\d+)\s+F=\s*([-\d.Ee+]+)"
    )

    with open(oszicar_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                energy = float(m.group(2))
                ionic_steps.append(step)
                energies.append(energy)

    if len(ionic_steps) == 0:
        raise RuntimeError("No ionic-step energies found in OSZICAR.")

    return np.array(ionic_steps, dtype=int), np.array(energies, dtype=float)


# =========================================================
# 3. Parse max force and volume from OUTCAR
# =========================================================
def parse_outcar_forces_volumes(outcar_path="OUTCAR"):
    """
    Parse OUTCAR for:
      - max force per ionic step from POSITION TOTAL-FORCE block
      - volume of cell per ionic step from 'volume of cell :'

    Notes:
    - We read one max-force value per POSITION TOTAL-FORCE block
    - We read one volume per 'volume of cell :' occurrence
    """
    if not os.path.isfile(outcar_path):
        raise FileNotFoundError(f"{outcar_path} not found.")

    with open(outcar_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    max_forces = []
    volumes = []

    re_force_header = re.compile(r"POSITION\s+TOTAL-FORCE", re.IGNORECASE)
    re_volume = re.compile(r"volume of cell\s*:\s*([-\d.Ee+]+)", re.IGNORECASE)

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # ---- parse force block ----
        if re_force_header.search(line):
            tmp_forces = []
            i += 1

            # skip separators / blank lines
            while i < n and (not lines[i].strip() or set(lines[i].strip()) <= set("- ")):
                i += 1

            # read atom rows until next separator / blank
            while i < n:
                s = lines[i].strip()
                if not s or set(s) <= set("- "):
                    break

                parts = s.split()
                # expected format: x y z fx fy fz
                if len(parts) >= 6:
                    fx = safe_float(parts[3])
                    fy = safe_float(parts[4])
                    fz = safe_float(parts[5])
                    if fx is not None and fy is not None and fz is not None:
                        fmag = math.sqrt(fx * fx + fy * fy + fz * fz)
                        tmp_forces.append(fmag)
                i += 1

            max_forces.append(max(tmp_forces) if tmp_forces else np.nan)
            continue

        # ---- parse volume ----
        m_vol = re_volume.search(line)
        if m_vol:
            volumes.append(float(m_vol.group(1)))

        i += 1

    return np.array(max_forces, dtype=float), np.array(volumes, dtype=float)


# =========================================================
# 4. Helpers
# =========================================================
def parse_force_threshold_from_incar(incar):
    """
    If EDIFFG < 0, VASP uses force criterion |EDIFFG| in eV/Å.
    """
    val = safe_float(incar.get("EDIFFG", None))
    if val is not None and val < 0:
        return abs(val)
    return None


def first_converged_step(max_forces, force_thr):
    if force_thr is None:
        return None
    for i, f in enumerate(max_forces, start=1):
        if np.isfinite(f) and f <= force_thr:
            return i
    return None


def choose_energy_break(energies):
    """
    Decide whether a broken y-axis is needed for energy.

    Logic:
    - if early-step outlier(s) are far above the main converged band,
      use broken axis.
    """
    e = np.asarray(energies, dtype=float)
    e = e[np.isfinite(e)]

    if len(e) < 6:
        return False, None, None

    # Ignore first 3-5 steps to estimate the main converged band
    tail = e[min(5, len(e)-1):]
    if len(tail) < 3:
        return False, None, None

    q01, q99 = np.percentile(tail, [1, 99])
    span = max(q99 - q01, 1e-6)

    global_max = np.max(e)
    global_min = np.min(e)

    if (global_max - q99) > 8.0 * span:
        lower_ylim = (q01 - 0.08 * span, q99 + 0.08 * span)

        upper_low = max(q99 + 2.0 * span, q99 + 0.25 * (global_max - q99))
        upper_high = global_max + 0.05 * (global_max - global_min + 1e-8)
        upper_ylim = (upper_low, upper_high)

        return True, lower_ylim, upper_ylim

    return False, None, None


def finite_limits(y, pad_ratio=0.08):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return None, None
    ymin = np.min(y)
    ymax = np.max(y)
    if np.isclose(ymin, ymax):
        delta = max(abs(ymin) * 0.03, 1e-3)
        return ymin - delta, ymax + delta
    pad = (ymax - ymin) * pad_ratio
    return ymin - pad, ymax + pad


# =========================================================
# 5. Main plot
# =========================================================
def plot_vasp_optimization(
    incar_path="INCAR",
    oszicar_path="OSZICAR",
    outcar_path="OUTCAR",
    output_path="opt_progress_pretty.png",
):
    incar = read_incar(incar_path)

    # --- energies from OSZICAR ---
    ionic_steps_e, energies = parse_oszicar(oszicar_path)

    # --- forces & volumes from OUTCAR ---
    max_forces, volumes = parse_outcar_forces_volumes(outcar_path)

    # Align all series by the minimum available ionic-step count
    n_energy = len(energies)
    n_force = len(max_forces)
    n_vol = len(volumes)

    nsteps = min(n_energy, n_force, n_vol)

    if nsteps == 0:
        raise RuntimeError("No aligned ionic-step data found among OSZICAR/OUTCAR.")

    steps = np.arange(1, nsteps + 1, dtype=int)
    energies = energies[:nsteps]
    max_forces = max_forces[:nsteps]
    volumes = volumes[:nsteps]

    # Convergence criterion
    force_thr = parse_force_threshold_from_incar(incar)
    conv_step = first_converged_step(max_forces, force_thr)

    # Optional NSW
    nsw = safe_float(incar.get("NSW", None))
    nsw = int(nsw) if nsw is not None else None

    # xlim: only show current actual range + small padding
    xmax = max(10, int(math.ceil(nsteps * 1.05)))
    xmin = 1

    # Decide if broken y-axis is needed
    use_break, lower_ylim, upper_ylim = choose_energy_break(energies)

    # ---------------- style ----------------
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_style("whitegrid", {
        "grid.linestyle": "--",
        "grid.alpha": 0.20,
        "axes.edgecolor": "0.15",
        "axes.linewidth": 1.1,
    })

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })

    # Colors
    c_energy = sns.color_palette("deep")[0]
    c_force = sns.color_palette("deep")[3]
    c_vol = sns.color_palette("deep")[2]
    c_ref = "0.35"

    # ---------------- figure layout ----------------
    if use_break:
        fig = plt.figure(figsize=(10.4, 10.8), constrained_layout=False)
        gs = GridSpec(4, 1, height_ratios=[0.95, 2.2, 2.35, 2.35], hspace=0.10, figure=fig)

        ax_e_top = fig.add_subplot(gs[0])
        ax_e_bot = fig.add_subplot(gs[1], sharex=ax_e_top)
        ax_f = fig.add_subplot(gs[2], sharex=ax_e_top)
        ax_v = fig.add_subplot(gs[3], sharex=ax_e_top)
    else:
        fig = plt.figure(figsize=(10.4, 9.6), constrained_layout=False)
        gs = GridSpec(3, 1, height_ratios=[2.35, 2.35, 2.35], hspace=0.12, figure=fig)

        ax_e_top = None
        ax_e_bot = fig.add_subplot(gs[0])
        ax_f = fig.add_subplot(gs[1], sharex=ax_e_bot)
        ax_v = fig.add_subplot(gs[2], sharex=ax_e_bot)

    # ---------------- energy ----------------
    if use_break:
        for ax in [ax_e_top, ax_e_bot]:
            ax.plot(
                steps, energies,
                color=c_energy, lw=2.2,
                marker="o", ms=3.5, alpha=0.95
            )

        ax_e_top.set_ylim(*upper_ylim)
        ax_e_bot.set_ylim(*lower_ylim)

        ax_e_top.spines["bottom"].set_visible(False)
        ax_e_bot.spines["top"].set_visible(False)
        ax_e_top.tick_params(labelbottom=False, bottom=False)

        # break marks
        d = 0.008
        kwargs = dict(transform=ax_e_top.transAxes, color="k", clip_on=False, lw=1.15)
        ax_e_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_e_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

        kwargs.update(transform=ax_e_bot.transAxes)
        ax_e_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_e_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        ax_e_top.set_title("VASP Optimization Progress", pad=10, weight="semibold")
        ax_e_bot.set_ylabel("TOTEN (eV)")
    else:
        ax_e_bot.plot(
            steps, energies,
            color=c_energy, lw=2.2,
            marker="o", ms=3.5, alpha=0.95
        )
        ymin, ymax = finite_limits(energies)
        ax_e_bot.set_ylim(ymin, ymax)
        ax_e_bot.set_title("VASP Optimization Progress", pad=10, weight="semibold")
        ax_e_bot.set_ylabel("TOTEN (eV)")

    # ---------------- force ----------------
    valid_f = np.isfinite(max_forces)
    ax_f.plot(
        steps[valid_f], max_forces[valid_f],
        color=c_force, lw=2.15,
        marker="o", ms=3.5, alpha=0.95
    )
    ax_f.set_ylabel("Max force (eV/Å)")

    if force_thr is not None:
        ax_f.axhline(force_thr, ls="--", lw=1.45, color=c_ref, alpha=0.95)
        ax_f.text(
            0.985, 0.92,
            f"Force criterion = {force_thr:.4f} eV/Å",
            transform=ax_f.transAxes,
            ha="right", va="top",
            fontsize=12, color="0.2"
        )

    # ---------------- volume ----------------
    valid_v = np.isfinite(volumes)
    ax_v.plot(
        steps[valid_v], volumes[valid_v],
        color=c_vol, lw=2.15,
        marker="o", ms=3.5, alpha=0.95
    )
    ax_v.set_ylabel(r"Volume ($\mathrm{\AA^3}$)")
    ax_v.set_xlabel("Ionic step")

    # ---------------- shared cosmetics ----------------
    all_axes = [ax_e_bot, ax_f, ax_v] if not use_break else [ax_e_top, ax_e_bot, ax_f, ax_v]

    for ax in all_axes:
        ax.set_xlim(xmin, xmax)
        ax.grid(True, which="major", axis="both")
        ax.spines["top"].set_alpha(0.95)
        ax.spines["right"].set_alpha(0.95)

    # Only show bottom x tick labels
    for ax in [ax_e_bot, ax_f]:
        plt.setp(ax.get_xticklabels(), visible=False)

    if use_break:
        plt.setp(ax_e_top.get_xticklabels(), visible=False)

    # Optional convergence-step marker
    if conv_step is not None:
        for ax in all_axes:
            ax.axvline(conv_step, ls="-.", lw=1.2, color="0.45", alpha=0.75, zorder=0)
        ax_f.text(
            0.985, 0.82,
            f"First converged step: {conv_step}",
            transform=ax_f.transAxes,
            ha="right", va="top",
            fontsize=12, color="0.28"
        )

    # Progress text
    if nsw is not None and nsw > 0:
        progress = 100.0 * nsteps / nsw
        progress_text = f"Progress: {nsteps}/{nsw} ionic steps ({progress:.1f}%)"
    else:
        progress_text = f"Progress: {nsteps} ionic steps"

    anchor_ax = ax_e_top if use_break else ax_e_bot
    anchor_ax.text(
        0.018, 0.92,
        progress_text,
        transform=anchor_ax.transAxes,
        ha="left", va="top",
        fontsize=13, color="0.15", weight="medium"
    )

    # Footer
    info_parts = []
    for key in ["IBRION", "ISIF", "EDIFF", "EDIFFG", "NSW"]:
        if key in incar:
            info_parts.append(f"{key}={incar[key]}")
    footer = " | ".join(info_parts)

    fig.text(
        0.5, 0.018,
        footer,
        ha="center", va="center",
        fontsize=12.2, color="0.20"
    )

    fig.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.08)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[Done] Saved figure to: {output_path}")
    print(f"[Info] Ionic steps used: {nsteps}")
    print(f"[Info] Energy source    : {oszicar_path}")
    print(f"[Info] Force/Volume src: {outcar_path}")


# =========================================================
# 6. Run
# =========================================================
if __name__ == "__main__":
    plot_vasp_optimization(
        incar_path="INCAR",
        oszicar_path="OSZICAR",
        outcar_path="OUTCAR",
        output_path="opt_progress_pretty.png",
    )