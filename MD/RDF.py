#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNIFIED RDF SCRIPT
==================
Supports:
- XDATCAR (multi-frame, Direct/Cartesian)
- LAMMPS dump (orthogonal / triclinic; x/xu or xs/ys/zs)
- consistent averaging & normalization
- total + partial RDF
- one TXT + one PNG output

Examples:
---------
python RDF.py XDATCAR --format xdatcar --frac 0.5 1.0 --cutoff 6
python RDF.py dump.lammpstrj --format lammps --type 1:Cu,2:Se,3:Al --frac 0.9 1.0
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from math import pi


# =========================================================
# PBC utilities
# =========================================================
def minimum_image(df_frac):
    return df_frac - np.round(df_frac)


def frac_to_cart(frac, lattice):
    return frac @ lattice


def cart_to_frac(cart, lattice):
    return cart @ np.linalg.inv(lattice)


# =========================================================
# XDATCAR reader (ALL frames)
# =========================================================
def read_xdatcar_all_frames(path):
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    if len(lines) < 8:
        raise ValueError("XDATCAR too short or malformed.")

    scale = float(lines[1])
    lattice = np.array([
        np.fromstring(lines[2], sep=" "),
        np.fromstring(lines[3], sep=" "),
        np.fromstring(lines[4], sep=" "),
    ], dtype=float) * scale

    def is_int_list(s):
        try:
            [int(x) for x in s.split()]
            return True
        except Exception:
            return False

    if is_int_list(lines[5]):
        species = None
        counts = np.array([int(x) for x in lines[5].split()], dtype=int)
        idx = 5
    else:
        species = lines[5].split()
        counts = np.array([int(x) for x in lines[6].split()], dtype=int)
        idx = 6

    N = int(counts.sum())
    volume = abs(np.linalg.det(lattice))

    types = []
    for i, c in enumerate(counts):
        types += [i + 1] * int(c)
    types = np.array(types, dtype=int)

    frames = []
    i = idx + 1
    while i < len(lines):
        low = lines[i].lower()
        if low.startswith("direct"):
            mode = "direct"
        elif low.startswith("cart"):
            mode = "cart"
        else:
            i += 1
            continue

        if i + N >= len(lines):
            break

        coords = np.array(
            [[float(x) for x in lines[i + 1 + j].split()[:3]] for j in range(N)],
            dtype=float
        )

        if mode == "direct":
            pos = frac_to_cart(coords, lattice)
        else:
            pos = coords * scale

        frames.append((pos, types, lattice, volume))
        i += N + 1

    if not frames:
        raise ValueError("No frames parsed from XDATCAR. Check file format.")

    return frames, species


# =========================================================
# LAMMPS dump reader (ALL frames)
# =========================================================
def _parse_lammps_cell(bounds_line, bound_rows):
    triclinic = ("xy" in bounds_line) or ("xz" in bounds_line) or ("yz" in bounds_line)

    bounds = []
    tilts = [0.0, 0.0, 0.0]  # xy, xz, yz
    for j in range(3):
        parts = list(map(float, bound_rows[j].split()))
        if triclinic:
            if len(parts) < 3:
                raise ValueError("Triclinic BOX BOUNDS row must have 3 numbers (lo hi tilt).")
            lo, hi, tilt = parts[:3]
            bounds.append((lo, hi))
            tilts[j] = tilt
        else:
            if len(parts) < 2:
                raise ValueError("Orthogonal BOX BOUNDS row must have 2 numbers (lo hi).")
            lo, hi = parts[:2]
            bounds.append((lo, hi))

    xlo, xhi = bounds[0]
    ylo, yhi = bounds[1]
    zlo, zhi = bounds[2]

    if triclinic:
        xy, xz, yz = tilts
        lx = xhi - xlo
        ly = yhi - ylo
        lz = zhi - zlo
        cell = np.array([
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz],
        ], dtype=float)
    else:
        cell = np.array([
            [xhi - xlo, 0.0, 0.0],
            [0.0, yhi - ylo, 0.0],
            [0.0, 0.0, zhi - zlo],
        ], dtype=float)

    volume = abs(np.linalg.det(cell))
    return cell, volume


def read_lammps_all_frames(path):
    frames = []
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        if i + 9 >= len(lines):
            break

        natoms = int(lines[i + 3].strip())

        bounds_line = lines[i + 4].strip()
        bound_rows = [lines[i + 5 + j].strip() for j in range(3)]
        cell, volume = _parse_lammps_cell(bounds_line, bound_rows)

        atoms_header = lines[i + 8].split()
        if len(atoms_header) < 3 or atoms_header[0] != "ITEM:" or atoms_header[1] != "ATOMS":
            raise ValueError("Unexpected LAMMPS dump format near ITEM: ATOMS.")
        header = atoms_header[2:]
        col = {h: k for k, h in enumerate(header)}

        if "type" not in col:
            raise KeyError(f"No 'type' column in ITEM: ATOMS. Have: {list(col.keys())}")

        has_xyz = all(k in col for k in ("x", "y", "z"))
        has_xu = all(k in col for k in ("xu", "yu", "zu"))
        has_xs = all(k in col for k in ("xs", "ys", "zs"))
        if not (has_xyz or has_xu or has_xs):
            raise KeyError(
                f"Cannot find coordinates (x/y/z, xu/yu/zu, or xs/ys/zs). Have: {list(col.keys())}"
            )

        pos = np.empty((natoms, 3), dtype=float)
        typ = np.empty((natoms,), dtype=int)

        for k in range(natoms):
            parts = lines[i + 9 + k].split()
            typ[k] = int(parts[col["type"]])

            if has_xyz:
                pos[k, 0] = float(parts[col["x"]])
                pos[k, 1] = float(parts[col["y"]])
                pos[k, 2] = float(parts[col["z"]])
            elif has_xu:
                pos[k, 0] = float(parts[col["xu"]])
                pos[k, 1] = float(parts[col["yu"]])
                pos[k, 2] = float(parts[col["zu"]])
            else:
                frac = np.array([
                    float(parts[col["xs"]]),
                    float(parts[col["ys"]]),
                    float(parts[col["zs"]]),
                ], dtype=float)
                pos[k] = frac_to_cart(frac, cell)

        frames.append((pos, typ, cell, volume))
        i += 9 + natoms

    if not frames:
        raise ValueError("No frames parsed from LAMMPS dump. Check file format.")

    return frames


# =========================================================
# RDF core
# =========================================================
def distances_pbc(pos, cell):
    frac = cart_to_frac(pos, cell)
    df = frac[:, None, :] - frac[None, :, :]
    df = minimum_image(df)
    dc = df @ cell
    return np.linalg.norm(dc, axis=-1)


def partial_dist(pos, cell, typ, ta, tb):
    pa = pos[typ == ta]
    pb = pos[typ == tb]
    if pa.size == 0 or pb.size == 0:
        return np.empty((0,), dtype=float)

    fa = cart_to_frac(pa, cell)
    fb = cart_to_frac(pb, cell)
    df = fa[:, None, :] - fb[None, :, :]
    df = minimum_image(df)
    dc = df @ cell
    d = np.linalg.norm(dc, axis=-1).ravel()
    return d[d > 0]


def rdf_normalize(hist, edges, Na, Nb, volume):
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges)
    shell = 4.0 * pi * r**2 * dr
    rho = Nb / volume
    shell = np.where(shell == 0.0, np.nan, shell)
    g = hist / (Na * rho * shell)
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return r, g


def parse_type_map(s):
    """
    Parse: '1:Cu,2:Se,3:Al' -> {1:'Cu',2:'Se',3:'Al'}
    Return None if s is None.
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    m = {}
    for chunk in s.split(","):
        k, v = chunk.split(":")
        m[int(k.strip())] = v.strip()
    return m


# =========================================================
# MAIN
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--format", choices=["xdatcar", "lammps"], required=True)
    # 这里按你的要求：用 --type 直接承载 type-map
    ap.add_argument("--type", default=None,
                    help="Optional type mapping like '1:Cu,2:Se,3:Al' (for labels).")
    ap.add_argument("--frac", nargs=2, type=float, default=[0.9, 1.0])
    ap.add_argument("--cutoff", type=float, default=6.0)
    ap.add_argument("--bins", type=int, default=300)
    ap.add_argument("--out", default="rdf.png")
    ap.add_argument("--txt", default="rdf.txt")
    args = ap.parse_args()

    # labels: prefer XDATCAR species if exists, then override with --type mapping if provided
    type_labels = None

    # ---------- load ----------
    if args.format == "xdatcar":
        frames, species = read_xdatcar_all_frames(args.input)
        if species is not None:
            type_labels = {i + 1: s for i, s in enumerate(species)}
    else:
        frames = read_lammps_all_frames(args.input)

    # override / provide labels from --type map (your request)
    user_map = parse_type_map(args.type)
    if user_map is not None:
        type_labels = user_map if type_labels is None else {**type_labels, **user_map}

    nframes = len(frames)
    print(f"[INFO] total frames = {nframes}")

    # avg window
    f0 = int(args.frac[0] * nframes)
    f1 = int(args.frac[1] * nframes)
    f0 = max(0, min(f0, nframes))
    f1 = max(0, min(f1, nframes))
    if f1 <= f0:
        raise ValueError(f"Empty averaging window: frames={nframes}, frac={args.frac}, f0={f0}, f1={f1}")
    use = frames[f0:f1]
    print(f"[INFO] using frames {f0}..{f1 - 1} ({len(use)})")

    typ0 = use[0][1]
    all_types = sorted(np.unique(typ0).tolist())

    edges = np.linspace(0.0, args.cutoff, args.bins + 1)
    total_hist = np.zeros(args.bins, dtype=float)
    partial_hist = {(i, j): np.zeros(args.bins, dtype=float)
                    for i in all_types for j in all_types if j >= i}

    # accumulate
    for pos, typ, cell, vol in use:
        d = distances_pbc(pos, cell).ravel()
        d = d[(d > 0) & (d < args.cutoff)]
        total_hist += np.histogram(d, edges)[0]

        for i in all_types:
            for j in all_types:
                if j < i:
                    continue
                dij = partial_dist(pos, cell, typ, i, j)
                if dij.size == 0:
                    continue
                dij = dij[dij < args.cutoff]
                partial_hist[(i, j)] += np.histogram(dij, edges)[0]

    nf = len(use)
    total_hist /= nf
    for k in partial_hist:
        partial_hist[k] /= nf

    # normalize
    volume = use[0][3]
    Na_all = len(typ0)
    r, gtot = rdf_normalize(total_hist, edges, Na_all, Na_all, volume)

    partial_rdf = {}
    for (i, j), h in partial_hist.items():
        Ni = int(np.sum(typ0 == i))
        Nj = int(np.sum(typ0 == j))
        _, g = rdf_normalize(h, edges, Ni, Nj, volume)
        partial_rdf[(i, j)] = g

    def _lab(t):
        return type_labels.get(t, str(t)) if type_labels else str(t)

    # output txt
    with open(args.txt, "w") as f:
        f.write("r g_total " +
                " ".join([f"g_{_lab(i)}-{_lab(j)}" for (i, j) in partial_rdf]) + "\n")
        for k in range(len(r)):
            row = [f"{r[k]:.6f}", f"{gtot[k]:.6f}"]
            for ij in partial_rdf:
                row.append(f"{partial_rdf[ij][k]:.6f}")
            f.write(" ".join(row) + "\n")

    # plot
    n = 1 + len(partial_rdf)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(10, 4 * nrow))
    axs = np.array(axs).reshape(-1)

    axs[0].plot(r, gtot, lw=2)
    axs[0].set_title("Total RDF")

    idx = 1
    for (i, j), g in partial_rdf.items():
        axs[idx].plot(r, g, lw=2)
        axs[idx].set_title(f"g({_lab(i)}-{_lab(j)})")
        idx += 1

    for k in range(idx, len(axs)):
        axs[k].set_visible(False)

    for ax in axs[:idx]:
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"[INFO] wrote {args.out}, {args.txt}")


if __name__ == "__main__":
    main()
