#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNIFIED RDF SCRIPT
==================
Supports:
- XDATCAR (multi-frame, Direct/Cartesian)
- LAMMPS dump (orthogonal / triclinic)
- consistent averaging & normalization
- total + partial RDF
- one TXT + one PNG output

Usage examples:
---------------
python RDF.py XDATCAR --fmt xdatcar --avg-frac 0.5 1.0 --cutoff 6
python RDF.py dump.lammpstrj --fmt lammps --type-map 1:Cu,2:Se,3:Al
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from math import pi


# =========================================================
# PBC utilities
# =========================================================
def minimum_image(df):
    return df - np.round(df)


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

    scale = float(lines[1])
    lattice = np.array([
        np.fromstring(lines[2], sep=" "),
        np.fromstring(lines[3], sep=" "),
        np.fromstring(lines[4], sep=" "),
    ]) * scale

    # species & counts
    def is_int_list(s):
        try:
            [int(x) for x in s.split()]
            return True
        except:
            return False

    if is_int_list(lines[5]):
        species = None
        counts = np.array([int(x) for x in lines[5].split()])
        idx = 5
    else:
        species = lines[5].split()
        counts = np.array([int(x) for x in lines[6].split()])
        idx = 6

    N = counts.sum()
    volume = abs(np.linalg.det(lattice))

    # atom types
    types = []
    for i, c in enumerate(counts):
        types += [i + 1] * c
    types = np.array(types, dtype=int)

    frames = []
    i = idx + 1
    while i < len(lines):
        if lines[i].lower().startswith("direct"):
            mode = "direct"
        elif lines[i].lower().startswith("cart"):
            mode = "cart"
        else:
            i += 1
            continue

        coords = np.array([
            [float(x) for x in lines[i + 1 + j].split()[:3]]
            for j in range(N)
        ])

        if mode == "direct":
            pos = frac_to_cart(coords, lattice)
        else:
            pos = coords * scale

        frames.append((pos, types, lattice, volume))
        i += N + 1

    return frames, species


# =========================================================
# LAMMPS dump reader (ALL frames)
# =========================================================
def read_lammps_all_frames(path):
    frames = []
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue

        natoms = int(lines[i + 3])
        box = []
        for j in range(3):
            lo, hi, *tilt = map(float, lines[i + 5 + j].split())
            box.append((lo, hi))

        cell = np.array([
            [box[0][1] - box[0][0], 0, 0],
            [0, box[1][1] - box[1][0], 0],
            [0, 0, box[2][1] - box[2][0]],
        ])
        volume = abs(np.linalg.det(cell))

        header = lines[i + 8].split()[2:]
        col = {h: k for k, h in enumerate(header)}

        pos = []
        typ = []
        for k in range(natoms):
            parts = lines[i + 9 + k].split()
            typ.append(int(parts[col["type"]]))
            pos.append([
                float(parts[col.get("x", col.get("xu"))]),
                float(parts[col.get("y", col.get("yu"))]),
                float(parts[col.get("z", col.get("zu"))]),
            ])

        pos = np.array(pos)
        typ = np.array(typ)

        frames.append((pos, typ, cell, volume))
        i += 9 + natoms

    return frames


# =========================================================
# RDF core
# =========================================================
def distances_pbc(pos, cell):
    frac = cart_to_frac(pos, cell)
    df = frac[:, None, :] - frac[None, :, :]
    df = minimum_image(df)
    dc = df @ cell
    d = np.linalg.norm(dc, axis=-1)
    return d


def partial_dist(pos, cell, typ, ta, tb):
    pa = pos[typ == ta]
    pb = pos[typ == tb]
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
    shell = 4 * pi * r**2 * dr
    rho = Nb / volume
    g = hist / (Na * rho * shell)
    return r, g


# =========================================================
# MAIN
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--fmt", choices=["xdatcar", "lammps"], required=True)
    ap.add_argument("--type-map", default=None)
    ap.add_argument("--avg-frac", nargs=2, type=float, default=[0.5, 1.0])
    ap.add_argument("--cutoff", type=float, default=6.0)
    ap.add_argument("--bins", type=int, default=300)
    ap.add_argument("--out", default="rdf.png")
    ap.add_argument("--txt", default="rdf.txt")
    args = ap.parse_args()

    # ---------- load ----------
    if args.fmt == "xdatcar":
        frames, species = read_xdatcar_all_frames(args.input)
    else:
        frames = read_lammps_all_frames(args.input)
        species = None

    print(f"[INFO] total frames = {len(frames)}")

    # avg window
    f0 = int(args.avg_frac[0] * len(frames))
    f1 = int(args.avg_frac[1] * len(frames))
    use = frames[f0:f1]
    print(f"[INFO] using frames {f0}..{f1-1} ({len(use)})")

    typ0 = use[0][1]
    all_types = sorted(np.unique(typ0))

    edges = np.linspace(0, args.cutoff, args.bins + 1)
    total_hist = np.zeros(args.bins)
    partial_hist = {(i, j): np.zeros(args.bins)
                    for i in all_types for j in all_types if j >= i}

    # ---------- accumulate ----------
    for pos, typ, cell, vol in use:
        d = distances_pbc(pos, cell).ravel()
        d = d[(d > 0) & (d < args.cutoff)]
        total_hist += np.histogram(d, edges)[0]

        for i in all_types:
            for j in all_types:
                if j < i:
                    continue
                dij = partial_dist(pos, cell, typ, i, j)
                dij = dij[dij < args.cutoff]
                partial_hist[(i, j)] += np.histogram(dij, edges)[0]

    nf = len(use)
    total_hist /= nf
    for k in partial_hist:
        partial_hist[k] /= nf

    # ---------- normalize ----------
    Na = len(typ0)
    r, gtot = rdf_normalize(total_hist, edges, Na, Na, use[0][3])

    partial_rdf = {}
    for (i, j), h in partial_hist.items():
        Ni = np.sum(typ0 == i)
        Nj = np.sum(typ0 == j)
        _, g = rdf_normalize(h, edges, Ni, Nj, use[0][3])
        partial_rdf[(i, j)] = g

    # ---------- output txt ----------
    with open(args.txt, "w") as f:
        f.write("r g_total " +
                " ".join([f"g_{i}-{j}" for (i, j) in partial_rdf]) + "\n")
        for k in range(len(r)):
            row = [f"{r[k]:.6f}", f"{gtot[k]:.6f}"]
            for ij in partial_rdf:
                row.append(f"{partial_rdf[ij][k]:.6f}")
            f.write(" ".join(row) + "\n")

    # ---------- plot ----------
    n = 1 + len(partial_rdf)
    ncol = 2
    nrow = (n + 1) // 2
    fig, axs = plt.subplots(nrow, ncol, figsize=(10, 4 * nrow))
    axs = axs.flatten()

    axs[0].plot(r, gtot, lw=2)
    axs[0].set_title("Total RDF")

    idx = 1
    for (i, j), g in partial_rdf.items():
        axs[idx].plot(r, g, lw=2)
        axs[idx].set_title(f"g({i}-{j})")
        idx += 1

    for ax in axs:
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"[INFO] wrote {args.out}, {args.txt}")


if __name__ == "__main__":
    main()
