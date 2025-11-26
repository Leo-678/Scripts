#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL RDF SCRIPT
Supports:
- XDATCAR or LAMMPS dump (your supplied format)
- --type-map 1:Al,2:N,3:Sc
- --mode avg with --avg-frac 0.2 0.3
- smoothing: off/moving/gaussian
- outputs total + partial RDFs
- 2-column subplot layout
- output RDFs to ONE txt file (Format A)
python RDF.py wrapped-1.traj --fmt lammps --mode avg --cutoff 10 --bins 150 --type-map 1:Cu,2:Se,3:Ag --avg-frac 0.5 1.0
"""

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from math import exp, sqrt, pi


# =========================================================
# Utility
# =========================================================
def minimum_image(dr):
    return dr - np.round(dr)


def frac_to_cart(frac, lattice):
    return frac @ lattice


def cart_to_frac(cart, lattice):
    return cart @ np.linalg.inv(lattice)


# =========================================================
# Read XDATCAR (only last frame)
# =========================================================
def read_xdatcar(xfile):
    with open(xfile, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    scale = float(lines[1])
    a = np.fromstring(lines[2], sep=" ")
    b = np.fromstring(lines[3], sep=" ")
    c = np.fromstring(lines[4], sep=" ")
    lattice = np.vstack([a, b, c]) * scale

    # species?
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

    # find last "Direct"/"Cartesian"
    last = None
    for i, l in enumerate(lines):
        if l.lower().startswith("direct") or l.lower().startswith("cart"):
            last = i
    mode = "direct" if "direct" in lines[last].lower() else "cart"
    N = counts.sum()

    coords = np.array([
        [float(x) for x in lines[last + 1 + j].split()[:3]]
        for j in range(N)
    ])

    types = []
    for i, c in enumerate(counts):
        types += [i + 1] * c
    types = np.array(types, dtype=int)

    if mode == "direct":
        pos = frac_to_cart(coords, lattice)
    else:
        pos = coords * scale

    volume = abs(np.linalg.det(lattice))
    return [pos], [types], [lattice], volume, species


# =========================================================
# Read LAMMPS dump (your exact format)
# =========================================================
def read_lammps_all_frames(path):
    """读取 LAMMPS dump，自动识别 2 列(lo/hi) 或 3 列(含 xy xz yz) box 格式"""
    frames = []
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    N = len(lines)
    while i < N:
        # TIMESTEP
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        step = int(lines[i+1].strip())

        # NUMBER OF ATOMS
        if not lines[i+2].startswith("ITEM: NUMBER OF ATOMS"):
            raise RuntimeError("Missing 'ITEM: NUMBER OF ATOMS'")
        natoms = int(lines[i+3].strip())

        # BOX BOUNDS
        if not lines[i+4].startswith("ITEM: BOX BOUNDS"):
            raise RuntimeError("Missing 'ITEM: BOX BOUNDS'")

        box = []
        for j in range(3):
            parts = lines[i+5+j].split()
            if len(parts) == 2:
                # lo hi
                lo, hi = map(float, parts)
                tilt = 0.0
            elif len(parts) == 3:
                # lo hi tilt
                lo, hi, tilt = map(float, parts)
            else:
                raise RuntimeError(f"BOX line format error: {parts}")
            box.append((lo, hi, tilt))

        # 构建正交盒（RDF 不关心倾斜，直接用长方体即可）
        ax = np.array([box[0][1] - box[0][0], 0.0, 0.0])
        by = np.array([0.0, box[1][1] - box[1][0], 0.0])
        cz = np.array([0.0, 0.0, box[2][1] - box[2][0]])
        cell = np.vstack([ax, by, cz])
        vol = float(abs(np.linalg.det(cell)))

        # ATOMS
        if not lines[i+8].startswith("ITEM: ATOMS"):
            raise RuntimeError("Missing 'ITEM: ATOMS'")
        header = lines[i+8].split()[2:]
        col = {h: k for k, h in enumerate(header)}
        if "type" not in col:
            raise RuntimeError("Missing type column")

        X = []
        T = []
        for k in range(natoms):
            parts = lines[i+9+k].split()
            T.append(int(parts[col["type"]]))
            X.append([float(parts[col[c]]) for c in ("x", "y", "z")])
        X = np.array(X, float)
        T = np.array(T, int)

        # 平移到 box 原点
        X[:, 0] -= box[0][0]
        X[:, 1] -= box[1][0]
        X[:, 2] -= box[2][0]

        # ✅ 这里一次性返回 4 个量：pos, typ, cell, vol
        frames.append((X, T, cell, vol))

        i += 9 + natoms

    return frames

# =========================================================
# pairwise distances
# =========================================================
def distances_pbc(pos, lattice):
    frac = cart_to_frac(pos, lattice)
    df = frac[:, None, :] - frac[None, :, :]
    df = minimum_image(df)
    dc = df @ lattice
    d = np.linalg.norm(dc, axis=-1)
    return d


def partial_dist(pos, lattice, types, ta, tb):
    fa = pos[types == ta]
    fb = pos[types == tb]
    if len(fa) == 0 or len(fb) == 0:
        return np.array([])
    frac_a = cart_to_frac(fa, lattice)
    frac_b = cart_to_frac(fb, lattice)
    df = frac_a[:, None, :] - frac_b[None, :, :]
    df = minimum_image(df)
    dc = df @ lattice
    d = np.linalg.norm(dc, axis=-1)
    d = d.reshape(-1)
    return d[d > 0]


# =========================================================
# RDF normalize
# =========================================================
def rdf_normalize(counts, edges, Na, Nb, volume):
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges)
    shell = 4 * pi * r**2 * dr
    rho = Nb / volume
    denom = Na * rho * shell
    g = counts / denom
    return r, g


# =========================================================
# smoothing
# =========================================================
def smooth(y, mode, w=5, sigma=1.0):
    if mode == "off":
        return y
    if mode == "moving":
        k = np.ones(w) / w
        return np.convolve(y, k, "same")
    if mode == "gaussian":
        hw = int(3 * sigma)
        xs = np.arange(-hw, hw + 1)
        k = np.exp(-0.5 * (xs / sigma)**2)
        k /= k.sum()
        return np.convolve(y, k, "same")
    return y


# =========================================================
# MAIN
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--fmt", choices=["xdatcar", "lammps"], required=True)
    ap.add_argument("--type-map", required=False,
                    help="e.g. 1:Al,2:N,3:Sc")
    ap.add_argument("--mode", choices=["avg"], default="avg")
    ap.add_argument("--avg-frac", nargs=2, type=float, default=[0.8, 1.0])
    ap.add_argument("--cutoff", type=float, default=6)
    ap.add_argument("--bins", type=int, default=300)
    ap.add_argument("--smooth", choices=["off", "moving", "gaussian"],
                    default="off")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--out", default="rdf.png")
    ap.add_argument("--txt", default="rdf.txt")
    args = ap.parse_args()

    # ---------------- LOAD ----------------
    if args.fmt == "xdatcar":
        posL, typL, cellL, vol, species = read_xdatcar(args.input)
        frames = list(zip(posL, typL, cellL, [vol]*len(posL)))
    else:
        frames = read_lammps_all_frames(args.input)
        species = None

    print(f"[INFO] total frames={len(frames)}")

    # type-map
    if args.type_map:
        tmap = {}
        for kv in args.type_map.split(","):
            k, v = kv.split(":")
            tmap[int(k)] = v
    else:
        # fallback numeric
        tmap = {}

    # avg-frac range
    f0 = int(args.avg_frac[0] * len(frames))
    f1 = int(args.avg_frac[1] * len(frames))
    if f1 <= f0:
        f1 = f0 + 1

    use = frames[f0:f1]
    print(f"[INFO] using frames {f0}..{f1-1} ({len(use)} frames)")

    # determine cutoff
    if args.cutoff is None:
        # use minimal box length / 2 of first frame
        _, _, cell0, _ = use[0]
        lens = [np.linalg.norm(cell0[i]) for i in range(3)]
        args.cutoff = min(lens) / 2.0
    print(f"[INFO] cutoff={args.cutoff}")

    edges = np.linspace(0, args.cutoff, args.bins + 1)

    # detect types
    all_types = np.unique(use[0][1])
    all_types = sorted(all_types)
    print(f"[INFO] atom types: {all_types}")

    # accumulate histograms
    total_hist = np.zeros(args.bins)
    partial_hist = {(ta, tb): np.zeros(args.bins)
                    for ta in all_types for tb in all_types if tb >= ta}

    # ---------------- accumulate over frames ----------------
    for pos, typ, cell, vol in use:
        # total RDF
        d_all = distances_pbc(pos, cell).ravel()
        d_all = d_all[(d_all > 0) & (d_all < args.cutoff)]
        h, _ = np.histogram(d_all, bins=edges)
        total_hist += h

        # partial RDF
        for ta in all_types:
            Na = np.sum(typ == ta)
            for tb in all_types:
                if tb < ta:
                    continue
                Nb = np.sum(typ == tb)
                d = partial_dist(pos, cell, typ, ta, tb)
                d = d[(d > 0) & (d < args.cutoff)]
                h, _ = np.histogram(d, bins=edges)
                partial_hist[(ta, tb)] += h

    # average
    nf = len(use)
    total_hist /= nf
    for k in partial_hist:
        partial_hist[k] /= nf

    # ---------------- normalize ----------------
    pos0, typ0, cell0, vol0 = use[0]
    Na_total = len(typ0)

    r, g_total = rdf_normalize(
        total_hist, edges, Na_total, Na_total, vol0
    )

    # partial
    partial_rdf = {}
    for (ta, tb), h in partial_hist.items():
        Na = np.sum(typ0 == ta)
        Nb = np.sum(typ0 == tb)
        r, g = rdf_normalize(h, edges, Na, Nb, vol0)
        partial_rdf[(ta, tb)] = g

    # smoothing
    g_total = smooth(g_total, args.smooth, args.window, args.sigma)
    for k in partial_rdf:
        partial_rdf[k] = smooth(
            partial_rdf[k], args.smooth, args.window, args.sigma
        )

    # =========================================================
    # TXT OUTPUT (Format A)
    # =========================================================
    with open(args.txt, "w") as f:
        header = ["r", "g_total"]
        for ta in all_types:
            for tb in all_types:
                if tb < ta:
                    continue
                nm = f"g_{tmap.get(ta,ta)}-{tmap.get(tb,tb)}"
                header.append(nm)
        f.write("# " + " ".join(header) + "\n")

        for i in range(len(r)):
            row = [f"{r[i]:.6f}", f"{g_total[i]:.6f}"]
            for ta in all_types:
                for tb in all_types:
                    if tb < ta:
                        continue
                    g = partial_rdf[(ta, tb)][i]
                    row.append(f"{g:.6f}")
            f.write(" ".join(row) + "\n")

    print(f"[INFO] wrote txt: {args.txt}")

    # =========================================================
    # Plot (2 columns layout)
    # =========================================================
    n_sub = 1 + len(partial_rdf)
    ncol = 2
    nrow = (n_sub + 1) // 2

    fig, axs = plt.subplots(nrow, ncol, figsize=(10, 4 * nrow))
    axs = axs.flatten()

    # total
    axs[0].plot(r, g_total, lw=2)
    axs[0].set_title("Total RDF")
    axs[0].set_xlabel("r (Å)")
    axs[0].set_ylabel("g(r)")

    # partial
    idx = 1
    for ta in all_types:
        for tb in all_types:
            if tb < ta:
                continue
            ax = axs[idx]
            ax.plot(r, partial_rdf[(ta, tb)], lw=2)
            ax.set_title(f"g({tmap.get(ta,ta)}-{tmap.get(tb,tb)})")
            ax.set_xlabel("r (Å)")
            ax.set_ylabel("g(r)")
            idx += 1

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"[INFO] wrote figure: {args.out}")


if __name__ == "__main__":
    main()
