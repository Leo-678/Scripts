#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast NVT RDF script
===================

Supports:
- VASP XDATCAR
- LAMMPS dump

Fast path:
- Orthogonal cells use scipy.spatial.cKDTree
- Triclinic cells fall back to brute-force calculation

Examples:
---------
python RDF.py XDATCAR --frac 0.9 1.0 --cut 10
python RDF.py produc.traj --type 1:Cu,2:Se --frac 0.9 1.0 --cut 10
"""

import argparse
from pathlib import Path
from math import pi

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# =========================================================
# Basic utilities
# =========================================================
def detect_format(path):
    name = Path(path).name.upper()
    return "vasp" if "XDATCAR" in name else "lmp"


def minimum_image(df_frac):
    return df_frac - np.round(df_frac)


def frac_to_cart(frac, lattice):
    return frac @ lattice


def cart_to_frac(cart, lattice):
    return cart @ np.linalg.inv(lattice)


def is_orthogonal(cell, tol=1e-10):
    offdiag = cell.copy()
    np.fill_diagonal(offdiag, 0.0)
    return np.max(np.abs(offdiag)) < tol


def parse_type_map(s):
    if s is None or not s.strip():
        return {}
    out = {}
    for item in s.split(","):
        k, v = item.split(":")
        out[int(k.strip())] = v.strip()
    return out


def label_type(t, type_labels):
    return type_labels.get(t, str(t))


def check_cutoff(cell, cutoff):
    lengths = [np.linalg.norm(cell[i]) for i in range(3)]
    half_min = 0.5 * min(lengths)
    if cutoff > half_min:
        print(f"[WARN] cutoff = {cutoff:.3f} Å > half minimum box length = {half_min:.3f} Å")
        print("[WARN] RDF may be unreliable. Reduce --cut.")


# =========================================================
# XDATCAR reader
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

    natoms = int(counts.sum())
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

        if i + natoms >= len(lines):
            break

        coords = np.array(
            [[float(x) for x in lines[i + 1 + j].split()[:3]] for j in range(natoms)],
            dtype=float
        )

        if mode == "direct":
            pos = frac_to_cart(coords, lattice)
        else:
            pos = coords * scale

        frames.append((pos, types, lattice, volume, np.zeros(3)))
        i += natoms + 1

    if not frames:
        raise ValueError("No frames parsed from XDATCAR.")

    return frames, species


# =========================================================
# LAMMPS dump reader
# =========================================================
def parse_lammps_cell(bounds_line, bound_rows):
    triclinic = ("xy" in bounds_line) or ("xz" in bounds_line) or ("yz" in bounds_line)

    bounds = []
    tilts = [0.0, 0.0, 0.0]

    for j in range(3):
        parts = list(map(float, bound_rows[j].split()))
        if triclinic:
            lo, hi, tilt = parts[:3]
            bounds.append((lo, hi))
            tilts[j] = tilt
        else:
            lo, hi = parts[:2]
            bounds.append((lo, hi))

    xlo, xhi = bounds[0]
    ylo, yhi = bounds[1]
    zlo, zhi = bounds[2]

    origin = np.array([xlo, ylo, zlo], dtype=float)

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
    return cell, volume, origin


def read_lammps_all_frames(path):
    with open(path, "r") as f:
        lines = f.readlines()

    frames = []
    i = 0

    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue

        natoms = int(lines[i + 3].strip())

        bounds_line = lines[i + 4].strip()
        bound_rows = [lines[i + 5 + j].strip() for j in range(3)]
        cell, volume, origin = parse_lammps_cell(bounds_line, bound_rows)

        atoms_header = lines[i + 8].split()
        if atoms_header[0] != "ITEM:" or atoms_header[1] != "ATOMS":
            raise ValueError("Unexpected LAMMPS dump format near ITEM: ATOMS.")

        header = atoms_header[2:]
        col = {h: k for k, h in enumerate(header)}

        if "type" not in col:
            raise KeyError("No 'type' column in LAMMPS dump.")

        has_x = all(k in col for k in ("x", "y", "z"))
        has_xu = all(k in col for k in ("xu", "yu", "zu"))
        has_xs = all(k in col for k in ("xs", "ys", "zs"))

        if not (has_x or has_xu or has_xs):
            raise KeyError("Need x/y/z, xu/yu/zu, or xs/ys/zs coordinates.")

        pos = np.zeros((natoms, 3), dtype=float)
        typ = np.zeros(natoms, dtype=int)

        for k in range(natoms):
            parts = lines[i + 9 + k].split()
            typ[k] = int(parts[col["type"]])

            if has_x:
                pos[k] = [
                    float(parts[col["x"]]),
                    float(parts[col["y"]]),
                    float(parts[col["z"]]),
                ]
            elif has_xu:
                pos[k] = [
                    float(parts[col["xu"]]),
                    float(parts[col["yu"]]),
                    float(parts[col["zu"]]),
                ]
            else:
                frac = np.array([
                    float(parts[col["xs"]]),
                    float(parts[col["ys"]]),
                    float(parts[col["zs"]]),
                ], dtype=float)
                pos[k] = origin + frac_to_cart(frac, cell)

        frames.append((pos, typ, cell, volume, origin))
        i += 9 + natoms

    if not frames:
        raise ValueError("No frames parsed from LAMMPS dump.")

    return frames


# =========================================================
# Fast RDF per frame
# =========================================================
def wrap_to_cell(pos, cell, origin):
    frac = cart_to_frac(pos - origin, cell)
    frac = frac - np.floor(frac)
    return frac_to_cart(frac, cell)


def frame_pairs_kdtree(pos, typ, cell, origin, cutoff):
    lengths = np.diag(cell).astype(float)
    wrapped = wrap_to_cell(pos, cell, origin)

    tree = cKDTree(wrapped, boxsize=lengths)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")

    if pairs.size == 0:
        return np.empty(0), np.empty((0, 2), dtype=int)

    delta = wrapped[pairs[:, 0]] - wrapped[pairs[:, 1]]
    delta -= lengths * np.round(delta / lengths)

    dist = np.linalg.norm(delta, axis=1)
    pair_types = np.sort(np.column_stack([typ[pairs[:, 0]], typ[pairs[:, 1]]]), axis=1)

    return dist, pair_types


def frame_pairs_bruteforce(pos, typ, cell, origin, cutoff):
    frac = cart_to_frac(pos - origin, cell)
    frac = frac - np.floor(frac)

    df = frac[:, None, :] - frac[None, :, :]
    df = minimum_image(df)
    dc = df @ cell
    dmat = np.linalg.norm(dc, axis=-1)

    iu = np.triu_indices(len(pos), k=1)
    dist = dmat[iu]

    mask = dist < cutoff
    dist = dist[mask]

    t1 = typ[iu[0]][mask]
    t2 = typ[iu[1]][mask]
    pair_types = np.sort(np.column_stack([t1, t2]), axis=1)

    return dist, pair_types


def normalize_total(hist_unique, edges, natoms, volume):
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges)
    shell = 4.0 * pi * r**2 * dr
    rho = natoms / volume

    # hist_unique contains each unordered pair only once.
    g = 2.0 * hist_unique / (natoms * rho * shell)
    return r, np.nan_to_num(g)


def normalize_partial(hist_unique, edges, Na, Nb, volume, same_type):
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges)
    shell = 4.0 * pi * r**2 * dr
    rho = Nb / volume

    if same_type:
        # Same-type histogram contains each unordered pair once.
        g = 2.0 * hist_unique / (Na * rho * shell)
    else:
        # Cross-type histogram contains Ni * Nj unordered pairs once.
        g = hist_unique / (Na * rho * shell)

    return r, np.nan_to_num(g)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Fast NVT RDF calculator.")

    parser.add_argument("input", help="XDATCAR or LAMMPS dump file.")
    parser.add_argument("--type", default="", help="Type labels, e.g. '1:Cu,2:Se'.")
    parser.add_argument("--frac", nargs=2, type=float, default=[0.9, 1.0],
                        help="Frame fraction range. Default: 0.9 1.0")
    parser.add_argument("--cut", type=float, default=10.0,
                        help="RDF cutoff in Angstrom. Default: 10.0")
    parser.add_argument("--bin", type=int, default=300,
                        help="Number of bins. Default: 300")
    parser.add_argument("--out", default="rdf.png",
                        help="Output PNG file. Default: rdf.png")
    parser.add_argument("--txt", default="rdf.txt",
                        help="Output TXT file. Default: rdf.txt")

    args = parser.parse_args()

    fmt = detect_format(args.input)
    type_labels = parse_type_map(args.type)

    if fmt == "vasp":
        frames, species = read_xdatcar_all_frames(args.input)
        if species is not None:
            for i, s in enumerate(species, start=1):
                type_labels.setdefault(i, s)
    else:
        frames = read_lammps_all_frames(args.input)

    nframes = len(frames)

    f0 = int(args.frac[0] * nframes)
    f1 = int(args.frac[1] * nframes)
    f0 = max(0, min(f0, nframes))
    f1 = max(0, min(f1, nframes))

    if f1 <= f0:
        raise ValueError(f"Empty frame window: frac={args.frac}, frames={nframes}")

    use = frames[f0:f1]

    pos0, typ0, cell0, volume0, origin0 = use[0]
    natoms = len(typ0)
    all_types = sorted(np.unique(typ0).tolist())

    check_cutoff(cell0, args.cut)

    use_kdtree = HAS_SCIPY and is_orthogonal(cell0)

    print(f"[INFO] Input format: {fmt}")
    print(f"[INFO] Total frames: {nframes}")
    print(f"[INFO] Using frames: {f0} to {f1 - 1} ({len(use)} frames)")
    print(f"[INFO] Method: {'cKDTree' if use_kdtree else 'brute force fallback'}")

    edges = np.linspace(0.0, args.cut, args.bin + 1)

    total_hist = np.zeros(args.bin, dtype=float)
    partial_hist = {
        (i, j): np.zeros(args.bin, dtype=float)
        for i in all_types
        for j in all_types
        if j >= i
    }

    nf = len(use)

    for iframe, (pos, typ, cell, volume, origin) in enumerate(use, start=1):
        if use_kdtree:
            dist, pair_types = frame_pairs_kdtree(pos, typ, cell, origin, args.cut)
        else:
            dist, pair_types = frame_pairs_bruteforce(pos, typ, cell, origin, args.cut)

        total_hist += np.histogram(dist, bins=edges)[0]

        for key in partial_hist:
            i, j = key
            mask = (pair_types[:, 0] == i) & (pair_types[:, 1] == j)
            if np.any(mask):
                partial_hist[key] += np.histogram(dist[mask], bins=edges)[0]

        if iframe % max(1, nf // 10) == 0 or iframe == nf:
            print(f"[INFO] Processed {iframe}/{nf} frames")

    total_hist /= nf
    for key in partial_hist:
        partial_hist[key] /= nf

    r, g_total = normalize_total(total_hist, edges, natoms, volume0)

    partial_rdf = {}
    for key, hist in partial_hist.items():
        i, j = key
        Ni = int(np.sum(typ0 == i))
        Nj = int(np.sum(typ0 == j))
        _, g = normalize_partial(
            hist,
            edges,
            Ni,
            Nj,
            volume0,
            same_type=(i == j)
        )
        partial_rdf[key] = g

    with open(args.txt, "w") as f:
        headers = ["r", "g_total"]
        headers += [
            f"g_{label_type(i, type_labels)}-{label_type(j, type_labels)}"
            for (i, j) in partial_rdf.keys()
        ]
        f.write(" ".join(headers) + "\n")

        for k in range(len(r)):
            row = [f"{r[k]:.6f}", f"{g_total[k]:.6f}"]
            row += [f"{partial_rdf[key][k]:.6f}" for key in partial_rdf.keys()]
            f.write(" ".join(row) + "\n")

    nplot = 1 + len(partial_rdf)
    ncol = 2
    nrow = (nplot + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 4 * nrow), dpi=200)
    axes = np.array(axes).reshape(-1)

    axes[0].plot(r, g_total, lw=2)
    axes[0].set_title("Total RDF")

    idx = 1
    for (i, j), g in partial_rdf.items():
        li = label_type(i, type_labels)
        lj = label_type(j, type_labels)

        axes[idx].plot(r, g, lw=2)
        axes[idx].set_title(f"g({li}-{lj})")
        idx += 1

    for ax in axes[:idx]:
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.grid(alpha=0.25)

    for ax in axes[idx:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close(fig)

    print(f"[INFO] Wrote {args.out}")
    print(f"[INFO] Wrote {args.txt}")


if __name__ == "__main__":
    main()