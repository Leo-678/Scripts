#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast NVT RDF script for VASP XDATCAR and LAMMPS dump files.

Supports:
    - VASP XDATCAR
    - LAMMPS dump (orthogonal and triclinic cells)

Fast path:
    - Orthogonal cells use scipy.spatial.cKDTree
    - Triclinic cells fall back to brute-force calculation

Examples:
    python RDF.py XDATCAR --frac 0.9 1.0 --cut 10
    python RDF.py produc.traj --type 1:Cu,2:Se --frac 0.9 1.0 --cut 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from math import pi
from typing import Tuple, Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Constants
DEFAULT_CUTOFF = 10.0
DEFAULT_BINS = 300
DEFAULT_FRAC = (0.9, 1.0)
ORTHOGONAL_TOL = 1e-10
RDF_FACTOR = 2.0 * pi  # Factor for RDF normalization


# =========================================================
# Basic utilities
# =========================================================

def detect_format(path: str) -> str:
    """Detect file format (VASP XDATCAR or LAMMPS dump)."""
    name = Path(path).name.upper()
    return "vasp" if "XDATCAR" in name else "lmp"


def minimum_image(df_frac: np.ndarray) -> np.ndarray:
    """Apply minimum image convention to fractional coordinates."""
    return df_frac - np.round(df_frac)


def frac_to_cart(frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert fractional to Cartesian coordinates."""
    return frac @ lattice


def cart_to_frac(cart: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert Cartesian to fractional coordinates."""
    return cart @ np.linalg.inv(lattice)


def is_orthogonal(cell: np.ndarray, tol: float = ORTHOGONAL_TOL) -> bool:
    """Check if cell is orthogonal."""
    offdiag = cell.copy()
    np.fill_diagonal(offdiag, 0.0)
    return np.max(np.abs(offdiag)) < tol


def parse_type_map(s: Optional[str]) -> Dict[int, str]:
    """Parse type labels from string format 'idx:label,idx:label'."""
    if s is None or not s.strip():
        return {}
    type_map = {}
    for item in s.split(","):
        idx, label = item.split(":")
        type_map[int(idx.strip())] = label.strip()
    return type_map


def label_type(type_id: int, type_labels: Dict[int, str]) -> str:
    """Get label for atom type, fallback to string representation."""
    return type_labels.get(type_id, str(type_id))


def check_cutoff(cell: np.ndarray, cutoff: float) -> None:
    """Warn if cutoff exceeds half the minimum box length."""
    lengths = np.linalg.norm(cell, axis=1)
    half_min = 0.5 * np.min(lengths)
    if cutoff > half_min:
        print(f"[WARN] cutoff = {cutoff:.3f} Å > half minimum box length = {half_min:.3f} Å")
        print("[WARN] RDF may be unreliable. Reduce --cut.")


# =========================================================
# XDATCAR reader
# =========================================================

def _parse_int_list(line: str) -> Tuple[bool, List[int]]:
    """Try to parse line as integers. Returns (success, values)."""
    try:
        values = [int(x) for x in line.split()]
        return True, values
    except ValueError:
        return False, []


def _parse_species_and_counts(
    lines: List[str], idx: int
) -> Tuple[Optional[List[str]], np.ndarray, int]:
    """Parse species names and atom counts from XDATCAR header."""
    success, counts = _parse_int_list(lines[idx])
    if success:
        return None, np.array(counts, dtype=int), idx
    # Next line must be counts
    success, counts = _parse_int_list(lines[idx + 1])
    if not success:
        raise ValueError("Cannot parse atom counts from XDATCAR")
    species = lines[idx].split()
    return species, np.array(counts, dtype=int), idx + 1


def read_xdatcar_all_frames(path: str) -> Tuple[List[Tuple], Optional[List[str]]]:
    """Read all frames from VASP XDATCAR file.
    
    Returns:
        (frames, species_names) where frames is list of tuples:
        (positions, types, lattice, volume, origin)
    """
    with open(path) as f:
        lines = [line.strip() for line in f.readlines()]

    if len(lines) < 8:
        raise ValueError("XDATCAR too short or malformed.")

    scale = float(lines[1])
    lattice = np.array(
        [np.fromstring(lines[i], sep=" ") for i in range(2, 5)],
        dtype=float
    ) * scale

    species, counts, idx = _parse_species_and_counts(lines, 5)
    natoms = int(counts.sum())
    volume = abs(np.linalg.det(lattice))

    # Build type array using np.repeat for efficiency
    types = np.repeat(np.arange(1, len(counts) + 1), counts, axis=0).astype(int)

    frames = []
    i = idx + 1

    while i < len(lines):
        low = lines[i].lower()
        mode = "direct" if low.startswith("direct") else "cart" if low.startswith("cart") else None

        if mode is None:
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

def parse_lammps_cell(
    bounds_line: str, bound_rows: List[str]
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Parse LAMMPS dump cell information.
    
    Returns:
        (cell_matrix, volume, origin)
    """
    triclinic = any(s in bounds_line for s in ("xy", "xz", "yz"))
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
    xy, xz, yz = tilts

    lx = xhi - xlo
    ly = yhi - ylo
    lz = zhi - zlo

    if triclinic:
        cell = np.array([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]], dtype=float)
    else:
        cell = np.diag([lx, ly, lz]).astype(float)

    volume = abs(np.linalg.det(cell))
    origin = np.array([xlo, ylo, zlo], dtype=float)
    return cell, volume, origin


def _parse_lammps_atoms(
    lines: List[str], idx: int, natoms: int, col: Dict[str, int],
    cell: np.ndarray, origin: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse atomic positions and types from LAMMPS dump."""
    pos = np.zeros((natoms, 3), dtype=float)
    typ = np.zeros(natoms, dtype=int)

    has_x = all(k in col for k in ("x", "y", "z"))
    has_xu = all(k in col for k in ("xu", "yu", "zu"))
    has_xs = all(k in col for k in ("xs", "ys", "zs"))

    if not (has_x or has_xu or has_xs):
        raise KeyError("Need x/y/z, xu/yu/zu, or xs/ys/zs coordinates.")

    coord_keys = ("x", "y", "z") if has_x else ("xu", "yu", "zu") if has_xu else ("xs", "ys", "zs")

    for k in range(natoms):
        parts = lines[idx + k].split()
        typ[k] = int(parts[col["type"]])

        if has_x or has_xu:
            pos[k] = [float(parts[col[key]]) for key in coord_keys]
        else:  # has_xs
            frac = np.array([float(parts[col[key]]) for key in coord_keys], dtype=float)
            pos[k] = origin + frac_to_cart(frac, cell)

    return pos, typ


def read_lammps_all_frames(path: str) -> List[Tuple]:
    """Read all frames from LAMMPS dump file.
    
    Returns:
        List of tuples: (positions, types, lattice, volume, origin)
    """
    with open(path) as f:
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

        pos, typ = _parse_lammps_atoms(lines, i + 9, natoms, col, cell, origin)
        frames.append((pos, typ, cell, volume, origin))
        i += 9 + natoms

    if not frames:
        raise ValueError("No frames parsed from LAMMPS dump.")

    return frames


# =========================================================
# Fast RDF per frame
# =========================================================

def wrap_to_cell(pos: np.ndarray, cell: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Wrap atomic positions into unit cell."""
    frac = cart_to_frac(pos - origin, cell)
    frac = frac - np.floor(frac)
    return frac_to_cart(frac, cell)


def frame_pairs_kdtree(
    pos: np.ndarray,
    typ: np.ndarray,
    cell: np.ndarray,
    origin: np.ndarray,
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate pair distances and types using KDTree (orthogonal cells only).
    
    Returns:
        (distances, pair_types) where pair_types is sorted atom type pairs
    """
    lengths = np.diag(cell).astype(float)
    wrapped = wrap_to_cell(pos, cell, origin)

    tree = cKDTree(wrapped, boxsize=lengths)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")

    if pairs.size == 0:
        return np.array([]), np.empty((0, 2), dtype=int)

    delta = wrapped[pairs[:, 0]] - wrapped[pairs[:, 1]]
    delta -= lengths * np.round(delta / lengths)
    dist = np.linalg.norm(delta, axis=1)
    pair_types = np.sort(
        np.column_stack([typ[pairs[:, 0]], typ[pairs[:, 1]]]), axis=1
    )

    return dist, pair_types


def frame_pairs_bruteforce(
    pos: np.ndarray,
    typ: np.ndarray,
    cell: np.ndarray,
    origin: np.ndarray,
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate pair distances and types using brute force (works for all cells).
    
    Returns:
        (distances, pair_types) where pair_types is sorted atom type pairs
    """
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


def normalize_total(
    hist_unique: np.ndarray,
    edges: np.ndarray,
    natoms: int,
    volume: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize total RDF histogram.
    
    Returns:
        (radii, g_values)
    """
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges)
    shell = RDF_FACTOR * pi * r**2 * dr
    rho = natoms / volume
    g = 2.0 * hist_unique / (natoms * rho * shell)
    return r, np.nan_to_num(g)


def normalize_partial(
    hist_unique: np.ndarray,
    edges: np.ndarray,
    na: int,
    nb: int,
    volume: float,
    same_type: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize partial RDF histogram.
    
    Returns:
        (radii, g_values)
    """
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges)
    shell = RDF_FACTOR * pi * r**2 * dr
    rho = nb / volume
    g = (2.0 if same_type else 1.0) * hist_unique / (na * rho * shell)
    return r, np.nan_to_num(g)


# =========================================================
# Main
# =========================================================

def _create_plots(
    r: np.ndarray,
    g_total: np.ndarray,
    partial_rdf: Dict[Tuple[int, int], np.ndarray],
    type_labels: Dict[int, str],
    output_path: str,
) -> None:
    """Create and save RDF plots."""
    nplot = 1 + len(partial_rdf)
    ncol = 2
    nrow = (nplot + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 4 * nrow), dpi=200)
    axes = np.array(axes).reshape(-1)

    # Total RDF
    axes[0].plot(r, g_total, lw=2)
    axes[0].set_title("Total RDF")

    # Partial RDFs
    for idx, ((i, j), g) in enumerate(partial_rdf.items(), start=1):
        li = label_type(i, type_labels)
        lj = label_type(j, type_labels)
        axes[idx].plot(r, g, lw=2)
        axes[idx].set_title(f"g({li}-{lj})")

    # Format all active plots
    for ax in axes[: 1 + len(partial_rdf)]:
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.grid(alpha=0.25)

    # Hide empty subplots
    for ax in axes[1 + len(partial_rdf) :]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def _write_rdf_output(
    output_txt: str,
    r: np.ndarray,
    g_total: np.ndarray,
    partial_rdf: Dict[Tuple[int, int], np.ndarray],
    type_labels: Dict[int, str],
) -> None:
    """Write RDF data to text file."""
    headers = ["r", "g_total"]
    headers += [
        f"g_{label_type(i, type_labels)}-{label_type(j, type_labels)}"
        for (i, j) in partial_rdf.keys()
    ]

    with open(output_txt, "w") as f:
        f.write(" ".join(headers) + "\n")
        for k in range(len(r)):
            row = [f"{r[k]:.6f}", f"{g_total[k]:.6f}"]
            row += [f"{partial_rdf[key][k]:.6f}" for key in partial_rdf.keys()]
            f.write(" ".join(row) + "\n")


def main() -> int:
    """Main RDF calculation workflow.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(description="Fast NVT RDF calculator.")
    parser.add_argument("input", help="XDATCAR or LAMMPS dump file.")
    parser.add_argument(
        "--type", default="", help="Type labels, e.g. '1:Cu,2:Se'."
    )
    parser.add_argument(
        "--frac",
        nargs=2,
        type=float,
        default=list(DEFAULT_FRAC),
        help=f"Frame fraction range. Default: {DEFAULT_FRAC[0]} {DEFAULT_FRAC[1]}",
    )
    parser.add_argument(
        "--cut",
        type=float,
        default=DEFAULT_CUTOFF,
        help=f"RDF cutoff in Angstrom. Default: {DEFAULT_CUTOFF}",
    )
    parser.add_argument(
        "--bin", type=int, default=DEFAULT_BINS, help=f"Number of bins. Default: {DEFAULT_BINS}"
    )
    parser.add_argument(
        "--out", default="rdf.png", help="Output PNG file. Default: rdf.png"
    )
    parser.add_argument(
        "--txt", default="rdf.txt", help="Output TXT file. Default: rdf.txt"
    )

    try:
        args = parser.parse_args()
        
        if not Path(args.input).exists():
            print(f"[ERROR] Input file not found: {args.input}")
            return 1
        
        if args.frac[0] < 0 or args.frac[1] > 1.0 or args.frac[0] >= args.frac[1]:
            print(f"[ERROR] Invalid --frac range: {args.frac}")
            return 1
            
        if args.cut <= 0:
            print(f"[ERROR] Invalid --cut: {args.cut}")
            return 1
            
        if args.bin <= 0:
            print(f"[ERROR] Invalid --bin: {args.bin}")
            return 1

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
        f1 = max(f0 + 1, min(f1, nframes))

        if f1 <= f0:
            print(f"[ERROR] Empty frame window: frac={args.frac}, nframes={nframes}")
            return 1

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
        print(f"[INFO] Atom types: {sorted(all_types)}")
        print(f"[INFO] Number of atoms: {natoms}")

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
            ni = int(np.sum(typ0 == i))
            nj = int(np.sum(typ0 == j))
            _, g = normalize_partial(hist, edges, ni, nj, volume0, same_type=(i == j))
            partial_rdf[key] = g

        _write_rdf_output(args.txt, r, g_total, partial_rdf, type_labels)
        _create_plots(r, g_total, partial_rdf, type_labels, args.out)

        print(f"[INFO] Wrote {args.out}")
        print(f"[INFO] Wrote {args.txt}")
        print("[INFO] Done!")
        return 0

    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        return 1
    except ValueError as e:
        print(f"[ERROR] Invalid input: {e}")
        return 1
    except KeyError as e:
        print(f"[ERROR] Missing required data: {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
