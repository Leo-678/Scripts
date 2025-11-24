#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_to_poscar.py

Description:
    Reads a custom trajectory file containing repeated frames of atomic configurations,
    and writes each frame to a VASP5-style POSCAR file using Cartesian coordinates.

Enhancement:
    `preferred_order` is now a command-line argument (--order Cu P S In).

Usage:
    python convert_to_poscar.py traj.xyz
    python convert_to_poscar.py traj.xyz --order Cu In P S
"""

import sys
import re
import numpy as np
import argparse
from collections import Counter


def parse_cell(line):
    """Extract the 3×3 lattice vectors from a line containing Lattice="..."."""
    match = re.search(r'Lattice="([^"]+)"', line)
    if not match:
        raise RuntimeError(f"Cannot find Lattice in line: {line}")
    values = list(map(float, match.group(1).split()))
    if len(values) != 9:
        raise RuntimeError(f"Expected 9 floats in Lattice, got {len(values)}: {values}")
    return [values[0:3], values[3:6], values[6:9]]


def write_poscar_cartesian(species, coords, cell, filename, preferred_order):
    """Write a POSCAR file in Cartesian mode, grouping atoms by element."""

    count = Counter(species)

    # --- Build final element sequence ---
    # Preferred first (but only those that appear)
    elements = [e for e in preferred_order if count[e] > 0]

    # Add remaining elements alphabetically
    for elem in sorted(count):
        if elem not in elements:
            elements.append(elem)

    numbers = [count[e] for e in elements]

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{filename}\n1.0\n")

        # Write lattice vectors
        for vec in cell:
            f.write(f"  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}\n")

        # Elements and counts
        f.write(" ".join(elements) + "\n")
        f.write(" ".join(map(str, numbers)) + "\n")
        f.write("Cartesian\n")

        # Group coordinates by element
        for elem in elements:
            for sp, pos in zip(species, coords):
                if sp == elem:
                    f.write(f"  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}\n")


def convert(input_file, preferred_order):
    """Main routine to generate POSCAR files per frame."""
    lines = open(input_file, 'r').read().splitlines()
    i = 0
    frame_index = 1

    while i < len(lines):
        line = lines[i].strip()

        if not line or not line.split()[0].isdigit():
            i += 1
            continue

        natoms = int(line.split()[0])
        i += 1
        if i >= len(lines):
            break

        # Parse lattice
        cell = parse_cell(lines[i])
        i += 1

        species = []
        coords  = []

        for _ in range(natoms):
            toks = lines[i].split()
            species.append(toks[0])
            coords.append(tuple(map(float, toks[1:4])))
            i += 1

        # Write POSCAR
        out_name = f"POSCAR-{frame_index}"
        write_poscar_cartesian(species, coords, cell, out_name, preferred_order)

        print(f"→ Frame {frame_index}: wrote {out_name} with {natoms} atoms")
        frame_index += 1


def main():
    parser = argparse.ArgumentParser(
        description="Convert multi-frame trajectory file to POSCAR frames."
    )
    parser.add_argument("input_file", help="Trajectory file path")
    parser.add_argument(
        "--order",
        nargs="+",
        default=["Cu", "P", "S", "In"],
        help="Preferred element ordering, e.g. --order Cu In P S",
    )

    args = parser.parse_args()
    convert(args.input_file, args.order)


if __name__ == "__main__":
    main()
