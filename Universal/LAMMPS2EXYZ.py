#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np


def parse_type_map(s):
    """
    Parse "1:Al,2:N,3:Sc" → {1:"Al", 2:"N", 3:"Sc"}
    """
    mp = {}
    for kv in s.split(","):
        k, v = kv.split(":")
        mp[int(k)] = v
    return mp


def read_lammps_dump(path):
    """
    Read LAMMPS dump with format:

    ITEM: TIMESTEP
    <int>
    ITEM: NUMBER OF ATOMS
    <natoms>
    ITEM: BOX BOUNDS xy xz yz
    <xlo xhi xy>
    <ylo yhi xz>
    <zlo zhi yz>
    ITEM: ATOMS id type x y z
    ...
    """
    frames = []

    with open(path, "r") as f:
        lines = [l.strip() for l in f]

    i = 0
    n = len(lines)
    while i < n:
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue

        i += 1
        timestep = int(lines[i]); i += 1

        if not lines[i].startswith("ITEM: NUMBER OF ATOMS"):
            raise RuntimeError("Missing NUMBER OF ATOMS")
        i += 1
        nat = int(lines[i]); i += 1

        if not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise RuntimeError("Missing BOX BOUNDS")
        i += 1

        # triclinic: each line = lo hi tilt
        xlo, xhi, xy = map(float, lines[i].split()); i += 1
        ylo, yhi, xz = map(float, lines[i].split()); i += 1
        zlo, zhi, yz = map(float, lines[i].split()); i += 1

        # Build Lattice matrix (OVITO/VASP convention)
        ax = xhi - xlo
        by = yhi - ylo
        cz = zhi - zlo

        a = np.array([ax, 0, 0], float)
        b = np.array([xy, by, 0], float)
        c = np.array([xz, yz, cz], float)

        lattice = np.vstack([a, b, c])  # 3×3

        # ATOMS:
        if not lines[i].startswith("ITEM: ATOMS"):
            raise RuntimeError("Missing ATOMS line")
        header = lines[i].split()[2:]
        i += 1

        colmap = {h: idx for idx, h in enumerate(header)}
        required = ["id", "type", "x", "y", "z"]
        for r in required:
            if r not in colmap:
                raise RuntimeError(f"Missing column {r} in dump")

        rows = lines[i:i+nat]
        i += nat

        ids = []
        types = []
        pos = []

        for r in rows:
            p = r.split()
            ids.append(int(p[colmap["id"]]))
            types.append(int(p[colmap["type"]]))
            pos.append([float(p[colmap["x"]]),
                        float(p[colmap["y"]]),
                        float(p[colmap["z"]])])

        ids = np.array(ids)
        types = np.array(types)
        pos = np.array(pos)

        # sort by id
        idx = np.argsort(ids)
        frames.append(
            (timestep, lattice, types[idx], pos[idx])
        )

    return frames


def write_extxyz(frames, type_map, outfile):
    with open(outfile, "w") as f:
        for (ts, lat, types, pos) in frames:
            nat = len(types)
            # Lattice="ax ay az bx by bz cx cy cz"
            lat_flat = " ".join(f"{x:.8f}" for x in lat.flatten())

            f.write(f"{nat}\n")
            f.write(
                f'Time={ts} pbc="T T T" Lattice="{lat_flat}" '
                f'Properties=species:S:1:pos:R:3\n'
            )

            for t, (x, y, z) in zip(types, pos):
                elem = type_map.get(t, f"T{t}")
                f.write(f"{elem} {x:.8f} {y:.8f} {z:.8f}\n")


def main():
    ap = argparse.ArgumentParser(description="Convert LAMMPS dump → extxyz")
    ap.add_argument("dump", help="LAMMPS dump file")
    ap.add_argument("--out", default="output.xyz", help="extxyz output file")
    ap.add_argument("--type-map", required=True,
                    help='e.g. "1:Al,2:N,3:Sc"')

    args = ap.parse_args()

    type_map = parse_type_map(args.type_map)
    frames = read_lammps_dump(args.dump)

    print(f"[INFO] Read {len(frames)} frames")
    write_extxyz(frames, type_map, args.out)
    print(f"[OK] Written extxyz → {args.out}")


if __name__ == "__main__":
    main()
