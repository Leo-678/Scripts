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
        kv = kv.strip()
        if not kv:
            continue
        try:
            k, v = kv.split(":")
        except ValueError:
            raise ValueError(f"Bad type-map entry: {kv!r}, expected like '1:Cu'")
        mp[int(k)] = v
    return mp


def _parse_box_line(line):
    """
    支持两种 BOX BOUNDS 行格式：
      lo hi
      lo hi tilt
    返回 (lo, hi, tilt)
    """
    parts = line.split()
    if len(parts) == 2:
        lo, hi = map(float, parts)
        tilt = 0.0
    elif len(parts) == 3:
        lo, hi, tilt = map(float, parts)
    else:
        raise RuntimeError(
            f"BOX BOUNDS 行无法解析：{line!r} (需要 2 或 3 个数)"
        )
    return lo, hi, tilt


def _choose_coord_columns(header):
    """
    从 ATOMS 头部中选择坐标列名，支持：
      - x y z
      - xs ys zs
      - xu yu zu
    返回 (cx, cy, cz)
    """
    candidates = [
        ("x", "y", "z"),
        ("xs", "ys", "zs"),
        ("xu", "yu", "zu"),
    ]
    for cx, cy, cz in candidates:
        if cx in header and cy in header and cz in header:
            return cx, cy, cz
    raise RuntimeError(
        f"找不到坐标列，头部为：{header}\n"
        f"需要包含以下三元组之一：x/y/z, xs/ys/zs 或 xu/yu/zu"
    )


def read_lammps_dump(path):
    """
    Read LAMMPS dump with format:

    ITEM: TIMESTEP
    <int>
    ITEM: NUMBER OF ATOMS
    <natoms>
    ITEM: BOX BOUNDS [xy xz yz] ...
    <xlo xhi [xy]>
    <ylo yhi [xz]>
    <zlo zhi [yz]>
    ITEM: ATOMS id type x y z ...
    ...
    """
    frames = []

    with open(path, "r") as f:
        lines = [l.strip() for l in f]

    i = 0
    n = len(lines)
    while i < n:
        # 寻找 "ITEM: TIMESTEP"
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue

        i += 1
        if i >= n:
            break
        timestep = int(lines[i])
        i += 1

        # NUMBER OF ATOMS
        if i >= n or not lines[i].startswith("ITEM: NUMBER OF ATOMS"):
            raise RuntimeError("Missing 'ITEM: NUMBER OF ATOMS'")
        i += 1
        nat = int(lines[i])
        i += 1

        # BOX BOUNDS
        if i >= n or not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise RuntimeError("Missing 'ITEM: BOX BOUNDS'")
        box_header = lines[i]
        i += 1

        # 兼容：lo hi / lo hi tilt
        xlo, xhi, xy = _parse_box_line(lines[i]); i += 1
        ylo, yhi, xz = _parse_box_line(lines[i]); i += 1
        zlo, zhi, yz = _parse_box_line(lines[i]); i += 1

        # 构造晶格矩阵 (与 OVITO/VASP 一致)
        ax = xhi - xlo
        by = yhi - ylo
        cz = zhi - zlo

        a = np.array([ax, 0.0, 0.0], float)
        b = np.array([xy, by, 0.0], float)
        c = np.array([xz, yz, cz], float)

        lattice = np.vstack([a, b, c])  # 3×3

        # ATOMS
        if i >= n or not lines[i].startswith("ITEM: ATOMS"):
            raise RuntimeError("Missing 'ITEM: ATOMS' line")

        header_tokens = lines[i].split()[2:]
        i += 1

        colmap = {h: idx for idx, h in enumerate(header_tokens)}

        # id / type 必须有
        for r in ["id", "type"]:
            if r not in colmap:
                raise RuntimeError(f"Dump 缺少列: {r!r}，当前列: {header_tokens}")

        # 选择坐标列
        cx, cy, cz_name = _choose_coord_columns(header_tokens)

        # 读取原子行
        rows = lines[i:i + nat]
        if len(rows) < nat:
            raise RuntimeError(
                f"期待 {nat} 行 ATOMS 数据，但文件只剩 {len(rows)} 行"
            )
        i += nat

        ids = []
        types = []
        pos = []

        for r in rows:
            p = r.split()
            ids.append(int(p[colmap["id"]]))
            types.append(int(p[colmap["type"]]))
            pos.append([
                float(p[colmap[cx]]),
                float(p[colmap[cy]]),
                float(p[colmap[cz_name]]),
            ])

        ids = np.array(ids)
        types = np.array(types)
        pos = np.array(pos)

        # 按 id 排序
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
    ap.add_argument(
        "--type-map",
        required=True,
        help='e.g. "1:Cu,2:Se,3:Ag"',
    )

    args = ap.parse_args()

    type_map = parse_type_map(args.type_map)
    frames = read_lammps_dump(args.dump)

    print(f"[INFO] Read {len(frames)} frames from {args.dump}")
    write_extxyz(frames, type_map, args.out)
    print(f"[OK] Written extxyz → {args.out}")


if __name__ == "__main__":
    main()
