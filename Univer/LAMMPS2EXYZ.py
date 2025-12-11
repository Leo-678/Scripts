#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import Dict, List

from ase.io import read, write
from ase.io.formats import UnknownFileTypeError


def parse_type_map(s: str) -> Dict[int, str]:
    """
    Parse "1:Al,2:N,3:Sc" → {1: "Al", 2: "N", 3: "Sc"}
    """
    mp: Dict[int, str] = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        try:
            k, v = kv.split(":")
        except ValueError:
            raise ValueError(
                f"Bad type-map entry: {kv!r}, expected like '1:Al,2:N'"
            )
        mp[int(k)] = v
    return mp


def convert_dump_to_extxyz(
    dump_file: str,
    out_file: str,
    type_map: Dict[int, str],
) -> None:
    """
    使用 ASE 读取 LAMMPS dump，并写出 extxyz。

    这里不依赖 frame.arrays['type']，而是用
    frame.get_atomic_numbers() 作为“类型编号”，再用
    用户提供的 type_map 进行映射。
    """
    # 读所有帧；你的 dump-0.xyz 有 5000 帧
    try:
        frames = read(dump_file, format="lammps-dump-text", index=":")
    except UnknownFileTypeError:
        # 某些 ASE 版本格式名是 lammps-dump
        frames = read(dump_file, format="lammps-dump", index=":")

    if not isinstance(frames, (list, tuple)):
        frames = [frames]

    print(f"[INFO] Read {len(frames)} frame(s) from {dump_file}")

    # 检查所有帧里的“类型编号”是否在 type_map 范围内
    all_unknown: List[int] = []
    for i, at in enumerate(frames):
        nums = at.get_atomic_numbers()  # 整数数组
        for z in set(nums):
            if z not in type_map and z not in all_unknown:
                all_unknown.append(z)

    if all_unknown:
        raise RuntimeError(
            "在 dump 中发现以下类型编号/atomic numbers，"
            f"但未在 --type-map 中提供映射: {sorted(all_unknown)}\n"
            f"当前 type-map: {type_map}"
        )

    # 逐帧把 atomic_numbers → 指定元素符号
    new_frames = []
    for i, at in enumerate(frames):
        nums = at.get_atomic_numbers()
        symbols = [type_map[int(z)] for z in nums]
        at.set_chemical_symbols(symbols)
        new_frames.append(at)

    # 写出 extxyz
    write(out_file, new_frames, format="extxyz")
    print(f"[OK] Written {len(new_frames)} frame(s) to {out_file}")


def main():
    ap = argparse.ArgumentParser(
        description="Convert LAMMPS dump → extxyz (using ASE, with type-map)"
    )
    ap.add_argument("dump", help="LAMMPS dump file")
    ap.add_argument(
        "--out", default="output.xyz",
        help="extxyz output file (default: output.xyz)"
    )
    ap.add_argument(
        "--type-map",
        required=True,
        help='e.g. "1:Al,2:N,3:Sc"  (编号 → 元素符号)',
    )

    args = ap.parse_args()

    type_map = parse_type_map(args.type_map)
    convert_dump_to_extxyz(args.dump, args.out, type_map)


if __name__ == "__main__":
    main()
