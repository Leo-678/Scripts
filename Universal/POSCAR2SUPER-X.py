#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
supercell_export.py

基于当前目录的结构文件生成超胞，并导出多种格式文件：
  - VASP (POSCAR)
  - LAMMPS 数据文件（含 Masses 段）
  - XYZ / extxyz
  - CIF

示例用法：
  # 最常见：从 POSCAR 读，扩成 2x2x2，写 lammps-data
  python supercell_export.py POSCAR -r 2 2 2 -f lammps-data

  # 多种输出格式
  python supercell_export.py POSCAR -r 2 2 1 -f lammps-data vasp xyz

  # 指定元素优先顺序（影响 LAMMPS Masses 顺序）
  python supercell_export.py POSCAR -r 3 3 1 -f lammps-data --specorder Al N

注意：
  * LAMMPS 写出时传入 masses=True 强制输出 Masses。
  * specorder 会自动补全所有元素：先按传入的顺序，再补结构中剩余元素。
"""

from ase.io import read, write
from ase.data import atomic_masses, atomic_numbers
from pathlib import Path
from collections import Counter
import argparse
import sys
import textwrap


# ===========================
# 工具函数
# ===========================
def unique_in_appearance(symbols):
    """按首次出现顺序返回去重后的元素序列"""
    seen, out = set(), []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_specorder(symbols, hint):
    """根据 hint 和结构实际包含的元素构造最终 specorder
       规则：先按 hint 中的出现顺序保留（若确实出现在结构中）
            再把剩余元素按结构首次出现顺序补到后面
    """
    uniq = unique_in_appearance(symbols)
    head = [s for s in hint if s in uniq]
    tail = [s for s in uniq if s not in head]
    return head + tail


def symbol_mass(sym):
    """返回元素的标准原子质量（amu）"""
    Z = atomic_numbers[sym]
    return float(atomic_masses[Z])


# ===========================
# 主逻辑
# ===========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="基于输入结构生成超胞，并导出多种格式文件。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            示例：
              python supercell_export.py POSCAR -r 2 2 2 -f lammps-data
              python supercell_export.py POSCAR -r 2 2 1 -f lammps-data vasp xyz
              python supercell_export.py POSCAR -r 3 3 1 -f lammps-data --specorder Al N
            """
        )
    )
    parser.add_argument(
        "input",
        help="输入结构文件（支持 POSCAR/CONTCAR/cif/xyz 等 ASE 能识别的格式）",
    )
    parser.add_argument(
        "-r", "--repeat",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        default=(1, 1, 1),
        help="超胞重复数 (nx ny nz)，默认 1 1 1",
    )
    parser.add_argument(
        "-f", "--formats",
        nargs="+",
        default=["lammps-data"],
        metavar="FMT",
        help="输出格式列表，默认: lammps-data；可用：vasp, lammps-data, xyz, cif, extxyz",
    )
    parser.add_argument(
        "--specorder",
        nargs="+",
        default=None,
        metavar="ELM",
        help="元素优先顺序，例如：--specorder Al N；未包含的元素会自动按出现顺序补到后面",
    )
    parser.add_argument(
        "-o", "--prefix",
        default="supercell",
        help="输出文件前缀，默认 supercell（POSCAR.supercell、supercell.lmp 等）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"[错误] 找不到输入文件：{input_path}")

    try:
        prim = read(input_path.as_posix())
    except Exception as e:
        sys.exit(f"[错误] 无法读取 {input_path}: {e}")

    repeat = tuple(args.repeat)
    supercell = prim.repeat(repeat)

    print("=== 超胞信息 ===")
    print(f"输入文件: {input_path}")
    print(f"重复数  : {repeat[0]} {repeat[1]} {repeat[2]}")
    print("晶格矩阵 (Å)：")
    print(supercell.get_cell())
    print(f"原子总数：{len(supercell)}")

    # 结构内元素
    symbols_all = supercell.get_chemical_symbols()
    species_in_struct = unique_in_appearance(symbols_all)

    # 最终 specorder
    if args.specorder is None:
        specorder_hint = []
    else:
        specorder_hint = args.specorder

    specorder = build_specorder(symbols_all, specorder_hint)

    print("\n=== 元素顺序 (specorder) ===")
    print("结构中元素（出现顺序）:", " ".join(species_in_struct))
    print("最终 specorder         :", " ".join(specorder))

    print("\n=== 元素质量 (amu) ===")
    for s in specorder:
        try:
            m = symbol_mass(s)
            print(f"{s:>3s} : {m:.6f}")
        except Exception:
            print(f"{s:>3s} : <未找到质量，请检查元素符号>")

    # ===========================
    # 写出多种格式
    # ===========================
    print("\n=== 写出文件 ===")
    for fmt in args.formats:
        fmt = fmt.lower()

        # 统一几个常见别名
        if fmt in ("vasp", "poscar"):
            fmt = "vasp"
            out_filename = f"{args.prefix}.POSCAR"
        elif fmt in ("lammps", "lammps-data", "data"):
            fmt = "lammps-data"
            out_filename = f"input.pos"
        elif fmt == "extxyz":
            out_filename = "model.xyz"    
        else:
            out_filename = f"{args.prefix}.{fmt}"

        try:
            if fmt == "lammps-data":
                write(
                    out_filename,
                    supercell,
                    format=fmt,
                    specorder=specorder,
                    masses=True,
                )
            else:
                write(out_filename, supercell, format=fmt)

            print(f"  [OK] {out_filename}  (format={fmt})")
        except Exception as e:
            print(f"  [失败] 写出 {fmt} → {out_filename} 失败：{e}")

    print("\n全部完成。若 LAMMPS data 中 Masses 与期望不一致，")
    print("请检查 --specorder 是否覆盖并匹配结构中的所有元素。")


if __name__ == "__main__":
    main()
