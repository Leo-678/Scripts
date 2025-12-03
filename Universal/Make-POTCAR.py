#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Concatenate element POTCAR files into one combined POTCAR.

Two usage patterns are supported:
1) Explicit element order (compatible with older usage):
    python Make-POTCAR.py Ag H.25 /path/to/PBE
   The last argument is the POTCAR base directory, and all preceding
   positional arguments are the elements to include in order.

2) Derive element order directly from POSCAR:
    python Make-POTCAR.py --poscar POSCAR /path/to/PBE
   The script reads the element symbols from the POSCAR element line
   (the line after the three lattice vectors) and merges POTCAR files
   in that sequence.
"""

import argparse
import os
import sys
from typing import List


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge element POTCAR files from a VASP pseudopotential library.\n"
            "Provide element names first and the POTCAR library directory last;"
            " or pass --poscar to read the element order from POSCAR.\n"
            "You can also point to the POTCAR directory explicitly with --library."
        ),
        epilog=(
            "示例：\n"
            "  python Make-POTCAR.py Ag H.25 /path/to/PBE\n"
            "  python Make-POTCAR.py O Si C /data/potcar/PBE\n"
            "  python Make-POTCAR.py --poscar POSCAR /path/to/PBE\n"
            "  python Make-POTCAR.py --poscar POSCAR --library /data/potcar/PBE"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "items",
        nargs="*",
        help=(
            "元素名称列表，最后一个参数为 POTCAR 库所在目录；"
            "若使用 --poscar，可仅提供 POTCAR 库目录。"
        ),
    )
    parser.add_argument(
        "-l",
        "--library",
        help="显式指定 POTCAR 库目录，可替代位置参数中的目录。",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="POTCAR",
        help="输出文件名（默认：POTCAR）",
    )
    parser.add_argument(
        "-p",
        "--poscar",
        help="从 POSCAR 中读取元素顺序（使用元素行）。",
    )
    return parser.parse_args(argv)


def ensure_dir_exists(path: str) -> str:
    abs_path = os.path.abspath(path)
    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"POTCAR 库目录不存在：{abs_path}")
    return abs_path


def locate_potcar(base_dir: str, element: str) -> str:
    path = os.path.join(base_dir, element, "POTCAR")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到 {element} 的 POTCAR：{path}")
    return path


def merge_potcars(base_dir: str, elements: List[str], output: str) -> None:
    output_path = os.path.abspath(output)
    with open(output_path, "wb") as out_f:
        for idx, elem in enumerate(elements, 1):
            potcar_path = locate_potcar(base_dir, elem)
            with open(potcar_path, "rb") as f:
                data = f.read()
            out_f.write(data)
            if idx != len(elements) and not data.endswith(b"\n"):
                out_f.write(b"\n")
            print(f"[Leo] ✅ 已添加 {elem}: {potcar_path}")
    print(f"[Leo] 🎉 合并完成，输出文件: {output_path}")


def parse_poscar_elements(poscar_path: str) -> List[str]:
    poscar_abs = os.path.abspath(poscar_path)
    if not os.path.isfile(poscar_abs):
        raise FileNotFoundError(f"POSCAR 文件不存在：{poscar_abs}")

    with open(poscar_abs, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if len(lines) < 6:
        raise ValueError("POSCAR 格式不完整，至少需要 6 行（包含元素行）。")

    element_line = lines[5].strip()
    # If the line is purely numeric (counts), there's no element information.
    tokens = element_line.split()
    if not tokens:
        raise ValueError("POSCAR 元素行为空，无法识别元素顺序。")

    if all(token.replace(".", "", 1).isdigit() for token in tokens):
        raise ValueError(
            "POSCAR 第 6 行为原子数而非元素符号，请使用包含元素行的 POSCAR (VASP5 格式)。"
        )

    print(f"[Leo] 🧾 从 POSCAR 读取元素顺序: {' '.join(tokens)}")
    return tokens


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    base_dir_arg = args.library
    if args.items:
        # If library is not specified, assume the last positional item is the library directory.
        # Otherwise, treat all positional items as elements.
        if args.library:
            elements_from_items = args.items
        else:
            elements_from_items = args.items[:-1]
            base_dir_arg = args.items[-1]
    else:
        elements_from_items = []

    if args.poscar:
        if args.items and args.library:
            raise SystemExit("使用 --poscar 且指定 --library 时，不需要再提供位置参数。")
        if base_dir_arg is None:
            raise SystemExit(
                "使用 --poscar 时需要提供 POTCAR 库目录，例如：python Make-POTCAR.py --poscar POSCAR /path/to/PBE"
            )

        elements = parse_poscar_elements(args.poscar)
    else:
        elements = elements_from_items
        if not elements:
            raise SystemExit("需要至少一个元素名和一个 POTCAR 目录，例如：Ag H.25 /path/to/PBE")

    if base_dir_arg is None:
        raise SystemExit("需要提供 POTCAR 库目录，例如：Ag H.25 /path/to/PBE 或使用 --library 指定。")

    base_dir = ensure_dir_exists(base_dir_arg)
    merge_potcars(base_dir, elements, args.output)


if __name__ == "__main__":
    main(sys.argv[1:])
