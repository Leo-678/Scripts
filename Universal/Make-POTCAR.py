#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Concatenate element POTCAR files into one combined POTCAR.

Usage example:
    python Make-POTCAR.py Ag H.25 /path/to/PBE
This treats the last argument as the POTCAR base directory, and all
preceding positional arguments as the elements to include in order.
"""

import argparse
import os
import sys
from typing import List


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge element POTCAR files from a VASP pseudopotential library.\n"
            "Provide element names first and the POTCAR library directory last."
        ),
        epilog=(
            "示例：\n"
            "  python Make-POTCAR.py Ag H.25 /path/to/PBE\n"
            "  python Make-POTCAR.py O Si C /data/potcar/PBE"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "items",
        nargs="+",
        help="元素名称列表，最后一个参数为 POTCAR 库所在目录。",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="POTCAR",
        help="输出文件名（默认：POTCAR）",
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


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    if len(args.items) < 2:
        raise SystemExit("需要至少一个元素名和一个 POTCAR 目录，例如：Ag H.25 /path/to/PBE")

    elements = args.items[:-1]
    base_dir = ensure_dir_exists(args.items[-1])

    merge_potcars(base_dir, elements, args.output)


if __name__ == "__main__":
    main(sys.argv[1:])
