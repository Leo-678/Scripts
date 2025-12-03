#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Concatenate element POTCAR files into one combined POTCAR.

支持两种使用方式：

1) 显式给出元素顺序（保持原有用法不变）：
    python Make-POTCAR.py Ag H.25 /path/to/PBE
   最后一个参数为 POTCAR 库目录，前面的参数为元素名，按给定顺序合并。

2) 只给 POTCAR 库目录，元素顺序自动从当前目录的 POSCAR 读取（VASP5 格式）：
    python Make-POTCAR.py /path/to/PBE
   此时脚本会从当前目录的 POSCAR 第 6 行读取元素符号并按该顺序生成 POTCAR。
"""

import argparse
import os
import sys
from typing import List


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge element POTCAR files from a VASP pseudopotential library.\n"
            "Provide element names first and the POTCAR library directory last.\n"
            "If only one argument is given, it is treated as the POTCAR library "
            "directory and the element order is read from the current POSCAR."
        ),
        epilog=(
            "示例：\n"
            "  # 旧用法：显式给出元素顺序\n"
            "  python Make-POTCAR.py Ag H.25 /path/to/PBE\n"
            "  python Make-POTCAR.py O Si C /data/potcar/PBE\n\n"
            "  # 新用法：从当前 POSCAR 自动读取元素顺序\n"
            "  python Make-POTCAR.py /path/to/PBE"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "items",
        nargs="+",
        help=(
            "旧用法：元素名称列表，最后一个参数为 POTCAR 库所在目录；\n"
            "新用法：若仅提供一个参数，则该参数作为 POTCAR 库目录，元素顺序自动从当前 POSCAR 读取。"
        ),
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
    if not elements:
        raise ValueError("元素列表为空，无法合并 POTCAR。")

    output_path = os.path.abspath(output)
    with open(output_path, "wb") as out_f:
        for idx, elem in enumerate(elements, 1):
            potcar_path = locate_potcar(base_dir, elem)
            with open(potcar_path, "rb") as f:
                data = f.read()
            out_f.write(data)
            # 元素之间保证换行分隔
            if idx != len(elements) and not data.endswith(b"\n"):
                out_f.write(b"\n")
            print(f"[Leo] ✅ 已添加 {elem}: {potcar_path}")
    print(f"[Leo] 🎉 合并完成，输出文件: {output_path}")


def parse_poscar_elements(poscar_path: str = "POSCAR") -> List[str]:
    poscar_abs = os.path.abspath(poscar_path)
    if not os.path.isfile(poscar_abs):
        raise FileNotFoundError(f"POSCAR 文件不存在：{poscar_abs}")

    with open(poscar_abs, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if len(lines) < 6:
        raise ValueError("POSCAR 格式不完整，至少需要 6 行（包含元素行）。")

    element_line = lines[5].strip()
    tokens = element_line.split()
    if not tokens:
        raise ValueError("POSCAR 第 6 行为空，无法识别元素顺序。")

    # 判断第 6 行是不是纯数字（即 VASP4 格式的原子数行）
    def _is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    if all(_is_number(tok) for tok in tokens):
        raise ValueError(
            "检测到 POSCAR 第 6 行是原子数（VASP4 格式），\n"
            "当前脚本需要 VASP5 格式（第 6 行为元素符号行）。"
        )

    print(f"[Leo] 🧾 从 POSCAR 读取元素顺序: {' '.join(tokens)}")
    return tokens


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    # 只给了一个参数：新用法 → 该参数是 POTCAR 库目录，元素顺序来自当前 POSCAR
    if len(args.items) == 1:
        base_dir = ensure_dir_exists(args.items[0])
        elements = parse_poscar_elements("POSCAR")
    else:
        # 两个及以上参数：旧用法 → 最后一个是目录，前面的是元素名
        elements = args.items[:-1]
        base_dir = ensure_dir_exists(args.items[-1])

    merge_potcars(base_dir, elements, args.output)


if __name__ == "__main__":
    main(sys.argv[1:])
