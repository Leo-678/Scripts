#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Leo：个人脚本统一入口

用法示例：
    leo universal substitute POSCAR POSCAR_new Al Sc 0.25
    leo universal substitute POSCAR POSCAR_new Al Sc 0.25 --seed 42
    leo universal substitute POSCAR POSCAR_new Al Sc 10 --mode count
"""

import argparse
import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===================== 1. 在这里配置所有脚本 =====================
# 结构：
# TOOLS = {
#   "组名": {
#       "命令名": {
#           "script": "相对 Scripts 的路径",
#           "help": "子命令在目录中的一句话说明",
#           "desc": "子命令的详细说明（-h 时显示）",
#           "examples": [ "示例命令1", "示例命令2", ... ],
#       },
#       ...
#   },
#   ...
# }

TOOLS = {
    "universal": {
        "substitute": {
            "script": os.path.join("Universal", "Substitute-POSCAR.py"),
            "help": "在 POSCAR 中随机替换元素",
            "desc": "封装 Universal/Substitute-POSCAR.py，用于在 POSCAR 中随机替换元素。",
            "examples": [
                "leo universal substitute POSCAR POSCAR_new Al Sc 0.25",
                "leo universal substitute POSCAR POSCAR_new Al Sc 0.25 --seed 42",
                "leo universal substitute POSCAR POSCAR_new Al Sc 10 --mode count",
            ],
        },
        "super": {
            "script": os.path.join("Universal", "VASP2SUPER-X.py"),
            "help": "VASP 结构生成超胞",
            "desc": "封装 Universal/VASP2SUPER-X.py，用于将 VASP 结构扩展为超胞。",
            "examples": [
                "leo universal super POSCAR POSCAR_2x2x2 --super 2 2 2",
            ],
        },
    },
########################################## NEP
    "NEP": {
        "plot": {
            "script": os.path.join("NEP", "NEP-plot.py"),
            "help": "绘制NEP的训练结果",
            "desc": "封装 NEP/NEP-plot.py，用于绘制NEP的训练结果。",
            "examples": [
                "leo nep-plot",
            ],
        },
}

# ===================== 2. 通用运行函数 =====================

def run_script(script_rel_path, extra_args):
    """调用某个子脚本，参数原样透传"""
    script_path = os.path.join(BASE_DIR, script_rel_path)
    cmd = [sys.executable, script_path] + extra_args
    # 把控制权交给子脚本（它自己处理 -h 等）
    subprocess.run(cmd, check=True)


# ===================== 3. 构建命令行解析 =====================

def build_parser():
    parser = argparse.ArgumentParser(
        prog="leo",
        description="Leo：个人 VASP/MD/后处理 脚本统一命令入口",
    )

    group_parsers = parser.add_subparsers(
        title="脚本分组",
        dest="group",
        metavar="group",
        help="先选择一个分组，再选择分组中的具体工具",
    )

    # 按照 TOOLS 自动创建分组和子命令
    for group_name, cmds in TOOLS.items():
        group_parser = group_parsers.add_parser(
            group_name,
            help=f"{group_name} 相关工具集",
        )
        subparsers = group_parser.add_subparsers(
            title=f"{group_name} 组可用命令",
            dest="command",
            metavar="command",
        )

        for cmd_name, info in cmds.items():
            desc = info.get("desc", info.get("help", ""))
            examples = info.get("examples", [])
            if examples:
                desc += "\n\n示例：\n  " + "\n  ".join(examples)

            p = subparsers.add_parser(
                cmd_name,
                help=info.get("help", ""),
                description=desc,
                add_help=True,
            )
            # 所有后续参数原样传给子脚本
            p.add_argument(
                "args",
                nargs=argparse.REMAINDER,
                help="后面所有参数会原样传递给对应脚本。",
            )

            # 绑定执行函数
            script_rel = info["script"]
            p.set_defaults(
                func=lambda ns, script_rel=script_rel: run_script(script_rel, ns.args)
            )

    return parser


def main():
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
