#!/usr/bin/env python
# -*- coding: utf-8 -*-
# update git pull origin main
"""
Leo：个人脚本统一入口
"""

import argparse
import os
import sys
import subprocess
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===================== 1. 在这里配置所有脚本 =====================
# 结构：
# TOOLS = {
#   "组名(group)": {
#       "命令名(command)": {
#           "script": "相对 Scripts 的路径",
#           "help": "子命令在目录中的一句话说明",
#           "desc": "子命令的详细说明（-h 时显示）",
#           "examples": [ "示例命令1", "示例命令2", ... ],
#           "copy_files": [  # 可选：运行前要复制到当前目录的文件列表（相对 Scripts）
#               os.path.join("Universal", "template.vasp"),
#               ...
#           ],
#       },
#       ...
#   },
#   ...
# }

TOOLS = {
    "universal": {
        "substitute": {
            "script": os.path.join("Universal", "Substitute-POSCAR.py"),
            "help": "Substitute atoms in POSCAR",
            "desc": "封装 Universal/Substitute-POSCAR.py，用于在 POSCAR 中随机替换元素。",
            "examples": [
                "leo universal substitute POSCAR POSCAR_new Al Sc 0.25",
                "leo universal substitute POSCAR POSCAR_new Al Sc 0.25 --seed 42",
                "leo universal substitute POSCAR POSCAR_new Al Sc 10 --mode count",
            ],
            # 示例（先注释着，等你真的有需要再填实际文件）
            # "copy_files": [
            #     os.path.join("Universal", "template.vasp"),
            #     os.path.join("Universal", "default_config.yaml"),
            # ],
        },
        "replicate": {
            "script": os.path.join("Universal", "POSCAR2SUPER-X.py"),
            "help": "Replicate POSCAR & convert format",
            "desc": "封装 Universal/POSCAR2SUPER-X.py，用于将 VASP 结构扩展为超胞。",
            "examples": [
                "leo universal replicate POSCAR -r 2 2 2 -f lammps-data",
            ],
            "copy_files": [
                os.path.join("NEP", f)
                for f in ["POSCAR2SUPER-X.py"]
            ],
        },
        "vacancy": {
            "script": os.path.join("Universal", "POS-Remove.py"),
            "help": "Reproduce vacancy in POSCAR",
            "desc": "封装 Universal/POS-Remove.py，用于删掉POSCAR原子",
            "examples": [
                "leo universal vacancy In 10 POSCAR",
            ],
            "copy_files": [
                os.path.join("NEP", f)
                for f in ["POS-Remove.py"]
            ],
        },
    },

    # NEP 相关工具
    "nep": {
        "plot": {
            "script": os.path.join("NEP", "NEP-plot.py"),
            "help": "Plot NEP training results",
            "desc": "封装 NEP/NEP-plot.py，用于绘制 NEP 的训练损失、误差等结果。",
            "examples": [
                "leo nep plot",
            ],
            # 需要的话可以这样加：
            # "copy_files": [
            #     os.path.join("NEP", "palette.json"),
            #     os.path.join("NEP", "style.yaml"),
            # ],
        },
        "single": {
            "script": os.path.join("NEP", "Xyz2poscar.py"),
            "help": "Single point related calculations",
            "desc": "封装 NEP/Xyz2poscar.py，用于绘制计算微扰生成的结构单点能。",
            "examples": [
                "leo nep single dump.xyz --order Cu In P S",
            ],
        "copy_files": [
            os.path.join("NEP", f)
            for f in ["INCAR_Single_point", "KPOINTS", "run.sh","Outcars2xyz.sh"]
        ],
        },
        "split": {
            "script": os.path.join("NEP", "Exyz-random-select.py"),
            "help": "Split training data",
            "desc": "封装 NEP/Exyz-random-select.py，按比例分开训练集和测试级",
            "examples": [
                "leo nep split total.xyz 0.9",
            ],
        "copy_files": [
            os.path.join("NEP", f)
            for f in ["Exyz-random-select.py"]
        ],
        },        
    },
    "MD": {
        "pdos": {
            "script": os.path.join("MD", "PDOS.py"),
            "help": "PDOS",
            "desc": "封装 MD/PDOS.py，用于处理。",
            "examples": [
                "leo MD pdos dump.velo",
            ],
            # 需要的话可以这样加：
            # "copy_files": [
            #     os.path.join("NEP", "palette.json"),
            #     os.path.join("NEP", "style.yaml"),
            # ],
        },     
        "plt": {
            "script": os.path.join("MD", "LAMMPS-Plot.py"),
            "help": "MD-lammps process parameters",
            "desc": "封装 MD/LAMMPS-Plot.py，用于处理。",
            "examples": [
                "leo MD plt log.lammps",
            ],
            # 需要的话可以这样加：
            # "copy_files": [
            #     os.path.join("NEP", "palette.json"),
            #     os.path.join("NEP", "style.yaml"),
            # ],
        },           
        },
}


# ===================== 2. 通用运行函数 =====================

def run_script(script_rel_path, extra_args, copy_files=None):
    """
    调用某个子脚本：
      1) 如有需要，先把 copy_files 里的文件复制到当前目录
      2) 然后以当前 Python 解释器运行脚本
    """
    # 1) 复制依赖文件
    if copy_files:
        for file_rel in copy_files:
            src = os.path.join(BASE_DIR, file_rel)
            dst = os.path.join(os.getcwd(), os.path.basename(file_rel))
            try:
                if not os.path.exists(src):
                    print(f"[Leo] ⚠ 需要复制的文件不存在: {src}")
                    continue
                # 如果目标已经存在，可以根据喜好选择是否覆盖，这里选择覆盖
                shutil.copy(src, dst)
                print(f"[Leo] ✅ 已复制: {src}  →  {dst}")
            except Exception as e:
                print(f"[Leo] ⚠ 复制文件时出错: {src} → {dst} | {e}")

    # 2) 运行脚本
    script_path = os.path.join(BASE_DIR, script_rel_path)
    cmd = [sys.executable, script_path] + extra_args
    subprocess.run(cmd, check=True)


# ===================== 3. 顶部猫猫头 + 总览文本 =====================

def build_overview():
    """猫猫头 + 一行一个完整命令示例的总览（命令和说明对齐）"""
    lines = []

    # 猫猫头 Banner
    lines.append("┌──────────────────────────────────────────────┐")
    lines.append("│   /\\_/\\                                       │")
    lines.append("│  ( o.o )   <  Miao!                           │")
    lines.append("│   > ^ <                                       │")
    lines.append("└──────────────────────────────────────────────┘")
    lines.append("【command】")

    # 1) 先收集所有“示例命令字符串”，用来算最大长度
    all_cmd_strs = []
    for group_name, cmds in TOOLS.items():
        for cmd_name, info in cmds.items():
            examples = info.get("examples", [])
            if examples:
                cmd_str = examples[0]
            else:
                cmd_str = f"leo {group_name} {cmd_name}"
            all_cmd_strs.append(cmd_str)

    max_len = max(len(s) for s in all_cmd_strs)

    # 2) 再按组打印，每条命令把左边命令部分补到同样长度
    for group_name, cmds in TOOLS.items():
        lines.append(f"组 {group_name}:")
        for cmd_name, info in cmds.items():
            help_txt = info.get("help", "")
            examples = info.get("examples", [])

            if examples:
                cmd_str = examples[0]
            else:
                cmd_str = f"leo {group_name} {cmd_name}"

            # 左侧命令补空格到 max_len，这样竖线就会对齐
            padded = cmd_str.ljust(max_len)
            lines.append(f"  {padded} |  {help_txt}")

        lines.append("")  # 组之间空一行

    return "\n".join(lines)


# ===================== 4. 构建命令行解析 =====================

def build_parser():
    """构建 argparse 的 parser，但顶层展示用我们自己的 overview"""
    parser = argparse.ArgumentParser(
        prog="leo",
        description="",  # 顶层我们自己打印 overview，就不靠 argparse 的 description 了
        add_help=True,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    group_parsers = parser.add_subparsers(
        title="脚本分组（group）",
        dest="group",
        metavar="group",
        help="例如：leo universal substitute ... 或 leo nep plot ...",
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
                formatter_class=argparse.RawTextHelpFormatter,
            )

            # 所有后续参数原样传给子脚本
            p.add_argument(
                "args",
                nargs=argparse.REMAINDER,
                help="后面所有参数会原样传递给对应脚本（保持与原脚本用法一致）。",
            )

            # 绑定执行函数（注意用默认参数避免 lambda 晚绑定坑）
            script_rel = info["script"]
            copy_files = info.get("copy_files", None)
            p.set_defaults(
                func=lambda ns, script_rel=script_rel, copy_files=copy_files:
                    run_script(script_rel, ns.args, copy_files)
            )

    return parser


# ===================== 5. 主入口 =====================

def main():
    parser = build_parser()

    # 直接 `python Leo.py` / `leo` 时，只打印猫猫头 + command 总览
    if len(sys.argv) == 1:
        print(build_overview())
        sys.exit(0)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        # 参数不完整或没匹配到命令时，也打印总览
        print(build_overview())


if __name__ == "__main__":
    main()
