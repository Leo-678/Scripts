#!/usr/bin/env python
# -*- coding: utf-8 -*-
# leo main entry

"""
Leo：个人脚本统一入口

新增命令：
    leo update
用于将本地 Scripts 仓库硬重置到远程 origin/main（全部替换本地修改，慎用）。
"""

import argparse
import os
import sys
import subprocess
import shutil
from datetime import datetime
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
    # 专门的系统工具组，目前只有 update
    "system": {
        "update": {
            "script": None,
            "help": "Update Leo scripts from GitHub (HARD reset)",
            "desc": (
                "将当前 Scripts 仓库硬重置到 origin/main：\n"
                "  git fetch origin\n"
                "  git reset --hard origin/main\n"
                "会丢弃本地未提交修改，请谨慎使用。"
            ),
            "examples": [
                "leo system update",
            ],
        },
    },

    "universal": {
        "substitute": {
            "script": os.path.join("Universal", "Substitute-POSCAR.py"),
            "help": "Substitute POSCAR Elements",
            "desc": "封装 Universal/Substitute-POSCAR.py，用于在 POSCAR 中随机替换元素。",
            "examples": [
                "leo universal substitute POSCAR POSCAR_new Al Sc 0.25",
            ],
        },
        "replicate": {
            "script": os.path.join("Universal", "POSCAR2SUPER-X.py"),
            "help": "Replicate POSCAR & convert format",
            "desc": "封装 Universal/POSCAR2SUPER-X.py，用于将 VASP 结构扩展为超胞，并可输出多种格式。",
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
            "help": "Make vacancy in POSCAR",
            "desc": "封装 Universal/POS-Remove.py，用于删掉 POSCAR 中指定元素/编号的原子。",
            "examples": [
                "leo universal vacancy In 10 POSCAR",
            ],
            "copy_files": [
                os.path.join("NEP", f)
                for f in ["POS-Remove.py"]
            ],
        },
        "LMP2XYZ": {
            "script": os.path.join("Universal", "LAMMPS2EXYZ.py"),
            "help": "Convert LAMMPS dump to extxyz",
            "desc": "封装 Universal/LAMMPS2EXYZ.py，将 LAMMPS dump 转换为 extxyz。",
            "examples": [
                "leo universal LMP2XYZ dump.xyz --type-map 1:Al,2:N",
            ],
        },
    },

    # NEP 相关工具
    "nep": {
        "plot": {
            "script": os.path.join("NEP", "NEP-plot.py"),
            "help": "Plot NEP training results",
            "desc": "封装 NEP/NEP-plot.py，用于绘制 NEP 的训练损失、力误差等结果。",
            "examples": [
                "leo nep plot",
            ],
        },
        "single": {
            "script": os.path.join("NEP", "Xyz2poscar.py"),
            "help": "Generate VASP single-point inputs from NEP data",
            "desc": "封装 NEP/Xyz2poscar.py，用于从 NEP 结构生成 VASP 单点能计算输入，并打包脚本。",
            "examples": [
                "leo nep single dump.xyz --order Cu In P S",
            ],
            "copy_files": [
                os.path.join("NEP", f)
                for f in ["INCAR_Single_point", "KPOINTS", "run.sh", "Outcars2xyz.sh"]
            ],
        },
        "split": {
            "script": os.path.join("NEP", "Exyz-random-select.py"),
            "help": "Split training/test exyz",
            "desc": "封装 NEP/Exyz-random-select.py，按比例分开训练集和测试集。",
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
            "help": "Compute VACF + PDOS from velocities",
            "desc": "封装 MD/PDOS.py，用于从 LAMMPS 速度 dump 计算 VACF 和 PDOS。",
            "examples": [
                "leo MD pdos dump.velo --ninitial 30 --corlength-steps 5000",
            ],
        },
        "plt": {
            "script": os.path.join("MD", "LAMMPS-Plot.py"),
            "help": "Plot LAMMPS thermo (step, T, P, E, cell…)",
            "desc": "封装 MD/LAMMPS-Plot.py，用于绘制 LAMMPS log 中的温度、压强、能量、晶格等随步长变化曲线。",
            "examples": [
                "leo MD plt log.lammps",
            ],
        },
        "plt-gpu": {
            "script": os.path.join("MD", "GPUMD-plot.py"),
            "help": "Plot GPUMD thermo (step, T, P, E, cell…)",
            "desc": "封装 MD/LAMMPS-Plot.py，用于绘制 LAMMPS log 中的温度、压强、能量、晶格等随步长变化曲线。",
            "examples": [
                "leo MD plt-gpu",
            ],
        },
        "rdf": {
            "script": os.path.join("MD", "RDF.py"),
            "help": "Compute RDF g(r)",
            "desc": "封装 MD/RDF.py，用于从 XDATCAR 或 LAMMPS dump 计算总 RDF 以及分类型 RDF。",
            "examples": [
                "leo MD rdf dump.xyz --fmt lammps --type-map 1:Cu,2:Se",
            ],
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
                shutil.copy(src, dst)
                print(f"[Leo] ✅ 已复制: {src}  →  {dst}")
            except Exception as e:
                print(f"[Leo] ⚠ 复制文件时出错: {src} → {dst} | {e}")

    # 2) 运行脚本
    script_path = os.path.join(BASE_DIR, script_rel_path)
    cmd = [sys.executable, script_path] + extra_args
    subprocess.run(cmd, check=True)

def get_git_last_update():
    """
    返回 git 仓库最后一次提交时间（YYYY-MM-DD HH:MM:SS）。
    若失败，则返回 'unknown'。
    """
    try:
        out = subprocess.check_output(
            ["git", "-C", BASE_DIR, "log", "-1", "--format=%cd", "--date=iso"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        # iso 格式示例： "2025-01-03 14:57:21 +0800"
        # 只保留前 19 字符： "2025-01-03 14:57:21"
        return out[:19] if len(out) >= 19 else out
    except Exception:
        return "unknown"

def run_update(branch="main"):
    """
    强制更新当前 Leo Scripts 仓库：
        git fetch origin
        git reset --hard origin/<branch>

    相当于“全部替换”为远程 main 分支内容，会丢弃本地未提交修改。
    """
    repo_dir = BASE_DIR
    git_dir = os.path.join(repo_dir, ".git")

    if not os.path.isdir(git_dir):
        print("[Leo] 当前目录不是 git 仓库，无法执行 leo update。")
        return

    print(f"[Leo] ⚠ 注意：即将把本地仓库硬重置为 origin/{branch}，本地未提交修改将丢失。")
    try:
        subprocess.run(
            ["git", "-C", repo_dir, "fetch", "origin"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", repo_dir, "reset", "--hard", f"origin/{branch}"],
            check=True,
        )
        print(f"[Leo] ✅ 已成功重置到 origin/{branch}。")
    except subprocess.CalledProcessError as e:
        print("[Leo] ❌ 更新失败，请检查网络或远程分支是否存在。")
        print("      详细错误：", e)


# ===================== 3. 顶部猫猫头 + 总览文本 =====================

def build_overview():
    """猫猫头 + 一行一个完整命令示例的总览（命令和说明对齐）"""
    lines = []

    # 计算内部宽度（当前边框字符串长度是 49，去掉左右边框 2 个字符）
    inner_width = 49 - 2
    last_git = get_git_last_update()
    # 构造一行：last git update: YYYY-MM-DD
    update_str = f"last git update: {last_git}"
    if len(update_str) > inner_width:
        update_str = update_str[:inner_width]

    # 猫猫头 Banner
    lines.append("┌──────────────────────────────────────────────┐")
    lines.append("│   /\\_/\\                                       │")
    lines.append("│  ( o.o )   <  Miao!                           │")
    # 新增一行更新时间
    lines.append("│" + update_str.ljust(inner_width) + "│")
    lines.append("│   > ^ <                                       │")
    lines.append("└──────────────────────────────────────────────┘")
    lines.append("【command】")

    # 收集所有“示例命令字符串”，用来算最大长度
    all_cmd_strs = []
    for group_name, cmds in TOOLS.items():
        for cmd_name, info in cmds.items():
            examples = info.get("examples", [])
            if examples:
                cmd_str = examples[0]
            else:
                cmd_str = f"leo {group_name} {cmd_name}"
            all_cmd_strs.append(cmd_str)

    max_len = max(len(s) for s in all_cmd_strs) if all_cmd_strs else 0

    # 再按组打印，每条命令把左边命令部分补到同样长度
    for group_name, cmds in TOOLS.items():
        lines.append(f"组 {group_name}:")
        for cmd_name, info in cmds.items():
            help_txt = info.get("help", "")
            examples = info.get("examples", [])

            if examples:
                cmd_str = examples[0]
            else:
                cmd_str = f"leo {group_name} {cmd_name}"

            padded = cmd_str.ljust(max_len)
            lines.append(f"  {padded} |  {help_txt}")

        lines.append("")  # 组之间空一行

    return "\n".join(lines)


# ===================== 4. 构建命令行解析 =====================

def build_parser():
    """构建 argparse 的 parser，但顶层展示用我们自己的 overview"""
    parser = argparse.ArgumentParser(
        prog="leo",
        description="",
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
            # 对 system/update 做特殊处理：不调用 run_script，而是 run_update
            if group_name == "system" and cmd_name == "update":
                p = subparsers.add_parser(
                    cmd_name,
                    help=info.get("help", ""),
                    description=info.get("desc", ""),
                    add_help=True,
                    formatter_class=argparse.RawTextHelpFormatter,
                )
                p.set_defaults(func=lambda ns: run_update())
                continue

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
