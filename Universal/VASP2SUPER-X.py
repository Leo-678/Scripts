#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于当前目录的 POSCAR 生成超胞，并导出多种格式文件：
  - VASP (POSCAR)
  - LAMMPS 数据文件（含 Masses 段）
  - XYZ / extxyz
  - CIF

要点：
  * LAMMPS 写出时传入 masses=True 强制输出 Masses。
  * 自动检查/补全 specorder，保证覆盖所有元素且顺序可控。
"""

from ase.io import read, write
from ase.data import atomic_masses, atomic_numbers
import sys
from pathlib import Path

# ===========================
# 可编辑变量
# ===========================
input_file = "POSCAR"          # 输入 VASP 结构文件
repeat     = (1, 1, 1)         # 超胞重复数 (nx, ny, nz)
formats    = ["lammps-data"]
#formats    = ["vasp", "lammps-data", "xyz", "cif", "extxyz"]
# 优先元素顺序（可留空 []，程序会自动以结构中首次出现顺序补齐）
# 例：["Cu", "Se", "Ag"]
specorder_hint = []


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
    # 只保留 hint 里结构实际包含的元素
    head = [s for s in hint if s in uniq]
    tail = [s for s in uniq if s not in head]
    return head + tail

def symbol_mass(sym):
    """返回元素的标准原子质量（amu）"""
    Z = atomic_numbers[sym]
    return float(atomic_masses[Z])


# ===========================
# 读取结构 & 生成超胞
# ===========================
if not Path(input_file).exists():
    sys.exit(f"找不到输入文件：{input_file}")

try:
    prim = read(input_file)
except Exception as e:
    sys.exit(f"无法读取 {input_file}: {e}")

supercell = prim.repeat(repeat)

print("=== 超胞信息 ===")
print("晶格矩阵 (Å)：")
print(supercell.get_cell())
print(f"原子总数：{len(supercell)}")

# 结构内包含的元素（按首次出现顺序）
symbols_all = supercell.get_chemical_symbols()
species_in_struct = unique_in_appearance(symbols_all)

# 构造最终 specorder
specorder = build_specorder(symbols_all, specorder_hint)
print("\n=== 元素顺序 (specorder) ===")
print(specorder)

# 打印质量表，便于核对
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
for fmt in formats:
    # 根据 ASE 习惯，lammps-data 通常建议扩展名为 .data 或 .lmp
    if fmt == "lammps-data":
        out_filename = "supercell.lmp"
    elif fmt == "vasp":
        out_filename = "POSCAR.supercell"
    else:
        out_filename = f"supercell.{fmt}"

    try:
        if fmt == "lammps-data":
            # 关键：masses=True 强制写 Masses 段；specorder 确保类型顺序与质量匹配
            write(out_filename, supercell,
                  format=fmt,
                  specorder=specorder,
                  masses=True)
        else:
            write(out_filename, supercell, format=fmt)
        print(f"已写出：{out_filename}")
    except Exception as e:
        print(f"[警告] 写出 {fmt} 失败：{e}")

print("\n全部完成。提示：若 LAMMPS data 中 Masses 与期望不一致，"
      "请检查 specorder 与结构实际元素是否一致。")
