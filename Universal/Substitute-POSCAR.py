#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Substitute atoms in a VASP POSCAR with a given fraction or count.

Features:
- Supports standard VASP5 POSCAR: line 6 = element symbols, line 7 = counts.
- Allows replacing a source element with a target element at a given fraction
  (0–1) or by an explicit number of atoms.
- Randomly selects which atoms to replace (optional seed for reproducibility).
- Automatically updates element list and counts, removes species with 0 atoms.
- Preserves lattice, Selective dynamics flags, and coordinate system.

Usage examples:
python Substitute-POSCAR.py POSCAR_origin POSCAR Al Sc 0.2
"""

import argparse
import random
import sys
from typing import List, Tuple


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Substitute atoms in a POSCAR with a given fraction or count."
    )
    parser.add_argument("poscar_in", help="Input POSCAR filename")
    parser.add_argument("poscar_out", help="Output POSCAR filename")
    parser.add_argument("src_element", help="Element symbol to be replaced, e.g. Al")
    parser.add_argument("dst_element", help="Element symbol to replace with, e.g. Sc")
    parser.add_argument(
        "amount",
        help="If --mode frac: fraction (0–1) of src atoms to replace; "
             "If --mode count: integer number of atoms to replace.",
        type=float,
    )
    parser.add_argument(
        "--mode",
        choices=["frac", "count"],
        default="frac",
        help="Interpret 'amount' as fraction of atoms (frac) or number of atoms (count). "
             "Default: frac.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for selecting which atoms to replace (optional).",
    )

    return parser.parse_args()


def read_poscar(filename: str) -> List[str]:
    with open(filename, "r") as f:
        lines = [line.rstrip("\n") for line in f]
    if len(lines) < 8:
        raise ValueError("POSCAR seems too short; not a valid VASP5 POSCAR.")
    return lines


def detect_poscar_layout(lines: List[str]) -> Tuple[List[str], List[int], int, bool]:
    """
    Detect VASP5-like POSCAR layout.

    Returns:
        species: list of element symbols
        counts: list of integer atom counts
        coord_start: index of the first coordinate line in 'lines'
        selective: whether Selective dynamics is present
    """
    # Standard VASP5:
    # 0: comment
    # 1: scale
    # 2–4: lattice vectors
    # 5: element names
    # 6: element counts
    # 7: (optional) "Selective dynamics"
    # 7 or 8: "Direct" / "Cartesian"
    # following: coordinates

    species_line_idx = 5
    counts_line_idx = 6

    try:
        species = lines[species_line_idx].split()
        counts = [int(x) for x in lines[counts_line_idx].split()]
    except Exception as e:
        raise ValueError(
            "Failed to parse species/counts lines. Make sure this is a VASP5-style POSCAR.\n"
            f"Error: {e}"
        )

    if len(species) != len(counts):
        raise ValueError(
            f"Number of species ({len(species)}) != number of counts ({len(counts)})."
        )

    # Detect Selective dynamics & coordinate system lines
    line_after_counts = counts_line_idx + 1
    selective = False

    if lines[line_after_counts].strip().lower().startswith("s"):
        selective = True
        coord_sys_line_idx = line_after_counts + 1
    else:
        coord_sys_line_idx = line_after_counts

    coord_sys = lines[coord_sys_line_idx].strip().lower()
    if not (coord_sys.startswith("d") or coord_sys.startswith("c")):
        raise ValueError(
            "Failed to detect coordinate system line (Direct/Cartesian). "
            "Check POSCAR format."
        )

    coord_start = coord_sys_line_idx + 1
    return species, counts, coord_start, selective


def build_atom_species_list(species: List[str], counts: List[int]) -> List[str]:
    """
    Build a per-atom species list based on species and counts.
    Example: species = ['Al', 'N'], counts = [2, 3]
             -> ['Al', 'Al', 'N', 'N', 'N']
    """
    atom_species = []
    for elem, n in zip(species, counts):
        atom_species.extend([elem] * n)
    return atom_species


def substitute_atoms(
    atom_species: List[str],
    src_element: str,
    dst_element: str,
    amount: float,
    mode: str,
    rng: random.Random,
) -> List[str]:
    """
    Perform the actual substitution on the per-atom species list.
    Returns the updated per-atom species list.
    """
    n_atoms = len(atom_species)
    src_indices = [i for i, elem in enumerate(atom_species) if elem == src_element]

    if not src_indices:
        raise ValueError(f"No atoms of source element '{src_element}' found in POSCAR.")

    n_src = len(src_indices)

    if mode == "frac":
        if not (0.0 <= amount <= 1.0):
            raise ValueError(
                f"Fraction mode selected but amount={amount} is not in [0, 1]."
            )
        n_replace = int(round(n_src * amount))
        # If amount>0 but rounding gives 0, ensure at least 1
        if amount > 0 and n_replace == 0:
            n_replace = 1
    else:  # mode == "count"
        n_replace = int(round(amount))
        if n_replace < 0:
            raise ValueError("Number of atoms to replace must be non-negative.")
        if n_replace > n_src:
            raise ValueError(
                f"Requested to replace {n_replace} atoms of '{src_element}', "
                f"but only {n_src} are available."
            )

    if n_replace == 0:
        # Nothing to do
        return atom_species

    chosen = rng.sample(src_indices, n_replace)

    new_atom_species = atom_species[:]
    for idx in chosen:
        new_atom_species[idx] = dst_element

    return new_atom_species


def rebuild_species_counts(atom_species: List[str], original_order: List[str], dst_element: str) -> Tuple[List[str], List[int]]:
    """
    From a per-atom species list, rebuild species list and counts.

    Strategy:
    - Start from original_order; if dst_element was not present, append it.
    - For each species, count how many atoms remain.
    - Drop species with zero atoms.
    """
    unique_elements_in_atoms = sorted(set(atom_species), key=lambda x: original_order.index(x) if x in original_order else len(original_order))

    # Ensure dst_element appears (if present in atom_species)
    if dst_element in atom_species and dst_element not in unique_elements_in_atoms:
        unique_elements_in_atoms.append(dst_element)

    # Final species list preserves as much original ordering as possible
    final_species = []
    final_counts = []

    # Start with all elements from original_order that still exist
    for elem in original_order:
        if elem in atom_species:
            final_species.append(elem)
            final_counts.append(atom_species.count(elem))

    # Then add any new elements that were not in original_order but appear now
    for elem in set(atom_species):
        if elem not in original_order:
            final_species.append(elem)
            final_counts.append(atom_species.count(elem))

    return final_species, final_counts


def reorder_coordinates(
    lines: List[str],
    coord_start: int,
    atom_species_old: List[str],
    atom_species_new: List[str],
    final_species: List[str],
) -> List[str]:
    """
    Rebuild coordinate lines so that atoms are grouped according to final_species
    order, while preserving each atom's coordinates and (if present) SD flags.
    """
    n_atoms = len(atom_species_old)
    coord_lines = lines[coord_start : coord_start + n_atoms]

    if len(coord_lines) != n_atoms:
        raise ValueError("Number of coordinate lines does not match total atom count.")

    # 为每个原子建立 (element, coord_line)
    atoms = list(zip(atom_species_new, coord_lines))

    # 按最终元素顺序重排
    new_coord_lines: List[str] = []
    for elem in final_species:
        for e, coord in atoms:
            if e == elem:
                new_coord_lines.append(coord)

    if len(new_coord_lines) != n_atoms:
        raise RuntimeError("Reordered coordinates length mismatch.")

    # 将新的坐标行写回
    new_lines = lines[:]
    new_lines[coord_start : coord_start + n_atoms] = new_coord_lines
    return new_lines


def main():
    args = parse_arguments()

    if args.seed is not None:
        rng = random.Random(args.seed)
    else:
        rng = random.Random()

    lines = read_poscar(args.poscar_in)
    species, counts, coord_start, selective = detect_poscar_layout(lines)

    total_atoms = sum(counts)
    atom_species_old = build_atom_species_list(species, counts)

    if len(atom_species_old) != total_atoms:
        raise ValueError("Inconsistent total atom count when building atom species list.")

    # 执行替换
    atom_species_new = substitute_atoms(
        atom_species_old,
        args.src_element,
        args.dst_element,
        args.amount,
        args.mode,
        rng,
    )

    # 重建 species + counts
    final_species, final_counts = rebuild_species_counts(
        atom_species_new, species, args.dst_element
    )

    # 更新 POSCAR 中的元素行和数量行
    species_line_idx = 5
    counts_line_idx = 6

    new_lines = lines[:]
    new_lines[species_line_idx] = " ".join(final_species)
    new_lines[counts_line_idx] = " ".join(str(n) for n in final_counts)

    # 按 final_species 重排坐标
    new_lines = reorder_coordinates(
        new_lines, coord_start, atom_species_old, atom_species_new, final_species
    )

    # 写出结果
    with open(args.poscar_out, "w") as f:
        for line in new_lines:
            f.write(line + "\n")

    print(f"Done. Wrote substituted POSCAR to '{args.poscar_out}'.")
    print("Summary:")
    print("  Original species:", species)
    print("  Original counts :", counts)
    print("  New species     :", final_species)
    print("  New counts      :", final_counts)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"[ERROR] {e}\n")
        sys.exit(1)
