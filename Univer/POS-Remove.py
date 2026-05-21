#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate vacancy structures for VASP POSCAR/CONTCAR or LAMMPS data files.

Main features
-------------
1. Supports VASP and LAMMPS data input.
2. Supports VASP and LAMMPS data output independently.
3. For LAMMPS output:
   - writes Masses section;
   - preserves atom IDs as consecutive IDs;
   - supports explicit type order with --specorder;
   - supports extra unused atom types with --extra-elements.

Examples
--------
VASP -> VASP:
    python POS-Remove.py POSCAR Cu 52 -o POSCAR_CuVac

VASP -> LAMMPS data:
    python POS-Remove.py perfect.pos Cu 52 -o supercell.lammps --output-format lammps-data --specorder Cu Se

VASP -> LAMMPS data with dopant type reserved:
    python POS-Remove.py perfect.pos Cu 52 -o supercell.lammps --output-format lammps-data --specorder Cu Se Ag

In your loop:
    python POS-Remove.py perfect.pos Cu 52 -o supercell.lammps \\
        --output-format lammps-data \\
        --specorder Cu Se ${elem} \\
        --seed ${seed}
"""

import argparse
import random
import re
import sys
from pathlib import Path

from ase.io import read, write
from ase.data import atomic_masses, atomic_numbers


# ============================================================
# Utilities
# ============================================================

def die(msg):
    sys.exit(f"[ERROR] {msg}")


def detect_input_format(path):
    """
    Detect input file format.
    """
    text = Path(path).read_text(errors="ignore")
    low = text.lower()

    if "xlo xhi" in low and "atoms" in low:
        return "lammps-data"

    return "vasp"


def detect_output_format(output_file, input_format):
    """
    Detect output format from output filename.

    If unclear, use same format as input.
    """
    if output_file is None:
        return input_format

    name = Path(output_file).name.lower()

    if (
        name.endswith(".lmp")
        or name.endswith(".lammps")
        or name.startswith("data")
        or ".data" in name
    ):
        return "lammps-data"

    if (
        name.startswith("poscar")
        or name.startswith("contcar")
        or name.endswith(".pos")
        or name.endswith(".vasp")
    ):
        return "vasp"

    return input_format


def first_occurrence_order(symbols):
    """
    Get element order by first appearance.
    """
    order = []

    for s in symbols:
        if s not in order:
            order.append(s)

    return order


def normalize_specorder(specorder, extra_elements, symbols_before_delete):
    """
    Determine final LAMMPS element/type order.

    Priority:
        1. user --specorder
        2. first appearance in input structure
        3. append --extra-elements
    """
    if specorder is not None and len(specorder) > 0:
        final = list(specorder)
    else:
        final = first_occurrence_order(symbols_before_delete)

    if extra_elements is not None:
        for e in extra_elements:
            if e not in final:
                final.append(e)

    return final


def sort_atoms_by_specorder(atoms, specorder):
    """
    Sort atoms according to specorder.

    This ensures type order is clean, e.g.
        Cu first, Se second, Ag third.
    """
    symbols = atoms.get_chemical_symbols()

    rank = {s: i for i, s in enumerate(specorder)}

    missing = sorted(set(symbols) - set(specorder))

    if missing:
        die(
            f"Atoms contain elements not in specorder: {missing}\n"
            f"Current specorder = {specorder}"
        )

    order = sorted(
        range(len(atoms)),
        key=lambda i: (rank[symbols[i]], i)
    )

    return atoms[order]


def element_mass(elem):
    """
    Return atomic mass from ASE database.
    """
    if elem not in atomic_numbers:
        die(f"Unknown element symbol: {elem}")

    z = atomic_numbers[elem]
    return float(atomic_masses[z])


def ensure_lammps_masses(data_file, specorder):
    """
    Ensure LAMMPS data file has Masses section.

    Also enforce the atom types count to be len(specorder).

    This is a fallback/safety layer in case ASE version writes no Masses.
    """
    data_file = Path(data_file)
    lines = data_file.read_text().splitlines()

    # Fix atom types count if needed.
    new_lines = []

    atom_types_fixed = False

    for line in lines:
        if re.match(r"^\s*\d+\s+atom\s+types\s*$", line):
            new_lines.append(f"{len(specorder)} atom types")
            atom_types_fixed = True
        else:
            new_lines.append(line)

    lines = new_lines

    if not atom_types_fixed:
        # Usually ASE writes this. If not, leave it alone.
        pass

    # If Masses already exists, do not insert again.
    has_masses = any(line.strip().startswith("Masses") for line in lines)

    if has_masses:
        data_file.write_text("\n".join(lines) + "\n")
        return

    # Find Atoms section and insert Masses before it.
    atoms_idx = None

    for i, line in enumerate(lines):
        if line.strip().startswith("Atoms"):
            atoms_idx = i
            break

    if atoms_idx is None:
        die("Cannot find Atoms section in written LAMMPS data file.")

    masses_block = []
    masses_block.append("")
    masses_block.append("Masses")
    masses_block.append("")

    for i, elem in enumerate(specorder, start=1):
        masses_block.append(f"{i} {element_mass(elem):.10f} # {elem}")

    masses_block.append("")

    lines = lines[:atoms_idx] + masses_block + lines[atoms_idx:]

    data_file.write_text("\n".join(lines) + "\n")


# ============================================================
# Read / Write
# ============================================================

def read_structure(input_file, input_format):
    """
    Read structure.
    """
    if input_format == "vasp":
        return read(input_file, format="vasp")

    if input_format == "lammps-data":
        return read(
            input_file,
            format="lammps-data",
            style="atomic"
        )

    die(f"Unsupported input format: {input_format}")


def write_structure(output_file, atoms, output_format, specorder):
    """
    Write structure.

    For LAMMPS:
        Masses are written and checked.
    """
    if output_format == "vasp":
        write(
            output_file,
            atoms,
            format="vasp",
            direct=True,
            vasp5=True,
            sort=False,
        )
        return

    if output_format == "lammps-data":
        atoms = sort_atoms_by_specorder(atoms, specorder)

        try:
            write(
                output_file,
                atoms,
                format="lammps-data",
                atom_style="atomic",
                masses=True,
                specorder=specorder,
                units="metal",
            )
        except TypeError:
            # Older ASE fallback.
            print("[WARN] ASE writer did not accept masses=True. Writing first, then inserting Masses manually.")
            write(
                output_file,
                atoms,
                format="lammps-data",
                atom_style="atomic",
                specorder=specorder,
                units="metal",
            )

        ensure_lammps_masses(output_file, specorder)
        return

    die(f"Unsupported output format: {output_format}")


# ============================================================
# Vacancy generation
# ============================================================

def generate_vacancy(
    input_file,
    element,
    num_remove,
    output_file=None,
    seed=None,
    input_format="auto",
    output_format="auto",
    specorder=None,
    extra_elements=None,
):
    """
    Generate random vacancies.
    """
    input_file = Path(input_file)

    if not input_file.exists():
        die(f"Input file not found: {input_file}")

    if num_remove < 0:
        die("num_remove must be non-negative.")

    if input_format == "auto":
        input_format = detect_input_format(input_file)

    if output_file is None:
        if output_format == "auto":
            output_format = input_format

        if output_format == "vasp":
            output_file = f"POSCAR_del-{element}-{num_remove}"
            if seed is not None:
                output_file += f"_seed{seed}"

        elif output_format == "lammps-data":
            output_file = f"data_del-{element}-{num_remove}"
            if seed is not None:
                output_file += f"_seed{seed}"
            output_file += ".lammps"

        else:
            die(f"Unsupported output format: {output_format}")

    else:
        if output_format == "auto":
            output_format = detect_output_format(output_file, input_format)

    print("========== Vacancy generation ==========")
    print(f"Input file    : {input_file}")
    print(f"Input format  : {input_format}")
    print(f"Output file   : {output_file}")
    print(f"Output format : {output_format}")
    print(f"Remove element: {element}")
    print(f"Remove count  : {num_remove}")
    print(f"Seed          : {seed}")

    atoms = read_structure(input_file, input_format)

    symbols_before = atoms.get_chemical_symbols()

    final_specorder = normalize_specorder(
        specorder=specorder,
        extra_elements=extra_elements,
        symbols_before_delete=symbols_before,
    )

    print(f"Element/type order for LAMMPS: {final_specorder}")

    if seed is None:
        rng = random.Random()
    else:
        rng = random.Random(seed)

    symbols = atoms.get_chemical_symbols()

    candidate_indices = [
        i for i, s in enumerate(symbols)
        if s == element
    ]

    n_available = len(candidate_indices)

    if n_available == 0:
        die(f"No element '{element}' found in input structure.")

    if num_remove > n_available:
        die(
            f"Cannot remove {num_remove} {element}; "
            f"only {n_available} available."
        )

    remove_indices = sorted(
        rng.sample(candidate_indices, num_remove),
        reverse=True
    )

    print(f"Available {element}: {n_available}")
    print(f"Removing atom indices, 0-based: {remove_indices}")

    for idx in remove_indices:
        del atoms[idx]

    # Sort for clean LAMMPS type ordering or clean POSCAR grouping.
    atoms = sort_atoms_by_specorder(atoms, final_specorder)

    write_structure(
        output_file=output_file,
        atoms=atoms,
        output_format=output_format,
        specorder=final_specorder,
    )

    print("========== Summary ==========")
    print(f"Final atom count : {len(atoms)}")
    print(f"Removed count    : {num_remove}")
    print(f"Saved to         : {output_file}")
    print("[DONE]")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate vacancy structures for VASP or LAMMPS data."
    )

    parser.add_argument(
        "input_file",
        help="Input file: POSCAR/CONTCAR or LAMMPS data."
    )

    parser.add_argument(
        "element",
        help="Element to remove, e.g. Cu."
    )

    parser.add_argument(
        "num_remove",
        type=int,
        help="Number of atoms to remove."
    )

    parser.add_argument(
        "--format",
        dest="input_format",
        choices=["auto", "vasp", "lammps-data"],
        default="auto",
        help="Input format. Default: auto. Kept for backward compatibility."
    )

    parser.add_argument(
        "--output-format",
        choices=["auto", "vasp", "lammps-data"],
        default="auto",
        help="Output format. Default: auto by output filename."
    )

    parser.add_argument(
        "--specorder",
        nargs="+",
        default=None,
        help=(
            "LAMMPS element/type order. "
            "Example: --specorder Cu Se Ag. "
            "This must match pair_coeff order."
        )
    )

    parser.add_argument(
        "--extra-elements",
        nargs="*",
        default=None,
        help=(
            "Extra elements to reserve in LAMMPS Masses even if absent now. "
            "Example: --extra-elements Ag."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed."
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output filename."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    generate_vacancy(
        input_file=args.input_file,
        element=args.element,
        num_remove=args.num_remove,
        output_file=args.output,
        seed=args.seed,
        input_format=args.input_format,
        output_format=args.output_format,
        specorder=args.specorder,
        extra_elements=args.extra_elements,
    )


if __name__ == "__main__":
    main()