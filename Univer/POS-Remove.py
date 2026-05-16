#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate vacancies for VASP POSCAR/CONTCAR or LAMMPS data files.

Usage examples:

    # VASP
    python make_vacancy.py POSCAR Cu 4 --format vasp --seed 123

    # LAMMPS
    python make_vacancy.py supercell.lammps Cu 4 --format lammps --seed 123

    # Auto detect format
    python make_vacancy.py POSCAR Cu 4 --seed 123
    python make_vacancy.py supercell.lammps Cu 4 --seed 123

    # Specify output
    python make_vacancy.py POSCAR Cu 4 -o POSCAR_CuVac4
    python make_vacancy.py supercell.lammps Cu 4 -o data_CuVac4.lammps

Requirements:
    - LAMMPS Masses section should contain element comments, e.g.
        1 63.546 # Cu
        2 78.960 # Se
"""

import argparse
import random
import re
import sys
from pathlib import Path


# ============================================================
# Generic helpers
# ============================================================

def die(msg):
    sys.exit(f"[ERROR] {msg}")


def is_float(s):
    try:
        float(s)
        return True
    except Exception:
        return False


def detect_format(path):
    """
    Roughly detect file format.

    VASP POSCAR:
        line 2 is scale factor
        lines 3-5 are lattice vectors
        line 6 usually element symbols

    LAMMPS data:
        contains lines such as:
            atoms
            xlo xhi
            Masses
            Atoms
    """
    lines = Path(path).read_text(errors="ignore").splitlines()

    joined_lower = "\n".join(lines[:80]).lower()

    if "xlo xhi" in joined_lower and "ylo yhi" in joined_lower and "atoms" in joined_lower:
        return "lammps"

    if len(lines) >= 8:
        if is_float(lines[1].split()[0]):
            return "vasp"

    die("Could not auto-detect format. Please use --format vasp or --format lammps.")


# ============================================================
# VASP POSCAR handling
# ============================================================

def read_poscar(path):
    """
    Read VASP POSCAR/CONTCAR.

    Supports:
        - VASP5 element line
        - Selective dynamics
        - Direct / Cartesian coordinates

    Returns dict.
    """
    lines = Path(path).read_text().splitlines()

    if len(lines) < 8:
        die("POSCAR seems too short.")

    title = lines[0].rstrip()
    scale = lines[1].rstrip()
    lattice = [lines[i].rstrip() for i in range(2, 5)]

    line5 = lines[5].split()
    line6 = lines[6].split()

    # VASP5 format: line 5 = element names, line 6 = counts
    if all(tok.isalpha() or re.match(r"^[A-Z][a-z]?$", tok) for tok in line5) and all(tok.isdigit() for tok in line6):
        elements = line5
        counts = list(map(int, line6))
        idx = 7
    else:
        die(
            "Only VASP5 POSCAR with element symbols is supported. "
            "Expected line 6 elements and line 7 counts."
        )

    selective = False
    selective_line = None

    if lines[idx].strip().lower().startswith("s"):
        selective = True
        selective_line = lines[idx].rstrip()
        idx += 1

    coord_type = lines[idx].rstrip()
    idx += 1

    total_atoms = sum(counts)

    if len(lines) < idx + total_atoms:
        die(f"POSCAR atom count mismatch. Expected {total_atoms} coordinate lines.")

    positions = [lines[i].rstrip() for i in range(idx, idx + total_atoms)]
    tail = [line.rstrip() for line in lines[idx + total_atoms:]]

    return {
        "title": title,
        "scale": scale,
        "lattice": lattice,
        "elements": elements,
        "counts": counts,
        "selective": selective,
        "selective_line": selective_line,
        "coord_type": coord_type,
        "positions": positions,
        "tail": tail,
    }


def write_poscar(path, data):
    with open(path, "w", encoding="utf-8") as f:
        f.write(data["title"] + "\n")
        f.write(data["scale"] + "\n")

        for line in data["lattice"]:
            f.write(line + "\n")

        f.write(" ".join(data["elements"]) + "\n")
        f.write(" ".join(str(x) for x in data["counts"]) + "\n")

        if data["selective"]:
            f.write(data["selective_line"] + "\n")

        f.write(data["coord_type"] + "\n")

        for line in data["positions"]:
            f.write(line + "\n")

        for line in data.get("tail", []):
            f.write(line + "\n")


def remove_vasp_atoms(input_path, element, num_remove, output_path, seed):
    data = read_poscar(input_path)

    elements = data["elements"]
    counts = data["counts"]
    positions = data["positions"]

    if element not in elements:
        die(f"Element {element} not found in POSCAR. Available: {elements}")

    elem_idx = elements.index(element)

    n_available = counts[elem_idx]

    if num_remove > n_available:
        die(f"Cannot remove {num_remove} {element}; only {n_available} available.")

    start = sum(counts[:elem_idx])
    end = start + counts[elem_idx]

    rng = random.Random(seed)
    remove_indices = set(rng.sample(range(start, end), num_remove))

    new_positions = [
        line for i, line in enumerate(positions)
        if i not in remove_indices
    ]

    new_counts = counts[:]
    new_counts[elem_idx] -= num_remove

    # Remove element completely if count becomes zero.
    new_elements = []
    compact_counts = []

    for elem, cnt in zip(elements, new_counts):
        if cnt > 0:
            new_elements.append(elem)
            compact_counts.append(cnt)

    data["elements"] = new_elements
    data["counts"] = compact_counts
    data["positions"] = new_positions
    data["title"] = f"{data['title']} | removed {num_remove} {element} vacancy seed={seed}"

    write_poscar(output_path, data)

    return {
        "format": "vasp",
        "element": element,
        "removed": num_remove,
        "available_before": n_available,
        "seed": seed,
        "output": output_path,
        "removed_indices_1based": sorted(i + 1 for i in remove_indices),
    }


# ============================================================
# LAMMPS data handling
# ============================================================

SECTION_NAMES = {
    "Masses",
    "Atoms",
    "Velocities",
    "Bonds",
    "Angles",
    "Dihedrals",
    "Impropers",
    "Pair Coeffs",
    "Bond Coeffs",
    "Angle Coeffs",
    "Dihedral Coeffs",
    "Improper Coeffs",
}


def find_section_indices(lines):
    """
    Return section name -> line index.
    """
    sec = {}

    for i, line in enumerate(lines):
        stripped = line.strip()

        for name in SECTION_NAMES:
            if stripped == name or stripped.startswith(name + " "):
                sec[name] = i

    return sec


def next_section_start(lines, start_idx):
    """
    Find next section after start_idx.
    """
    for i in range(start_idx + 1, len(lines)):
        stripped = lines[i].strip()

        for name in SECTION_NAMES:
            if stripped == name or stripped.startswith(name + " "):
                return i

    return len(lines)


def parse_lammps_header_counts(lines):
    """
    Parse header count lines:
        384 atoms
        2 atom types
    """
    atom_count_idx = None
    atoms_count = None

    for i, line in enumerate(lines):
        s = line.strip()
        m = re.match(r"^(\d+)\s+atoms\b", s)
        if m:
            atom_count_idx = i
            atoms_count = int(m.group(1))
            break

    if atom_count_idx is None:
        die("Could not find '<N> atoms' in LAMMPS data header.")

    return atom_count_idx, atoms_count


def parse_lammps_masses(lines, masses_idx):
    """
    Parse Masses section.

    Requires element labels in comments:
        1 63.546 # Cu

    Returns:
        type_to_element dict
    """
    end = next_section_start(lines, masses_idx)
    type_to_element = {}

    for i in range(masses_idx + 1, end):
        s = lines[i].strip()

        if not s or s.startswith("#"):
            continue

        if "#" not in s:
            continue

        left, comment = s.split("#", 1)
        toks = left.split()

        if len(toks) < 2:
            continue

        try:
            atom_type = int(toks[0])
        except Exception:
            continue

        elem = comment.strip().split()[0]

        if not re.match(r"^[A-Z][a-z]?$", elem):
            continue

        type_to_element[atom_type] = elem

    if not type_to_element:
        die(
            "Failed to parse element labels from Masses section. "
            "Need comments like: 1 63.546 # Cu"
        )

    return type_to_element


def parse_lammps_atoms(lines, atoms_idx):
    """
    Parse Atoms section lines.

    We preserve each raw line but infer:
        atom id = first column
        atom type = detected column

    Supported common styles:
        atomic: id type x y z
        charge: id type q x y z
        full:   id mol type q x y z

    Detection:
        If Atoms line says '# atomic', '# charge', '# full', use it.
        Otherwise infer:
            len 5 -> atomic
            len 6 -> charge
            len >=7 -> full
    """
    section_header = lines[atoms_idx].strip().lower()
    atom_style = None

    if "#" in section_header:
        atom_style = section_header.split("#", 1)[1].strip().split()[0]

    end = next_section_start(lines, atoms_idx)

    atom_rows = []
    row_line_indices = []

    for i in range(atoms_idx + 1, end):
        raw = lines[i]

        s = raw.strip()

        if not s or s.startswith("#"):
            continue

        toks = s.split()

        if len(toks) < 5:
            continue

        # Remove trailing inline comment for parsing.
        if "#" in toks:
            hash_idx = toks.index("#")
            toks_parse = toks[:hash_idx]
        else:
            # safer split by '#'
            toks_parse = s.split("#", 1)[0].split()

        if len(toks_parse) < 5:
            continue

        try:
            atom_id = int(toks_parse[0])
        except Exception:
            continue

        if atom_style == "atomic":
            type_col = 1
        elif atom_style == "charge":
            type_col = 1
        elif atom_style == "full":
            type_col = 2
        else:
            if len(toks_parse) == 5:
                type_col = 1
            elif len(toks_parse) == 6:
                type_col = 1
            else:
                type_col = 2

        try:
            atom_type = int(toks_parse[type_col])
        except Exception:
            die(f"Could not parse atom type from atom line: {raw}")

        atom_rows.append({
            "line_index": i,
            "raw": raw,
            "tokens": toks_parse,
            "atom_id": atom_id,
            "atom_type": atom_type,
        })

        row_line_indices.append(i)

    if not atom_rows:
        die("No atom rows parsed from LAMMPS Atoms section.")

    return atom_rows, atom_style


def remove_lammps_atoms(input_path, element, num_remove, output_path, seed):
    lines = Path(input_path).read_text().splitlines()

    sections = find_section_indices(lines)

    if "Masses" not in sections:
        die("LAMMPS data file has no Masses section.")

    if "Atoms" not in sections:
        die("LAMMPS data file has no Atoms section.")

    atom_count_idx, atoms_count = parse_lammps_header_counts(lines)

    type_to_element = parse_lammps_masses(lines, sections["Masses"])

    target_types = [
        t for t, e in type_to_element.items()
        if e == element
    ]

    if not target_types:
        die(
            f"Element {element} not found in Masses section. "
            f"Available mapping: {type_to_element}"
        )

    atom_rows, atom_style = parse_lammps_atoms(lines, sections["Atoms"])

    candidate_rows = [
        row for row in atom_rows
        if row["atom_type"] in target_types
    ]

    n_available = len(candidate_rows)

    if num_remove > n_available:
        die(f"Cannot remove {num_remove} {element}; only {n_available} available.")

    rng = random.Random(seed)
    chosen_rows = rng.sample(candidate_rows, num_remove)

    remove_line_indices = set(row["line_index"] for row in chosen_rows)
    removed_atom_ids = sorted(row["atom_id"] for row in chosen_rows)

    new_lines = []
    for i, line in enumerate(lines):
        if i in remove_line_indices:
            continue

        if i == atom_count_idx:
            old_line = line.strip()
            new_count = atoms_count - num_remove
            suffix = ""
            if "#" in line:
                suffix = " " + line.split("#", 1)[1]
            new_lines.append(f"{new_count} atoms")
        else:
            new_lines.append(line)

    # Remove corresponding velocities if Velocities section exists.
    # Velocities format starts with atom id. Remove same atom ids.
    if "Velocities" in sections:
        # Since line indices changed after removing atom lines, easier second-pass.
        new_lines = remove_lammps_velocities_by_ids(new_lines, set(removed_atom_ids))

    Path(output_path).write_text("\n".join(new_lines) + "\n")

    return {
        "format": "lammps",
        "element": element,
        "removed": num_remove,
        "available_before": n_available,
        "seed": seed,
        "output": output_path,
        "removed_atom_ids": removed_atom_ids,
        "target_types": target_types,
        "atom_style": atom_style,
    }


def remove_lammps_velocities_by_ids(lines, remove_atom_ids):
    """
    Remove velocity rows for removed atom ids.
    """
    sections = find_section_indices(lines)

    if "Velocities" not in sections:
        return lines

    v_idx = sections["Velocities"]
    v_end = next_section_start(lines, v_idx)

    new_lines = []

    for i, line in enumerate(lines):
        if v_idx < i < v_end:
            s = line.strip()

            if not s or s.startswith("#"):
                new_lines.append(line)
                continue

            toks = s.split()

            try:
                atom_id = int(toks[0])
            except Exception:
                new_lines.append(line)
                continue

            if atom_id in remove_atom_ids:
                continue

        new_lines.append(line)

    return new_lines


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate vacancies for VASP POSCAR or LAMMPS data files."
    )

    parser.add_argument(
        "input",
        help="Input file: POSCAR/CONTCAR or LAMMPS data file."
    )

    parser.add_argument(
        "element",
        help="Element to remove, e.g. Cu, Se, Ag."
    )

    parser.add_argument(
        "num_remove",
        type=int,
        help="Number of atoms to remove."
    )

    parser.add_argument(
        "--format",
        choices=["auto", "vasp", "lammps"],
        default="auto",
        help="Input format. Default: auto."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=677677,
        help="Random seed. Default: 677677."
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file name. Default: auto-generated."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        die(f"Input file not found: {input_path}")

    if args.num_remove < 0:
        die("num_remove must be non-negative.")

    fmt = args.format

    if fmt == "auto":
        fmt = detect_format(input_path)

    if args.output is None:
        if fmt == "vasp":
            output_path = f"POSCAR_del-{args.element}-{args.num_remove}_seed{args.seed}"
        elif fmt == "lammps":
            output_path = f"data_del-{args.element}-{args.num_remove}_seed{args.seed}.lammps"
        else:
            die(f"Unknown format: {fmt}")
    else:
        output_path = args.output

    print("----- Vacancy Generation -----")
    print(f"Input file   : {input_path}")
    print(f"Format       : {fmt}")
    print(f"Element      : {args.element}")
    print(f"Remove count : {args.num_remove}")
    print(f"Seed         : {args.seed}")
    print(f"Output file  : {output_path}")

    if args.num_remove == 0:
        print("[WARN] num_remove = 0. No atom will be removed.")

    if fmt == "vasp":
        summary = remove_vasp_atoms(
            input_path=input_path,
            element=args.element,
            num_remove=args.num_remove,
            output_path=output_path,
            seed=args.seed,
        )

    elif fmt == "lammps":
        summary = remove_lammps_atoms(
            input_path=input_path,
            element=args.element,
            num_remove=args.num_remove,
            output_path=output_path,
            seed=args.seed,
        )

    else:
        die(f"Unsupported format: {fmt}")

    print("----- Summary -----")
    for k, v in summary.items():
        print(f"{k:22s}: {v}")

    print("[DONE] Vacancy structure generated.")


if __name__ == "__main__":
    main()