import matplotlib
matplotlib.use("Agg")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NEP descriptor based structure selection.

Goal:
    Select structures from sample.xyz to supplement an existing train.xyz set.

Main usage:
    python NEP-select.py sample.xyz train.xyz nep.txt 300

Design:
    1. Compute structure-level NEP descriptors.
    2. Project sample and train into the same PCA space.
    3. Use train distribution in PCA space as the existing coverage.
    4. Select structures that:
        - are far from train as much as possible;
        - are mutually diverse, avoiding dense selected clusters.
    5. User only needs to specify how many structures to select.

Outputs:
    selected.xyz
    selected_report.txt
    sample_distance_report.txt
    select_pca_elements.png
    select_pca_exact.png

Label displays:
    - elements: element-set label, e.g. Ag-Cu-Se
    - exact: exact composition label, e.g. Ag1Cu200Se108
"""

import argparse
import sys
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from ase.io import read, write
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

try:
    from calorine.nep import get_descriptors
except ImportError as e:
    print("[ERROR] Cannot import calorine.nep.get_descriptors")
    print("        Please install calorine, for example:")
    print("        pip install calorine")
    raise e


# ============================================================
# Basic settings
# ============================================================

DEFAULT_TOP_EXACT_LABELS = 40
DEFAULT_LEGEND_COLS = 8


# ============================================================
# Progress bar and descriptor calculation
# ============================================================

def print_progress_bar(iteration, total, prefix='', suffix='',
                       decimals=1, length=50, fill='█'):
    if total <= 0:
        return

    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total))
    )

    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')

    if iteration == total:
        print()


def calculate_descriptors(structures, model_file, label="structures"):
    """
    Calculate structure-level descriptors by averaging atomic NEP descriptors.

    Parameters
    ----------
    structures : list[ase.Atoms]
    model_file : str
    label : str

    Returns
    -------
    des_all : np.ndarray
        Shape: (Nstructure, Ndescriptor)
    """
    des_all = []
    total = len(structures)

    for i, atoms in enumerate(structures):
        des_atom = get_descriptors(atoms, model_filename=model_file)
        des_all.append(np.mean(des_atom, axis=0))

        print_progress_bar(
            i + 1,
            total,
            prefix=f" Processing {label}:",
            suffix="Complete",
            length=50
        )

    return np.asarray(des_all, dtype=float)


# ============================================================
# Structure labels
# ============================================================

def composition_counter(atoms):
    return Counter(atoms.get_chemical_symbols())


def label_elements(atoms):
    """
    Element-set label:
        Ag-Cu-Se
        Cu-Se-Te
    """
    counter = composition_counter(atoms)
    return "-".join(sorted(counter.keys()))


def label_exact(atoms):
    """
    Exact composition label:
        Ag1Cu200Se108
    """
    counter = composition_counter(atoms)
    return "".join(f"{elem}{counter[elem]}" for elem in sorted(counter.keys()))


def make_labels(structures):
    """
    Return both elements labels and exact composition labels.
    """
    labels_elements = [label_elements(at) for at in structures]
    labels_exact = [label_exact(at) for at in structures]

    return labels_elements, labels_exact


def compress_labels(labels, top_labels=None):
    """
    Keep most frequent labels and merge the rest into Others.

    Parameters
    ----------
    labels : list[str]
    top_labels : int or None

    Returns
    -------
    compressed : list[str]
    unique_labels : list[str]
    """
    if top_labels is None or top_labels <= 0:
        return labels, sorted(set(labels))

    counts = Counter(labels)
    keep = [lab for lab, _ in counts.most_common(top_labels)]

    compressed = [lab if lab in keep else "Others" for lab in labels]
    unique_labels = sorted(set(compressed))

    return compressed, unique_labels


# ============================================================
# Distance utilities
# ============================================================

def min_dist_to_reference(points, reference):
    """
    Compute nearest-neighbor distance from each point to reference.

    Parameters
    ----------
    points : np.ndarray, shape (N, D)
    reference : np.ndarray, shape (M, D)

    Returns
    -------
    dmin : np.ndarray, shape (N,)
    """
    if len(points) == 0:
        return np.array([], dtype=float)

    if reference is None or len(reference) == 0:
        return np.full(len(points), np.inf, dtype=float)

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(reference)

    d, _ = nn.kneighbors(points, return_distance=True)

    return d[:, 0]


def train_internal_nn_distance(points_train):
    """
    Compute nearest-neighbor distance among train points themselves.

    For each train point, the nearest other train point is used.
    """
    if len(points_train) < 2:
        raise ValueError("Need at least two train structures to estimate train coverage.")

    nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn.fit(points_train)

    d, _ = nn.kneighbors(points_train, return_distance=True)

    # d[:, 0] is self-distance = 0
    # d[:, 1] is nearest other train structure
    return d[:, 1]


def robust_percentile(values, p):
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, p))


# ============================================================
# PCA projection
# ============================================================

def build_pca_projection(des_sample, des_train, pca_fit="sample"):
    """
    Build PCA projection.

    pca_fit:
        sample:
            PCA.fit(des_sample)
            Same projection logic as the old script.

        all:
            PCA.fit(train + sample)
            Sometimes better for global comparison, but changes old coordinates.

        train:
            PCA.fit(des_train)
            Uses train as reference basis.

    Returns
    -------
    proj_sample, proj_train, reducer
    """
    reducer = PCA(n_components=2)

    if pca_fit == "sample":
        reducer.fit(des_sample)
    elif pca_fit == "all":
        reducer.fit(np.vstack([des_train, des_sample]))
    elif pca_fit == "train":
        reducer.fit(des_train)
    else:
        raise ValueError(f"Unknown pca_fit: {pca_fit}")

    proj_sample = reducer.transform(des_sample)
    proj_train = reducer.transform(des_train)

    return proj_sample, proj_train, reducer


# ============================================================
# Selection logic
# ============================================================

def select_structures_to_enrich_train(
    proj_sample,
    proj_train,
    n_select,
    novelty_power=2.0,
    coverage_percentile=90.0,
    strict_first=True,
):
    """
    Select sample structures to enrich train in PCA space.

    Core idea:
        score = distance_to_train_plus_selected * novelty_weight

    where:
        distance_to_train_plus_selected:
            keeps selected structures mutually sparse.

        novelty_weight:
            penalizes sample points close to train.

    The algorithm first tries to select from structures outside train coverage.
    If the requested number is too large, it relaxes and fills the remaining
    structures with the best available points, still penalizing train-overlap.

    Parameters
    ----------
    proj_sample : np.ndarray, shape (Ns, 2)
    proj_train : np.ndarray, shape (Nt, 2)
    n_select : int
    novelty_power : float
        Larger value penalizes train-near structures more strongly.
    coverage_percentile : float
        Percentile of train internal NN distance used as automatic coverage radius.
    strict_first : bool
        If True, first select structures outside automatic train coverage.

    Returns
    -------
    selected_idx : list[int]
    d_train : np.ndarray
        sample-to-train distance in PCA space.
    coverage_radius : float
    phase_tags : dict[int, str]
        selected index -> "strict" or "relaxed"
    """
    n_sample = len(proj_sample)
    n_select = int(n_select)

    if n_select <= 0:
        return [], np.array([]), 0.0, {}

    n_select = min(n_select, n_sample)

    # Distance from sample to existing train.
    d_train = min_dist_to_reference(proj_sample, proj_train)

    # Estimate how dense train is in PCA space.
    train_nn = train_internal_nn_distance(proj_train)

    coverage_radius = robust_percentile(train_nn, coverage_percentile)

    if coverage_radius <= 0:
        positive = train_nn[train_nn > 0]
        if len(positive) > 0:
            coverage_radius = float(np.median(positive))
        else:
            coverage_radius = 1e-12

    print("[INFO] Train PCA-space internal nearest-neighbor distance:")
    print(f"       min  = {np.min(train_nn):.8f}")
    print(f"       mean = {np.mean(train_nn):.8f}")
    print(f"       p50  = {np.percentile(train_nn, 50):.8f}")
    print(f"       p75  = {np.percentile(train_nn, 75):.8f}")
    print(f"       p90  = {np.percentile(train_nn, 90):.8f}")
    print(f"       p95  = {np.percentile(train_nn, 95):.8f}")
    print(f"[INFO] Automatic train coverage radius p{coverage_percentile:g} = {coverage_radius:.8f}")

    # Initial distance to train + selected.
    d_ref = d_train.copy()

    # Novelty weight:
    # close to train -> small weight
    # far from train -> larger weight
    novelty_ratio = d_train / coverage_radius
    novelty_ratio = np.maximum(novelty_ratio, 1e-12)
    novelty_weight = novelty_ratio ** novelty_power

    selected = []
    phase_tags = {}

    used = np.zeros(n_sample, dtype=bool)

    # Strict candidates are those outside estimated train coverage.
    strict_mask = d_train >= coverage_radius

    print(f"[INFO] Sample structures outside automatic train coverage: {np.sum(strict_mask)}")
    print("[INFO] Sample-to-train PCA distance statistics:")
    print(f"       min  = {np.min(d_train):.8f}")
    print(f"       mean = {np.mean(d_train):.8f}")
    print(f"       p50  = {np.percentile(d_train, 50):.8f}")
    print(f"       p75  = {np.percentile(d_train, 75):.8f}")
    print(f"       p90  = {np.percentile(d_train, 90):.8f}")
    print(f"       p95  = {np.percentile(d_train, 95):.8f}")
    print(f"       max  = {np.max(d_train):.8f}")

    # ---------- Phase 1: strict supplement region ---------- #
    if strict_first:
        while len(selected) < n_select:
            candidate_mask = (~used) & strict_mask

            if not np.any(candidate_mask):
                break

            score = d_ref * novelty_weight
            score[~candidate_mask] = -np.inf

            best_idx = int(np.argmax(score))

            if not np.isfinite(score[best_idx]):
                break

            selected.append(best_idx)
            phase_tags[best_idx] = "strict"
            used[best_idx] = True

            # Update distance to train + selected.
            diff = proj_sample - proj_sample[best_idx]
            dist_new = np.sqrt(np.sum(diff * diff, axis=1))
            d_ref = np.minimum(d_ref, dist_new)

            if len(selected) % 100 == 0:
                print(f"[INFO] strict selected: {len(selected)}")

    # ---------- Phase 2: relaxed fill if user requested more ---------- #
    if len(selected) < n_select:
        print(
            f"[WARN] Strict region provided {len(selected)} structures, "
            f"but requested {n_select}."
        )
        print("[WARN] Filling remaining structures using penalized relaxed selection.")

    while len(selected) < n_select:
        candidate_mask = ~used

        if not np.any(candidate_mask):
            break

        score = d_ref * novelty_weight
        score[~candidate_mask] = -np.inf

        best_idx = int(np.argmax(score))

        if not np.isfinite(score[best_idx]):
            break

        selected.append(best_idx)
        phase_tags[best_idx] = "relaxed"
        used[best_idx] = True

        # Update distance to train + selected.
        diff = proj_sample - proj_sample[best_idx]
        dist_new = np.sqrt(np.sum(diff * diff, axis=1))
        d_ref = np.minimum(d_ref, dist_new)

        if len(selected) % 100 == 0:
            print(f"[INFO] selected: {len(selected)}")

    print(f"[INFO] Final selected structures: {len(selected)}")
    strict_count = sum(1 for i in selected if phase_tags.get(i) == "strict")
    relaxed_count = sum(1 for i in selected if phase_tags.get(i) == "relaxed")
    print(f"[INFO] Strict selected : {strict_count}")
    print(f"[INFO] Relaxed selected: {relaxed_count}")

    return selected, d_train, coverage_radius, phase_tags


# ============================================================
# Reports
# ============================================================

def write_selected_report(
    filename,
    selected_idx,
    sampledata,
    d_train_pca,
    phase_tags,
):
    """
    Write selected structures with both element-set and exact composition labels.
    """
    labels_elements, labels_exact = make_labels(sampledata)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# order sample_index d_train_pca phase elements_label exact_label\n")

        for order, idx in enumerate(selected_idx, start=1):
            phase = phase_tags.get(idx, "NA")
            f.write(
                f"{order:8d} "
                f"{idx:12d} "
                f"{d_train_pca[idx]:.10f} "
                f"{phase:8s} "
                f"{labels_elements[idx]:20s} "
                f"{labels_exact[idx]}\n"
            )


def write_sample_distance_report(
    filename,
    sampledata,
    d_train_pca,
    coverage_radius,
):
    """
    Write distance-to-train for all sample structures.
    """
    labels_elements, labels_exact = make_labels(sampledata)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# sample_index d_train_pca covered_by_train elements_label exact_label\n")

        for idx, d in enumerate(d_train_pca):
            covered = "yes" if d < coverage_radius else "no"
            f.write(
                f"{idx:12d} "
                f"{d:.10f} "
                f"{covered:5s} "
                f"{labels_elements[idx]:20s} "
                f"{labels_exact[idx]}\n"
            )


def print_selected_summary(selected_idx, d_train_pca, phase_tags):
    if len(selected_idx) == 0:
        print("[WARN] No structures selected.")
        return

    selected_idx = np.asarray(selected_idx, dtype=int)
    dsel = d_train_pca[selected_idx]

    print("[INFO] Selected sample-to-train PCA distance:")
    print(f"       count= {len(selected_idx)}")
    print(f"       min  = {np.min(dsel):.8f}")
    print(f"       mean = {np.mean(dsel):.8f}")
    print(f"       p50  = {np.percentile(dsel, 50):.8f}")
    print(f"       p75  = {np.percentile(dsel, 75):.8f}")
    print(f"       p90  = {np.percentile(dsel, 90):.8f}")
    print(f"       max  = {np.max(dsel):.8f}")

    strict_count = sum(1 for i in selected_idx if phase_tags.get(int(i)) == "strict")
    relaxed_count = sum(1 for i in selected_idx if phase_tags.get(int(i)) == "relaxed")

    print(f"[INFO] Strict selected : {strict_count}")
    print(f"[INFO] Relaxed selected: {relaxed_count}")


# ============================================================
# Plotting
# ============================================================

def make_pca_plot(
    fig_file,
    proj_sample,
    proj_train,
    sampledata,
    traindata,
    selected_idx,
    d_train_pca,
    coverage_radius,
    phase_tags,
    label_mode="elements",
    top_exact_labels=DEFAULT_TOP_EXACT_LABELS,
    legend_cols=DEFAULT_LEGEND_COLS,
):
    """
    Make four-panel PCA figure.

    label_mode:
        elements
        exact
    """
    selected_idx_arr = np.asarray(selected_idx, dtype=int)

    if len(selected_idx_arr) > 0:
        proj_selected = proj_sample[selected_idx_arr]
    else:
        proj_selected = np.empty((0, 2), dtype=float)

    if label_mode == "elements":
        train_labels_raw, _ = make_labels(traindata)
        selected_labels_raw, _ = make_labels([sampledata[i] for i in selected_idx_arr])
        top_labels = None
    elif label_mode == "exact":
        _, train_labels_raw = make_labels(traindata)
        _, selected_labels_raw = make_labels([sampledata[i] for i in selected_idx_arr])
        top_labels = top_exact_labels
    else:
        raise ValueError("label_mode must be 'elements' or 'exact'.")

    all_labels_raw = train_labels_raw + selected_labels_raw
    all_labels_compressed, _ = compress_labels(all_labels_raw, top_labels=top_labels)

    train_labels = all_labels_compressed[:len(train_labels_raw)]
    selected_labels = all_labels_compressed[len(train_labels_raw):]

    unique_classes = sorted(set(train_labels + selected_labels))
    class_map = {lab: i for i, lab in enumerate(unique_classes)}

    train_ids = np.array([class_map[x] for x in train_labels], dtype=int)

    if len(selected_labels) > 0:
        selected_ids = np.array([class_map[x] for x in selected_labels], dtype=int)
    else:
        selected_ids = np.array([], dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=200)
    (ax1, ax2), (ax3, ax4) = axes

    # ---------- (a) Overall ---------- #
    ax1.scatter(
        proj_sample[:, 0],
        proj_sample[:, 1],
        s=12,
        alpha=0.25,
        label="sample"
    )

    ax1.scatter(
        proj_train[:, 0],
        proj_train[:, 1],
        s=12,
        alpha=0.40,
        label="train"
    )

    if len(proj_selected) > 0:
        strict_mask = np.array(
            [phase_tags.get(int(i)) == "strict" for i in selected_idx_arr],
            dtype=bool
        )
        relaxed_mask = ~strict_mask

        if np.any(strict_mask):
            ax1.scatter(
                proj_selected[strict_mask, 0],
                proj_selected[strict_mask, 1],
                s=22,
                color="red",
                alpha=0.90,
                label="selected-strict"
            )

        if np.any(relaxed_mask):
            ax1.scatter(
                proj_selected[relaxed_mask, 0],
                proj_selected[relaxed_mask, 1],
                s=22,
                marker="x",
                color="black",
                alpha=0.80,
                label="selected-relaxed"
            )

    ax1.set_title("(a) Overall")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)

    # ---------- (b) Sample + Selected ---------- #
    ax2.scatter(
        proj_sample[:, 0],
        proj_sample[:, 1],
        s=12,
        alpha=0.20,
        label="sample"
    )

    if len(proj_selected) > 0:
        strict_mask = np.array(
            [phase_tags.get(int(i)) == "strict" for i in selected_idx_arr],
            dtype=bool
        )
        relaxed_mask = ~strict_mask

        if np.any(strict_mask):
            ax2.scatter(
                proj_selected[strict_mask, 0],
                proj_selected[strict_mask, 1],
                s=22,
                color="red",
                alpha=0.90,
                label="selected-strict"
            )

        if np.any(relaxed_mask):
            ax2.scatter(
                proj_selected[relaxed_mask, 0],
                proj_selected[relaxed_mask, 1],
                s=22,
                marker="x",
                color="black",
                alpha=0.80,
                label="selected-relaxed"
            )

    ax2.set_title("(b) Sample + Selected")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.3)

    # ---------- (c) Train only ---------- #
    ax3.scatter(
        proj_train[:, 0],
        proj_train[:, 1],
        s=12,
        alpha=0.40,
        color="orange",
        label="train"
    )

    ax3.set_title("(c) Train Only")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.legend(frameon=False)
    ax3.grid(alpha=0.3)

    # ---------- (d) Train + Selected by label ---------- #
    cmap = plt.cm.tab20
    ncls = max(1, len(unique_classes) - 1)

    handles = []
    labels = []

    # Train background by label.
    for cid, cls in enumerate(unique_classes):
        mask_t = train_ids == cid

        if not np.any(mask_t):
            continue

        color = cmap(cid / ncls)

        ax4.scatter(
            proj_train[mask_t, 0],
            proj_train[mask_t, 1],
            s=10,
            alpha=0.14,
            color=color
        )

    # Selected by label.
    if len(proj_selected) > 0:
        for cid, cls in enumerate(unique_classes):
            mask_s = selected_ids == cid

            if not np.any(mask_s):
                continue

            color = cmap(cid / ncls)

            sc = ax4.scatter(
                proj_selected[mask_s, 0],
                proj_selected[mask_s, 1],
                s=30,
                alpha=0.95,
                edgecolor="k",
                linewidths=0.4,
                color=color
            )

            handles.append(sc)
            labels.append(cls)

    ax4.set_title(f"(d) Train + Selected by {label_mode}")
    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.grid(alpha=0.3)

    if len(selected_idx_arr) > 0:
        dsel = d_train_pca[selected_idx_arr]
        strict_count = sum(
            1 for i in selected_idx_arr
            if phase_tags.get(int(i)) == "strict"
        )
        relaxed_count = len(selected_idx_arr) - strict_count

        info_text = (
            f"selected={len(selected_idx_arr)} "
            f"(strict={strict_count}, relaxed={relaxed_count}) | "
            f"d_train_pca min/mean/max = "
            f"{np.min(dsel):.5f}/{np.mean(dsel):.5f}/{np.max(dsel):.5f} | "
            f"coverage_radius={coverage_radius:.5f}"
        )
    else:
        info_text = f"selected=0 | coverage_radius={coverage_radius:.5f}"

    fig.suptitle(info_text, fontsize=10)

    if len(handles) > 0:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=max(1, int(legend_cols)),
            fontsize=7,
            frameon=False,
            title=f"Selected Structure Type ({label_mode})"
        )

        plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    else:
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    plt.savefig(fig_file, dpi=300)
    plt.close()

    print(f"[OK] PCA figure saved: {fig_file}")


# ============================================================
# Argument parser
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Select structures from sample.xyz to enrich train.xyz. "
            "Only the number of structures is required."
        )
    )

    parser.add_argument(
        "sample_file",
        help="Candidate structures, e.g. sample.xyz"
    )

    parser.add_argument(
        "train_file",
        help="Existing training structures, e.g. train.xyz"
    )

    parser.add_argument(
        "model_file",
        help="NEP model file, e.g. nep.txt"
    )

    parser.add_argument(
        "n_select",
        type=int,
        help="Number of structures to select."
    )

    parser.add_argument(
        "--pca-fit",
        choices=["sample", "all", "train"],
        default="sample",
        help=(
            "PCA fitting basis. Default: sample, consistent with the old script. "
            "Use 'all' if you want train+sample global projection."
        )
    )

    parser.add_argument(
        "--coverage-percentile",
        type=float,
        default=90.0,
        help=(
            "Percentile of train internal PCA nearest-neighbor distance used "
            "as train coverage radius. Default: 90."
        )
    )

    parser.add_argument(
        "--novelty-power",
        type=float,
        default=2.0,
        help=(
            "Power used to penalize structures close to train. "
            "Larger value means less train-overlap. Default: 2.0."
        )
    )

    parser.add_argument(
        "--output",
        default="selected.xyz",
        help="Output selected structures. Default: selected.xyz"
    )

    parser.add_argument(
        "--report",
        default="selected_report.txt",
        help="Output selected report. Default: selected_report.txt"
    )

    parser.add_argument(
        "--all-distance-report",
        default="sample_distance_report.txt",
        help="Output sample distance report. Default: sample_distance_report.txt"
    )

    parser.add_argument(
        "--elements-fig",
        default="select_pca_elements.png",
        help="Output PCA figure with element labels. Default: select_pca_elements.png"
    )

    parser.add_argument(
        "--exact-fig",
        default="select_pca_exact.png",
        help="Output PCA figure with exact-composition labels. Default: select_pca_exact.png"
    )

    parser.add_argument(
        "--top-exact-labels",
        type=int,
        default=DEFAULT_TOP_EXACT_LABELS,
        help=(
            "Maximum exact-composition labels shown in legend. "
            "Other exact labels are merged into Others. Default: 40."
        )
    )

    parser.add_argument(
        "--legend-cols",
        type=int,
        default=DEFAULT_LEGEND_COLS,
        help="Legend columns. Default: 8."
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_arguments()

    print("[INFO] Reading structures...")
    sampledata = read(args.sample_file, ":")
    traindata = read(args.train_file, ":")

    print(f"[INFO] sample structures : {len(sampledata)}")
    print(f"[INFO] train  structures : {len(traindata)}")
    print(f"[INFO] NEP model file    : {args.model_file}")
    print(f"[INFO] requested select  : {args.n_select}")
    print(f"[INFO] PCA fit basis     : {args.pca_fit}")

    if len(sampledata) == 0:
        print("[ERROR] No structures found in sample file.")
        sys.exit(1)

    if len(traindata) < 2:
        print("[ERROR] Need at least two train structures.")
        sys.exit(1)

    if args.n_select <= 0:
        print("[ERROR] n_select must be positive.")
        sys.exit(1)

    if args.n_select > len(sampledata):
        print(
            f"[WARN] Requested n_select={args.n_select}, "
            f"but only {len(sampledata)} sample structures are available."
        )
        args.n_select = len(sampledata)

    print("[INFO] Calculating sample descriptors...")
    des_sample = calculate_descriptors(
        sampledata,
        args.model_file,
        label="sampledata"
    )

    print("[INFO] Calculating train descriptors...")
    des_train = calculate_descriptors(
        traindata,
        args.model_file,
        label="traindata"
    )

    print("[INFO] Building PCA projection...")
    proj_sample, proj_train, reducer = build_pca_projection(
        des_sample=des_sample,
        des_train=des_train,
        pca_fit=args.pca_fit
    )

    print("[INFO] Selecting structures to enrich train...")
    selected_idx, d_train_pca, coverage_radius, phase_tags = (
        select_structures_to_enrich_train(
            proj_sample=proj_sample,
            proj_train=proj_train,
            n_select=args.n_select,
            novelty_power=args.novelty_power,
            coverage_percentile=args.coverage_percentile,
            strict_first=True
        )
    )

    print_selected_summary(
        selected_idx=selected_idx,
        d_train_pca=d_train_pca,
        phase_tags=phase_tags
    )

    selected_atoms = [sampledata[i] for i in selected_idx]

    if len(selected_atoms) > 0:
        write(args.output, selected_atoms)
        print(f"[INFO] Written selected structures: {args.output}")
    else:
        print("[WARN] No selected structures written.")

    write_selected_report(
        filename=args.report,
        selected_idx=selected_idx,
        sampledata=sampledata,
        d_train_pca=d_train_pca,
        phase_tags=phase_tags
    )

    print(f"[INFO] Written selected report: {args.report}")

    write_sample_distance_report(
        filename=args.all_distance_report,
        sampledata=sampledata,
        d_train_pca=d_train_pca,
        coverage_radius=coverage_radius
    )

    print(f"[INFO] Written all-sample distance report: {args.all_distance_report}")

    print("[INFO] Plotting PCA with element labels...")
    make_pca_plot(
        fig_file=args.elements_fig,
        proj_sample=proj_sample,
        proj_train=proj_train,
        sampledata=sampledata,
        traindata=traindata,
        selected_idx=selected_idx,
        d_train_pca=d_train_pca,
        coverage_radius=coverage_radius,
        phase_tags=phase_tags,
        label_mode="elements",
        top_exact_labels=args.top_exact_labels,
        legend_cols=args.legend_cols
    )

 
    print("[DONE] Selection finished.")


if __name__ == "__main__":
    main()