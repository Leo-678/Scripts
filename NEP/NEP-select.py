#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select representative structures based on NEP descriptors
and visualize descriptor space with PCA (4 subplots + global legend).

Usage:
    python pynep_select_structs.py dump.xyz train.xyz nep.txt
"""

import sys
import numpy as np
from collections import Counter

from ase.io import read, write
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from pynep.calculate import NEP
from pynep.select import FarthestPointSample


# ======================= 工具函数 ======================= #

def print_progress_bar(iteration, total, prefix='', suffix='',
                       decimals=1, length=50, fill='█'):
    """命令行进度条"""
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total))
    )
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()


def calculate_descriptors(sampledata, traindata, calc):
    """
    对每个结构计算 NEP descriptor，并取所有原子的平均值。

    返回:
        des_sample : (Ns, D)
        des_train  : (Nt, D)
    """
    total_sample = len(sampledata)
    total_train = len(traindata)

    des_sample = []
    for i, at in enumerate(sampledata, 1):
        des = calc.get_property('descriptor', at)  # (Nat, D)
        des_sample.append(np.mean(des, axis=0))
        print_progress_bar(i, total_sample,
                           prefix=' Processing sampledata:',
                           suffix='Complete', length=50)
    des_sample = np.array(des_sample)

    des_train = []
    for i, at in enumerate(traindata, 1):
        des = calc.get_property('descriptor', at)
        des_train.append(np.mean(des, axis=0))
        print_progress_bar(i, total_train,
                           prefix=' Processing traindata: ',
                           suffix='Complete', length=50)
    des_train = np.array(des_train)

    return des_sample, des_train


def classify_structure(atoms):
    """
    根据元素及其个数生成结构类型字符串，例如:
        Cu32Se32Ag1 之类
    """
    symbols = atoms.get_chemical_symbols()
    counter = Counter(symbols)
    # 按元素名排序，保证稳定
    label = "".join(f"{elem}{counter[elem]}" for elem in sorted(counter.keys()))
    return label


# ======================= 主程序 ======================= #

def main():

    # -------- 参数检查 --------
    if len(sys.argv) < 4:
        print(" Usage:")
        print("   python pynep_select_structs.py dump.xyz train.xyz nep.txt")
        sys.exit(1)

    sample_file = sys.argv[1]
    train_file = sys.argv[2]
    model_file = sys.argv[3]

    # -------- 读入数据 --------
    sampledata = read(sample_file, ':')
    traindata = read(train_file, ':')

    print(f"[INFO] sample structures : {len(sampledata)}")
    print(f"[INFO] train  structures : {len(traindata)}")

    # -------- 初始化 NEP --------
    calc = NEP(model_file)
    print("[INFO] Loaded NEP model:")
    print(calc)

    # -------- 计算 descriptor --------
    des_sample, des_train = calculate_descriptors(sampledata, traindata, calc)

    # -------- 选择方式 --------
    print("\n Choose selection method:")
    print(" 1) min_distance (descriptor space)")
    print(" 2) min_select / max_select (number of structures)")
    choice = input(" ------------>> ").strip()

    sampler = FarthestPointSample()

    if choice == '1':
        min_dist = float(input(" Enter min_dist (e.g., 0.02): ").strip())
        selected_idx = sampler.select(
            des_sample,
            des_train,
            min_distance=min_dist,
            max_select=None
        )
    elif choice == '2':
        min_select, max_select = map(
            int,
            input(" Enter min_select and max_select (e.g., '50 100'): ").split()
        )
        selected_idx = sampler.select(
            des_sample,
            des_train,
            min_select=min_select,
            max_select=max_select
        )
    else:
        print(" Invalid choice. Exit.")
        sys.exit(1)

    print(f"[INFO] Selected {len(selected_idx)} structures")

    # 写出 selected.xyz
    selected_atoms = [sampledata[i] for i in selected_idx]
    write("selected.xyz", selected_atoms)
    print("[INFO] Written selected structures to selected.xyz")

    # ======================= PCA ======================= #
    reducer = PCA(n_components=2)
    reducer.fit(des_sample)

    proj_sample = reducer.transform(des_sample)
    proj_train = reducer.transform(des_train)
    proj_selected = reducer.transform(des_sample[selected_idx])

    # ======================= 结构类型分类 ======================= #
    train_labels = [classify_structure(at) for at in traindata]
    selected_labels = [classify_structure(at) for at in selected_atoms]

    all_labels = train_labels + selected_labels
    unique_classes = sorted(set(all_labels))
    class_map = {cls: i for i, cls in enumerate(unique_classes)}

    train_ids = np.array([class_map[c] for c in train_labels])
    selected_ids = np.array([class_map[c] for c in selected_labels])

    # ======================= 四子图 + 全局 legend ======================= #
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=200)
    (ax1, ax2), (ax3, ax4) = axes

    # ---------- (a) Overall ---------- #
    ax1.scatter(proj_sample[:, 0], proj_sample[:, 1],
                s=12, alpha=0.3, label='sample')
    ax1.scatter(proj_train[:, 0], proj_train[:, 1],
                s=12, alpha=0.4, label='train')
    ax1.scatter(proj_selected[:, 0], proj_selected[:, 1],
                s=20, color='red', alpha=0.9, label='selected')

    ax1.set_title('(a) Overall')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)

    # ---------- (b) Sample + Selected ---------- #
    ax2.scatter(proj_sample[:, 0], proj_sample[:, 1],
                s=12, alpha=0.25, label='sample')
    ax2.scatter(proj_selected[:, 0], proj_selected[:, 1],
                s=20, color='red', alpha=0.9, label='selected')

    ax2.set_title('(b) Sample + Selected')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.3)

    # ---------- (c) Train Only ---------- #
    ax3.scatter(proj_train[:, 0], proj_train[:, 1],
                s=12, alpha=0.4, color='orange', label='train')

    ax3.set_title('(c) Train Only')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.legend(frameon=False)
    ax3.grid(alpha=0.3)

    # ---------- (d) Train + Selected (by type) ---------- #
    cmap = plt.cm.tab20
    ncls = max(1, len(unique_classes) - 1)

    handles = []
    labels = []

    # 先画 train（淡色背景）
    for cid, cls in enumerate(unique_classes):
        mask_t = (train_ids == cid)
        if not np.any(mask_t):
            continue
        color = cmap(cid / ncls)
        ax4.scatter(proj_train[mask_t, 0],
                    proj_train[mask_t, 1],
                    s=10, alpha=0.2, color=color)

    # 再画 selected（有黑边，加入 legend）
    for cid, cls in enumerate(unique_classes):
        mask_s = (selected_ids == cid)
        if not np.any(mask_s):
            continue
        color = cmap(cid / ncls)
        sc = ax4.scatter(proj_selected[mask_s, 0],
                         proj_selected[mask_s, 1],
                         s=28, alpha=0.95,
                         edgecolor='k', linewidths=0.4,
                         color=color)
        handles.append(sc)
        labels.append(cls)

    ax4.set_title('(d) Train + Selected (by type)')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.grid(alpha=0.3)

    # ---------- 全局 legend 放在底部 ---------- #
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=8,              # 一行显示多少个，可根据类型数量调整
        fontsize=7,
        frameon=False,
        title="Structure Type"
    )

    # 给底部 legend 留空间
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    plt.savefig("select_pca.png", dpi=300)
    plt.close()
    print("[OK] Four-panel PCA figure saved as select_pca.png")


if __name__ == "__main__":
    main()
