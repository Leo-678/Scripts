#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select representative structures based on NEP descriptors
using calorine.nep.get_descriptors (no pynep), and visualize
descriptor space with PCA (4 subplots + global legend).

Usage:
    # 按最小距离筛选
    python select_calor4.py sample.xyz train.xyz nep.txt mindist 0.02

    # （可选）按数量筛选
    python select_calor4.py sample.xyz train.xyz nep.txt nselect 100
"""

import sys
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ase.io import read, write

# 用 calorine 计算 NEP 描述子
try:
    from calorine.nep import get_descriptors
except ImportError as e:
    print("[ERROR] Cannot import calorine.nep.get_descriptors")
    print("        Please install calorine, e.g.:")
    print("        pip install calorine")
    raise e


# ======================= 工具函数 ======================= #

def print_progress_bar(iteration, total, prefix='', suffix='',
                       decimals=1, length=50, fill='█'):
    """命令行进度条"""
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


def calculate_descriptors():
    """
    使用 calorine.nep.get_descriptors 计算结构描述子，
    对每个结构的所有原子取平均。

    使用全局变量：
        sampledata, traindata, MODEL_FILE
    返回:
        des_sample : (Ns, D)
        des_train  : (Nt, D)
    """
    total_sample = len(sampledata)
    total_train = len(traindata)

    des_sample = []
    for i in range(total_sample):
        des = get_descriptors(sampledata[i], model_filename=MODEL_FILE)  # (Nat, D)
        des_sample.append(np.mean(des, axis=0))
        print_progress_bar(i + 1, total_sample,
                           prefix=' Processing sampledata:',
                           suffix='Complete', length=50)
    des_sample = np.array(des_sample)

    des_train = []
    for i in range(total_train):
        des = get_descriptors(traindata[i], model_filename=MODEL_FILE)
        des_train.append(np.mean(des, axis=0))
        print_progress_bar(i + 1, total_train,
                           prefix=' Processing traindata: ',
                           suffix='Complete', length=50)
    des_train = np.array(des_train)

    return des_sample, des_train


def classify_structure(atoms):
    """
    根据元素及其个数生成结构类型字符串，例如:
        Cu32Se32Ag1
    """
    symbols = atoms.get_chemical_symbols()
    counter = Counter(symbols)
    label = "".join(f"{elem}{counter[elem]}" for elem in sorted(counter.keys()))
    return label


# =================== 最远点采样（替代 pynep.select） =================== #

def _pairwise_min_dist_to_set(points, ref):
    """
    对每个 points[i] 计算其到 ref 中所有点的最小欧氏距离。
    points: (N, D)
    ref   : (M, D)
    返回:
        dmin: (N,)
    """
    if ref.size == 0:
        return np.zeros(points.shape[0], dtype=float)

    diff = points[:, None, :] - ref[None, :, :]   # (N, M, D)
    dist2 = np.sum(diff * diff, axis=-1)          # (N, M)
    dmin = np.sqrt(np.min(dist2, axis=1))         # (N,)
    return dmin


def farthest_point_sample_min_distance(des_sample, des_train,
                                       min_distance, max_select=None):
    """
    模仿 FarthestPointSample(min_distance) 的行为：
    - 初始参考集为 train
    - 每次从 sample 中选出“到 train+已选集合距离最大的点”
    - 直到最大距离 < min_distance 或选满 max_select
    """
    n_sample = des_sample.shape[0]
    if n_sample == 0:
        return []

    dmin = _pairwise_min_dist_to_set(des_sample, des_train)

    selected = []
    used = np.zeros(n_sample, dtype=bool)

    while True:
        remaining_idx = np.where(~used)[0]
        if remaining_idx.size == 0:
            break

        best_local = np.argmax(dmin[remaining_idx])
        best_idx = remaining_idx[best_local]
        best_dist = dmin[best_idx]

        if best_dist < min_distance and len(selected) > 0:
            break

        selected.append(best_idx)
        used[best_idx] = True

        if max_select is not None and len(selected) >= max_select:
            break

        # 更新未选点到新的参考集距离
        new_ref = des_sample[best_idx:best_idx + 1]  # (1, D)
        rem_idx = np.where(~used)[0]
        if rem_idx.size == 0:
            break
        diff = des_sample[rem_idx] - new_ref
        dist2 = np.sum(diff * diff, axis=1)
        dist = np.sqrt(dist2)
        dmin[rem_idx] = np.minimum(dmin[rem_idx], dist)

    return selected


def farthest_point_sample_by_number(des_sample, des_train,
                                    n_target):
    """
    简化版：按数量选结构。
    - 首先选出“离 train 最远”的一个点
    - 然后每次加入当前离 train+已选集合最远的点
    """
    n_sample = des_sample.shape[0]
    if n_sample == 0 or n_target <= 0:
        return []

    n_target = min(n_target, n_sample)

    dmin = _pairwise_min_dist_to_set(des_sample, des_train)
    selected = []
    used = np.zeros(n_sample, dtype=bool)

    # 第一个点：距离 train 最大
    first_idx = int(np.argmax(dmin))
    selected.append(first_idx)
    used[first_idx] = True

    while len(selected) < n_target:
        rem_idx = np.where(~used)[0]
        if rem_idx.size == 0:
            break

        new_ref = des_sample[selected[-1]:selected[-1] + 1]
        diff = des_sample[rem_idx] - new_ref
        dist2 = np.sum(diff * diff, axis=1)
        dist = np.sqrt(dist2)
        dmin[rem_idx] = np.minimum(dmin[rem_idx], dist)

        best_local = np.argmax(dmin[rem_idx])
        best_idx = rem_idx[best_local]
        selected.append(best_idx)
        used[best_idx] = True

    return selected


# ======================= 主程序 ======================= #

if len(sys.argv) < 6:
    print(" Usage:")
    print("   python select_calor4.py sample.xyz train.xyz nep.txt mindist 0.02")
    print("   python select_calor4.py sample.xyz train.xyz nep.txt nselect 100")
    sys.exit(1)

sample_file = sys.argv[1]
train_file = sys.argv[2]
MODEL_FILE = sys.argv[3]
mode = sys.argv[4].lower()
param = sys.argv[5]

# 读入数据
sampledata = read(sample_file, ':')
traindata = read(train_file, ':')

print(f"[INFO] sample structures : {len(sampledata)}")
print(f"[INFO] train  structures : {len(traindata)}")
print(f"[INFO] NEP model file    : {MODEL_FILE}")
print(f"[INFO] selection mode    : {mode}")
print(f"[INFO] parameter         : {param}")

# 先统一算描述子（后面 PCA 也要用）
des_sample, des_train = calculate_descriptors()

if mode == 'mindist':
    min_dist = float(param)
    selected_idx = farthest_point_sample_min_distance(
        des_sample, des_train,
        min_distance=min_dist,
        max_select=None
    )
elif mode == 'nselect':
    n_target = int(param)
    selected_idx = farthest_point_sample_by_number(
        des_sample, des_train,
        n_target=n_target
    )
else:
    print(" [ERROR] Unknown mode. Use 'mindist' or 'nselect'.")
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

# 先画 train（淡底）
for cid, cls in enumerate(unique_classes):
    mask_t = (train_ids == cid)
    if not np.any(mask_t):
        continue
    color = cmap(cid / ncls)
    ax4.scatter(proj_train[mask_t, 0],
                proj_train[mask_t, 1],
                s=10, alpha=0.2, color=color)

# 再画 selected（加黑边 + 写入 legend）
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
    ncol=8,
    fontsize=7,
    frameon=False,
    title="Structure Type"
)

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.savefig("select_pca.png", dpi=300)
plt.close()
print("[OK] Four-panel PCA figure saved as select_pca.png")
