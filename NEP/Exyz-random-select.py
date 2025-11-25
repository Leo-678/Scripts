#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import random

def usage():
    print("Usage:")
    print("  python split_xyz.py <input.xyz> <train_ratio> [prefix]")
    print("\nExample:")
    print("  python split_xyz.py out.xyz 0.9")
    print("  python split_xyz.py out.xyz 0.8 dataset")
    sys.exit(1)

# -------------------------
# 读取命令行参数
# -------------------------
if len(sys.argv) < 3:
    usage()

input_file = sys.argv[1]

try:
    train_ratio = float(sys.argv[2])
    assert 0.0 < train_ratio < 1.0
except:
    print("Error: train_ratio must be a float between 0 and 1.")
    usage()

prefix = sys.argv[3] if len(sys.argv) >= 4 else ""

train_file = f"{prefix}train.xyz" if prefix else "train.xyz"
test_file  = f"{prefix}test.xyz"  if prefix else "test.xyz"

# -------------------------
# 读取所有帧
# -------------------------
frames = []
with open(input_file, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    try:
        n_atoms = int(lines[i].strip())
    except:
        print(f"Error: cannot read number of atoms in line {i+1}")
        sys.exit(1)

    frame = lines[i : i + 2 + n_atoms]
    frames.append(frame)
    i += 2 + n_atoms

# -------------------------
# 随机打乱
# -------------------------
random.shuffle(frames)

# -------------------------
# 划分
# -------------------------
num_train = int(len(frames) * train_ratio)
train_frames = frames[:num_train]
test_frames = frames[num_train:]

# -------------------------
# 输出
# -------------------------
with open(train_file, "w") as f:
    for frame in train_frames:
        f.writelines(frame)

with open(test_file, "w") as f:
    for frame in test_frames:
        f.writelines(frame)

print(f"输入文件：{input_file}")
print(f"总帧数：{len(frames)}")
print(f"训练集：{len(train_frames)} -> {train_file}")
print(f"测试集：{len(test_frames)} -> {test_file}")
print("Done.")
