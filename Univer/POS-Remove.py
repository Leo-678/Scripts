import random
import sys

def read_poscar(file_path):
    """读取 POSCAR 文件并解析其内容"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    title = lines[0].strip()
    scale = lines[1].strip()
    lattice_vectors = lines[2:5]
    atom_types = lines[5].split()
    atom_counts = list(map(int, lines[6].split()))
    coord_type = lines[7].strip()
    atomic_positions = lines[8:]
    
    return title, scale, lattice_vectors, atom_types, atom_counts, coord_type, atomic_positions

def write_poscar(file_path, title, scale, lattice_vectors, atom_types, atom_counts, coord_type, atomic_positions):
    """将修改后的 POSCAR 文件写入新文件"""
    with open(file_path, 'w') as f:
        f.write(title + '\n')
        f.write(scale + '\n')
        f.writelines(lattice_vectors)
        f.write(' '.join(atom_types) + '\n')
        f.write(' '.join(map(str, atom_counts)) + '\n')
        f.write(coord_type + '\n')
        f.writelines(atomic_positions)

def remove_random_atoms(atom_type, num_remove, poscar_path):
    """从 POSCAR 文件中随机删除指定类型的原子，并保存新的 POSCAR"""
    # 读取 POSCAR
    title, scale, lattice_vectors, atom_types, atom_counts, coord_type, atomic_positions = read_poscar(poscar_path)
    
    if atom_type not in atom_types:
        raise ValueError(f"Atom type {atom_type} not found in POSCAR.")
    
    index = atom_types.index(atom_type)
    
    # 计算原子索引范围
    start_index = sum(atom_counts[:index])
    end_index = start_index + atom_counts[index]
    
    if num_remove > atom_counts[index]:
        raise ValueError(f"Cannot remove {num_remove} atoms, only {atom_counts[index]} available.")
    
    # 随机选择要删除的原子索引
    remove_indices = set(random.sample(range(start_index, end_index), num_remove))
    
    # 过滤掉被删除的原子
    new_atomic_positions = [line for i, line in enumerate(atomic_positions) if i not in remove_indices]
    
    # 更新原子计数
    new_atom_counts = atom_counts[:]
    new_atom_counts[index] -= num_remove
    
    # 生成输出文件名
    output_path = f"POSCAR_del-{atom_type}-{num_remove}"
    
    # 写入新的 POSCAR
    write_poscar(output_path, title, scale, lattice_vectors, atom_types, new_atom_counts, coord_type, new_atomic_positions)
    print(f"New POSCAR saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <atom_type> <num_remove> <poscar_path>")
        sys.exit(1)
    
    atom_type = sys.argv[1]
    num_remove = int(sys.argv[2])
    poscar_path = sys.argv[3]
    
    remove_random_atoms(atom_type, num_remove, poscar_path)
