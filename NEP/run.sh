START=135
END=268

for (( i=START; i<=END; i++ )); do
    file="POSCAR-$i"
    [ -f "$file" ] || { echo "跳过不存在的 $file"; continue; }

    # 使用数字 i 作为目录名
    mkdir -p "$i"
    mv "$file" "$i/POSCAR"
    cp POTCAR INCAR KPOINTS "$i/"
    (
      cd "$i"
      mpiexec -n 64 vasp_gam > vasp.log 2>&1
    )
done


