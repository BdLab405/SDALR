#!/bin/bash

# 定义 s 和 t 的组合（用户可在此处修改参数组合）
param_combinations=(
    "0,1"
    "0,2"
    "1,0"
    "1,2"
    "2,0"
    "2,1"
)

# 遍历 s 和 t 的组合
for st_combination in "${param_combinations[@]}"; do
    s=$(echo "$st_combination" | cut -d',' -f1)
    t=$(echo "$st_combination" | cut -d',' -f2)

    echo "Running with --s=$s --t=$t"

    # 遍历 threshold 的值
    for threshold in $(seq 0.1 0.05 0.5); do
        echo "Running with threshold=$threshold"

        # 运行 Python 脚本并传递参数
        python tar_adaptation_RES_PU_Ablation.py --s="$s" --t="$t" --mix=1 --threshold=2 --ent_par=-1 --aad_par=0
        python tar_adaptation_RES_PU_Ablation.py --s="$s" --t="$t" --mix=1 --threshold=2 --ent_par=-1 --aad_par=1 --K=0.1
        python tar_adaptation_RES_PU_Ablation.py --s="$s" --t="$t" --mix=2 --threshold="$threshold" --ent_par=-1 --aad_par=1 --K=0.1
        python tar_adaptation_RES_PU_Ablation.py --s="$s" --t="$t" --mix=2 --threshold=0.4 --ent_par=1 --aad_par=1

        # 检查 Python 脚本是否成功运行
        if [ $? -ne 0 ]; then
            echo "Error: Python script failed with --s=$s --t=$t and threshold=$threshold"
        fi
    done
done

echo "All tests completed successfully."
