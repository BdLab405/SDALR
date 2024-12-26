#!/bin/bash

# 定义参数和值
declare -A parameters  # 定义关联数组存放参数及其对应的值列表
parameters["lr"]="0.0005"
parameters["threshold"]="0.4"
parameters["K"]="0.6"

# 定义 s 和 t 的组合（用户可在此处修改参数组合）
param_combinations=(
    "0,1"
    "0,2"
    "1,0"
    "1,2"
    "2,0"
    "2,1"
)

# 创建笛卡尔积函数
cartesian_product() {
    local arrays=("$@") result=()
    local indices=($(for i in "${!arrays[@]}"; do echo 0; done))
    local sizes=($(for array in "${arrays[@]}"; do echo "$array" | wc -w; done))
    while true; do
        # 生成当前组合
        local combination=()
        for i in "${!arrays[@]}"; do
            combination+=($(echo "${arrays[i]}" | awk "{print \$$((indices[i] + 1))}"))
        done
        result+=("$(IFS=,; echo "${combination[*]}")")

        # 更新索引
        for i in $(seq $(( ${#indices[@]} - 1 )) -1 0); do
            indices[i]=$((indices[i] + 1))
            if [ "${indices[i]}" -lt "${sizes[i]}" ]; then
                break
            else
                indices[i]=0
                if [ "$i" -eq 0 ]; then
                    echo "${result[@]}"
                    return
                fi
            fi
        done
    done
}

# 提取参数和值列表
param_keys=("${!parameters[@]}")
param_values=("${parameters[@]}")

# 计算笛卡尔积，生成所有参数组合
combinations=$(cartesian_product "${param_values[@]}")

# 遍历参数组合
for combination in $combinations; do
    # 分解组合到各参数
    IFS=',' read -r -a values <<< "$combination"

    # 构建参数字符串
    param_str=""
    for i in "${!values[@]}"; do
        param_str+="--${param_keys[i]}=${values[i]} "
    done

    echo "Testing with parameters: $param_str"

    # 遍历 s 和 t 的组合
    for st_combination in "${param_combinations[@]}"; do
        s=$(echo "$st_combination" | cut -d',' -f1)
        t=$(echo "$st_combination" | cut -d',' -f2)

        echo "Running with $param_str --s=$s --t=$t"

        # 运行 Python 脚本并传递参数
        python tar_adaptation_RES_JNU.py $param_str --s="$s" --t="$t"

        # 检查 Python 脚本是否成功运行
        if [ $? -ne 0 ]; then
            echo "Error: Python script failed with $param_str --s=$s --t=$t"
        fi
    done
done

echo "All tests completed successfully."
