#!/bin/bash

# 通用测评数据集生成脚本
# 为3d、boolean、causal三个任务生成测评数据集

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================="
echo "开始生成所有任务的测评数据集 (时间戳: $TIMESTAMP)"
echo "========================================="

# 激活conda环境
echo "激活hypospace环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hypospace

# 创建数据集目录（如果不存在）
mkdir -p /opt/data/private/HypoSpace-40DA/datasets/3d
mkdir -p /opt/data/private/HypoSpace-40DA/datasets/boolean
mkdir -p /opt/data/private/HypoSpace-40DA/datasets/causal

# 1. 生成3D任务数据集
echo ""
echo "-----------------------------------------"
echo "生成3D任务数据集..."
echo "-----------------------------------------"
cd /opt/data/private/HypoSpace-40DA/3d

# 生成默认配置的数据集 (3x3网格，最多3个方块，最大高度3)
python generate_3d_dataset_complete.py \
    --grid-size 3 \
    --max-blocks 3 \
    --max-height 3 \
    --output ../datasets/3d/3d_grid3_h3_$TIMESTAMP.json

# 生成固定2个方块的数据集
python generate_3d_dataset_complete.py \
    --grid-size 3 \
    --max-blocks 2 \
    --fixed \
    --max-height 3 \
    --output ../datasets/3d/3d_grid3_h3_b2_$TIMESTAMP.json

# 合并3D数据集
echo "合并3D任务数据集..."
python3 << EOF
import json
import glob
import os

# 查找当前时间戳的3D数据集
pattern = "../datasets/3d/3d_grid3_h3*$TIMESTAMP.json"
files = glob.glob(pattern)

# 读取所有匹配的3D数据集
observation_sets = []
metadata_list = []

for filename in files:
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            if "observation_sets" in data:
                observation_sets.extend(data["observation_sets"])
                metadata_list.append(data["metadata"])
    except FileNotFoundError:
        print(f"文件 {filename} 未找到，跳过...")
        continue

# 创建合并后的数据集，保持标准3D格式
merged_metadata = {
    "description": "Merged 3D datasets",
    "source_datasets": files,
    "merged_at": "$TIMESTAMP",
    "grid_size": 3,
    "max_blocks": 3,
    "max_height": 3,
    "n_observation_sets": len(observation_sets)
}

merged_dataset = {
    "metadata": merged_metadata,
    "observation_sets": observation_sets
}

# 保存合并后的数据集
output_file = "../datasets/3d/3d_merged_dataset_$TIMESTAMP.json"
with open(output_file, 'w') as f:
    json.dump(merged_dataset, f, indent=2)

print(f"3D数据集合并完成，共包含 {len(observation_sets)} 个观察集")
print(f"合并后的数据集保存在: {output_file}")
EOF

echo "3D任务数据集生成完成!"

# 2. 生成布尔逻辑任务数据集
echo ""
echo "-----------------------------------------"
echo "生成布尔逻辑任务数据集..."
echo "-----------------------------------------"
cd /opt/data/private/HypoSpace-40DA/boolean

# 生成完整的数据集集合（包含1-4个观察值的所有组合）
python boolean_dataset.py \
    --operators extended \
    --max-depth 1 \
    --output ../datasets/boolean/boolean_complete_dataset_$TIMESTAMP.json

# 合并布尔逻辑数据集（如果需要）
echo "处理布尔逻辑任务数据集..."
python3 << EOF
import json

# 读取布尔逻辑数据集
filename = "../datasets/boolean/boolean_complete_dataset_$TIMESTAMP.json"
try:
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # 如果是按观察数分组的格式，保持该格式
    if "datasets_by_n_observations" in data:
        # 添加合并信息到元数据
        data["metadata"]["description"] = "Merged boolean datasets"
        data["metadata"]["source"] = f"boolean_complete_dataset_$TIMESTAMP.json"
        data["metadata"]["merged_at"] = "$TIMESTAMP"
        
        # 保存合并后的数据集（实际上是保持原格式）
        output_file = "../datasets/boolean/boolean_merged_dataset_$TIMESTAMP.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # 统计数据集数量
        total_count = sum(len(datasets) for datasets in data["datasets_by_n_observations"].values())
        print(f"布尔逻辑数据集处理完成，共包含 {total_count} 个数据集")
        print(f"数据集保存在: {output_file}")
    else:
        print("数据集格式不符合预期")
except FileNotFoundError:
    print(f"文件 {filename} 未找到")
EOF

echo "布尔逻辑任务数据集生成完成!"

# 3. 生成因果关系任务数据集
echo ""
echo "-----------------------------------------"
echo "生成因果关系任务数据集..."
echo "-----------------------------------------"
cd /opt/data/private/HypoSpace-40DA/causal

# 生成3节点的完整数据集集合（1-3个观察值）
python generate_causal_dataset.py \
    --nodes 3 \
    --output ../datasets/causal/causal_complete_dataset_$TIMESTAMP.json

# 合并因果关系数据集（如果需要）
echo "处理因果关系任务数据集..."
python3 << EOF
import json

# 读取因果关系数据集
filename = "../datasets/causal/causal_complete_dataset_$TIMESTAMP.json"
try:
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # 如果是按观察数分组的格式，合并所有数据集
    if "datasets_by_n_observations" in data:
        merged_datasets = []
        total_count = 0
        for n_obs, datasets in data["datasets_by_n_observations"].items():
            merged_datasets.extend(datasets)
            total_count += len(datasets)
        
        # 创建统一格式的数据集，保持标准因果关系格式
        merged_data = {
            "metadata": {
                "description": "Merged causal datasets",
                "source": f"causal_complete_dataset_$TIMESTAMP.json",
                "total_datasets": total_count,
                "merged_at": "$TIMESTAMP",
                "nodes": data["metadata"]["nodes"],
                "max_n_observations": data["metadata"]["max_n_observations"],
                "max_edges": data["metadata"]["max_edges"],
                "hypothesis_space_size": data["metadata"]["hypothesis_space_size"],
                "total_observation_sets": total_count
            },
            "datasets": merged_datasets
        }
        
        # 保存合并后的数据集
        output_file = "../datasets/causal/causal_merged_dataset_$TIMESTAMP.json"
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        print(f"因果关系数据集处理完成，共包含 {total_count} 个数据集")
        print(f"合并后的数据集保存在: {output_file}")
    else:
        print("数据集已经是统一格式，无需合并")
        # 复制为合并格式
        output_file = "../datasets/causal/causal_merged_dataset_$TIMESTAMP.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"数据集已复制为合并格式: {output_file}")
except FileNotFoundError:
    print(f"文件 {filename} 未找到")
EOF

echo "因果关系任务数据集生成完成!"

# 总结
echo ""
echo "========================================="
echo "所有任务的测评数据集生成完成! (时间戳: $TIMESTAMP)"
echo "数据集位置:"
echo "  - 3D任务: /opt/data/private/HypoSpace-40DA/datasets/3d/"
echo "  - 布尔逻辑任务: /opt/data/private/HypoSpace-40DA/datasets/boolean/"
echo "  - 因果关系任务: /opt/data/private/HypoSpace-40DA/datasets/causal/"
echo "每个任务目录下都包含带时间戳的原始数据集和合并后的数据集"
echo "========================================="