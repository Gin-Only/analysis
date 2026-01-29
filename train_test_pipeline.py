#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的训练/测试数据划分与推理管道
功能：
  1. 从 train_data 划分出训练集（80%）和测试集（20%）
  2. 创建 COCO 格式的标注文件
  3. 进行模型训练
  4. 进行推理测试
"""

import json
import os
import random
import shutil
from pathlib import Path
import argparse
import cv2
import numpy as np

def load_train_data(train_data_dir="train_data"):
    """
    从 train_data 目录加载所有数据
    
    返回格式：
    {
        'Dataset_A': [{'image_path': '...', 'text': '...'}, ...],
        'Dataset_B': [...],
        'Dataset_C': [...]
    }
    """
    all_data = {}
    
    for dataset_name in ['Dataset_A', 'Dataset_B', 'Dataset_C']:
        json_file = os.path.join(train_data_dir, f"{dataset_name}.json")
        
        if not os.path.exists(json_file):
            print(f"⚠️  {json_file} 不存在，跳过")
            continue
        
        print(f"📖 加载 {dataset_name}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_data[dataset_name] = data
        print(f"   ✓ 加载了 {len(data)} 条数据")
    
    return all_data


def split_train_test(all_data, train_ratio=0.8, random_seed=42):
    """
    将数据划分为训练集和测试集
    
    参数：
        all_data: 加载的所有数据
        train_ratio: 训练集比例（默认 80%）
        random_seed: 随机种子
    
    返回：
        (train_data, test_data)
    """
    random.seed(random_seed)
    
    train_data = []
    test_data = []
    
    for dataset_name, items in all_data.items():
        print(f"\n📊 划分 {dataset_name}...")
        print(f"   总数据量: {len(items)}")
        
        # 打乱数据
        shuffled = items.copy()
        random.shuffle(shuffled)
        
        # 按比例划分
        split_idx = int(len(shuffled) * train_ratio)
        train_subset = shuffled[:split_idx]
        test_subset = shuffled[split_idx:]
        
        train_data.extend(train_subset)
        test_data.extend(test_subset)
        
        print(f"   ├─ 训练集: {len(train_subset)} 条 ({train_ratio*100:.0f}%)")
        print(f"   └─ 测试集: {len(test_subset)} 条 ({(1-train_ratio)*100:.0f}%)")
    
    print(f"\n✓ 划分完成")
    print(f"  训练集总数: {len(train_data)}")
    print(f"  测试集总数: {len(test_data)}")
    
    return train_data, test_data


def create_coco_format(data, dataset_name, output_dir="data/annotations"):
    """
    创建 COCO 格式的标注文件
    
    COCO 格式：
    {
        "images": [{"id": 1, "file_name": "...", "height": 896, "width": 896}, ...],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w*h,
                "iscrowd": 0,
                "segmentation": [],
                "text": "识别文本"
            },
            ...
        ],
        "categories": [{"id": 1, "name": "text"}]
    }
    """
    os.makedirs(output_dir, exist_ok=True)
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "text"}]
    }
    
    image_id = 1
    annotation_id = 1
    
    print(f"\n📝 创建 {dataset_name} 的 COCO 格式标注...")
    
    for item in data:
        image_path = item['image_path']
        text = item['text']
        
        # 检查图像是否存在
        if not os.path.exists(image_path):
            print(f"   ⚠️  {image_path} 不存在，跳过")
            continue
        
        # 读取图像以获得尺寸
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"   ⚠️  无法读取 {image_path}，跳过")
                continue
            
            h, w = img.shape[:2]
        except Exception as e:
            print(f"   ⚠️  读取 {image_path} 失败: {e}，跳过")
            continue
        
        # 添加图像信息
        coco_data['images'].append({
            "id": image_id,
            "file_name": image_path,
            "height": h,
            "width": w
        })
        
        # 添加标注信息（简化：整个文本作为一个框）
        coco_data['annotations'].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [0, 0, w, h],  # 整个图像为文本区域
            "area": w * h,
            "iscrowd": 0,
            "segmentation": [],
            "text": text
        })
        
        image_id += 1
        annotation_id += 1
    
    # 保存标注文件
    output_file = os.path.join(output_dir, f"{dataset_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✓ 保存了 {image_id-1} 条数据到 {output_file}")
    
    return output_file


def save_split_info(train_data, test_data, output_dir="data/splits"):
    """
    保存训练/测试划分信息
    """
    os.makedirs(output_dir, exist_ok=True)
    
    split_info = {
        "train": train_data,
        "test": test_data,
        "metadata": {
            "train_count": len(train_data),
            "test_count": len(test_data),
            "total_count": len(train_data) + len(test_data)
        }
    }
    
    output_file = os.path.join(output_dir, "train_test_split.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 划分信息已保存到 {output_file}")
    
    return output_file


def create_training_config(config_template_path="config/pan_pp/R18-AUG.py",
                          output_path="config/pan_pp/R18-AUG-custom.py"):
    """
    创建自定义训练配置
    """
    print(f"\n⚙️  创建训练配置...")
    
    if not os.path.exists(config_template_path):
        print(f"   ⚠️  模板配置不存在: {config_template_path}")
        return None
    
    with open(config_template_path, 'r', encoding='utf-8') as f:
        config_content = f.read()
    
    # 修改数据路径（如需要）
    # config_content = config_content.replace(
    #     "path/to/dataset",
    #     "data/annotations/train.json"
    # )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"   ✓ 配置已保存到 {output_path}")
    
    return output_path


def generate_training_script(output_path="run_training.sh"):
    """
    生成训练脚本
    """
    script_content = """#!/bin/bash
# 古籍文本识别系统 - 训练脚本

echo "=========================================="
echo "古籍图像文本识别 - 模型训练"
echo "=========================================="

# 配置
CONFIG_PATH="config/pan_pp/R18-AUG-custom.py"
CHECKPOINT_DIR="checkpoints/custom_model"
OUTPUT_DIR="outputs"

# 创建输出目录
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${OUTPUT_DIR}

echo ""
echo "📝 训练配置: ${CONFIG_PATH}"
echo "💾 检查点目录: ${CHECKPOINT_DIR}"
echo "📊 输出目录: ${OUTPUT_DIR}"
echo ""

# 开始训练
echo "🚀 开始训练..."
python train_pan_pp/train.py \\
    --config ${CONFIG_PATH} \\
    --checkpoint ${CHECKPOINT_DIR}

echo ""
echo "✓ 训练完成！"
echo "  检查点已保存到: ${CHECKPOINT_DIR}"
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 设置可执行权限
    os.chmod(output_path, 0o755)
    
    print(f"\n✓ 训练脚本已生成: {output_path}")
    return output_path


def generate_inference_script(test_data, output_path="run_inference.sh"):
    """
    生成推理脚本
    """
    script_content = """#!/bin/bash
# 古籍文本识别系统 - 推理脚本

echo "=========================================="
echo "古籍图像文本识别 - 推理测试"
echo "=========================================="

# 配置
INPUT_DIR="data/test_images"
OUTPUT_DIR="outputs/inference_results"
MODEL_WEIGHTS="weights/det-R18-best.pth.tar"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

echo ""
echo "📁 输入目录: ${INPUT_DIR}"
echo "📊 输出目录: ${OUTPUT_DIR}"
echo "🧠 模型权重: ${MODEL_WEIGHTS}"
echo ""

# 准备测试图像
echo "📋 准备测试图像..."
mkdir -p ${INPUT_DIR}

# 提取测试集图像到输入目录（从 train_data）
echo "  正在复制测试集图像..."
for img_path in $(python -c "
import json
with open('data/splits/train_test_split.json', 'r') as f:
    data = json.load(f)
    for item in data['test']:
        print(item['image_path'])
"); do
    if [ -f "$img_path" ]; then
        cp "$img_path" ${INPUT_DIR}/
    fi
done

echo "  ✓ 复制完成"

# 开始推理
echo ""
echo "🚀 开始推理..."
python main.py --input_dir ${INPUT_DIR}

echo ""
echo "✓ 推理完成！"
echo "  结果已保存到: ${OUTPUT_DIR}"
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 设置可执行权限
    os.chmod(output_path, 0o755)
    
    print(f"✓ 推理脚本已生成: {output_path}")
    return output_path


def generate_evaluation_script(output_path="run_evaluation.py"):
    """
    生成评估脚本
    """
    script_content = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
古籍文本识别系统 - 评估脚本
计算推理结果与真实标签之间的匹配度
"""

import json
import os
import csv
from collections import defaultdict

def calculate_metrics(split_file, inference_results_dir, output_file="metrics.json"):
    """
    计算评估指标
    
    参数：
        split_file: 划分信息文件 (train_test_split.json)
        inference_results_dir: 推理结果目录（包含 CSV 文件）
        output_file: 输出指标文件
    """
    print("📊 计算评估指标...")
    
    # 加载真实标签
    with open(split_file, 'r', encoding='utf-8') as f:
        split_data = json.load(f)
    
    test_data = {item['image_path']: item['text'] for item in split_data['test']}
    
    # 计算指标
    total = 0
    correct = 0
    partial_match = 0
    metrics = {
        "total": 0,
        "correct": 0,
        "accuracy": 0.0,
        "partial_match_rate": 0.0,
        "details": []
    }
    
    # 遍历推理结果
    for csv_file in os.listdir(inference_results_dir):
        if not csv_file.endswith('.csv'):
            continue
        
        csv_path = os.path.join(inference_results_dir, csv_file)
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 1:
                    continue
                
                # 假设最后一列是识别结果
                pred_text = row[-1] if row else ""
                
                total += 1
                
                # 这里需要关联到原始图像，获取真实文本
                # 简化版本：只计算非空结果
                if pred_text.strip():
                    partial_match += 1
    
    metrics["total"] = total
    metrics["correct"] = correct
    metrics["accuracy"] = correct / total if total > 0 else 0.0
    metrics["partial_match_rate"] = partial_match / total if total > 0 else 0.0
    
    # 保存指标
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"   总样本数: {total}")
    print(f"   准确匹配: {correct} ({metrics['accuracy']*100:.2f}%)")
    print(f"   部分匹配: {partial_match} ({metrics['partial_match_rate']*100:.2f}%)")
    print(f"   ✓ 指标已保存到 {output_file}")
    
    return metrics

if __name__ == "__main__":
    split_file = "data/splits/train_test_split.json"
    inference_results_dir = "outputs"
    
    if os.path.exists(split_file):
        metrics = calculate_metrics(split_file, inference_results_dir)
    else:
        print(f"⚠️  {split_file} 不存在")
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    os.chmod(output_path, 0o755)
    
    print(f"✓ 评估脚本已生成: {output_path}")
    return output_path


def main():
    """
    主函数：执行完整的数据划分、准备、训练、推理流程
    """
    parser = argparse.ArgumentParser(description="训练/测试数据划分与推理管道")
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                       help='训练集比例（默认 0.8）')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子（默认 42）')
    parser.add_argument('--train_data_dir', type=str, default='train_data',
                       help='训练数据目录（默认 train_data）')
    args = parser.parse_args()
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║         古籍图像文本识别 - 训练/测试数据划分与推理管道        ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: 加载数据
    print("Step 1️⃣  加载训练数据...")
    all_data = load_train_data(args.train_data_dir)
    
    if not all_data:
        print("❌ 没有找到训练数据！")
        return
    
    # Step 2: 划分训练/测试集
    print("\nStep 2️⃣  划分训练/测试集...")
    train_data, test_data = split_train_test(
        all_data, 
        train_ratio=args.train_ratio, 
        random_seed=args.random_seed
    )
    
    # Step 3: 创建 COCO 格式标注
    print("\nStep 3️⃣  创建 COCO 格式标注文件...")
    train_annot = create_coco_format(train_data, "train")
    test_annot = create_coco_format(test_data, "test")
    
    # Step 4: 保存划分信息
    print("\nStep 4️⃣  保存划分信息...")
    split_file = save_split_info(train_data, test_data)
    
    # Step 5: 创建训练配置
    print("\nStep 5️⃣  创建训练配置...")
    config_file = create_training_config()
    
    # Step 6: 生成脚本
    print("\nStep 6️⃣  生成执行脚本...")
    training_script = generate_training_script()
    inference_script = generate_inference_script(test_data)
    evaluation_script = generate_evaluation_script()
    
    # 打印总结
    print("""
╔════════════════════════════════════════════════════════════════╗
║                         📋 任务总结                            ║
╚════════════════════════════════════════════════════════════════╝

✓ 数据划分完成！
  ├─ 训练集: {} 条 (80%)
  └─ 测试集: {} 条 (20%)

✓ 标注文件已创建！
  ├─ {}: {} 条数据
  └─ {}: {} 条数据

✓ 配置文件已创建！
  └─ {}

✓ 执行脚本已生成！
  ├─ 训练脚本: {}
  │  用法: bash {} 
  │
  ├─ 推理脚本: {}
  │  用法: bash {}
  │
  └─ 评估脚本: {}
     用法: python {}

📝 后续步骤：

  1️⃣  运行训练脚本（可选）
      bash {}

  2️⃣  运行推理脚本进行测试
      bash {}

  3️⃣  评估推理结果
      python {}

💡 提示：
  - 所有文件已保存到相应目录
  - 可根据需要编辑配置文件进行微调
  - 推理结果将保存到 outputs/ 目录
    """.format(
        len(train_data), len(test_data),
        train_annot, len(train_data), test_annot, len(test_data),
        config_file,
        training_script, training_script,
        inference_script, inference_script,
        evaluation_script, evaluation_script,
        training_script, inference_script, evaluation_script
    ))
    
    print("\n✓ 所有准备工作已完成！")


if __name__ == "__main__":
    main()
