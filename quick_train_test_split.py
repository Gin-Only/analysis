#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版：train_data 数据划分与推理
直接可运行，无需 bash 环境
"""

import json
import os
import random
import shutil
from pathlib import Path

def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║    古籍文本识别 - 快速数据划分与推理工具                ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # ===== Step 1: 加载数据 =====
    print("📖 Step 1: 加载训练数据...\n")
    
    all_data = {}
    train_data_dir = "train_data"
    
    for dataset_name in ['Dataset_A', 'Dataset_B', 'Dataset_C']:
        json_file = os.path.join(train_data_dir, f"{dataset_name}.json")
        
        if not os.path.exists(json_file):
            print(f"   ⚠️  {json_file} 不存在")
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 确保每个 item 的 image_path 包含完整路径
            for item in data:
                if not item['image_path'].startswith('train_data'):
                    item['image_path'] = os.path.join('train_data', item['image_path'])
            
            all_data[dataset_name] = data
            print(f"   ✓ {dataset_name}: {len(data)} 条数据")
        except Exception as e:
            print(f"   ❌ 读取 {json_file} 失败: {e}")
    
    if not all_data:
        print("\n❌ 没有找到任何数据!")
        return
    
    total_items = sum(len(v) for v in all_data.values())
    print(f"\n   总计: {total_items} 条数据\n")
    
    # ===== Step 2: 划分训练/测试集 =====
    print("📊 Step 2: 划分训练/测试集 (80% / 20%)...\n")
    
    random.seed(42)
    train_data = []
    test_data = []
    
    for dataset_name, items in all_data.items():
        shuffled = items.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * 0.8)
        train_subset = shuffled[:split_idx]
        test_subset = shuffled[split_idx:]
        
        train_data.extend(train_subset)
        test_data.extend(test_subset)
        
        print(f"   {dataset_name}:")
        print(f"      训练: {len(train_subset)} 条")
        print(f"      测试: {len(test_subset)} 条")
    
    print(f"\n   📌 总训练集: {len(train_data)} 条")
    print(f"   📌 总测试集: {len(test_data)} 条\n")
    
    # ===== Step 3: 创建数据目录 =====
    print("📁 Step 3: 创建数据目录...\n")
    
    os.makedirs("data/annotations", exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)
    os.makedirs("data/test_images", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    print("   ✓ 目录结构已创建\n")
    
    # ===== Step 4: 保存划分信息 =====
    print("💾 Step 4: 保存划分信息...\n")
    
    split_info = {
        "train": train_data,
        "test": test_data,
        "metadata": {
            "train_count": len(train_data),
            "test_count": len(test_data),
            "total_count": len(train_data) + len(test_data)
        }
    }
    
    split_file = "data/splits/train_test_split.json"
    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    print(f"   ✓ 划分信息已保存到: {split_file}\n")
    
    # ===== Step 5: 创建 COCO 格式标注 =====
    print("📝 Step 5: 创建 COCO 格式标注文件...\n")
    
    def create_coco_format(data, dataset_type):
        """创建 COCO 格式的标注"""
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "text"}]
        }
        
        image_id = 1
        annotation_id = 1
        valid_count = 0
        
        for item in data:
            image_path = item['image_path']
            text = item.get('text', '')  # 如果没有 'text' 字段，使用空字符串
            
            # 检查图像是否存在
            if not os.path.exists(image_path):
                continue
            
            try:
                import cv2
                img = cv2.imread(image_path)
                if img is None:
                    continue
                h, w = img.shape[:2]
            except:
                continue
            
            # 添加图像信息
            coco_data['images'].append({
                "id": image_id,
                "file_name": image_path,
                "height": h,
                "width": w
            })
            
            # 添加标注信息
            coco_data['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [0, 0, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [],
                "text": text
            })
            
            image_id += 1
            annotation_id += 1
            valid_count += 1
        
        return coco_data, valid_count
    
    # 创建训练集标注
    train_coco, train_count = create_coco_format(train_data, "train")
    train_annot_file = "data/annotations/train.json"
    with open(train_annot_file, 'w', encoding='utf-8') as f:
        json.dump(train_coco, f, ensure_ascii=False, indent=2)
    print(f"   ✓ 训练标注: {train_annot_file} ({train_count} 条)")
    
    # 创建测试集标注
    test_coco, test_count = create_coco_format(test_data, "test")
    test_annot_file = "data/annotations/test.json"
    with open(test_annot_file, 'w', encoding='utf-8') as f:
        json.dump(test_coco, f, ensure_ascii=False, indent=2)
    print(f"   ✓ 测试标注: {test_annot_file} ({test_count} 条)\n")
    
    # ===== Step 6: 复制测试图像 =====
    print("📋 Step 6: 复制测试集图像...\n")
    
    test_image_count = 0
    for item in test_data:
        image_path = item['image_path']
        if os.path.exists(image_path):
            try:
                filename = os.path.basename(image_path)
                dest_path = os.path.join("data/test_images", filename)
                shutil.copy(image_path, dest_path)
                test_image_count += 1
            except Exception as e:
                print(f"   ⚠️  复制 {image_path} 失败: {e}")
    
    print(f"   ✓ 已复制 {test_image_count} 张测试图像到 data/test_images/\n")
    
    # ===== Step 7: 生成推理命令 =====
    print("🚀 Step 7: 生成推理命令...\n")
    
    inference_cmd = f"""
# 运行以下命令进行推理测试：

python main.py --input_dir data/test_images

# 输出结果将保存到 outputs/ 目录
# - *.csv 文件：识别结果（坐标 + 文本）
# - res_*.jpg 文件：可视化结果
"""
    
    print(inference_cmd)
    
    # ===== 总结 =====
    print("""
╔═══════════════════════════════════════════════════════════╗
║                      ✓ 任务完成！                        ║
╚═══════════════════════════════════════════════════════════╝

📊 数据统计：
   • 总数据量: {} 条
   • 训练集: {} 条 (80%)
   • 测试集: {} 条 (20%)

📁 生成的文件：
   ✓ data/annotations/train.json        训练标注（COCO 格式）
   ✓ data/annotations/test.json         测试标注（COCO 格式）
   ✓ data/splits/train_test_split.json  划分信息
   ✓ data/test_images/                  复制的测试集图像

📝 后续步骤：

  1️⃣  运行推理测试
      python main.py --input_dir data/test_images

  2️⃣  查看结果
      ls -la outputs/

  3️⃣  查看识别结果详情
      cat outputs/image_001.csv

  4️⃣  查看可视化图像
      显示 outputs/res_*.jpg

💡 说明：
  • 推理结果保存到 outputs/ 目录
  • *.csv 文件包含坐标和识别文本
  • res_*.jpg 是带标注的可视化图像
  • 测试图像已复制到 data/test_images/

✅ 所有准备工作已完成！
    """.format(
        total_items, 
        len(train_data), 
        len(test_data)
    ))

if __name__ == "__main__":
    main()
