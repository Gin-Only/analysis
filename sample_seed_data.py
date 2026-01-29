#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import random
import os
import sys

if __name__ == "__main__":
    random.seed(42)
    
    train_data_dir = "train_data"
    datasets = ["Dataset_A", "Dataset_B", "Dataset_C"]
    sample_size = 50
    all_samples = []
    
    for dataset in datasets:
        json_file = os.path.join(train_data_dir, f"{dataset}.json")
        
        print(f"正在处理 {dataset}...")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"  总数据量: {len(data)}")
        
        samples = random.sample(data, min(sample_size, len(data)))
        
        print(f"  已采样: {len(samples)}")
        
        for sample in samples:
            sample['dataset'] = dataset
        
        all_samples.extend(samples)
    
    output_dir = "seed_data_for_review"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "seed_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 种子数据已保存到: {output_file}")
    print(f"✓ 总共采样: {len(all_samples)} 条数据")
    print(f"✓ 每个数据集采样: {sample_size} 条")
    
    output_summary = os.path.join(output_dir, "summary.txt")
    with open(output_summary, 'w', encoding='utf-8') as f:
        f.write("种子数据采样摘要\n")
        f.write("=" * 50 + "\n\n")
        for dataset in datasets:
            count = sum(1 for s in all_samples if s['dataset'] == dataset)
            f.write(f"{dataset}: {count} 条\n")
        f.write(f"\n总计: {len(all_samples)} 条数据\n")
        f.write("\n用途: 人工检查标注错误率\n")
    
    print(f"✓ 摘要已保存到: {output_summary}")
