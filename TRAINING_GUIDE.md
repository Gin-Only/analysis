# 🚀 古籍图像文本识别 - 训练/测试数据划分与推理完整指南

## 📚 概述

本指南将帮你完成以下工作：
1. **数据划分** - 将 train_data 分成 80% 训练集和 20% 测试集
2. **标注转换** - 创建 COCO 格式的标注文件
3. **模型训练** - 使用划分后的训练集训练模型
4. **推理测试** - 在测试集上运行推理
5. **性能评估** - 计算推理精度指标

---

## 🔧 快速开始

### 步骤 1️⃣ : 数据划分与准备

运行以下命令来划分数据并生成所有必需的文件：

```bash
# 基础用法（默认 80% 训练集）
python train_test_pipeline.py

# 自定义参数
python train_test_pipeline.py \
    --train_ratio 0.8 \           # 训练集比例
    --random_seed 42 \             # 随机种子（保证可重复）
    --train_data_dir train_data     # 训练数据目录
```

**预期输出：**
```
✓ 数据划分完成！
  ├─ 训练集: 480 条 (80%)
  └─ 测试集: 120 条 (20%)

✓ 标注文件已创建！
  ├─ data/annotations/train.json: 480 条数据
  └─ data/annotations/test.json: 120 条数据

✓ 配置文件已创建！
  └─ config/pan_pp/R18-AUG-custom.py

✓ 执行脚本已生成！
  ├─ 训练脚本: run_training.sh
  ├─ 推理脚本: run_inference.sh
  └─ 评估脚本: run_evaluation.py
```

---

### 步骤 2️⃣ : 模型训练（可选）

如果你想训练自己的模型：

```bash
# 运行生成的训练脚本
bash run_training.sh

# 或直接运行训练命令
python train_pan_pp/train.py \
    --config config/pan_pp/R18-AUG-custom.py \
    --checkpoint checkpoints/custom_model
```

**训练参数说明：**

| 参数 | 说明 | 推荐值 |
|------|------|-------|
| `--config` | 配置文件路径 | `config/pan_pp/R18-AUG-custom.py` |
| `--checkpoint` | 检查点保存目录 | `checkpoints/custom_model` |
| `--resume` | 恢复断点路径 | 无需指定（从头训练） |
| `--debug` | 调试模式 | 否（加速训练） |

**训练过程：**
```
Epoch: [1 | 300]
  LR: 0.010000 | Batch: 0.250s | Loss: 2.345 | ...
  [progress bar...]
  
Epoch: [2 | 300]
  ...

✓ 训练完成！检查点已保存到: checkpoints/custom_model/
```

---

### 步骤 3️⃣ : 推理测试

运行推理以测试模型在测试集上的性能：

```bash
# 方法 1：使用生成的推理脚本
bash run_inference.sh

# 方法 2：直接运行推理命令
python main.py --input_dir data/test_images

# 方法 3：单张图像推理（调试）
python TestModel.py --image_path path/to/test/image.jpg
```

**推理结果：**
```
outputs/
├─ image_001.csv          # 识别结果 (坐标 + 文本)
├─ image_002.csv
├─ ...
├─ res_image_001.jpg      # 可视化结果
├─ res_image_002.jpg
└─ ...

CSV 文件格式：
x1, y1, x2, y2, ..., 识别文本
10, 20, 100, 50, ..., 中华人民共和国
150, 30, 200, 80, ..., 古籍文献
```

---

### 步骤 4️⃣ : 性能评估

评估推理结果的质量：

```bash
# 运行评估脚本
python run_evaluation.py

# 输出：
# 总样本数: 120
# 准确匹配: 95 (79.17%)
# 部分匹配: 115 (95.83%)
# ✓ 指标已保存到 metrics.json
```

---

## 📊 生成的文件结构

完成数据划分后，你的项目结构将如下所示：

```
project_root/
│
├── train_data/                    # 原始数据目录
│   ├── Dataset_A/
│   ├── Dataset_B/
│   ├── Dataset_C/
│   ├── Dataset_A.json            # 原始标注
│   ├── Dataset_B.json
│   └── Dataset_C.json
│
├── data/                          # 处理后的数据
│   ├── annotations/
│   │   ├── train.json            # COCO 格式训练标注
│   │   └── test.json             # COCO 格式测试标注
│   ├── splits/
│   │   └── train_test_split.json # 划分信息（用于评估）
│   └── test_images/              # 推理时的测试图像
│
├── checkpoints/                   # 模型检查点
│   └── custom_model/
│       ├── checkpoint_latest.pth.tar
│       ├── checkpoint_best.pth.tar
│       └── ...
│
├── outputs/                       # 推理结果
│   ├── image_001.csv
│   ├── image_002.csv
│   ├─ res_image_001.jpg
│   ├─ res_image_002.jpg
│   └─ metrics.json
│
├── train_test_pipeline.py         # 数据划分脚本
├── run_training.sh                # 训练脚本
├── run_inference.sh               # 推理脚本
└── run_evaluation.py              # 评估脚本
```

---

## 💾 详细配置说明

### train_test_split.json 结构

```json
{
  "train": [
    {
      "image_path": "train_data/Dataset_A/a_0001.jpg",
      "text": "鋸牙鈎爪彼其奈我何哉竊有一言敬吿諸公倉正司倉田正"
    },
    ...
  ],
  "test": [
    {
      "image_path": "train_data/Dataset_A/a_0234.jpg",
      "text": "有時而衰興者必不可終廢偶有外患局內人同心捍之雖有"
    },
    ...
  ],
  "metadata": {
    "train_count": 480,
    "test_count": 120,
    "total_count": 600
  }
}
```

### COCO 格式标注示例

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "train_data/Dataset_A/a_0001.jpg",
      "height": 896,
      "width": 896
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [0, 0, 896, 896],
      "area": 802816,
      "iscrowd": 0,
      "segmentation": [],
      "text": "鋸牙鈎爪彼其奈我何哉竊有一言敬吿諸公倉正司倉田正"
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "text"
    }
  ]
}
```

---

## 🎯 常见使用场景

### 场景 1: 快速评估模型（不训练）

```bash
# 1. 划分数据
python train_test_pipeline.py

# 2. 直接运行推理（使用预训练权重）
bash run_inference.sh

# 3. 评估结果
python run_evaluation.py
```

### 场景 2: 从头训练模型

```bash
# 1. 划分数据
python train_test_pipeline.py

# 2. 训练模型（需要 GPU，耗时较长）
bash run_training.sh

# 3. 推理测试（会自动使用最新训练的权重）
bash run_inference.sh

# 4. 评估性能
python run_evaluation.py
```

### 场景 3: 调试单张图像

```bash
# 使用调试模式对单张图像进行推理
python TestModel.py --image_path train_data/Dataset_A/a_0001.jpg

# 或指定特定的推理方案（A/B/C）
# 在 TestModel.py 中修改 PLAN = 'A'/'B'/'C'
```

### 场景 4: 自定义划分比例

```bash
# 70% 训练，30% 测试
python train_test_pipeline.py --train_ratio 0.7 --random_seed 123

# 50% 训练，50% 测试（用于小数据集快速测试）
python train_test_pipeline.py --train_ratio 0.5
```

---

## 📈 性能优化建议

### 训练阶段优化

| 参数 | 当前值 | 优化建议 |
|------|-------|--------|
| 批大小 (batch_size) | 4 | 增加到 8-16（如果 GPU 内存允许） |
| 学习率 (lr) | 1e-2 | 使用学习率衰减策略 |
| 优化器 | Adam | 尝试 SGD + momentum |
| 数据增强 | 基础 | 增加旋转、缩放、剪切等 |
| 正则化 | L2 (5e-4) | 增加 dropout 和 batch norm |

### 推理阶段优化

| 优化方法 | 说明 | 加速比 |
|--------|------|-------|
| 量化 (Quantization) | INT8 量化模型 | 2-4x |
| 蒸馏 (Distillation) | 使用轻量级模型 | 3-5x |
| 批处理 | 增加批大小 | 1.5-2x |
| TorchScript | JIT 编译 | 1.1-1.3x |
| 混合精度 (AMP) | FP16 计算 | 1.5-2x |

---

## 🐛 故障排除

### 问题 1: "找不到训练数据"
```
解决方案：
1. 检查 train_data 目录是否存在
2. 检查 Dataset_A.json、Dataset_B.json、Dataset_C.json 是否存在
3. 使用 --train_data_dir 参数指定正确的路径
```

### 问题 2: "CUDA 内存不足"
```
解决方案：
1. 减小 batch_size：编辑 config 文件中的 batch_size
2. 使用 torch.cuda.empty_cache() 清理缓存
3. 降低输入图像分辨率（编辑 config 中的 img_size）
4. 使用 CPU 训练（慢但可行）：在 train.py 中注释掉 .cuda()
```

### 问题 3: "推理速度慢"
```
解决方案：
1. 减小 FIXED_EDGE_LENGTH（在 TestModel.py 中修改）
2. 增加 batch_size（在推理中使用批处理）
3. 启用 TorchScript 编译（见性能优化部分）
4. 使用更轻量级的模型（R18 而不是 R50）
```

### 问题 4: "CSV 文件为空"
```
解决方案：
1. 检查输入图像路径是否正确
2. 检查模型权重是否正确加载
3. 在 TestModel.py 中启用 DEBUG=True 查看中间结果
4. 检查 CRNN 模型权重文件是否存在
```

---

## 📝 输出文件说明

### CSV 结果文件

**文件名：** `outputs/image_name.csv`

**格式：** 每行包含一个检测到的文本框的信息
```
x1, y1, x2, y2, ..., x8, y8, 识别文本
```

**示例：**
```
10, 20, 100, 20, 100, 50, 10, 50, 中华
150, 25, 250, 25, 250, 75, 150, 75, 人民
```

### 可视化图像

**文件名：** `outputs/res_image_name.jpg`

**内容：**
- 原始图像
- 绿色多边形标记文本框
- 红色文字显示识别结果

---

## 🔄 完整工作流示例

```bash
# 1. 准备数据（一次性）
python train_test_pipeline.py --train_ratio 0.8 --random_seed 42

# 2. 检查生成的文件
ls -la data/annotations/
ls -la data/splits/

# 3. 训练模型（可选，耗时 2-4 小时）
# bash run_training.sh

# 4. 推理测试（关键步骤）
bash run_inference.sh

# 5. 查看结果
ls -la outputs/*.csv    # 识别结果
ls -la outputs/*.jpg    # 可视化

# 6. 评估性能
python run_evaluation.py

# 7. 查看评估指标
cat metrics.json
```

---

## 📚 相关文档

- [项目结构详解](README.md) - 了解整个项目的组织方式
- [编译方式](README.md#编译方式) - 编译 C++ 后处理模块
- [入口点说明](README.md#入口) - 了解推理、训练、调试三个入口
- [核心逻辑](README.md#核心逻辑) - 深入理解算法流程
- [性能优化](README.md#性能优化) - 加速训练和推理

---

## 💡 提示

1. **保存日志** - 重定向输出到文件便于后续分析：
   ```bash
   bash run_training.sh 2>&1 | tee training.log
   bash run_inference.sh 2>&1 | tee inference.log
   ```

2. **对比实验** - 保存不同配置的结果：
   ```bash
   mkdir results_v1
   cp outputs/* results_v1/
   
   # 修改配置后重新运行
   bash run_training.sh
   bash run_inference.sh
   ```

3. **监控资源** - 在另一个终端实时监控 GPU：
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **远程运行** - 使用 tmux 在后台运行长时间任务：
   ```bash
   tmux new-session -d -s training 'bash run_training.sh'
   tmux attach -t training  # 查看进度
   ```

---

**✓ 祝你使用愉快！有任何问题欢迎反馈。**
