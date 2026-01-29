# 📖 古籍图像文本识别系统 - 训练/推理完整指南

## 🎯 快速开始（3 步）

### 1️⃣ 划分数据

```bash
python quick_train_test_split.py
```

这会自动：
- ✓ 将 train_data 分成 80% 训练集和 20% 测试集
- ✓ 创建 COCO 格式的标注文件
- ✓ 复制测试集图像到 `data/test_images/`
- ✓ 保存划分信息到 `data/splits/train_test_split.json`

**预期输出：**
```
✓ 总数据量: 600 条
  • 训练集: 480 条 (80%)
  • 测试集: 120 条 (20%)

✓ 生成的文件：
  data/annotations/train.json
  data/annotations/test.json
  data/splits/train_test_split.json
  data/test_images/ (120 张测试图像)
```

---

### 2️⃣ 运行推理测试

```bash
python main.py --input_dir data/test_images
```

这会：
- ✓ 对测试集中的所有图像进行推理
- ✓ 生成识别结果（CSV 格式）
- ✓ 生成可视化图像（带文本框和识别结果）
- ✓ 保存到 `outputs/` 目录

**输出结果：**
```
outputs/
├── image_name_001.csv      # 识别结果：x1,y1,x2,...,文本
├── image_name_002.csv
├── ...
├── res_image_name_001.jpg  # 可视化：带文本框和识别文本的图像
├── res_image_name_002.jpg
└── ...
```

---

### 3️⃣ 查看结果

```bash
# 查看识别结果
cat outputs/*.csv

# 查看可视化图像
# （在文件管理器中打开 outputs/ 目录查看 res_*.jpg）
```

**CSV 文件格式：**
```
坐标信息, 识别文本
x1,y1,x2,y2,x3,y3,x4,y4, 识别的文本内容
10,20,100,20,100,50,10,50, 中华人民
```

---

## 📊 数据分析

### 查看划分信息

```bash
cat data/splits/train_test_split.json
```

输出示例：
```json
{
  "metadata": {
    "train_count": 480,
    "test_count": 120,
    "total_count": 600
  },
  "train": [...],
  "test": [...]
}
```

### 查看标注文件

```bash
# COCO 格式标注
cat data/annotations/train.json | python -m json.tool | head -50
```

---

## 🔄 完整工作流

### 场景 A：仅做推理测试（快速）

```bash
# 1. 划分数据（一次性）
python quick_train_test_split.py

# 2. 推理测试（使用预训练权重）
python main.py --input_dir data/test_images

# 3. 查看结果
ls -la outputs/
cat outputs/image_001.csv
```

**耗时：** ~1-2 分钟（依赖 GPU）

---

### 场景 B：从头训练新模型

```bash
# 1. 划分数据
python quick_train_test_split.py

# 2. 训练模型（需要 GPU，耗时较长）
python train_pan_pp/train.py --config config/pan_pp/R18-AUG.py

# 3. 推理测试（会使用新训练的权重）
python main.py --input_dir data/test_images

# 4. 查看结果和性能
ls -la outputs/
```

**耗时：** 2-4 小时（取决于数据量和 GPU 性能）

---

### 场景 C：对比实验

```bash
# 方案 A：轻量级模型 (R18)
python quick_train_test_split.py
python train_pan_pp/train.py --config config/pan_pp/R18-AUG.py
python main.py --input_dir data/test_images
mkdir results_R18
cp outputs/*.csv results_R18/
cp outputs/res_*.jpg results_R18/

# 方案 B：高精度模型 (R50)
python quick_train_test_split.py
python train_pan_pp/train.py --config config/pan_pp/R50-AUG.py
python main.py --input_dir data/test_images
mkdir results_R50
cp outputs/*.csv results_R50/
cp outputs/res_*.jpg results_R50/

# 对比两个结果
# 查看 results_R18/ 和 results_R50/ 中的识别精度和速度
```

---

## 🛠️ 高级配置

### 修改数据划分比例

编辑 `quick_train_test_split.py` 的这一行：

```python
split_idx = int(len(shuffled) * 0.8)  # 改为 0.7 表示 70% 训练
```

然后重新运行：
```bash
python quick_train_test_split.py
```

---

### 修改推理参数

编辑 `TestModel.py` 中的全局变量：

```python
# 选择推理方案（影响精度和速度）
PLAN = 'A'  # A: R50 (精度高), B: R18 (快速), C: R50 大尺寸 (最精确)

# 固定边长（影响处理速度）
FIXED_EDGE_LENGTH = 1024  # 减小以加快速度

# 调试模式（用于可视化）
DEBUG = False  # 改为 True 查看中间过程

# 处理方式
PROCESS_ON_SMALLER_PIC = True  # True: 在缩小图上处理 (快), False: 原图处理 (精确)
```

重新运行推理：
```bash
python main.py --input_dir data/test_images
```

---

## 📈 性能评估

### 计算推理精度

```bash
python run_evaluation.py
```

这会比较：
- 识别结果与真实标签的匹配度
- 计算准确率和匹配率
- 生成 `metrics.json` 性能报告

---

### 分析性能指标

生成的 `metrics.json` 包含：

```json
{
  "total": 120,           // 总测试图像数
  "correct": 95,          // 完全正确的个数
  "accuracy": 0.7917,     // 准确率
  "partial_match_rate": 0.9583,  // 部分匹配率
  "details": [...]
}
```

---

## 🐛 常见问题

### Q1: "找不到训练数据"

**A:** 检查以下几点：
```bash
# 1. train_data 目录是否存在
ls -la train_data/

# 2. JSON 文件是否存在
ls -la train_data/Dataset_A.json
ls -la train_data/Dataset_B.json
ls -la train_data/Dataset_C.json

# 3. 图像目录是否存在
ls -la train_data/Dataset_A/
ls -la train_data/Dataset_B/
ls -la train_data/Dataset_C/
```

---

### Q2: "推理结果为空"

**A:** 检查以下几点：
```bash
# 1. 测试图像是否存在
ls -la data/test_images/

# 2. 模型权重是否存在
ls -la weights/

# 3. 查看推理日志
python main.py --input_dir data/test_images 2>&1 | tee inference.log
```

---

### Q3: "CUDA 内存不足"

**A:** 解决方案：
```bash
# 1. 减小处理图像尺寸（在 TestModel.py 中修改）
FIXED_EDGE_LENGTH = 512  # 原来是 1024

# 2. 使用 CPU（会很慢）
# 在 TestModel.py 中修改
device = torch.device('cpu')  # 改为 CPU

# 3. 清理 GPU 缓存
python -c "import torch; torch.cuda.empty_cache()"
```

---

### Q4: "推理速度很慢"

**A:** 优化方案：
```python
# 在 TestModel.py 中修改

# 1. 使用轻量级模型
PLAN = 'B'  # R18 而不是 R50

# 2. 减小图像尺寸
FIXED_EDGE_LENGTH = 512  # 原来是 1024

# 3. 在缩小的图上处理
PROCESS_ON_SMALLER_PIC = True

# 4. 增加批处理大小（推理时）
batch_size = 128  # 在数据加载器中修改
```

---

## 📁 目录结构

完成数据划分后的项目结构：

```
project_root/
├── train_data/                    # 原始数据
│   ├── Dataset_A/
│   ├── Dataset_B/
│   ├── Dataset_C/
│   ├── Dataset_A.json
│   ├── Dataset_B.json
│   └── Dataset_C.json
│
├── data/
│   ├── annotations/
│   │   ├── train.json            # COCO 格式训练标注
│   │   └── test.json             # COCO 格式测试标注
│   ├── splits/
│   │   └── train_test_split.json # 划分信息
│   ├── test_images/              # 测试集图像
│   └── ...
│
├── outputs/                       # 推理结果
│   ├── image_001.csv            # 识别结果
│   ├── res_image_001.jpg        # 可视化
│   └── ...
│
├── checkpoints/                   # 模型权重（如果训练）
├── quick_train_test_split.py     # 数据划分工具
├── train_test_pipeline.py        # 完整管道脚本
├── TRAINING_GUIDE.md             # 详细指南
└── main.py                        # 推理入口
```

---

## 💡 最佳实践

### 1. 保存实验日志

```bash
# 推理时保存日志
python main.py --input_dir data/test_images 2>&1 | tee inference_$(date +%Y%m%d_%H%M%S).log

# 查看日志
cat inference_20260129_070000.log
```

### 2. 版本管理

```bash
# 为每个实验创建版本目录
mkdir -p results/v1_baseline
mkdir -p results/v2_enhanced
mkdir -p results/v3_optimized

# 运行不同的实验并保存结果
python main.py --input_dir data/test_images
cp outputs/*.csv results/v1_baseline/
cp outputs/res_*.jpg results/v1_baseline/
```

### 3. 性能对比

```bash
# 创建对比脚本
cat > compare_results.py << 'EOF'
import os
import csv

for version in ['v1_baseline', 'v2_enhanced', 'v3_optimized']:
    csv_count = len([f for f in os.listdir(f'results/{version}') if f.endswith('.csv')])
    print(f"{version}: {csv_count} CSV 文件")
EOF

python compare_results.py
```

---

## 🚀 快速命令汇总

```bash
# 一键执行完整流程
python quick_train_test_split.py && python main.py --input_dir data/test_images

# 只查看结果
ls -la outputs/

# 查看统计信息
cat data/splits/train_test_split.json | python -m json.tool

# 评估性能
python run_evaluation.py

# 清理输出
rm -rf outputs/*
```

---

## 📞 需要帮助？

如果遇到问题：

1. **查看日志** - 完整的错误信息通常在日志中
2. **检查路径** - 确保所有文件路径正确
3. **查看文档** - 参考项目中的 README.md 和相关注释
4. **测试单张** - 用 `TestModel.py` 对单张图像进行测试

---

**✅ 祝你使用愉快！**
