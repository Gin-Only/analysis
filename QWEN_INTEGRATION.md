# Qwen2.5-VL-7B 集成说明

## 概述

本项目已成功集成 Qwen2.5-VL-7B 视觉-语言模型作为文本识别的新选项，与原有的 CRNN 模型并存，用户可以根据需要选择使用。

## 变更说明

### 原有流程
```
输入图片 → PAN++ 检测 → 文本框坐标 → ROI 提取 + 预处理 → CRNN 识别 → 排序 + 后处理 → CSV 输出
```

### 改造后流程
```
输入图片 → PAN++ 检测 → 文本框坐标 → ROI 提取 + 预处理 → Qwen2.5-VL-7B 识别 → 排序 + 后处理 → CSV 输出
```

**核心变化**：第5步文本识别模块支持两种模型选择。

## 文件修改清单

### 1. `requirements.txt`
新增依赖：
- `transformers>=4.37.0` - Hugging Face Transformers 库
- `accelerate` - 模型加载加速
- `qwen-vl-utils` - Qwen 视觉处理工具

### 2. `Infer_Utils.py`
新增函数（约 145 行）：
- `QWEN_OCR_PROMPT` - OCR 提示词常量
- `load_qwen_model()` - 加载 Qwen2.5-VL-7B 模型和处理器
- `qwen_rec()` - 单张图像识别
- `qwen_seq_rec()` - 批量图像识别

### 3. `TestModel.py`
修改内容（约 60 行变更）：
- 新增 `USE_QWEN` 和 `QWEN_MODEL_NAME` 配置参数
- 修改 `Infer.__init__()` 支持条件加载不同模型
- 修改 `test()` 函数签名，增加 `use_qwen` 和 `qwen_processor` 参数
- 更新 DEBUG 和非 DEBUG 模式的识别代码路径

### 4. `README.md`
文档更新：
- 更新功能特点说明
- 新增模型选择章节
- 更新算法流程说明

## 使用方法

### 选择识别模型

在 `TestModel.py` 中设置：

```python
# 使用 CRNN 模型（默认，向后兼容）
USE_QWEN = False

# 使用 Qwen2.5-VL-7B 模型（新功能）
USE_QWEN = True
QWEN_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"  # 或本地模型路径
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行推理

使用与之前相同的命令：

```bash
# 批量处理
python main.py

# 单张图像
python TestModel.py --image_path /path/to/image.jpg
```

## 模型对比

| 特性 | CRNN | Qwen2.5-VL-7B |
|------|------|---------------|
| **模型大小** | 约 50MB | 约 7GB |
| **推理速度** | 快 | 较慢 |
| **识别准确率** | 良好 | 优秀 |
| **泛化能力** | 中等 | 强 |
| **依赖** | 少 | 多（transformers） |
| **硬件要求** | 低 | 高（建议 GPU） |
| **适用场景** | 快速批量处理 | 高精度识别 |

## 技术实现细节

### 模型加载
- 使用 Hugging Face Transformers 库自动下载和加载模型
- 首次使用会自动从 HuggingFace Hub 下载模型（约 7GB）
- 支持本地模型路径以避免重复下载

### 推理流程
1. 将 ROI 图像转换为 PIL Image 格式
2. 构建包含图像和 OCR 提示词的消息
3. 使用 processor 处理输入
4. 调用模型生成文本
5. 解码并返回识别结果

### 批量处理优化
- 虽然 Qwen 模型较大，但通过 `qwen_seq_rec` 函数仍支持批量处理
- 每个 patch 独立推理以确保稳定性
- 可根据显存大小调整 batch size

## 注意事项

1. **首次运行**：首次使用 Qwen 模型时会自动下载，需要稳定的网络连接
2. **硬件要求**：Qwen2.5-VL-7B 建议使用 GPU，否则推理速度会很慢
3. **显存需求**：至少需要 16GB 显存以加载 7B 模型
4. **向后兼容**：默认使用 CRNN 模型，不影响现有用户
5. **提示词调整**：可通过修改 `QWEN_OCR_PROMPT` 常量调整 OCR 提示词

## 故障排除

### 问题：模型下载失败
**解决方案**：
- 检查网络连接
- 使用镜像站点：`export HF_ENDPOINT=https://hf-mirror.com`
- 手动下载模型到本地，然后设置 `QWEN_MODEL_NAME` 为本地路径

### 问题：显存不足
**解决方案**：
- 减小 batch size
- 使用量化版本的模型
- 切换回 CRNN 模型

### 问题：识别结果不理想
**解决方案**：
- 调整 `QWEN_OCR_PROMPT` 提示词
- 检查输入图像质量
- 尝试调整 `max_new_tokens` 参数

## 未来改进方向

1. **模型量化**：支持 INT8/INT4 量化以减少显存占用
2. **批量优化**：实现真正的批量推理以提升效率
3. **提示词工程**：针对古籍文本优化提示词
4. **模型微调**：基于古籍数据集微调 Qwen 模型
5. **多模型集成**：支持更多视觉-语言模型选择

## 贡献

如有问题或建议，欢迎提交 Issue 或 Pull Request。

## 参考资料

- [Qwen2-VL GitHub](https://github.com/QwenLM/Qwen2-VL)
- [Qwen2-VL HuggingFace](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [Transformers 文档](https://huggingface.co/docs/transformers)
