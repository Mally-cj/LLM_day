# 中文诗歌生成模型

基于 GPT-2 模型和 Chinese-Poems 数据集的中文诗歌生成系统，支持训练和推理。

## 功能特点

- 🎯 基于 GPT-2 模型微调
- 📚 使用 Million/Chinese-Poems 数据集
- 🚀 支持训练、生成和交互式模式
- 💡 智能诗歌续写功能
- 🔧 完整的训练和推理流程

## 文件说明

- `poetry_trainer.py` - 主要的训练和推理脚本
- `quick_test.py` - 快速测试脚本，验证环境和数据
- `gpt2.py` - 原始的训练代码（已修复）
- `requirements.txt` - 依赖包列表

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 快速测试

首先运行快速测试，验证环境和数据集：

```bash
python quick_test.py
```

### 2. 训练模型

```bash
# 使用默认参数训练
python poetry_trainer.py --mode train

# 自定义训练参数
python poetry_trainer.py --mode train --epochs 5 --batch-size 8 --learning-rate 3e-5
```

### 3. 生成诗歌

```bash
# 使用训练好的模型生成诗歌
python poetry_trainer.py --mode generate
```

### 4. 交互式生成

```bash
# 启动交互式诗歌生成
python poetry_trainer.py --mode interactive
```

## 训练参数说明

- `--epochs`: 训练轮数（默认：3）
- `--batch-size`: 批次大小（默认：4）
- `--learning-rate`: 学习率（默认：5e-5）
- `--model-path`: 模型保存路径（默认：./poetry_model）

## 数据集格式

数据集包含以下字段：
- `instruction`: 指令描述
- `input`: 输入要求
- `output`: 诗歌输出

## 生成示例

训练完成后，模型可以生成各种风格的诗歌：

```
指令: 你是一个诗词创作的AI助手
要求: 请你模仿李白的风格，创作一首关于月亮的七言绝句

生成的诗词:
明月几时有，把酒问青天。
不知天上宫阙，今夕是何年。
```

## 注意事项

1. 首次运行会下载数据集和模型，需要一定时间
2. 训练需要 GPU 支持以获得更好的效果
3. 模型文件较大，请确保有足够的存储空间
4. 建议先运行 `quick_test.py` 验证环境

## 故障排除

如果遇到问题，请检查：

1. 网络连接是否正常（需要下载模型和数据集）
2. Python 版本是否 >= 3.7
3. 依赖包是否正确安装
4. GPU 内存是否足够（如果使用 GPU）

## 技术细节

- 模型：GPT-2 (openai-community/gpt2)
- 数据集：Million/Chinese-Poems
- 框架：PyTorch + Transformers
- 训练策略：指令微调 (Instruction Tuning)
