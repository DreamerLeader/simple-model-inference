# Simple Model Inference Framework

一个轻量级的模型推理框架，支持 PyTorch 和 ONNX 模型，可在 CPU 和 GPU 上运行。

## 功能特性

- 🚀 支持多种模型格式（PyTorch, ONNX）
- 💻 支持 CPU 和 GPU 推理
- 🔧 可配置的预处理和后处理
- 📊 简单的性能监控
- 📝 清晰的代码结构和文档

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```python
from inference import InferenceEngine
from config import InferenceConfig

# 创建配置
config = InferenceConfig(
    model_path="path/to/your/model.pt",
    device="cuda"  # 或 "cpu"
)

# 初始化推理引擎
engine = InferenceEngine(config)

# 运行推理
result = engine.predict(input_data)
```

## 项目结构

```
simple-model-inference/
├── inference/          # 核心推理模块
│   ├── __init__.py
│   ├── engine.py       # 推理引擎
│   ├── loader.py       # 模型加载器
│   └── processor.py    # 预处理/后处理
├── examples/           # 示例代码
├── tests/              # 单元测试
├── config.py           # 配置类
├── requirements.txt    # 依赖
└── README.md           # 本文档
```

## 许可证

MIT License
