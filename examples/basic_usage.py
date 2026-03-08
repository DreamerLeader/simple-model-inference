"""
使用示例
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import InferenceConfig
from inference import InferenceEngine
import numpy as np


def example_basic_usage():
    """基本使用示例"""
    print("=" * 50)
    print("基本使用示例")
    print("=" * 50)
    
    # 创建配置
    config = InferenceConfig(
        model_path="dummy_model.pt",  # 替换为实际模型路径
        model_type="pytorch",
        device="cpu",
        input_shape=(1, 3, 224, 224),
        enable_profiling=True
    )
    
    # 初始化引擎
    engine = InferenceEngine(config)
    
    # 打印模型信息
    print("\n模型信息:")
    for key, value in engine.get_model_info().items():
        print(f"  {key}: {value}")
    
    # 创建随机输入数据进行演示
    dummy_input = np.random.rand(224, 224, 3) * 255
    dummy_input = dummy_input.astype(np.uint8)
    
    # 运行推理
    print("\n运行推理...")
    result = engine.predict(dummy_input)
    
    print("\n推理结果:")
    print(f"  预测类别: {result['top_class']}")
    print(f"  置信度: {result['top_confidence']:.4f}")
    print(f"  Top-{len(result['indices'])} 类别: {result['indices']}")
    print(f"  概率: {[f'{p:.4f}' for p in result['probabilities']]}")
    
    # 打印性能统计
    engine.print_profiling_stats()


def example_batch_inference():
    """批量推理示例"""
    print("\n" + "=" * 50)
    print("批量推理示例")
    print("=" * 50)
    
    config = InferenceConfig(
        model_path="dummy_model.pt",
        model_type="pytorch",
        device="cpu",
        input_shape=(1, 3, 224, 224),
        enable_profiling=True
    )
    
    engine = InferenceEngine(config)
    
    # 创建批量输入
    batch_size = 4
    inputs = [np.random.rand(224, 224, 3).astype(np.uint8) for _ in range(batch_size)]
    
    print(f"\n批量推理 {batch_size} 个样本...")
    results = engine.batch_predict(inputs)
    
    print(f"\n批量推理结果:")
    for i, result in enumerate(results):
        print(f"  样本 {i+1}: 类别={result['top_class']}, 置信度={result['top_confidence']:.4f}")


def example_with_image():
    """图像推理示例"""
    print("\n" + "=" * 50)
    print("图像推理示例")
    print("=" * 50)
    
    print("""
    # 从文件加载图像进行推理
    from PIL import Image
    
    config = InferenceConfig(
        model_path="path/to/your/model.pt",
        device="cuda",  # 使用 GPU
        input_shape=(1, 3, 224, 224)
    )
    
    engine = InferenceEngine(config)
    
    # 加载图像
    image = Image.open("path/to/image.jpg")
    
    # 推理
    result = engine.predict(image)
    print(f"预测结果: 类别 {result['top_class']}, 置信度 {result['top_confidence']:.4f}")
    """)


def example_onnx_inference():
    """ONNX 模型推理示例"""
    print("\n" + "=" * 50)
    print("ONNX 模型推理示例")
    print("=" * 50)
    
    print("""
    # ONNX 模型推理
    config = InferenceConfig(
        model_path="path/to/model.onnx",
        model_type="onnx",  # 明确指定为 ONNX
        device="cpu",
        input_shape=(1, 3, 224, 224)
    )
    
    engine = InferenceEngine(config)
    
    # 查看 ONNX 模型信息
    info = engine.get_model_info()
    print("输入信息:", info.get("inputs"))
    print("输出信息:", info.get("outputs"))
    """)


if __name__ == "__main__":
    # 运行示例
    example_basic_usage()
    example_batch_inference()
    example_with_image()
    example_onnx_inference()
    
    print("\n" + "=" * 50)
    print("示例运行完成!")
    print("=" * 50)
