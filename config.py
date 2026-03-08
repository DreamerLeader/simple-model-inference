"""
配置管理模块
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class InferenceConfig:
    """推理配置类"""
    
    # 模型配置
    model_path: str
    model_type: str = "auto"  # "pytorch", "onnx", "auto"
    
    # 设备配置
    device: str = "cpu"  # "cpu", "cuda", "auto"
    
    # 推理配置
    batch_size: int = 1
    num_workers: int = 0
    
    # 预处理配置
    input_shape: Optional[tuple] = None
    normalize: bool = True
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    
    # 后处理配置
    top_k: int = 5
    threshold: float = 0.5
    
    # 性能配置
    enable_profiling: bool = False
    warmup_iterations: int = 3
    
    def __post_init__(self):
        """自动检测配置"""
        if self.model_type == "auto":
            if self.model_path.endswith(".pt") or self.model_path.endswith(".pth"):
                self.model_type = "pytorch"
            elif self.model_path.endswith(".onnx"):
                self.model_type = "onnx"
            else:
                raise ValueError(f"无法自动检测模型类型: {self.model_path}")
        
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
