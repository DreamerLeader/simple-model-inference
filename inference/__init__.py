"""
Simple Model Inference Framework
一个轻量级的模型推理框架
"""

from .engine import InferenceEngine
from .loader import ModelLoader
from .processor import PreProcessor, PostProcessor

__version__ = "0.1.0"
__all__ = ["InferenceEngine", "ModelLoader", "PreProcessor", "PostProcessor"]
