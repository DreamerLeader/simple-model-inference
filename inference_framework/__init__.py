"""
Simple Model Inference Framework

A lightweight framework for model inference supporting PyTorch and ONNX.
"""

from .model_loader import ModelLoader
from .inference_engine import InferenceEngine
from .preprocess import Preprocessor
from .postprocess import Postprocessor

__version__ = "0.1.0"
__all__ = ["ModelLoader", "InferenceEngine", "Preprocessor", "Postprocessor"]
