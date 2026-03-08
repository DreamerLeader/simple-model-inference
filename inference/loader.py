"""
模型加载器模块
支持 PyTorch 和 ONNX 模型加载
"""

import torch
import onnxruntime as ort
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """模型加载器"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.model_type = None
        
    def load(self, model_path: Union[str, Path], model_type: Optional[str] = None) -> Union[torch.nn.Module, ort.InferenceSession]:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
            model_type: 模型类型 ("pytorch", "onnx")，None 则自动检测
            
        Returns:
            加载的模型对象
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 自动检测模型类型
        if model_type is None:
            if model_path.suffix in ['.pt', '.pth']:
                model_type = "pytorch"
            elif model_path.suffix == '.onnx':
                model_type = "onnx"
            else:
                raise ValueError(f"不支持的模型格式: {model_path.suffix}")
        
        self.model_type = model_type
        
        if model_type == "pytorch":
            return self._load_pytorch(model_path)
        elif model_type == "onnx":
            return self._load_onnx(model_path)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def _load_pytorch(self, model_path: Path) -> torch.nn.Module:
        """加载 PyTorch 模型"""
        logger.info(f"加载 PyTorch 模型: {model_path}")
        
        try:
            # 尝试加载完整模型
            model = torch.load(model_path, map_location=self.device)
        except Exception:
            # 尝试加载 state_dict
            model = torch.load(model_path, map_location=self.device)
            logger.warning("加载的是 state_dict，需要手动构建模型结构")
        
        if isinstance(model, torch.nn.Module):
            model.eval()
            if self.device == "cuda" and torch.cuda.is_available():
                model = model.cuda()
        
        self.model = model
        logger.info(f"PyTorch 模型加载完成，设备: {self.device}")
        return model
    
    def _load_onnx(self, model_path: Path) -> ort.InferenceSession:
        """加载 ONNX 模型"""
        logger.info(f"加载 ONNX 模型: {model_path}")
        
        # 配置 ONNX Runtime 会话选项
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 选择执行提供程序
        if self.device == "cuda" and ort.get_device() == "GPU":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger.info("使用 CUDA 执行提供程序")
        else:
            providers = ['CPUExecutionProvider']
            logger.info("使用 CPU 执行提供程序")
        
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        self.model = session
        logger.info(f"ONNX 模型加载完成，输入: {[i.name for i in session.get_inputs()]}")
        return session
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = {
            "model_type": self.model_type,
            "device": self.device
        }
        
        if self.model_type == "onnx" and self.model is not None:
            info["inputs"] = [
                {"name": i.name, "shape": i.shape, "type": i.type}
                for i in self.model.get_inputs()
            ]
            info["outputs"] = [
                {"name": o.name, "shape": o.shape, "type": o.type}
                for o in self.model.get_outputs()
            ]
        
        return info
