"""
预处理和后处理模块
"""

import numpy as np
import torch
from PIL import Image
from typing import Union, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PreProcessor:
    """数据预处理器"""
    
    def __init__(
        self,
        input_shape: Optional[Tuple[int, ...]] = None,
        normalize: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        to_tensor: bool = True
    ):
        self.input_shape = input_shape
        self.normalize = normalize
        self.mean = np.array(mean).reshape(-1, 1, 1)
        self.std = np.array(std).reshape(-1, 1, 1)
        self.to_tensor = to_tensor
    
    def __call__(self, data: Union[np.ndarray, Image.Image, str]) -> Union[torch.Tensor, np.ndarray]:
        """
        预处理数据
        
        Args:
            data: 输入数据（numpy数组、PIL图像或图像路径）
            
        Returns:
            预处理后的数据
        """
        # 加载图像
        if isinstance(data, str):
            data = Image.open(data).convert('RGB')
        
        # 转换为 numpy 数组
        if isinstance(data, Image.Image):
            data = np.array(data)
        
        # 调整大小
        if self.input_shape is not None and len(self.input_shape) >= 2:
            target_size = (self.input_shape[-2], self.input_shape[-1])
            if data.shape[:2] != target_size:
                data = self._resize(data, target_size)
        
        # 归一化到 [0, 1]
        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0
        
        # 标准化
        if self.normalize:
            data = self._normalize(data)
        
        # 调整通道顺序 (H, W, C) -> (C, H, W)
        if len(data.shape) == 3 and data.shape[-1] in [1, 3]:
            data = np.transpose(data, (2, 0, 1))
        
        # 添加 batch 维度
        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=0)
        
        # 转换为 PyTorch tensor
        if self.to_tensor:
            data = torch.from_numpy(data)
        
        return data
    
    def _resize(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """调整图像大小"""
        pil_img = Image.fromarray(image.astype(np.uint8))
        pil_img = pil_img.resize(size, Image.BILINEAR)
        return np.array(pil_img)
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """标准化"""
        if len(image.shape) == 3:
            image = (image - self.mean.T) / self.std.T
        return image
    
    def batch_process(self, data_list: List) -> Union[torch.Tensor, np.ndarray]:
        """批量预处理"""
        processed = [self(d) for d in data_list]
        if self.to_tensor:
            return torch.cat(processed, dim=0)
        else:
            return np.concatenate(processed, axis=0)


class PostProcessor:
    """结果后处理器"""
    
    def __init__(
        self,
        top_k: int = 5,
        threshold: float = 0.5,
        apply_softmax: bool = True
    ):
        self.top_k = top_k
        self.threshold = threshold
        self.apply_softmax = apply_softmax
    
    def __call__(self, output: Union[torch.Tensor, np.ndarray]) -> dict:
        """
        后处理推理结果
        
        Args:
            output: 模型输出
            
        Returns:
            处理后的结果字典
        """
        # 转换为 numpy
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        
        # 确保是 2D (batch, classes)
        if len(output.shape) > 2:
            output = output.reshape(output.shape[0], -1)
        
        # 应用 softmax
        if self.apply_softmax:
            output = self._softmax(output)
        
        results = []
        for batch_idx in range(output.shape[0]):
            probs = output[batch_idx]
            
            # 获取 top-k
            top_indices = np.argsort(probs)[::-1][:self.top_k]
            top_probs = probs[top_indices]
            
            # 过滤低于阈值的
            valid_mask = top_probs >= self.threshold
            
            results.append({
                "indices": top_indices[valid_mask].tolist(),
                "probabilities": top_probs[valid_mask].tolist(),
                "top_class": int(top_indices[0]),
                "top_confidence": float(top_probs[0])
            })
        
        return results[0] if len(results) == 1 else results
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax 函数"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
