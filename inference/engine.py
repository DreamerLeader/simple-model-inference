"""
推理引擎模块
"""

import time
import torch
import numpy as np
from typing import Union, Any, Dict, List
import logging

from .loader import ModelLoader
from .processor import PreProcessor, PostProcessor
from config import InferenceConfig

logger = logging.getLogger(__name__)


class InferenceEngine:
    """推理引擎主类"""
    
    def __init__(self, config: InferenceConfig):
        """
        初始化推理引擎
        
        Args:
            config: 推理配置
        """
        self.config = config
        self.device = config.device
        
        # 初始化组件
        self.loader = ModelLoader(device=self.device)
        self.preprocessor = PreProcessor(
            input_shape=config.input_shape,
            normalize=config.normalize,
            mean=config.mean,
            std=config.std,
            to_tensor=(config.model_type == "pytorch")
        )
        self.postprocessor = PostProcessor(
            top_k=config.top_k,
            threshold=config.threshold,
            apply_softmax=True
        )
        
        # 加载模型
        self.model = self.loader.load(config.model_path, config.model_type)
        self.model_type = config.model_type
        
        # 性能统计
        self.profiling_data = {
            "preprocess_time": [],
            "inference_time": [],
            "postprocess_time": [],
            "total_time": []
        }
        
        # 预热
        if config.warmup_iterations > 0:
            self._warmup(config.warmup_iterations)
    
    def _warmup(self, iterations: int = 3):
        """模型预热"""
        logger.info(f"开始模型预热 ({iterations} 次)...")
        
        # 创建虚拟输入
        if self.config.input_shape:
            dummy_input = torch.randn(self.config.input_shape).to(self.device)
        else:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        for i in range(iterations):
            _ = self._inference(dummy_input)
        
        logger.info("预热完成")
    
    def predict(self, input_data: Union[np.ndarray, Any]) -> Dict:
        """
        单样本推理
        
        Args:
            input_data: 输入数据
            
        Returns:
            推理结果字典
        """
        return self.batch_predict([input_data])[0]
    
    def batch_predict(self, input_data_list: List) -> List[Dict]:
        """
        批量推理
        
        Args:
            input_data_list: 输入数据列表
            
        Returns:
            推理结果列表
        """
        total_start = time.time()
        
        # 预处理
        preprocess_start = time.time()
        processed_input = self.preprocessor.batch_process(input_data_list)
        if self.config.enable_profiling:
            self.profiling_data["preprocess_time"].append(time.time() - preprocess_start)
        
        # 推理
        inference_start = time.time()
        raw_output = self._inference(processed_input)
        if self.config.enable_profiling:
            self.profiling_data["inference_time"].append(time.time() - inference_start)
        
        # 后处理
        postprocess_start = time.time()
        results = self.postprocessor(raw_output)
        if self.config.enable_profiling:
            self.profiling_data["postprocess_time"].append(time.time() - postprocess_start)
            self.profiling_data["total_time"].append(time.time() - total_start)
        
        # 确保结果是列表
        if not isinstance(results, list):
            results = [results]
        
        return results
    
    def _inference(self, input_data: Union[torch.Tensor, np.ndarray]):
        """执行推理"""
        if self.model_type == "pytorch":
            return self._pytorch_inference(input_data)
        elif self.model_type == "onnx":
            return self._onnx_inference(input_data)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _pytorch_inference(self, input_data: torch.Tensor) -> torch.Tensor:
        """PyTorch 推理"""
        with torch.no_grad():
            if self.device == "cuda":
                input_data = input_data.cuda()
            output = self.model(input_data)
        return output
    
    def _onnx_inference(self, input_data: np.ndarray) -> np.ndarray:
        """ONNX 推理"""
        input_name = self.model.get_inputs()[0].name
        output = self.model.run(None, {input_name: input_data})
        return output[0] if len(output) == 1 else output
    
    def get_profiling_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        if not self.config.enable_profiling:
            return {"error": "性能分析未启用"}
        
        stats = {}
        for key, values in self.profiling_data.items():
            if values:
                stats[f"{key}_mean"] = np.mean(values) * 1000  # 转换为毫秒
                stats[f"{key}_std"] = np.std(values) * 1000
                stats[f"{key}_min"] = np.min(values) * 1000
                stats[f"{key}_max"] = np.max(values) * 1000
        
        return stats
    
    def print_profiling_stats(self):
        """打印性能统计"""
        stats = self.get_profiling_stats()
        
        if "error" in stats:
            logger.warning(stats["error"])
            return
        
        logger.info("=" * 50)
        logger.info("性能统计 (毫秒)")
        logger.info("=" * 50)
        for key, value in stats.items():
            if "_mean" in key:
                stage = key.replace("_time_mean", "")
                mean = value
                std = stats.get(f"{stage}_time_std", 0)
                logger.info(f"{stage:12s}: {mean:8.2f} ± {std:6.2f} ms")
        logger.info("=" * 50)
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return self.loader.get_model_info()
