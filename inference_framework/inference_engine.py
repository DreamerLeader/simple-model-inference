"""
Inference Engine Module

Handles model inference on CPU/GPU devices.
"""

import logging
from typing import Any, Union, Optional, List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Inference engine supporting CPU and GPU execution."""
    
    def __init__(self, 
                 model: Any,
                 device: str = "auto",
                 batch_size: int = 1,
                 **kwargs):
        """
        Initialize inference engine.
        
        Args:
            model: Loaded model object
            device: Device to use ('cpu', 'cuda', 'auto')
            batch_size: Default batch size for inference
            **kwargs: Additional configuration
        """
        self.model = model
        self.device = self._setup_device(device)
        self.batch_size = batch_size
        self.framework = self._detect_framework()
        
        logger.info(f"InferenceEngine initialized: framework={self.framework}, device={self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device."""
        if device == "auto":
            # Try CUDA first, then CPU
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return device
    
    def _detect_framework(self) -> str:
        """Detect the framework of the loaded model."""
        model_class = self.model.__class__.__module__
        
        if 'torch' in model_class:
            return 'pytorch'
        elif 'onnxruntime' in model_class:
            return 'onnx'
        elif isinstance(self.model, dict):
            # Likely a PyTorch state dict
            return 'pytorch'
        else:
            return 'unknown'
    
    def predict(self, 
                input_data: Union[np.ndarray, List, Any],
                **kwargs) -> Any:
        """
        Run inference on input data.
        
        Args:
            input_data: Input data for inference
            **kwargs: Additional inference parameters
            
        Returns:
            Model predictions
        """
        if self.framework == 'pytorch':
            return self._predict_pytorch(input_data, **kwargs)
        elif self.framework == 'onnx':
            return self._predict_onnx(input_data, **kwargs)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
    
    def _predict_pytorch(self, input_data: Any, **kwargs) -> np.ndarray:
        """Run PyTorch inference."""
        import torch
        
        # Convert input to tensor if needed
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data)
        elif isinstance(input_data, torch.Tensor):
            input_tensor = input_data
        else:
            input_tensor = torch.tensor(input_data)
        
        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        
        # Handle batch dimension
        if input_tensor.dim() == 3:  # Add batch dimension for single image
            input_tensor = input_tensor.unsqueeze(0)
        
        # Set model to eval mode if it's a nn.Module
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        # Run inference
        with torch.no_grad():
            if callable(self.model):
                output = self.model(input_tensor)
            else:
                raise RuntimeError("Model is not callable. For state dicts, load into a model class first.")
        
        # Convert output to numpy
        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()
        
        return output
    
    def _predict_onnx(self, input_data: Any, **kwargs) -> np.ndarray:
        """Run ONNX inference."""
        session = self.model
        
        # Get input info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        # Convert input to numpy if needed
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)
        
        # Ensure correct dtype
        input_dtype = session.get_inputs()[0].type
        if 'float16' in input_dtype:
            input_data = input_data.astype(np.float16)
        elif 'float' in input_dtype:
            input_data = input_data.astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {input_name: input_data})
        
        return outputs[0] if len(outputs) == 1 else outputs
    
    def predict_batch(self, 
                      input_data: List[np.ndarray],
                      **kwargs) -> List[np.ndarray]:
        """
        Run batch inference.
        
        Args:
            input_data: List of input arrays
            **kwargs: Additional inference parameters
            
        Returns:
            List of predictions
        """
        results = []
        
        # Process in batches
        for i in range(0, len(input_data), self.batch_size):
            batch = input_data[i:i + self.batch_size]
            
            # Stack batch
            if self.framework == 'pytorch':
                import torch
                batch_tensor = torch.stack([torch.from_numpy(x) for x in batch])
                result = self.predict(batch_tensor, **kwargs)
            else:
                batch_array = np.stack(batch)
                result = self.predict(batch_array, **kwargs)
            
            # Split results
            if isinstance(result, np.ndarray):
                results.extend([result[j] for j in range(result.shape[0])])
            else:
                results.append(result)
        
        return results
    
    def warmup(self, input_shape: Tuple[int, ...], runs: int = 3):
        """
        Warm up the model with dummy inputs.
        
        Args:
            input_shape: Shape of dummy input
            runs: Number of warmup runs
        """
        logger.info(f"Warming up model with shape {input_shape}")
        
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        for _ in range(runs):
            _ = self.predict(dummy_input)
        
        logger.info("Warmup complete")
    
    def get_info(self) -> Dict[str, Any]:
        """Get engine information."""
        info = {
            'framework': self.framework,
            'device': self.device,
            'batch_size': self.batch_size
        }
        
        if self.framework == 'onnx':
            info['providers'] = self.model.get_providers()
            info['input_names'] = [inp.name for inp in self.model.get_inputs()]
            info['output_names'] = [out.name for out in self.model.get_outputs()]
        
        return info
