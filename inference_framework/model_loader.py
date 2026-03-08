"""
Model Loader Module

Supports loading PyTorch and ONNX models.
"""

import os
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Universal model loader supporting multiple frameworks."""
    
    def __init__(self):
        self.supported_formats = {
            '.pt': 'pytorch',
            '.pth': 'pytorch',
            '.onnx': 'onnx'
        }
    
    def load(self, 
             model_path: Union[str, Path], 
             framework: Optional[str] = None,
             **kwargs) -> Any:
        """
        Load a model from the specified path.
        
        Args:
            model_path: Path to the model file
            framework: Framework to use ('pytorch', 'onnx', or None for auto-detect)
            **kwargs: Additional arguments passed to the loader
            
        Returns:
            Loaded model object
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Auto-detect framework if not specified
        if framework is None:
            framework = self._detect_framework(model_path)
        
        logger.info(f"Loading model from {model_path} using {framework}")
        
        if framework == 'pytorch':
            return self._load_pytorch(model_path, **kwargs)
        elif framework == 'onnx':
            return self._load_onnx(model_path, **kwargs)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def _detect_framework(self, model_path: Path) -> str:
        """Auto-detect framework from file extension."""
        ext = model_path.suffix.lower()
        if ext in self.supported_formats:
            return self.supported_formats[ext]
        raise ValueError(f"Cannot detect framework for file: {model_path}")
    
    def _load_pytorch(self, model_path: Path, **kwargs):
        """Load PyTorch model."""
        try:
            import torch
            
            # Load model
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Handle different model formats
            if isinstance(model, dict):
                # If it's a state dict, you might need to instantiate the model class
                state_dict = model.get('state_dict', model)
                logger.info("Loaded PyTorch state dict. Model class must be instantiated separately.")
                return state_dict
            
            logger.info("PyTorch model loaded successfully")
            return model
            
        except ImportError:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {e}")
    
    def _load_onnx(self, model_path: Path, **kwargs):
        """Load ONNX model."""
        try:
            import onnx
            import onnxruntime as ort
            
            # Verify model
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            
            # Create inference session
            providers = kwargs.get('providers', None)
            if providers is None:
                # Auto-select providers
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers
            )
            
            logger.info(f"ONNX model loaded successfully. Providers: {session.get_providers()}")
            return session
            
        except ImportError:
            raise ImportError("ONNX or ONNX Runtime is not installed. Install with: pip install onnx onnxruntime")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def get_model_info(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about a model file."""
        model_path = Path(model_path)
        
        info = {
            'path': str(model_path),
            'exists': model_path.exists(),
            'size_mb': model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0,
            'framework': self._detect_framework(model_path) if model_path.exists() else None
        }
        
        return info
