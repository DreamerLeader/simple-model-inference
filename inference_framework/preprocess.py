"""
Preprocessing Module

Common preprocessing operations for model inputs.
"""

import numpy as np
from typing import Union, Tuple, List, Optional, Callable
from PIL import Image
import cv2


class Preprocessor:
    """Input data preprocessor."""
    
    def __init__(self, operations: Optional[List[Callable]] = None):
        """
        Initialize preprocessor.
        
        Args:
            operations: List of preprocessing operations to apply
        """
        self.operations = operations or []
    
    def add_operation(self, operation: Callable):
        """Add a preprocessing operation."""
        self.operations.append(operation)
    
    def process(self, data: Any) -> np.ndarray:
        """
        Apply all preprocessing operations.
        
        Args:
            data: Input data
            
        Returns:
            Preprocessed array
        """
        for op in self.operations:
            data = op(data)
        return data
    
    def __call__(self, data: Any) -> np.ndarray:
        """Call process method."""
        return self.process(data)


class ImagePreprocessor(Preprocessor):
    """Preprocessor for image data."""
    
    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225),
                 channel_first: bool = True,
                 to_float: bool = True):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image size (H, W)
            normalize: Whether to normalize
            mean: Normalization mean
            std: Normalization std
            channel_first: Convert to channel-first format (C, H, W)
            to_float: Convert to float32
        """
        super().__init__()
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array(mean).reshape(-1, 1, 1)
        self.std = np.array(std).reshape(-1, 1, 1)
        self.channel_first = channel_first
        self.to_float = to_float
    
    def process(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image.
        
        Args:
            image: Image path, array, or PIL Image
            
        Returns:
            Preprocessed image array
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        
        # Convert BGR to RGB if needed (OpenCV loads as BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transpose to channel-first
        if self.channel_first:
            image = image.transpose(2, 0, 1)
        
        # Convert to float
        if self.to_float:
            image = image.astype(np.float32) / 255.0
        
        # Normalize
        if self.normalize:
            image = (image - self.mean) / self.std
        
        return image


class TextPreprocessor(Preprocessor):
    """Preprocessor for text data."""
    
    def __init__(self, 
                 max_length: int = 512,
                 tokenizer: Optional[Callable] = None,
                 pad_token_id: int = 0):
        """
        Initialize text preprocessor.
        
        Args:
            max_length: Maximum sequence length
            tokenizer: Tokenization function
            pad_token_id: Padding token ID
        """
        super().__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
    
    def process(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Preprocess text.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Tokenized array
        """
        if isinstance(text, str):
            text = [text]
        
        if self.tokenizer is None:
            # Simple whitespace tokenization as fallback
            tokens = [t.split()[:self.max_length] for t in text]
            # Pad sequences
            max_len = min(max(len(t) for t in tokens), self.max_length)
            padded = []
            for t in tokens:
                t = t + [self.pad_token_id] * (max_len - len(t))
                padded.append(t[:max_len])
            return np.array(padded)
        else:
            return self.tokenizer(text)


# Utility functions
def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size."""
    return cv2.resize(image, (size[1], size[0]))


def normalize_image(image: np.ndarray, 
                   mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                   std: Tuple[float, ...] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """Normalize image with mean and std."""
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)
    return (image - mean) / std


def center_crop(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Center crop image."""
    h, w = image.shape[:2]
    th, tw = size
    
    i = (h - th) // 2
    j = (w - tw) // 2
    
    return image[i:i+th, j:j+tw]
