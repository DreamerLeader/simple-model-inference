"""
Postprocessing Module

Common postprocessing operations for model outputs.
"""

import numpy as np
from typing import Any, Tuple, List, Optional, Callable


class Postprocessor:
    """Output data postprocessor."""
    
    def __init__(self, operations: Optional[List[Callable]] = None):
        """
        Initialize postprocessor.
        
        Args:
            operations: List of postprocessing operations to apply
        """
        self.operations = operations or []
    
    def add_operation(self, operation: Callable):
        """Add a postprocessing operation."""
        self.operations.append(operation)
    
    def process(self, data: Any) -> Any:
        """
        Apply all postprocessing operations.
        
        Args:
            data: Model output data
            
        Returns:
            Postprocessed data
        """
        for op in self.operations:
            data = op(data)
        return data
    
    def __call__(self, data: Any) -> Any:
        """Call process method."""
        return self.process(data)


class ClassificationPostprocessor(Postprocessor):
    """Postprocessor for classification outputs."""
    
    def __init__(self, 
                 apply_softmax: bool = True,
                 top_k: int = 5,
                 class_names: Optional[List[str]] = None):
        """
        Initialize classification postprocessor.
        
        Args:
            apply_softmax: Whether to apply softmax
            top_k: Number of top predictions to return
            class_names: List of class names
        """
        super().__init__()
        self.apply_softmax = apply_softmax
        self.top_k = top_k
        self.class_names = class_names
    
    def process(self, logits: np.ndarray) -> List[Tuple[int, float, Optional[str]]]:
        """
        Process classification logits.
        
        Args:
            logits: Model output logits
            
        Returns:
            List of (class_id, probability, class_name) tuples
        """
        # Apply softmax if needed
        if self.apply_softmax:
            probs = softmax(logits)
        else:
            probs = logits
        
        # Get top-k predictions
        if probs.ndim == 1:
            top_indices = np.argsort(probs)[-self.top_k:][::-1]
            results = []
            for idx in top_indices:
                name = self.class_names[idx] if self.class_names else None
                results.append((int(idx), float(probs[idx]), name))
        else:
            # Batch processing
            results = []
            for prob in probs:
                top_indices = np.argsort(prob)[-self.top_k:][::-1]
                batch_results = []
                for idx in top_indices:
                    name = self.class_names[idx] if self.class_names else None
                    batch_results.append((int(idx), float(prob[idx]), name))
                results.append(batch_results)
        
        return results


class DetectionPostprocessor(Postprocessor):
    """Postprocessor for object detection outputs."""
    
    def __init__(self,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 max_detections: int = 100):
        """
        Initialize detection postprocessor.
        
        Args:
            confidence_threshold: Minimum confidence score
            nms_threshold: NMS IoU threshold
            max_detections: Maximum number of detections
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
    
    def process(self, predictions: np.ndarray) -> List[dict]:
        """
        Process detection predictions.
        
        Args:
            predictions: Model predictions [N, 6] format: [x1, y1, x2, y2, conf, class]
            
        Returns:
            List of detection dictionaries
        """
        # Filter by confidence
        mask = predictions[:, 4] >= self.confidence_threshold
        predictions = predictions[mask]
        
        if len(predictions) == 0:
            return []
        
        # Apply NMS per class
        detections = []
        unique_classes = np.unique(predictions[:, 5])
        
        for cls in unique_classes:
            cls_mask = predictions[:, 5] == cls
            cls_preds = predictions[cls_mask]
            
            # Sort by confidence
            order = cls_preds[:, 4].argsort()[::-1]
            cls_preds = cls_preds[order]
            
            # Apply NMS
            keep = nms(cls_preds[:, :4], cls_preds[:, 4], self.nms_threshold)
            cls_preds = cls_preds[keep]
            
            for pred in cls_preds[:self.max_detections]:
                detections.append({
                    'bbox': pred[:4].tolist(),
                    'confidence': float(pred[4]),
                    'class_id': int(pred[5])
                })
        
        # Sort all detections by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections[:self.max_detections]


class SegmentationPostprocessor(Postprocessor):
    """Postprocessor for segmentation outputs."""
    
    def __init__(self, 
                 apply_sigmoid: bool = False,
                 threshold: float = 0.5,
                 resize_to: Optional[Tuple[int, int]] = None):
        """
        Initialize segmentation postprocessor.
        
        Args:
            apply_sigmoid: Whether to apply sigmoid activation
            threshold: Threshold for binary segmentation
            resize_to: Target size for resizing (H, W)
        """
        super().__init__()
        self.apply_sigmoid = apply_sigmoid
        self.threshold = threshold
        self.resize_to = resize_to
    
    def process(self, logits: np.ndarray) -> np.ndarray:
        """
        Process segmentation logits.
        
        Args:
            logits: Model output logits [C, H, W] or [N, C, H, W]
            
        Returns:
            Segmentation masks
        """
        # Apply sigmoid if needed
        if self.apply_sigmoid:
            masks = sigmoid(logits)
        else:
            masks = logits
        
        # Get class predictions
        if masks.ndim == 3:  # [C, H, W]
            masks = np.argmax(masks, axis=0)
        else:  # [N, C, H, W]
            masks = np.argmax(masks, axis=1)
        
        # Apply threshold for binary case
        if masks.max() == 1:
            masks = (masks > self.threshold).astype(np.uint8)
        
        # Resize if needed
        if self.resize_to is not None:
            import cv2
            if masks.ndim == 2:
                masks = cv2.resize(masks.astype(np.float32), 
                                 (self.resize_to[1], self.resize_to[0]),
                                 interpolation=cv2.INTER_NEAREST)
            else:
                resized = []
                for mask in masks:
                    r = cv2.resize(mask.astype(np.float32),
                                 (self.resize_to[1], self.resize_to[0]),
                                 interpolation=cv2.INTER_NEAREST)
                    resized.append(r)
                masks = np.array(resized)
        
        return masks


# Utility functions
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Apply softmax activation."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid activation."""
    return 1 / (1 + np.exp(-x))


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
    """
    Non-Maximum Suppression.
    
    Args:
        boxes: Bounding boxes [N, 4] in format [x1, y1, x2, y2]
        scores: Confidence scores [N]
        threshold: IoU threshold
        
    Returns:
        Indices of boxes to keep
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]
    
    return keep
