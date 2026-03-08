"""
Example: Image Classification Inference

This example demonstrates how to use the inference framework
for image classification tasks.
"""

import numpy as np
from inference_framework import ModelLoader, InferenceEngine
from inference_framework.preprocess import ImagePreprocessor
from inference_framework.postprocess import ClassificationPostprocessor


def main():
    # Configuration
    model_path = "path/to/your/model.pt"  # or .onnx
    image_path = "path/to/your/image.jpg"
    
    # 1. Load model
    print("Loading model...")
    loader = ModelLoader()
    model = loader.load(model_path, framework="pytorch")  # or "onnx"
    
    # 2. Create inference engine
    print("Creating inference engine...")
    engine = InferenceEngine(model, device="auto")
    print(f"Engine info: {engine.get_info()}")
    
    # 3. Warm up (optional but recommended)
    print("Warming up...")
    engine.warmup(input_shape=(1, 3, 224, 224))
    
    # 4. Create preprocessor
    print("Creating preprocessor...")
    preprocessor = ImagePreprocessor(
        target_size=(224, 224),
        normalize=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        channel_first=True
    )
    
    # 5. Create postprocessor
    print("Creating postprocessor...")
    postprocessor = ClassificationPostprocessor(
        apply_softmax=True,
        top_k=5
    )
    
    # 6. Preprocess image
    print("Preprocessing image...")
    input_data = preprocessor(image_path)
    
    # 7. Run inference
    print("Running inference...")
    output = engine.predict(input_data)
    
    # 8. Postprocess results
    print("Postprocessing results...")
    results = postprocessor(output)
    
    # 9. Display results
    print("\nTop predictions:")
    for class_id, prob, name in results:
        print(f"  Class {class_id}: {prob:.4f}")


def batch_inference_example():
    """Example of batch inference."""
    # Load model and create engine
    loader = ModelLoader()
    model = loader.load("model.pt")
    engine = InferenceEngine(model, batch_size=4)
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    
    # Load and preprocess multiple images
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
    input_data = [preprocessor(img) for img in image_paths]
    
    # Run batch inference
    results = engine.predict_batch(input_data)
    
    print(f"Processed {len(results)} images")


if __name__ == "__main__":
    # Note: This example requires a model file and image to run
    # Update the paths above with your actual files
    print("Image Classification Example")
    print("=" * 50)
    
    # Uncomment to run (requires actual model and image)
    # main()
    
    print("\nExample code is ready. Update the file paths and uncomment main() to run.")
