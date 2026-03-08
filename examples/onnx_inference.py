"""
Example: ONNX Model Inference

This example demonstrates how to use the inference framework
with ONNX models.
"""

import numpy as np
from inference_framework import ModelLoader, InferenceEngine


def main():
    # Configuration
    model_path = "path/to/your/model.onnx"
    
    # 1. Load ONNX model
    print("Loading ONNX model...")
    loader = ModelLoader()
    
    # Option 1: Auto-detect from file extension
    model = loader.load(model_path)
    
    # Option 2: Specify providers for GPU acceleration
    # model = loader.load(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # 2. Create inference engine
    print("Creating inference engine...")
    engine = InferenceEngine(model)
    
    # Print engine info
    info = engine.get_info()
    print(f"\nEngine Information:")
    print(f"  Framework: {info['framework']}")
    print(f"  Providers: {info['providers']}")
    print(f"  Input names: {info['input_names']}")
    print(f"  Output names: {info['output_names']}")
    
    # 3. Warm up
    print("\nWarming up...")
    engine.warmup(input_shape=(1, 3, 224, 224))
    
    # 4. Create dummy input (replace with actual data)
    print("\nRunning inference...")
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # 5. Run inference
    output = engine.predict(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
    print("\nInference completed successfully!")


def benchmark_example():
    """Benchmark inference speed."""
    import time
    
    # Load model
    loader = ModelLoader()
    model = loader.load("model.onnx")
    engine = InferenceEngine(model)
    
    # Prepare input
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Warm up
    engine.warmup(input_shape=(1, 3, 224, 224), runs=10)
    
    # Benchmark
    num_runs = 100
    start_time = time.time()
    
    for _ in range(num_runs):
        _ = engine.predict(input_data)
    
    elapsed = time.time() - start_time
    avg_time = elapsed / num_runs * 1000  # Convert to ms
    
    print(f"\nBenchmark Results ({num_runs} runs):")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  FPS: {num_runs / elapsed:.2f}")


if __name__ == "__main__":
    print("ONNX Model Inference Example")
    print("=" * 50)
    
    # Uncomment to run (requires actual ONNX model)
    # main()
    # benchmark_example()
    
    print("\nExample code is ready. Update the model path and uncomment to run.")
