### Inference Runtimes
- Engines that execute a neural network's computation graph
- The model weights remain the same, but serialized in different formats

**NOTE:** This project uses ONNX as TensforFlow's SavedModel format has a compatibility issue with Python 3.12

#### ONNX Runtime 
Lightweight (roughly 50MB) compared to TF frameworks with about 500MB. 

**Reasons ONNX is chosen for HarmoneyNet:**
- Compatibility: basic-pitch was trained in TF, but the TF's `tf-keras` package failed to install on Python 3.12 (macOS) due to numpy version conflicts. Whereas, ONNX runtime has no such contraint and installs cleanly. 
- Size: Model only needs to be run forward for inference, and v1 requires no training. ONNX Runtime ships execution engine without training machinery, making it a smaller project dependency.

**Ref and more info:** https://medium.com/@digvijay17july/ml-inference-runtimes-in-2026-an-architects-guide-to-choosing-the-right-engine-d3989a87d052
