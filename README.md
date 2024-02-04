# Lane detection with ONNX models on MetaDrive

## Installation

On Apple Silicon:
```
CONDA_SUBDIR=osx-arm64 conda create -n ld3.9 python=3.9 
conda activate ld3.9
pip3 install -r requirements-arm.txt
```

Requirements: panda3d metadrive-simulator opencv-python numpy pillow torch torchvision onnxruntime-gpu