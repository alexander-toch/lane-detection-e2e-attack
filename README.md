# Lane detection with ONNX models on MetaDrive

## Requirements

Conda: https://docs.conda.io/projects/miniconda/en/latest/

## Installation

Requirements: `panda3d metadrive-simulator opencv-python numpy pillow torch torchvision`

On Apple Silicon:
```
CONDA_SUBDIR=osx-arm64 conda create -n ld3.9 python=3.9 
conda activate ld3.9
pip3 install -r requirements-arm.txt
```

On Linux:
```
conda create -n ld3.9 python=3.9 
conda activate ld3.9
pip3 install panda3d metadrive-simulator opencv-python numpy pillow torch torchvision
```

Commands for SLURM cluster:
```
srun -N 1 -n 1 --gpus-per-node=1 --ntasks-per-node=1 --pty bash -i
conda activate ld3.9
pip3 install panda3d metadrive-simulator opencv-python numpy pillow torch torchvision panda3d onnxruntime-gpu cupy-cuda11x
conda list -e > requirements.txt
python3 -m metadrive.examples.verify_headless_installation --cuda --camera rgb
```

ONNX Runtime:
- Intel GPUs: `pip3 install onnxruntime`
- Nvidia: `pip3 install onnxruntime-gpu``