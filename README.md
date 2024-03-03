# Lane detection with ONNX models on MetaDrive

## Requirements

Conda: https://docs.conda.io/projects/miniconda/en/latest/

## Installation

Requirements: `panda3d metadrive-simulator opencv-python numpy pillow torch torchvision adversarial-robustness-toolbox[pytorch_image] timm`
`pip install -U openmim && mim install mmcv`

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
conda create -n ld python=3.9 
conda activate ld
conda install cuda-toolkit==11.4 cudatoolkit=11.4 -c pytorch -c nvidia
pip install panda3d metadrive-simulator opencv-python numpy pillow torch ==2.0.1 torchvision adversarial-robustness-toolbox[pytorch_image] timm mmcv tensorboard importmagician
pip install numpy --upgrade
pip install -U openmim
mim install  mmcv
conda list -e > requirements.txt
python3 -m metadrive.examples.verify_headless_installation --cuda --camera rgb
```

ONNX Runtime:
- Intel GPUs: `pip3 install onnxruntime`
- Nvidia: `pip3 install onnxruntime-gpu``