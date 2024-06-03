# End-to-end robustness testing of DNN lane detection models

This project is part of my master thesis (TODO: add title/link) and enables robustness testing of DNN-based lane detection models (semantic segmentation) of [PytorchAutoDrive](https://github.com/voldemortX/pytorch-auto-drive) with attacks taken from the [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox).

The robustness testing is performed in an end-to-end setting within the [MetaDrive](https://github.com/metadriverse/metadrive) simulator.

![animated gif of an example attack](example-attack.gif)

Currently supported models: RESA ([paper](https://arxiv.org/abs/2008.13719))

Currently supported attacks: modified Robust DPatch ([paper](https://arxiv.org/abs/1906.11897))


## Requirements

Conda: https://docs.conda.io/projects/miniconda/en/latest/

## Installation

Tested on Ubuntu Bullseye (SLURM node) with a Nvidia Tesla T4 (headless rendering).

Setup conda environment with:

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

On Windows 11:

Install CUDA: https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local

```
conda create -n ld python=3.11
conda activate ld
git clone git@github.com:alexander-toch/metadrive.git ../metadrive
cd ../metadrive
pip install -e .
cd ../lane-detection-e2e-attack
python -m pip install --upgrade pywin32
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install panda3d metadrive-simulator opencv-python numpy pillow torch torchvision adversarial-robustness-toolbox[pytorch_image] timm mmcv tensorboard importmagician cupy-cuda12x 
pip install cuda-python PyOpenGL PyOpenGL_accelerate
```

## Execution

Currently, the attack is started via `rm -rf camera_observations/*.{jpg,npy} && python metadrive_bridge_selfdrive.py`

## Resources / Links

- [MetaDrive](https://github.com/metadriverse/metadrive)
- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [PytorchAutoDrive](https://github.com/voldemortX/pytorch-auto-drive)