from dataclasses import dataclass
from random import randint
import cv2
from attack.pytorch_auto_drive.utils.runners.base import BaseTrainer
from config import *
import os
import sys
import numpy as np
from PIL import Image
import torch
from config import *
import pytorch_auto_drive.functional as F
from lanefitting import get_offset_center
from attack.pytorch_auto_drive.utils.models import MODELS
from attack.pytorch_auto_drive.utils.losses import LOSSES
from attack.pytorch_auto_drive.dpatch_robust import MyRobustDPatch
from attack.pytorch_auto_drive.estimator import MyPyTorchClassifier
from attack.pytorch_auto_drive.utils.common import load_checkpoint
from attack.pytorch_auto_drive.utils.args import read_config

from pytorch_auto_drive.utils import (
    lane_as_segmentation_inference,
    lane_detection_visualize_batched,
)

MODEL="resa"
script_dir=os.path.dirname(os.path.realpath(__file__))

if MODEL == "baseline":
    CONFIG=os.path.join(script_dir, 'attack/pytorch_auto_drive/configs/lane_detection/baseline/resnet50_culane.py').replace('\\','/')
    CHECKPOINT=os.path.join(script_dir, '../resnet50_baseline_culane_20210308.pt').replace('\\','/')
elif MODEL == "resa":
    CONFIG=os.path.join(script_dir, 'attack/pytorch_auto_drive/configs/lane_detection/resa/resnet50_culane.py').replace('\\','/')
    CHECKPOINT=os.path.join(script_dir, '../resnet50_resa_culane_20211016.pt').replace('\\','/')
elif MODEL == "scnn":
    CONFIG=os.path.join(script_dir, 'attack/pytorch_auto_drive/configs/lane_detection/scnn/resnet50_culane.py').replace('\\','/')
    CHECKPOINT=os.path.join(script_dir, '../resnet50_scnn_culane_20210311.pt').replace('\\','/')

@dataclass
class DirtyRoadPatch:
    model_in: np.ndarray
    patch: np.ndarray
    location: tuple
    probmaps: torch.Tensor
    original_image_sizes: tuple = None
    input_image_sizes: tuple = None

class PyTorchPipeline:
    def __init__(self, model_path=ONNX_MODEL_PATH, targeted=False):
        self.cfg = read_config(CONFIG)
        self.model = MODELS.from_dict(self.cfg['model'])
        self.current_patch = None

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:  # pragma: no cover
            cuda_idx = torch.cuda.current_device()
            self.device = torch.device(f"cuda:{cuda_idx}")


        # config for 1280x720
        # self.patch_size = (120,300) # (height, width)
        # self.patch_location=(200,160) # in format (W, H). (800, 288) is input size for resa

        # config for 800x288
        w, h = 800, 288
        PATCH_SCALE_FACTOR = 0.0875
        self.patch_size = (int(w * PATCH_SCALE_FACTOR),int(w * PATCH_SCALE_FACTOR)) # only support quadratic patches for now
        self.patch_location=(int(w/2 - self.patch_size[1] / 2), int(h/2 + self.patch_size[0])) # in format (W, H). (800, 288) is input size for resa

        brightness_range= (0.8, 1.0)
        rotation_weights = (0.4, 0.2, 0.2, 0.2)
        optimizer = BaseTrainer.get_optimizer(self.cfg['optimizer'], self.model)

        loss_config = loss = dict(
            name='LaneLossSeg',
            ignore_index=255,
            weight=[0.4, 1, 1, 1, 1]
        )

        loss = LOSSES.from_dict(loss_config)
        num_classes = self.cfg['train']['num_classes']
        input_size = self.cfg['train']['input_size']
        load_checkpoint(net=self.model, optimizer=None, lr_scheduler=None, filename=CHECKPOINT, strict=False)

        clip_values = (0, 255)


        classifier = MyPyTorchClassifier(
            model=self.model,
            loss=loss,
            clip_values=clip_values,
            optimizer=optimizer,
            input_shape=(1, input_size[0], input_size[1]),
            nb_classes=num_classes,
            channels_first=True,
        )

        self.targeted = targeted
        self.attack = MyRobustDPatch(estimator=classifier, 
                        max_iter=200,
                        sample_size=1,
                        patch_shape=(3, self.patch_size[0], self.patch_size[1]), 
                        patch_location=self.patch_location,
                        brightness_range=brightness_range,
                        learning_rate=5.0,
                        targeted=self.targeted
                        # rotation_weights=rotation_weights,
                    )

    def inference(self, model_in, original_img, orig_sizes, keypoints_only=False):
        onnx_out = self.ort_sess.run(None, {"input1": model_in})
        outputs = {"out": torch.Tensor(onnx_out[0]), "lane": torch.Tensor(onnx_out[1])}

        keypoints = lane_as_segmentation_inference(
            None,
            outputs,
            [input_sizes, orig_sizes],
            gap,
            ppl,
            thresh,
            dataset,
            max_lane,
            forward=False,  # already called model
        )

        assert len(keypoints[0]) > 0, "No lanes detected"
        keypoints = [[np.array(lane) for lane in image] for image in keypoints]

        if keypoints_only:
            return None, keypoints

        results = lane_detection_visualize_batched(
            original_img, keypoints=keypoints, style="point"
        )
        return results, keypoints

    def infer_offset_center(self, image, orig_sizes, control_object, image_on_cuda=False, ipm=None):
        if image_on_cuda:
            # image arrives in (H, W, C), needs to have [C, H, W] format
            image = torch.as_tensor(image, device='cuda').permute(2, 0, 1)
            # image = torch.from_numpy(image).permute(2, 0, 1)


        if orig_sizes != input_sizes:
            image = F.resize(image, size=input_sizes) #, interpolation=Image.NEAREST)


        if image_on_cuda:
            model_in = image.unsqueeze(0)
            results = self.model(model_in)
            # self.save_image(model_in[0].cpu().numpy(), f'camera_observations/{control_object.engine.episode_step}_model_input.jpg')
        else:
            model_in = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
            model_in = model_in.view(image.size[1], image.size[0], len(image.getbands()))
            model_in = (
                model_in.permute((2, 0, 1))
                .contiguous()
                .float()
                .div(255)
                .unsqueeze(0)
                .numpy()
            )

            # self.save_image(model_in[0], f'camera_observations/{control_object.engine.episode_step}_model_input.jpg')
            results = self.model(torch.from_numpy(model_in).to(self.device))

        keypoints = lane_as_segmentation_inference(
            None,
            results,
            [input_sizes, orig_sizes],
            gap,
            ppl,
            thresh,
            dataset,
            max_lane,
            forward=False,  # already called model
        )

        off_center, lane_heading_theta, _ = get_offset_center(
            keypoints[0], (orig_sizes[1], orig_sizes[0]), ipm
        )

        debug_info = {
            'probmaps': results,
        }

        return off_center, lane_heading_theta, keypoints[0], debug_info
    
    def save_image(self, image, path, sizes=input_sizes):
        image = image.transpose((1, 2, 0))
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        image = cv2.resize(image, (sizes[1], sizes[0]))
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)
    
    def infer_offset_center_with_dpatch(self, image, orig_sizes, control_object, generate_patch=True, target=None, image_on_cuda=False, ipm=None):
        if self.targeted and target is None:
            raise ValueError("Targeted attack requires a target!")
        
        if image_on_cuda:
            # image arrives in (H, W, C), needs to have [C, H, W] format
            image = torch.as_tensor(image, device='cuda').permute(2, 0, 1)
            # image = torch.from_numpy(image).permute(2, 0, 1)


        image = F.resize(image, size=input_sizes) #, interpolation=Image.NEAREST)


        if image_on_cuda:
            model_in = image.unsqueeze(0)
        else:
            model_in = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
            model_in = model_in.view(image.size[1], image.size[0], len(image.getbands()))
            model_in = (
                model_in.permute((2, 0, 1))
                .contiguous()
                .float()
                .div(255)
                .unsqueeze(0)
                .numpy()
            ) 
        
        if generate_patch or self.current_patch is None:
            if self.targeted:
                self.current_patch = self.attack.generate(x=model_in.cpu().numpy() if image_on_cuda else model_in.copy(), y=target)[0]
            else:
                self.current_patch = self.attack.generate(x=model_in.cpu().numpy() if image_on_cuda else model_in.copy())[0]
            # self.save_image(self.current_patch, f'camera_observations/{control_object.engine.episode_step}_patch.jpg', sizes=(self.current_patch.shape[1], self.current_patch.shape[2]))
        
        patch = self.current_patch

        # place patch
        x_1, y_1 = self.patch_location
        x_2, y_2 = x_1 + patch.shape[2], y_1 + patch.shape[1]
        model_in[0][:, y_1:y_2, x_1:x_2] = torch.from_numpy(patch).to(model_in.device) if image_on_cuda else patch

        # self.save_image(model_in[0].cpu().numpy(), f'camera_observations/{control_object.engine.episode_step}_model_input.jpg')

        results = self.model(model_in) if image_on_cuda else self.model(torch.from_numpy(model_in).to(self.device))

        keypoints = lane_as_segmentation_inference(
            None,
            results,
            [input_sizes, orig_sizes],
            gap,
            ppl,
            thresh,
            dataset,
            max_lane,
            forward=False,  # already called model
        )

        off_center, lane_heading_theta, _ = get_offset_center(
            keypoints[0], (orig_sizes[1], orig_sizes[0]), ipm
        )

        # scale patch to match orig_sizes proportionally
        scale_factor_height = orig_sizes[0]/input_sizes[0]
        scale_factor_width = orig_sizes[1]/input_sizes[1]
        patch = patch.transpose((1, 2, 0))
        patch = np.clip(patch, 0, 1)
        patch = (patch * 255).astype(np.uint8)
        patch = cv2.resize(patch, (int(patch.shape[1] * scale_factor_width), int(patch.shape[0] * scale_factor_height)))

        if generate_patch:
            cv2.imwrite(f'camera_observations/patch_{control_object.engine.episode_step}.jpg', patch)

        scaled_location = (int(self.patch_location[0] * scale_factor_width), int(self.patch_location[1] * scale_factor_height)) # format (x, y)
        patch_object: DirtyRoadPatch = DirtyRoadPatch(
            model_in[0].cpu().numpy(),
            patch, 
            scaled_location, 
            results,
            orig_sizes,
            input_sizes
        )

        return off_center, lane_heading_theta, keypoints[0], patch_object
