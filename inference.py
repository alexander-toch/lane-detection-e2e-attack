from config import *
import onnxruntime as ort
import numpy as np
import torch
from lanefitting import get_steering_angle

from pytorch_auto_drive.utils import (
    lane_as_segmentation_inference,
    lane_detection_visualize_batched,
)


class ONNXPipeline:
    def __init__(self, model_path=ONNX_MODEL_PATH):
        self.ort_sess = ort.InferenceSession(
            model_path,
            providers=ort.get_available_providers(),
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

    def infer_steering_angle(self, model_in, orig_sizes):
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
        steering_angle = get_steering_angle(
            keypoints[0], (orig_sizes[1], orig_sizes[0])
        )

        return steering_angle
