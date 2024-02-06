from config import *
import onnxruntime as ort
import numpy as np
import torch

from pytorch_auto_drive.utils import (
    lane_as_segmentation_inference,
    lane_detection_visualize_batched,
)


class ONNXPipeline:
    def __init__(self):
        self.ort_sess = ort.InferenceSession(
            ONNX_MODEL_PATH,
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
            original_img, keypoints=keypoints, style="line"
        )
        return results, keypoints
