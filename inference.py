from config import *
import onnxruntime as ort
import numpy as np
import torch
from config import *
import pytorch_auto_drive.functional as F
from lanefitting import get_steering_angle

from pytorch_auto_drive.utils import (
    lane_as_segmentation_inference,
    lane_detection_visualize_batched,
)
class ONNXPipeline:
    def __init__(self, model_path=ONNX_MODEL_PATH):
        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = 8

        self.ort_sess = ort.InferenceSession(
            model_path,
            providers=ort.get_available_providers(),
            sess_options=sess_opt,
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

    def infer_steering_angle(self, image, orig_sizes):
        image = F.resize(image, size=input_sizes)
        model_in = torch.ByteTensor(
            torch.ByteStorage.from_buffer(image.tobytes())
        )

        model_in = model_in.view(
            image.size[1], image.size[0], len(image.getbands())
        )
        model_in = (
            model_in.permute((2, 0, 1))
            .contiguous()
            .float()
            .div(255)
            .unsqueeze(0)
            .numpy()
        )


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

        return steering_angle, keypoints[0]
