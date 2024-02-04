import onnxruntime as ort
import numpy as np
import cv2
import sys
import torch
from PIL import Image
import numpy as np
from time import perf_counter
from config import *
from inference import ONNXPipeline

import pytorch_auto_drive.functional as F
from pytorch_auto_drive.utils import (
    lane_as_segmentation_inference,
    lane_detection_visualize_batched,
)


def run():

    # parse first cli argument as image path
    img_path = sys.argv[1]
    time_start = perf_counter()
    image = Image.open(img_path).convert("RGB")
    orig_sizes = (image.height, image.width)
    original_img = F.to_tensor(image).clone().unsqueeze(0)
    image = F.resize(image, size=input_sizes)

    model_in = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))

    model_in = model_in.view(image.size[1], image.size[0], len(image.getbands()))
    model_in = (
        model_in.permute((2, 0, 1)).contiguous().float().div(255).unsqueeze(0).numpy()
    )
    if BENCHMARK:
        print(f"Image preprocessing took {perf_counter() - time_start} seconds")

    pipeline = ONNXPipeline()
    results = pipeline.inference(model_in, original_img, orig_sizes)

    cv2.imshow("Inferred image", results[0])
    cv2.waitKey(5000)


if __name__ == "__main__":
    run()
