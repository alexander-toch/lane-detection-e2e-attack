# ONNX_MODEL_PATH = "../resnet50_resa_tusimple_20211019.onnx"
ONNX_MODEL_PATH = "../resnet50_resa_culane_20211016.onnx"
BENCHMARK = False

# input_sizes = (360, 640)  # tusimple
input_sizes = (288, 800)  # culane
max_lane = 4
gap = 10
ppl = 56
thresh = 0.3
dataset = "llamas"
