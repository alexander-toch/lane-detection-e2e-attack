# ONNX_MODEL_PATH = "../resnet50_resa_tusimple_20211019.onnx"
ONNX_MODEL_PATH = "../resnet50_resa_culane_20211016.onnx"
BENCHMARK = False

# input_sizes = (360, 640)    # tusimple
input_sizes = (288, 800)  # culane
max_lane = 0  # maximum number of lanes to detect (0 for no limit)
gap = 5  # y pixel gap for sampling
ppl = 128  # how many points for one lane
thresh = 0.2  # probability threshold
dataset = "llamas"
