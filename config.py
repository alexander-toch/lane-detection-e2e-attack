ONNX_MODEL_PATH = "../resnet50_resa_tusimple_20211019.onnx"
BENCHMARK = False

# input_sizes = (360, 640)  # defined in the pretrained model
input_sizes = (288, 800)  # defined in the pretrained model
max_lane = 4
gap = 10
ppl = 56
thresh = 0.3
dataset = "llamas"
