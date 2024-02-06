"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""

import argparse
from panda3d.core import Texture, GraphicsOutput

import cv2
import numpy as np

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE
from metadrive.component.map.base_map import BaseMap
from metadrive.component.sensors.base_camera import _cuda_enable
from metadrive.component.map.pg_map import MapGenerateMethod

import torch
from inference import ONNXPipeline
from config import *
from PIL import Image
import pytorch_auto_drive.functional as F

W, H = 1280, 720  #  Desired output size of annotated images

SAVE_IMAGES = True
HEADLESS = True
STEPS = 20 * 300

print(f"Using CUDA: {_cuda_enable}")
print(f"Headless mode: {HEADLESS}")

def dummy_env():
    env = MetaDriveEnv({"use_render": False, "image_observation": False})
    try:
        env.reset()
        for i in range(1, 100):
            o, r, tm, tc, info = env.step([0, 1])
    except:
        print("Error happens in Bullet physics world !")
        sys.exit()
    else:
        print("Bullet physics world is launched successfully!")
    finally:
        env.close()

# from https://metadrive-simulator.readthedocs.io/en/latest/points_and_lines.html#points 
def make_line(x_offset, height, y_dir=1, color=(1,105/255,180/255)):
    points = [(x_offset+x,x*y_dir,height*x/10+height) for x in range(10)]
    colors = [np.clip(np.array([*color,1])*(i+1)/11, 0., 1.0) for i in range(10)]
    if y_dir<0:
        points = points[::-1]
        colors = colors[::-1]
    return points, colors

def draw_keypoints(drawer, keypoints):
    line_1, color_1 = make_line(6, 0.5, 0.01*i) # define line 1 for test
    line_2, color_2 = make_line(6, 0.5, -0.01*i) # define line 2 for test
    points = line_1 + line_2 # create point list
    colors = color_1+ color_2
    drawer.reset()
    drawer.draw_points(points, colors) # draw points

if __name__ == "__main__":
    # Adapted from OpenPilot's bridge
    inference_pipeline = ONNXPipeline()

    map_config = {
        "config": "SSCrSCS",
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
        # BaseMap.GENERATE_CONFIG: 3,
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 2,
    }

    config = dict(
        use_render=not HEADLESS,
        window_size=(W, H),
         sensors={
            "rgb_camera": (RGBCamera, W, H),
        },
        interface_panel=[],
        vehicle_config={
            "image_source": "rgb_camera",
        },
        image_on_cuda=False,
        image_observation=True,       
        out_of_route_done=False,
        on_continuous_line_done=False,
        crash_vehicle_done=False,
        crash_object_done=False,
        traffic_density=0.0,  # traffic is incredibly expensive
        # map_config=create_map(),
        map_config=map_config,
        # map=4,  # seven block
        num_scenarios=10,
        decision_repeat=1,
        # physics_world_step_size=self.TICKS_PER_FRAME / 100,
        # preload_models=False,
        manual_control=True,
    )

    dummy_env()

    env = MetaDriveEnv(config)

    try:
        env.reset()
        for i in range(10):
            o, r, tm, tc, i = env.step([0, 1])
        assert isinstance(o, dict)
        point_drawer = env.engine.make_point_drawer(scale=1) # create a point drawer
        print(HELP_MESSAGE)

        env.agent.expert_takeover = True

        for i in range(1, STEPS):
            o, r, tm, tc, info = env.step([0, 0])
            
            if not HEADLESS:
                env.render(
                    text={
                        "Auto-Drive (Switch mode: T)": (
                            "on" if env.current_track_agent.expert_takeover else "off"
                        ),
                        "Keyboard Control": "W,A,S,D",
                        }
                )

            try:
                if i % 20 == 0:
                    # image = Image.fromarray(cv2.cvtColor((o["image"][..., -1]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
                    image = Image.fromarray((o["image"][..., -1]*255).astype(np.uint8))
                    orig_sizes = (image.height, image.width)
                    original_img = F.to_tensor(image).clone().unsqueeze(0)
                    image = F.resize(image, size=input_sizes)

                    model_in = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))

                    model_in = model_in.view(image.size[1], image.size[0], len(image.getbands()))
                    model_in = (
                        model_in.permute((2, 0, 1)).contiguous().float().div(255).unsqueeze(0).numpy()
                    )

                    results, keypoints = inference_pipeline.inference(model_in, original_img, orig_sizes)

                    # TODO: visualize keypoints: https://metadrive-simulator.readthedocs.io/en/latest/points_and_lines.html#points 
                    # draw_keypoints(point_drawer, keypoints)

                    # cv2.imshow("Inferred image", results[0])
                    if SAVE_IMAGES:
                        cv2.imwrite(
                                f"camera_observations/{str(i)}_inf.jpg",
                                results[0],
                        )

            except Exception as e:
                print(e)

            if SAVE_IMAGES:
                if i % 20 == 0:
                    cv2.imwrite(
                        f"camera_observations/{str(i)}.jpg",
                        (
                            o["image"].get()[..., -1]
                            if env.config["image_on_cuda"]
                            else o["image"][..., -1]
                        )
                        * 255,
                    )

            # cv2.waitKey(1)

            if (tm or tc) and info["arrive_dest"]:
                env.reset(env.current_seed + 1)
                env.current_track_agent.expert_takeover = True
    finally:
        env.close()

