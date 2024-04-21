"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""

import argparse, sys
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
from config import *
from PIL import Image
ATTACK = True
if ATTACK:
    from metadrive_policy.lanedetection_policy_dpatch import LaneDetectionPolicy
else:
    from metadrive_policy.lanedetection_policy import LaneDetectionPolicy

import pytorch_auto_drive.functional as F
from utils import dummy_env

W, H = 1280, 720  #  Desired output size of annotated images

HEADLESS = True
SAVE_IMAGES = True
SEED=1234
MAP_CONFIG = "SCS"

print(f"Using CUDA: {_cuda_enable}")
print(f"Headless mode: {HEADLESS}")


# from https://metadrive-simulator.readthedocs.io/en/latest/points_and_lines.html#points
def make_line(x_offset, height, y_dir=1, color=(1, 105 / 255, 180 / 255)):
    points = [(x_offset + x, x * y_dir, height * x / 10 + height) for x in range(10)]
    colors = [
        np.clip(np.array([*color, 1]) * (i + 1) / 11, 0.0, 1.0) for i in range(10)
    ]
    if y_dir < 0:
        points = points[::-1]
        colors = colors[::-1]
    return points, colors


def draw_keypoints(drawer, keypoints):
    line_1, color_1 = make_line(6, 0.5, 0.01 * i)  # define line 1 for test
    line_2, color_2 = make_line(6, 0.5, -0.01 * i)  # define line 2 for test
    points = line_1 + line_2  # create point list
    colors = color_1 + color_2
    drawer.reset()
    drawer.draw_points(points, colors)  # draw points


if __name__ == "__main__":

    
    map_config = {
        "config": MAP_CONFIG, # S=Straight, C=Circular/Curve
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
        agent_policy=LaneDetectionPolicy,
        start_seed=SEED,
        image_on_cuda=False, # TODO: check if this is fixable
        image_observation=True,
        out_of_route_done=True,
        on_continuous_line_done=True,
        crash_vehicle_done=True,
        crash_object_done=True,
        crash_human_done=True,
        traffic_density=0.0,
        map_config=map_config,
        num_scenarios=1,
        decision_repeat=1,
        # physics_world_step_size=self.TICKS_PER_FRAME / 100, # Physics world step is 0.02s and will be repeated for decision_repeat times per env.step()
        preload_models=False,
        manual_control=False,
    )

    dummy_env()

    env = MetaDriveEnv(config)

    try:
        env.reset()
        for i in range(10):
            o, r, tm, tc, infos = env.step([0, 1])
        assert isinstance(o, dict)
        point_drawer = env.engine.make_point_drawer(scale=1)  # create a point drawer
        print(HELP_MESSAGE)

        step_index = 0
        while True:
            o, r, tm, tc, info = env.step([0,0])

            if not HEADLESS:
                env.render(
                    text={
                        "Auto-Drive (Switch mode: T)": (
                            "on" if env.current_track_agent.expert_takeover else "off"
                        ),
                        "Keyboard Control": "W,A,S,D",
                    }
                )

            if SAVE_IMAGES:
                if step_index % 20 == 0:
                    cv2.imwrite(
                        f"camera_observations/{str(step_index)}.jpg",
                        (
                            o["image"].get()[..., -1]
                            if env.config["image_on_cuda"]
                            else o["image"][..., -1]
                        )
                        * 255,
                    )

            if tm or tc:
                env.reset(env.current_seed + 1)            
            step_index += 1
    finally:
        env.close()
