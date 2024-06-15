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
from metadrive_policy.lanedetection_policy_patch_e2e import LaneDetectionPolicyE2E
import pytorch_auto_drive.functional as F
from utils import dummy_env

# W, H = 1640, 590  #  Desired output size of annotated images
W, H = 1280, 720  #  Desired output size of annotated images

HEADLESS = False
SAVE_IMAGES = False
SEED=1235 # was 1234
MAP_CONFIG = "SCS" # SCS worked quite well
ATTACK_INDEX = 100
MAX_STEPS = ATTACK_INDEX + 200

print(f"Using CUDA: {_cuda_enable}")
print(f"Headless mode: {HEADLESS}")

def run_simulation(env):
    env.reset()
    env.current_track_agent.expert_takeover = True
    for i in range(10):
        o, r, tm, tc, infos = env.step([0, 1])
    assert isinstance(o, dict)

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

        if tm or tc or step_index >= MAX_STEPS:
            # env.reset(env.current_seed + 1)
            print(f"Simulation ended at step {step_index}")
            break            
        step_index += 1


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
        agent_policy=LaneDetectionPolicyE2E,
        start_seed=SEED,
        image_on_cuda=True,
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
        manual_control=True,
    )

    # ATTACK STEP 1: Drive without attack and generate the 
    config["enable_dirty_road_patch_attack"] = False
    config["dirty_road_patch_attack_step_index"] = ATTACK_INDEX

    env = MetaDriveEnv(config)
    run_simulation(env)
    env.close()

    config["enable_dirty_road_patch_attack"] = True
    env = MetaDriveEnv(config)
    run_simulation(env)
