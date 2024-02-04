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

W, H = 1280, 720  #  Desired output size of annotated images

SAVE_IMAGES = True


class CopyRamRGBCamera(RGBCamera):
    """Camera which copies its content into RAM during the render process, for faster image grabbing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpu_texture = Texture()
        self.buffer.addRenderTexture(self.cpu_texture, GraphicsOutput.RTMCopyRam)

    def get_rgb_array_cpu(self):
        origin_img = self.cpu_texture
        img = np.frombuffer(origin_img.getRamImage().getData(), dtype=np.uint8)
        img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), -1))
        img = img[:, :, :3]  # RGBA to RGB
        # img = np.swapaxes(img, 1, 0)
        img = img[::-1]  # Flip on vertical axis
        return img


class RGBCameraRoad(CopyRamRGBCamera):
    def __init__(self, *args, **kwargs):
        super(RGBCameraRoad, self).__init__(*args, **kwargs)
        # lens = self.get_lens()
        # lens.setFov(40)
        # lens.setNear(0.1)


if __name__ == "__main__":
    # Adapted from OpenPilot's bridge

    map_config = {
        "config": "SSCrSCS",
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
        # BaseMap.GENERATE_CONFIG: 3,
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 2,
    }

    config = dict(
        use_render=True,
        vehicle_config=dict(
            enable_reverse=False,
            image_source="rgb_road",
        ),
        sensors={
            "rgb_road": (
                RGBCameraRoad,
                W,
                H,
            )
        },
        image_on_cuda=_cuda_enable,
        image_observation=True,
        interface_panel=[],
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

    env = MetaDriveEnv(config)
    try:
        o, _ = env.reset()
        print(HELP_MESSAGE)
        env.vehicle.expert_takeover = True

        assert isinstance(o, dict)
        print(
            "The observation is a dict with numpy arrays as values: ",
            {k: v.shape for k, v in o.items()},
        )

        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 0])
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": (
                        "on" if env.current_track_vehicle.expert_takeover else "off"
                    ),
                    "Keyboard Control": "W,A,S,D",
                }
            )

            # Show the last RGB image in the observation
            cv2.imshow("RGB Image in Observation", o["image"][..., -1])

            if SAVE_IMAGES:
                if i % 20 == 0:
                    cv2.imwrite(
                        f"{str(i)}.jpg",
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
                env.current_track_vehicle.expert_takeover = True
    finally:
        env.close()
