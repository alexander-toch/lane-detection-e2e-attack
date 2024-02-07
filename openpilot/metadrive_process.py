import math
import numpy as np

from collections import namedtuple
from panda3d.core import Vec3
from multiprocessing.connection import Connection

from metadrive.engine.core.engine_core import EngineCore
from metadrive.engine.core.image_buffer import ImageBuffer
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.image_obs import ImageObservation

from common import Ratekeeper, vec3

C3_POSITION = Vec3(0.0, 0, 1.22)
C3_HPR = Vec3(0, 0, 0)


metadrive_state = namedtuple(
    "metadrive_state", ["velocity", "position", "bearing", "steering_angle"]
)


def metadrive_process(
    config: dict,
    image_lock,
    controls_recv: Connection,
    state_send: Connection,
    exit_event,
):

    env = MetaDriveEnv(config)

    def reset():
        env.reset()
        env.vehicle.config["max_speed_km_h"] = 1000

    reset()

    rk = Ratekeeper(100, None)

    steer_ratio = 8
    vc = [0, 0]

    while not exit_event.is_set():
        state = metadrive_state(
            velocity=vec3(
                x=float(env.vehicle.velocity[0]), y=float(env.vehicle.velocity[1]), z=0
            ),
            position=env.vehicle.position,
            bearing=float(math.degrees(env.vehicle.heading_theta)),
            steering_angle=env.vehicle.steering * env.vehicle.MAX_STEERING,
        )

        state_send.send(state)

        if controls_recv.poll(0):
            while controls_recv.poll(0):
                steer_angle, gas, should_reset = controls_recv.recv()

            steer_metadrive = steer_angle * 1 / (env.vehicle.MAX_STEERING * steer_ratio)
            steer_metadrive = np.clip(steer_metadrive, -1, 1)

            vc = [steer_metadrive, gas]

            if should_reset:
                reset()

        if rk.frame % 5 == 0:
            obs, _, terminated, _, info = env.step(vc)

            if terminated:
                reset()

            # TODO: get image from obs
            image_lock.release()

        rk.keep_time()
