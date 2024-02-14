from metadrive.examples import expert
from metadrive.policy.base_policy import BasePolicy
from metadrive.engine.engine_utils import get_global_config
from metadrive.obs.image_obs import ImageStateObservation
from metadrive.utils.math import not_zero
from PIL import Image
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.getcwd()))
from inference import ONNXPipeline


class LaneDetectionPolicy(BasePolicy):

    def __init__(self):
        super(LaneDetectionPolicy, self).__init__()
        self.onnx_pipeline = ONNXPipeline()
        self.camera_observation = ImageStateObservation(get_global_config().copy())

    def act(self, agent_id=None):
        action = expert()
        self.action_info["action"] = action
        return action

    def expert(self):

        print(self.control_object)

        # get RGB camera image from vehicle
        observation = self.camera_observation.observe()
        image = Image.fromarray((observation["image"][..., -1] * 255).astype(np.uint8))
        image_size = (image.width, image.height)

        # infer steering angle from image
        steering_angle = self.onnx_pipeline.infer_steering_angle(image, image_size)

        action = [steering_angle, self.acceleration()]

        print(action)
        # TODO: check steering_control() in metadrive.policy.idm_policy

        return action

    def acceleration(self) -> float:
        ego_vehicle = self.control_object
        ego_target_speed = not_zero(self.target_speed, 0)
        acceleration = self.ACC_FACTOR * (
            1 - np.power(max(ego_vehicle.speed_km_h, 0) / ego_target_speed, self.DELTA)
        )
        return acceleration
