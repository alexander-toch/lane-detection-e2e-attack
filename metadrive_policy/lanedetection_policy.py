import cv2
from metadrive.policy.base_policy import BasePolicy
from metadrive.engine.engine_utils import get_global_config
from metadrive.obs.image_obs import ImageStateObservation
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.utils.math import not_zero, wrap_to_pi
from PIL import Image
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.getcwd()))
from inference import ONNXPipeline
from lanefitting import draw_lane


class LaneDetectionPolicy(BasePolicy):
    MAX_SPEED = 100  # km/h
    NORMAL_SPEED = 30  # km/h
    ACC_FACTOR = 1.0
    DEACC_FACTOR = -5
    DELTA = 10.0  # Exponent of the velocity term

    def __init__(self, control_object, random_seed=None, config=None):
        super(LaneDetectionPolicy, self).__init__(control_object, random_seed, config)
        self.onnx_pipeline = ONNXPipeline()
        self.camera_observation = ImageStateObservation(get_global_config().copy())
        self.target_speed = self.NORMAL_SPEED
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, 0.002, 0.05)

    def act(self, agent_id=None):
        action = self.expert()
        self.action_info["action"] = action
        return action

    def expert(self):

        # get RGB camera image from vehicle
        observation = self.camera_observation.observe(self.control_object)
        image = Image.fromarray((observation["image"][..., -1] * 255).astype(np.uint8))
        image_size = (image.width, image.height)

        offset_center, lane_heading_theta, keypoints = (
            self.onnx_pipeline.infer_offset_center(image, (image_size[1], image_size[0])) # important: swap image_size order
        )

        
        if offset_center is None:
            return [0, self.acceleration()]

        v_heading = self.control_object.heading_theta  # current vehicle heading
        # steering = self.heading_pid.get_result(
        #     -wrap_to_pi(lane_heading_theta - v_heading)
        # )

        STEERING_VALUE_RAD = np.deg2rad(15)
        if offset_center > 0.01:
            steering = self.lateral_pid.get_result(-wrap_to_pi(-STEERING_VALUE_RAD)) # radian in range (-pi, pi]
        elif offset_center < -0.01:
            steering = self.lateral_pid.get_result(-wrap_to_pi(+STEERING_VALUE_RAD)) # radian in range (-pi, pi]
        else:
            steering = self.lateral_pid.get_result(0)

        action = [steering, self.acceleration()]
        # action = [0, self.acceleration()] # for disbling steering

        # TODO: add a flag to enable image saving
        if self.control_object.engine.episode_step % 10 == 0:
            print(f"Step: {self.control_object.engine.episode_step}, offset_center: {offset_center}, lane_heading_theta: {lane_heading_theta}, v_heading: {v_heading}, steering: {steering}")
            lane_image = draw_lane(image, keypoints, image_size) # swap image_size
            if lane_image is not None:
                cv2.imwrite(
                    f"camera_observations/lane_{str(self.control_object.engine.episode_step)}.jpg",
                    lane_image
                )

        return action

    def acceleration(self) -> float:
        ego_vehicle = self.control_object
        ego_target_speed = not_zero(self.target_speed, 0)
        acceleration = self.ACC_FACTOR * (
            1 - np.power(max(ego_vehicle.speed_km_h, 0) / ego_target_speed, self.DELTA)
        )
        return acceleration
