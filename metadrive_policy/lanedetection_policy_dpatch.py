import cv2
from metadrive.policy.base_policy import BasePolicy
from metadrive.engine.engine_utils import get_global_config
from metadrive.obs.image_obs import ImageStateObservation
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.utils.math import not_zero, wrap_to_pi
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.dirname(os.getcwd()))
from inference_pytorch import PyTorchPipeline
from lanefitting import draw_lane


class LaneDetectionPolicy(BasePolicy):
    MAX_SPEED = 100  # km/h
    NORMAL_SPEED = 30  # km/h
    ACC_FACTOR = 1.0
    DEACC_FACTOR = -5
    DELTA = 10.0  # Exponent of the velocity term

    def __init__(self, control_object, random_seed=None, config=None):
        super(LaneDetectionPolicy, self).__init__(control_object, random_seed, config)
        self.pipeline = PyTorchPipeline()
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

        if self.control_object.engine.episode_step < 100: # start attack after 100 steps # TODO: use variable
            offset_center, lane_heading_theta, keypoints, debug_info = (
                self.pipeline.infer_offset_center(image, (image_size[1], image_size[0]), self.control_object) # important: swap image_size order
            )
        else:
            # generate a fresh patch every 20 steps
            if self.control_object.engine.episode_step % 20 == 0:
                offset_center, lane_heading_theta, keypoints, debug_info = (
                    self.pipeline.infer_offset_center_with_dpatch(image, (image_size[1], image_size[0]), self.control_object, True) # important: swap image_size order
                )
            else:
                offset_center, lane_heading_theta, keypoints, debug_info = (
                    self.pipeline.infer_offset_center_with_dpatch(image, (image_size[1], image_size[0]), self.control_object, False) # important: swap image_size order
                )

        v_heading = self.control_object.heading_theta  # current vehicle heading
        # steering = self.heading_pid.get_result(
        #     -wrap_to_pi(lane_heading_theta - v_heading)
        # )

        STEERING_VALUE_RAD = np.deg2rad(15)
        if offset_center is None:
            steering = self.lateral_pid.get_result(0)
        elif offset_center > 0.01:
            steering = self.lateral_pid.get_result(-wrap_to_pi(-STEERING_VALUE_RAD)) # radian in range (-pi, pi]
        elif offset_center < -0.01:
            steering = self.lateral_pid.get_result(-wrap_to_pi(+STEERING_VALUE_RAD)) # radian in range (-pi, pi]
        else:
            steering = self.lateral_pid.get_result(0)

        action = [steering, self.acceleration()]
        # action = [0, self.acceleration()] # for disbling steering

        # TODO: add a flag to enable image saving and interval
        if self.control_object.engine.episode_step % 10 == 0 or True:
            print(f"Step: {self.control_object.engine.episode_step}, offset_center: {offset_center}, lane_heading_theta: {lane_heading_theta}, v_heading: {v_heading}, steering: {steering}")
            lane_image = draw_lane(image, keypoints, image_size) # swap image_size

            # lane iamge format is (H, W, C), (720, 1280, 3)
            if 'patch' in debug_info and 'location' in debug_info:
                x_1, y_1 = debug_info['location']
                x_2, y_2 = x_1 + debug_info['patch'].shape[0], y_1 + debug_info['patch'].shape[1]
                lane_image[x_1:x_2, y_1:y_2, :] = debug_info['patch']

            if lane_image is not None:
                cv2.imwrite(
                    f"camera_observations/lane_{str(self.control_object.engine.episode_step)}.jpg",
                    lane_image
                )
            else:
                print(f"step {str(self.control_object.engine.episode_step)} lane_image is None")



            if 'probmaps' in debug_info:
                prob_maps = torch.nn.functional.interpolate(debug_info['probmaps']['out'], 
                                                            size=(image_size[1], image_size[0]), mode='bilinear', align_corners=True)
                prob_maps_softmax = prob_maps.detach().clone().softmax(dim=1)
                
                merged = np.zeros_like(prob_maps[0][1].detach().cpu().numpy())
                merged_softmax = np.zeros_like(prob_maps_softmax[0][1].detach().cpu().numpy())

                for i, lane in enumerate(prob_maps[0]):
                    if i == 0: # skip first iteration (background class)
                        continue
                    pred = lane.detach().cpu().numpy()
                    pred_softmax = prob_maps_softmax[0][i].detach().cpu().numpy()
                    # plt.imsave(f"camera_observations/probmap_{str(self.control_object.engine.episode_step)}_{i}.jpg", pred, cmap='seismic')
                    merged = np.maximum(merged, pred)
                    merged_softmax = np.maximum(merged_softmax, pred_softmax)

                np.save(f"camera_observations/probmap_{str(self.control_object.engine.episode_step)}.npy", debug_info['probmaps'])

                im = Image.fromarray(merged)
                im_softmax = Image.fromarray(merged)
                plt.imsave(f"camera_observations/probmap_{str(self.control_object.engine.episode_step)}_merged.jpg", im, cmap='seismic')
                plt.imsave(f"camera_observations/softmax_probmap_{str(self.control_object.engine.episode_step)}_merged.jpg", im_softmax, cmap='seismic')

        return action

    def acceleration(self) -> float:
        ego_vehicle = self.control_object
        ego_target_speed = not_zero(self.target_speed, 0)
        acceleration = self.ACC_FACTOR * (
            1 - np.power(max(ego_vehicle.speed_km_h, 0) / ego_target_speed, self.DELTA)
        )
        return acceleration
