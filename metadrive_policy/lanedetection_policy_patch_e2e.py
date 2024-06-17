import pickle
import queue
import threading
import cv2
from metadrive.policy.base_policy import BasePolicy
from metadrive.engine.engine_utils import get_global_config
from metadrive.obs.image_obs import ImageStateObservation
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.policy.manual_control_policy import KeyboardController, get_controller
from metadrive.utils.math import not_zero, wrap_to_pi
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys, os

from metadrive_policy.lanedetection_policy_dpatch import LaneDetectionPolicy

sys.path.append(os.path.dirname(os.getcwd()))
from inference_pytorch import PyTorchPipeline
from lanefitting import draw_lane

TARGET=np.load("attack/targets/turn_right.npy", allow_pickle=True).item()

class LaneDetectionPolicyE2E(LaneDetectionPolicy):
    def __init__(self, control_object, random_seed=None, config=None):
        super(LaneDetectionPolicyE2E, self).__init__(control_object, random_seed, config)
        self.io_tasks = queue.Queue()
        self.stop_event = threading.Event()
        self.io_thread = threading.Thread(target=self.io_worker, daemon=True)
        self.io_thread.start()

    def io_worker(self):
        while not self.stop_event.isSet():
            try:
                task = self.io_tasks.get(block=True, timeout=1)
                task()
                self.io_tasks.task_done()
            except queue.Empty:
                pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        self.io_thread.join()

    def expert(self):

        if not self.engine.current_track_agent.expert_takeover:
            action = self.controller.process_input(self.engine.current_track_agent)
            self.action_info["manual_control"] = True
            return action
        
        # get RGB camera image from vehicle
        observation = self.camera_observation.observe(self.control_object)

        image_on_cuda = get_global_config()["image_on_cuda"]
        attack_step_index = get_global_config()["dirty_road_patch_attack_step_index"]

        if not image_on_cuda:
            image = Image.fromarray((observation["image"][..., -1] * 255).astype(np.uint8))
            image_size = (image.width, image.height)
        else:
            image = observation["image"][..., -1]
            image_size = (image.shape[1], image.shape[0])


        offset_center, lane_heading_theta, keypoints, debug_info = (
                self.pipeline.infer_offset_center(image, (image_size[1], image_size[0]), self.control_object, image_on_cuda) # important: swap image_size order
            )

        if self.control_object.engine.episode_step == attack_step_index:
            _, _, _, patch_object = (
                self.pipeline.infer_offset_center_with_dpatch(image, (image_size[1], image_size[0]), self.control_object, True, target=self.target, image_on_cuda=image_on_cuda) # important: swap image_size order
            )

            self.control_object.engine.dirty_road_patch_object = patch_object

            # pickle.dump(patch, open(f"camera_observations/seed_{self.control_object.engine.seed}_patch.pkl", "wb"))


        v_heading = self.control_object.heading_theta  # current vehicle heading
        # steering = self.heading_pid.get_result(
        #     -wrap_to_pi(lane_heading_theta - v_heading)
        # )

        STEERING_VALUE_RAD = np.deg2rad(15)
        self.target_speed = self.NORMAL_SPEED
        if offset_center is None:
            steering = self.lateral_pid.get_result(0)
            # brake if no lane detected
            self.target_speed = 0.01
        elif offset_center > 0.01:
            steering = self.lateral_pid.get_result(-wrap_to_pi(-STEERING_VALUE_RAD)) # radian in range (-pi, pi]
        elif offset_center < -0.01:
            steering = self.lateral_pid.get_result(-wrap_to_pi(+STEERING_VALUE_RAD)) # radian in range (-pi, pi]
        else:
            steering = self.lateral_pid.get_result(0)

        action = [steering, self.acceleration()]
        # action = [0, self.acceleration()] # for disbling steering

        # TODO: add a flag to enable image saving and interval
        if self.control_object.engine.episode_step % 10 == 0:
            print(f"Step: {self.control_object.engine.episode_step}, offset_center: {offset_center}, lane_heading_theta: {lane_heading_theta}, v_heading: {v_heading}, steering: {steering}")
            lane_image = draw_lane(image.get() * 255 if image_on_cuda else image, keypoints, image_size) # swap image_size
            if lane_image is not None:
                self.io_tasks.put(lambda: cv2.imwrite(
                        f"camera_observations/lane_{str(self.control_object.engine.episode_step)}.jpg",
                        lane_image
                    )
                )
            else:
                print(f"step {str(self.control_object.engine.episode_step)} lane_image is None")



        #     if 'probmaps' in debug_info:
        #         prob_maps = torch.nn.functional.interpolate(debug_info['probmaps']['out'], 
        #                                                     size=(image_size[1], image_size[0]), mode='bilinear', align_corners=True)
        #         prob_maps_softmax = prob_maps.detach().clone().softmax(dim=1)
                
        #         merged = np.zeros_like(prob_maps[0][1].detach().cpu().numpy())
        #         merged_softmax = np.zeros_like(prob_maps_softmax[0][1].detach().cpu().numpy())

        #         for i, lane in enumerate(prob_maps[0]):
        #             if i == 0: # skip first iteration (background class)
        #                 continue
        #             pred = lane.detach().cpu().numpy()
        #             pred_softmax = prob_maps_softmax[0][i].detach().cpu().numpy()
        #             merged = np.maximum(merged, pred)
        #             merged_softmax = np.maximum(merged_softmax, pred_softmax)

        #         im = Image.fromarray(merged)
        #         im_softmax = Image.fromarray(merged_softmax)
        #         plt.imsave(f"camera_observations/probmap_{str(self.control_object.engine.episode_step)}_merged.jpg", im, cmap='seismic')
        #         plt.imsave(f"camera_observations/softmax_probmap_{str(self.control_object.engine.episode_step)}_merged.jpg", im_softmax, cmap='seismic')

        return action
