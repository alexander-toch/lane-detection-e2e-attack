import pickle
import queue
import threading
import cv2
from metadrive.engine.engine_utils import get_global_config
from metadrive.utils.math import wrap_to_pi
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import torch

from metadrive_policy.lanedetection_policy_dpatch import LaneDetectionPolicy

sys.path.append(os.path.dirname(os.getcwd()))
from inference_pytorch import PyTorchPipeline
from lanefitting import draw_lane, draw_lane_bev, get_ipm_via_camera_config

TARGET=np.load("attack/targets/turn_right.npy", allow_pickle=True).item()
REGENERATE_INTERVAL = 150

class LaneDetectionPolicyE2E(LaneDetectionPolicy):
    def __init__(self, control_object, random_seed=None, config=None):
        super(LaneDetectionPolicyE2E, self).__init__(control_object, random_seed, config, init_pipeline=False)
        self.io_tasks = queue.Queue()
        self.stop_event = threading.Event()
        self.io_thread = threading.Thread(target=self.io_worker, daemon=True)
        self.io_thread.start()
        self.ipm = None
        
        w, h = get_global_config()["window_size"][0], get_global_config()["window_size"][1]
        patch_w, patch_h = get_global_config()["patch_size_meters"]
        PX_PER_METER = (80, 80) # TODO: find these values dynamically
        self.patch_size = (int(patch_w * PX_PER_METER[0]), int(patch_h * PX_PER_METER[1]))  # (width, height), will be swapped for Pipeline
        self.patch_location=(int(w/2 - self.patch_size[0] / 2), int(h - self.patch_size[1])) # in format (W, H). (800, 288) is input size for resa

        self.pipeline = PyTorchPipeline(targeted=True, patch_size=(self.patch_size[1], self.patch_size[0]), 
                                        patch_location=self.patch_location, max_iterations=get_global_config()["patch_geneneration_iterations"])

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

    def get_probmap_images(self, probmaps, image_size):
        prob_maps = torch.nn.functional.interpolate(probmaps['out'], 
                                                    size=(image_size[1], image_size[0]), mode='bilinear', align_corners=True)
        prob_maps_softmax = prob_maps.detach().clone().softmax(dim=1)
        
        merged = np.zeros_like(prob_maps[0][1].detach().cpu().numpy())
        merged_softmax = np.zeros_like(prob_maps_softmax[0][1].detach().cpu().numpy())

        for i, lane in enumerate(prob_maps[0]):
            if i == 0: # skip first iteration (background class)
                continue
            pred = lane.detach().cpu().numpy()
            pred_softmax = prob_maps_softmax[0][i].detach().cpu().numpy()
            merged = np.maximum(merged, pred)
            merged_softmax = np.maximum(merged_softmax, pred_softmax)

        im = Image.fromarray(merged)
        im_softmax = Image.fromarray(merged_softmax)

        return im, im_softmax

    def expert(self):

        if not self.engine.current_track_agent.expert_takeover:
            action = self.controller.process_input(self.engine.current_track_agent)
            self.action_info["manual_control"] = True
            return action
        
        # get RGB camera image from vehicle
        observation = self.camera_observation.observe(self.control_object)

        # top down camera (# 10 in driving direction, 50 height, yaw -90 degree)
        # observation = self.camera_observation.observe(self.control_object, position=(0., 10., 50.), hpr=(0., -90.0, 0.0)) 


        image_on_cuda = get_global_config()["image_on_cuda"]
        attack_step_index = get_global_config()["dirty_road_patch_attack_step_index"]

        if not image_on_cuda:
            image = Image.fromarray((observation["image"][..., -1] * 255).astype(np.uint8))
            image_size = (image.width, image.height)
        else:
            image = observation["image"][..., -1]
            image_size = (image.shape[1], image.shape[0])

        sensor = self.control_object.engine.get_sensor("rgb_camera")
        lens = sensor.get_lens()
        fov_angle = lens.getFov()

        fx =  get_global_config()["window_size"][0]  / (2 * np.tan(fov_angle[0] * np.pi / 360))
        fy = get_global_config()["window_size"][1] / (2 * np.tan(fov_angle[1] * np.pi / 360))

        if self.ipm is None:
            ipm_input_image = None
            if image_on_cuda:
                ipm_input_image = image.get() * 255
            else:
                ipm_input_image = image.permute((2, 0, 1)).contiguous().float().div(255).unsqueeze(0).numpy()
            self.ipm = get_ipm_via_camera_config(ipm_input_image, fx, fy)

        patch_object = None
        if self.control_object.engine.episode_step == attack_step_index or get_global_config()["place_patch_in_image_stream"] is True:
            regenerate_patch = True if get_global_config()["place_patch_in_image_stream"] is True and self.control_object.engine.episode_step % REGENERATE_INTERVAL == 0 else False
            offset_center, _, keypoints, patch_object = (
                self.pipeline.infer_offset_center_with_dpatch(image, (image_size[1], image_size[0]), self.control_object, regenerate_patch, target=self.target, image_on_cuda=image_on_cuda) # important: swap image_size order
            )
            debug_info = {
                'probmaps': patch_object.probmaps,
            }

            self.control_object.engine.dirty_road_patch_object = patch_object

            self.io_tasks.put(lambda: self.pipeline.save_image(
                    patch_object.model_in,
                    f"camera_observations/patched_input_{str(self.control_object.engine.episode_step)}.png",
                )
            )

            im, im_softmax = self.get_probmap_images(patch_object.probmaps, image_size)
            plt.imsave(f"camera_observations/patched_probmap_{str(self.control_object.engine.episode_step)}_merged.png", im, cmap='seismic')
            plt.imsave(f"camera_observations/patched_softmax_probmap_{str(self.control_object.engine.episode_step)}_merged.png", im_softmax, cmap='seismic')
        else: 
            offset_center, _, keypoints, debug_info = (
                self.pipeline.infer_offset_center(image, (image_size[1], image_size[0]), self.control_object, image_on_cuda, self.ipm) # important: swap image_size order
            )

        # v_heading = self.control_object.heading_theta  # current vehicle heading
        # steering = self.heading_pid.get_result(
        #     -wrap_to_pi(lane_heading_theta - v_heading)
        # )

        STEERING_VALUE_RAD = np.deg2rad(15)
        self.target_speed = self.NORMAL_SPEED
        if offset_center is None:
            steering = self.lateral_pid.get_result(0)
            # brake if no lane detected
            self.target_speed = 0.01
        elif offset_center > 2:
            # steer to the left
            steering = self.lateral_pid.get_result(-wrap_to_pi(-STEERING_VALUE_RAD)) # radian in range (-pi, pi]
        elif offset_center < -2:
            # steer to the right
            steering = self.lateral_pid.get_result(-wrap_to_pi(+STEERING_VALUE_RAD)) # radian in range (-pi, pi]
        else:
            steering = self.lateral_pid.get_result(0)

        action = [steering, self.acceleration()]
        # action = [0, self.acceleration()] # for disbling steering

        # TODO: add a flag to enable image saving and interval
        if self.control_object.engine.episode_step % 10 == 0:
            print(f"Step: {self.control_object.engine.episode_step}, offset_center: {offset_center}, steering: {steering}")
            if patch_object is not None and patch_object.input_image_sizes == tuple(reversed(image_size)):
                im = np.array(patch_object.model_in).transpose((1, 2, 0))
                im = (im * 255).astype(np.uint8)
                # I don't know why this is necessary 2 times, but it is
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            else:
                im = image.get() * 255 if image_on_cuda else image
            lane_image = draw_lane(im, keypoints, image_size, self.ipm, draw_lane_overlay=False) # swap image_size 
            lane_image_bev = draw_lane_bev(im, keypoints, image_size, self.ipm) 
            if lane_image is not None:
                self.io_tasks.put(lambda: cv2.imwrite(
                        f"camera_observations/lane_{str(self.control_object.engine.episode_step)}.png",
                        lane_image
                    )
                )
                # cv2.imshow("lane", lane_image_bev)
                # cv2.waitKey(10)
                self.io_tasks.put(lambda: cv2.imwrite(
                        f"camera_observations/lane_bev_{str(self.control_object.engine.episode_step)}.png",
                        lane_image_bev
                    )
                )
            else:
                print(f"step {str(self.control_object.engine.episode_step)} lane_image is None")

            if 'probmaps' in debug_info and get_global_config()["save_probmaps"] is True:
                im, im_softmax = self.get_probmap_images(debug_info['probmaps'], image_size)
                plt.imsave(f"camera_observations/probmap_{str(self.control_object.engine.episode_step)}_merged.png", im, cmap='seismic')
                plt.imsave(f"camera_observations/softmax_probmap_{str(self.control_object.engine.episode_step)}_merged.png", im_softmax, cmap='seismic')
                np.save(f"camera_observations/probmap_{str(self.control_object.engine.episode_step)}.npy", debug_info['probmaps'])

        return action
