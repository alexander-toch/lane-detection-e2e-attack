import queue
import threading
import cv2
import datetime
from metadrive.engine.engine_utils import get_global_config
from metadrive.utils.math import wrap_to_pi
from PIL import Image
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import sys, os
import torch

from metadrive_policy.lanedetection_policy_dpatch import LaneDetectionPolicy
from metadrive.obs.image_obs import ImageObservation

sys.path.append(os.path.dirname(os.getcwd()))
from inference_pytorch import PyTorchPipeline
from lanefitting import draw_lane, draw_lane_bev, get_ipm_via_camera_config

TARGET=np.load("attack/targets/right_new.npy", allow_pickle=True).item()
REGENERATE_INTERVAL = 1000

class LaneDetectionPolicyE2E(LaneDetectionPolicy):
    NORMAL_SPEED = 100  # km/h
    ACC_FACTOR = 3.0

    def __init__(self, control_object, random_seed=None, config=None):
        super(LaneDetectionPolicyE2E, self).__init__(control_object, random_seed, config, init_pipeline=False)
        self.io_tasks = queue.Queue()
        self.stop_event = threading.Event()
        self.io_thread = threading.Thread(target=self.io_worker, daemon=True)
        self.io_thread.start()
        self.ipm = None
        self.step_infos = []
        
        w, h = get_global_config()["window_size"][0], get_global_config()["window_size"][1]
        patch_w, patch_h = get_global_config()["patch_size_meters"]
        PX_PER_METER = (80, 80) # TODO: find these values dynamically
        self.patch_size = (int(patch_w * PX_PER_METER[0]), int(patch_h * PX_PER_METER[1]))  # (width, height), will be swapped for Pipeline
        # self.patch_location=(int(w/2 - self.patch_size[0] / 2), int(h - self.patch_size[1])) # in format (W, H). (800, 288) is input size for resa

        # print(f"Window size: {w}, {h}, Patch size (px): {self.patch_size}, location: {self.patch_location}")

        self.lane_detection_model = get_global_config()["lane_detection_model"]
        model_path = get_global_config()["custom_model_path"]
        self.pipeline = PyTorchPipeline(targeted=True, patch_size=(self.patch_size[1], self.patch_size[0]), 
                                        max_iterations=get_global_config()["patch_geneneration_iterations"],
                                        model=self.lane_detection_model,
                                        model_path=model_path)

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


        image_on_cuda = get_global_config()["image_on_cuda"]
        attack_at_meter = get_global_config()["dirty_road_patch_attack_at_meter"]
        attack_active = get_global_config()["enable_dirty_road_patch_attack"]
        place_patch_in_image_stream = get_global_config()["place_patch_in_image_stream"]
        generate_training_data = get_global_config()["generate_training_data"]

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
        current_car_pos_meter = self.control_object.origin.getPos()[0] # FIXME: this only works for straight roads, use navigation.travelled_length ?
        current_car_center_offset = self.control_object.dist_to_center

        if (not place_patch_in_image_stream and int(current_car_pos_meter) == attack_at_meter and self.control_object.engine.dirty_road_patch_object is None) or \
            (place_patch_in_image_stream and attack_active and int(current_car_pos_meter) >= attack_at_meter):
            regenerate_patch = True if place_patch_in_image_stream and self.control_object.engine.episode_step % REGENERATE_INTERVAL == 0 else False
            offset_center, _, keypoints, patch_object = (
                self.pipeline.infer_offset_center_with_dpatch(image, (image_size[1], image_size[0]), self.control_object, regenerate_patch, target=self.target, image_on_cuda=image_on_cuda, optimize_runtime=place_patch_in_image_stream) # important: swap image_size order
            )
                
            debug_info = {}
            if not place_patch_in_image_stream:
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

            if not place_patch_in_image_stream:
                offset_center, _, keypoints, debug_info = (
                    self.pipeline.infer_offset_center(image, (image_size[1], image_size[0]), self.control_object, image_on_cuda, self.ipm, return_model_input=generate_training_data) # important: swap image_size order
                )
        else: 
            if attack_active and get_global_config()["patch_color_replace"]:
                # find white pixels
                image_cpu = image.get() * 255 if image_on_cuda else image
                white = np.where(np.all(image_cpu == [255, 255, 255], axis=-1))

                # find bounding box if we have white pixel
                if white[0].shape[0] > 1 and white[1].shape[0] > 1 and self.control_object.engine.dirty_road_patch_object is not None:
                    min_x = np.min(white[1])
                    max_x = np.max(white[1])
                    min_y = np.min(white[0])
                    max_y = np.max(white[0])

                    # only place patch if the bounding box is greater 10x10
                    if max_x - min_x > 10 and max_y - min_y > 10:
                        # resize patch to fit bounding box
                        patch = cp.asarray(cv2.resize((self.control_object.engine.dirty_road_patch_object.patch * 255).astype(np.uint8), (max_x - min_x, max_y - min_y), interpolation=cv2.INTER_NEAREST_EXACT))
                        # fill area with patch
                        # (H, W, C)
                        image[min_y:max_y, min_x:max_x] = patch

                del white
                del image_cpu


            offset_center, _, keypoints, debug_info = (
                self.pipeline.infer_offset_center(image, (image_size[1], image_size[0]), self.control_object, image_on_cuda, self.ipm, return_model_input=generate_training_data) # important: swap image_size order
            )


        # TODO: scale steering value based on offset_center

        STEERING_VALUE_RAD = np.deg2rad(20)
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

        self.step_infos.append({
            "step": self.control_object.engine.episode_step,
            "time": datetime.datetime.now(),
            "offset_center": offset_center,	
            "offset_center_simulator": current_car_center_offset,
            "car_position_x_meter": current_car_pos_meter,
            "steering": steering,
            "speed": self.control_object.speed_km_h,
            "throttle_brake": self.control_object.throttle_brake,
            "model_input": debug_info["model_input"] if "model_input" in debug_info else None,
        })

        # TODO: add a flag to enable image saving and interval
        if self.control_object.engine.episode_step % 100 == 0:
            print(f"Step: {self.control_object.engine.episode_step}, offset_center: {offset_center} (sim: {current_car_center_offset}), steering: {steering}")
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

    def destroy(self):
        super(LaneDetectionPolicyE2E, self).destroy()
        self.pipeline.destroy()
