
import datetime
import gc
import os
import shutil
import glob
import pickle
from enum import Enum
from dataclasses import dataclass, field
import queue
import threading
import cv2
import numpy as np
from metadrive import MetaDriveEnv, constants
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import TerminationState
from metadrive.constants import HELP_MESSAGE, MetaDriveType
from metadrive.component.map.base_map import BaseMap
from metadrive.component.sensors.base_camera import _cuda_enable
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock

from metadrive_policy.TopDownCamera import TopDownCamera
from metadrive_policy.lanedetection_policy_patch_e2e import LaneDetectionPolicyE2E
from metadrive_policy.lanedetection_policy_dpatch import LaneDetectionPolicy
from metadrive_policy.lane_camera import LaneCamera

from panda3d.core import Point2, Point3

from database import ExperimentDatabase

@dataclass
class AttackConfig:
    attack_at_meter: int = 6000 # set to -1 to disable patch rendering in MetaDrive
    two_pass_attack: bool = False
    place_patch_in_image_stream: bool = False
    patch_color_replace: bool = False
    load_patch_from_file: bool = False
    use_blur_defense: bool = False

@dataclass
class Settings:
    seed: int = 1235
    num_scenarios: int = 1
    map_config: str = "SCS"
    headless_rendering: bool = False
    save_images: bool = False
    save_probmaps: bool = False
    max_steps: int = 5000
    lane_detection_model: str = "resa"
    custom_model_path: str | None = None
    patch_size_meters: tuple[float, float] = (1.0, 1.0) # (width, height) in meters
    patch_geneneration_iterations: int = 90
    start_with_manual_control: bool = False
    simulator_window_size: tuple = (1280, 720) # (width, height)
    policy: str = "LaneDetectionPolicyE2E"
    attack_config: AttackConfig | None = field(default_factory=AttackConfig)
    lanes_per_direction: int = 2
    use_lane_camera: bool = False
    debug_lane_labels: bool = False
    generate_training_data: bool = False
    generate_training_data_interval: int = 10

class AttackType(str, Enum):
    none = 'None'
    onePassSimple = 'onePassSimple'
    onePassImageStream = 'onePassImageStream'
    onePassColorReplace = 'onePassColorReplace'
    twoPassSimple = 'twoPassSimple'
    twoPassImageStream = 'twoPassImageStream'
    twoPassColorReplace = 'twoPassColorReplace'

class MetaDriveBridge:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.map_config = {
            "config": self.settings.map_config, # S=Straight, C=Circular/Curve
            BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
            # BaseMap.GENERATE_CONFIG: 3,
            BaseMap.LANE_WIDTH: 3.5,
            BaseMap.LANE_NUM: self.settings.lanes_per_direction,
        }

        self.database = ExperimentDatabase("experiments.db")
        self.io_tasks = queue.Queue()
        self.stop_event = threading.Event()
        self.io_thread = threading.Thread(target=self.io_worker, daemon=True)
        self.io_thread.start()

        self.attack_type = AttackType.none
        if self.settings.attack_config is not None:
            if self.settings.attack_config.two_pass_attack:
                if self.settings.attack_config.place_patch_in_image_stream:
                    self.attack_type = AttackType.twoPassImageStream
                elif self.settings.attack_config.patch_color_replace:
                    self.attack_type = AttackType.twoPassColorReplace
                else:
                    self.attack_type = AttackType.twoPassSimple
            else:
                if self.settings.attack_config.place_patch_in_image_stream:
                    self.attack_type = AttackType.onePassImageStream
                elif self.settings.attack_config.patch_color_replace:
                    self.attack_type = AttackType.onePassColorReplace
                else:
                    self.attack_type = AttackType.onePassSimple

        self.experiment_id = self.database.add_experiment(datetime.datetime.now(), None, self.attack_type, self.settings.num_scenarios)
        self.train_ground_truth = []

        self.policy = LaneDetectionPolicyE2E if self.settings.policy == "LaneDetectionPolicyE2E" else LaneDetectionPolicy

        self.config  = dict(
            use_render=not self.settings.headless_rendering,
            window_size=self.settings.simulator_window_size,
            sensors={
                "rgb_camera": (RGBCamera, self.settings.simulator_window_size[0], self.settings.simulator_window_size[1]),
            },
            vehicle_config={
                "image_source": "rgb_camera",
                "show_navi_mark": False,
            },
            agent_configs={
                "default_agent": {
                    "spawn_lane_index": (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0 if self.settings.lanes_per_direction == 1 else 1),
                }
            },
            show_coordinates=False,
            random_spawn_lane_index=False,
            agent_policy=self.policy,
            start_seed=self.settings.seed,
            image_on_cuda=True,
            image_observation=True,
            out_of_route_done=False,
            on_continuous_line_done=True,
            crash_vehicle_done=True,
            crash_object_done=True,
            crash_human_done=True,
            crash_sidewalk_done=True,
            relax_out_of_road_done=False,
            traffic_density=0.0,
            map_config=self.map_config,
            num_scenarios=self.settings.num_scenarios,
            save_probmaps=self.settings.save_probmaps,
            patch_size_meters=self.settings.patch_size_meters,
            place_patch_in_image_stream=self.settings.attack_config.place_patch_in_image_stream if self.settings.attack_config is not None else False,
            patch_geneneration_iterations=self.settings.patch_geneneration_iterations,
            patch_color_replace=self.settings.attack_config.patch_color_replace if self.settings.attack_config is not None else False,
            use_blur_defense=self.settings.attack_config.use_blur_defense if self.settings.attack_config is not None else False,
            lane_detection_model=self.settings.lane_detection_model,
            custom_model_path=self.settings.custom_model_path,
            generate_training_data=self.settings.generate_training_data,
            decision_repeat=1,
            preload_models=False,
            manual_control=True,
            force_map_generation=True, # disables the PG Map cache
            show_fps=True,
            show_interface_navi_mark=False,
            interface_panel=["dashboard", "rgb_camera", "lane_camera"],
        )

        if self.settings.use_lane_camera:
            self.config["sensors"]["lane_camera"] = (LaneCamera, self.settings.simulator_window_size[0], self.settings.simulator_window_size[1])

    def cleanup(self):
        # delete all the previous camera observations
        for f in glob.glob("./camera_observations/*.jpg"):
            os.remove(f)
        for f in glob.glob("./camera_observations/*.png"):
            os.remove(f)
        for f in glob.glob("./camera_observations/*.npy"):
            os.remove(f)

    def run(self):
        self.cleanup()

        if self.settings.attack_config is not None:
            self.config["dirty_road_patch_attack_at_meter"]= self.settings.attack_config.attack_at_meter
            if self.settings.attack_config.two_pass_attack:
                self.run_two_pass_attack()
            else:
                self.config["enable_dirty_road_patch_attack"] = True if self.settings.attack_config.attack_at_meter > 0 else False
                env = MetaDriveEnv(self.config)
                self.run_simulation(env, generate_training_data=self.settings.generate_training_data)
        else:
            env = MetaDriveEnv(self.config)
            self.run_simulation(env, generate_training_data=self.settings.generate_training_data)

        self.database.finish_experiment(self.experiment_id, datetime.datetime.now())

        # wait for io tasks to finish
        while not self.io_tasks.empty():
            pass

    def run_two_pass_attack(self):
        self.cleanup()
        env = MetaDriveEnv(self.config)

        current_seed = self.settings.seed
        while current_seed < self.settings.seed + self.settings.num_scenarios:
            # ATTACK PASS 1: Drive without attack and generate the patch
            print(f"Pass 1: Running simulation with seed {current_seed} with attack disabled.")
            # self.config["enable_dirty_road_patch_attack"] = False
            if self.settings.attack_config.place_patch_in_image_stream:
                env.dirty_road_patch_attack_at_meter = -1
            if self.settings.generate_training_data:
                env.generate_training_data = False
            env.start_seed = current_seed
            env.start_index = current_seed
            env.num_scenarios = 1

            if not self.settings.attack_config.load_patch_from_file:
                self.run_simulation(env)
            else:
                pkl_path = f"./camera_observations/training_data/{self.settings.lane_detection_model}_seed_{current_seed}/patch_object.pkl"
                if not os.path.exists(pkl_path):
                    print(f"Pass 1+2 skipped due to no patch file found for seed {current_seed}.")
                    current_seed += 1
                    continue
                with open(pkl_path, 'rb') as f:
                    obj = pickle.load(f)
                env.reset(env.start_seed)
                env.engine.get_policy(env.agent.name).control_object.engine.dirty_road_patch_object = obj
                print(f"Pass 1 skipped and patch loaded from file for seed {current_seed}.")


            # ATTACK PASS 2: Drive with mounted patch
            print(f"Pass 2: Running simulation with seed {current_seed} with attack enabled.")
            env.engine.global_config["enable_dirty_road_patch_attack"] = True
            if self.settings.attack_config.place_patch_in_image_stream:
                env.engine.global_config["dirty_road_patch_attack_at_meter"] = self.settings.attack_config.attack_at_meter
            if self.settings.generate_training_data:
                env.engine.global_config["generate_training_data"] = True
            self.run_simulation(env, generate_training_data=self.settings.generate_training_data)

            if self.settings.generate_training_data and env.engine.get_policy(env.agent.name).control_object.engine.dirty_road_patch_object is not None:
                with open(f"./camera_observations/training_data/{self.settings.lane_detection_model}_seed_{current_seed}/patch_object.pkl", 'wb') as f:
                    pickle.dump(env.engine.get_policy(env.agent.name).control_object.engine.dirty_road_patch_object, f)
            
            current_seed += 1
            env.engine.global_config["enable_dirty_road_patch_attack"] = False
            env.engine.get_policy(env.agent.name).control_object.engine.dirty_road_patch_object = None

    def get_end_reason(self, info, step_index, steps, early_exit=False):
        if info[TerminationState.SUCCESS] or info[TerminationState.MAX_STEP] or step_index >= self.settings.max_steps or early_exit:
            last_steps_to_check = 20 if len(steps) > 10 else max(int(len(steps) / 10), 1)
            if len(list(filter(lambda x: x["offset_center"] is None, steps[-last_steps_to_check:]))) / last_steps_to_check >= 0.5:
                return "NO_LANES_DETECTED"
            return "SUCCESS"
        elif info[TerminationState.OUT_OF_ROAD]:
            return "OUT_OF_ROAD"
        elif info[TerminationState.CRASH_OBJECT]:
            return "CRASH_OBJECT"
        elif info[TerminationState.CRASH_SIDEWALK]:
            return "OUT_OF_ROAD"
        else:
            return "UNKNOWN"

    def run_simulation(self, env: MetaDriveEnv, generate_training_data=False):
        self.train_ground_truth = []
        env.reset(env.start_seed)
        env.current_track_agent.expert_takeover = not self.settings.start_with_manual_control
        
        for i in range(15):
            o, r, tm, tc, infos = env.step([0, 1])
        assert isinstance(o, dict)

        start_time = datetime.datetime.now()
        step_index = 0
        no_move_count = 0
        current_car_pos_meter = 0
        offsets_center_simulator = []

        if generate_training_data:
            folder = f"./camera_observations/training_data/{self.settings.lane_detection_model}_seed_{env.current_seed}"
            if not os.path.exists(folder):
                os.makedirs(folder)
            lanes = self.get_lanes_from_navigation_map(env)

        while True:
            try:
                o, r, tm, tc, info = env.step([0,0])

                if not self.settings.headless_rendering:
                    env.render(
                        text={
                            "Auto-Drive (Switch mode: T)": (
                                "on" if env.current_track_agent.expert_takeover else "off"
                            ),
                            "Keyboard Control": "W,A,S,D",
                        }
                    )

                if self.settings.save_images:
                    if step_index % 20 == 0:
                        self.io_tasks.put(lambda: cv2.imwrite(
                            f"camera_observations/{str(step_index)}.png",
                            (
                                o["image"].get()[..., -1]
                                if env.config["image_on_cuda"]
                                else o["image"][..., -1]
                            )
                            * 255,
                        ))
                        if self.settings.use_lane_camera:
                            self.io_tasks.put(lambda: cv2.imwrite(
                                f"camera_observations/{str(step_index)}_label.png",
                                (
                                    o["lane"].get()[..., -1]
                                    if env.config["image_on_cuda"]
                                    else o["lane"][..., -1]
                                )
                                * 255,
                            ))

                policy = env.engine.get_policy(env.agent.name)

                if env.agent.position[0] - current_car_pos_meter < 0.01:
                    no_move_count += 1
                else:
                    no_move_count = 0

                current_car_pos_meter = env.agent.position[0]

                # Used for calculation of E2E-LD metric
                model_input = None
                if "step_infos" in policy.__dict__ and len(policy.step_infos) > 0:
                    offsets_center_simulator.append(abs(policy.step_infos[-1]["offset_center_simulator"]))
                    model_input = policy.step_infos[-1]["model_input"]

                if generate_training_data and step_index % self.settings.generate_training_data_interval == 0: 
                    self.generate_training_label(lanes, env, step_index, o, model_input)
                    del model_input

                if tm or tc or step_index >= self.settings.max_steps or no_move_count > 50: 
                    end_reason = self.get_end_reason(info, step_index, policy.step_infos, no_move_count > 50)
                    if generate_training_data and len(self.train_ground_truth) > 0:
                        with open(f"{folder}/{env.current_seed}.txt", "w") as f:
                            f.write("\n".join(self.train_ground_truth))
                
                    print(f"Simulation with seed {env.current_seed} ended at step {step_index} with reason: {end_reason}. Attack active: {env.engine.global_config['enable_dirty_road_patch_attack']}")
                    sim_id = self.database.add_simulation(self.experiment_id, 
                                        env.current_seed, 
                                        self.settings.map_config, 
                                        self.settings.max_steps,
                                        start_time, 
                                        datetime.datetime.now(), 
                                        env.engine.global_config["enable_dirty_road_patch_attack"], 
                                        self.settings.patch_geneneration_iterations, 
                                        self.settings.attack_config.attack_at_meter if self.settings.attack_config is not None else -1, 
                                        self.settings.simulator_window_size[0], 
                                        self.settings.simulator_window_size[1], 
                                        self.settings.lane_detection_model, 
                                        end_reason,
                                        step_index,
                                        current_car_pos_meter,
                                        max(offsets_center_simulator) if len(offsets_center_simulator) > 0 else None)
                    self.database.add_simulation_steps(sim_id, policy.step_infos)

                    if env.current_seed + 1 < env.start_seed + env.num_scenarios:   
                        env.reset(env.current_seed + 1)
                        gc.collect()
                        step_index = 0
                        start_time = datetime.datetime.now()
                        env.current_track_agent.expert_takeover = not self.settings.start_with_manual_control
                        if generate_training_data:
                            folder = f"./camera_observations/training_data/{self.settings.lane_detection_model}_seed_{env.current_seed}"
                            if not os.path.exists(folder):
                                os.makedirs(folder)
                            lanes = self.get_lanes_from_navigation_map(env)
                    else:
                        break 

            except KeyboardInterrupt as e:
                raise e
            except Exception:
                if generate_training_data and len(self.train_ground_truth) > 0:
                        with open(f"{folder}/{env.current_seed}.txt", "w") as f:
                            f.write("\n".join(self.train_ground_truth))
                sim_id = self.database.add_simulation(self.experiment_id, 
                    env.current_seed, 
                    self.settings.map_config, 
                    self.settings.max_steps,
                    start_time, 
                    datetime.datetime.now(), 
                    env.engine.global_config["enable_dirty_road_patch_attack"], 
                    self.settings.patch_geneneration_iterations, 
                    self.settings.attack_config.attack_at_meter if self.settings.attack_config is not None else -1, 
                    self.settings.simulator_window_size[0], 
                    self.settings.simulator_window_size[1], 
                    self.settings.lane_detection_model, 
                    "ERROR",
                    step_index,
                    current_car_pos_meter,
                    max(offsets_center_simulator) if len(offsets_center_simulator) > 0 else None)      
                self.database.add_simulation_steps(sim_id, policy.step_infos)

                import traceback
                print(traceback.format_exc())
                gc.collect()
                break
            
            step_index += 1

    def io_worker(self):
        while not self.stop_event.isSet():
            try:
                task = self.io_tasks.get(block=True, timeout=1)
                task()
                self.io_tasks.task_done()
            except queue.Empty:
                pass
    
    def process_training_data(self):
        data_root = "./camera_observations/training_data"
        root = f"{data_root}/culane"
        labels_root = f"{root}/laneseg_label_w16"
        list_root = f"{root}/lists"
        if not os.path.exists(root):
            os.makedirs(root)
            os.makedirs(labels_root)
            os.makedirs(list_root)

        entries = []
        train = []
        test = []

        for gt in glob.glob("./camera_observations/training_data/*_seed_*/*.txt"):
            # TODO: read model name from the file
            seed = gt.replace("\\", "/").split("/")[-1].split(".")[0]

            if not os.path.exists(f"{root}/{seed}"):
                os.makedirs(f"{root}/{seed}")
            if not os.path.exists(f"{labels_root}/{seed}"):
                os.makedirs(f"{labels_root}/{seed}")

            with open(gt, "r") as f:
                lines = f.readlines()
                for line in lines:
                    cam_path, lane_path, *lanes_existence = line.strip().split(" ")
                    step = f"{cam_path.split('/')[-1].split('.')[0]}"
                    shutil.copyfile(f"{data_root}/{cam_path}", f"{root}/{seed}/{step}.exr")
                    shutil.copyfile(f"{data_root}/{lane_path}", f"{labels_root}/{seed}/{step}.png")
                    entries.append(f"{seed}/{step} {' '.join(lanes_existence)}")

        # make a 70/30 train/test split for the entries (shuffled)
        import random
        random.shuffle(entries)
        train = entries[:int(len(entries) * 0.7)]
        test = entries[int(len(entries) * 0.7):]

        # test format is different
        test = list(map(lambda x: f"{x.split(' ')[0]}", test))

        with open(f"{list_root}/train.txt", "w") as f:
            f.write("\n".join(train))

        with open(f"{list_root}/test.txt", "w") as f:
            f.write("\n".join(test))

        # valfast.txt is used for IoU eval
        with open(f"{list_root}/valfast.txt", "w") as f:
            f.write("\n".join(test))
        
        print("Training data processed.")


    def save_image(self, image, path, sizes):
        image = image.transpose((1, 2, 0))
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        image = cv2.resize(image, (sizes[1], sizes[0]))
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)

    def get_lanes_from_navigation_map(self, env):
        all_lanes = list(filter(lambda x:  MetaDriveType.is_road_line(x["type"]) or MetaDriveType.is_road_boundary_line(x["type"]), 
                                env.engine.current_map.get_map_features(1).values()))
        checkpoints = env.agent.navigation.checkpoints

        final_lanes = [[] for _ in range(self.settings.lanes_per_direction * 2 + 1)] # i.e. 5 lane lines in total including the middle line


        for cid, checkpoint in enumerate(checkpoints):
            for lane in all_lanes:
                is_white = not MetaDriveType.is_yellow_line(lane["type"])
                is_broken = MetaDriveType.is_broken_line(lane["type"])
                idx = 1 if is_broken else 0 if is_white else -1
                if lane["index"][0] == checkpoint and lane["index"][1] == checkpoints[cid + 1]:
                    final_lanes[idx].extend(lane['polyline'])
                # lanes in negative direction
                elif lane['index'][1] == f"-{checkpoint}" and lane['index'][0] == f"-{checkpoints[cid + 1]}":
                    if is_broken: # skip the left border line
                        final_lanes[self.settings.lanes_per_direction + idx if is_white else -1].extend(lane['polyline'])

        # sort points by x coordinate (meters from bottom)
        final_lanes = list(map(lambda lane: sorted(lane, key=lambda x: x[0]), final_lanes))

        # filter out zer length lanes
        final_lanes = list(filter(lambda lane: len(lane) > 1, final_lanes))

        # sort the lane array by the y coord of the first point
        final_lanes = sorted(final_lanes, key=lambda lane: lane[0][1], reverse=True)

        return final_lanes

    def generate_training_label(self, lanes, env, step_index, observation, model_input):
        debug = self.settings.debug_lane_labels
        cam = env.engine.get_sensor("rgb_camera").get_cam()
        lens = env.engine.get_sensor("rgb_camera").get_lens()
        # Get the image dimensions
        image_width = env.engine.win.getXSize()
        image_height = env.engine.win.getYSize()

        # Transform lane coordinates
        lane_coordinates = [[] for _ in range(len(lanes))]

        # assign each lane a different color in a uint8 grayscale image
        image = np.zeros((env.engine.win.getYSize(), env.engine.win.getXSize(), 1 if not debug else 3), dtype=np.uint8)
        pos_meter = env.agent.position[0]

        # only consider points near the car
        lane_color_debug = [(255, 255, 255), (200, 200, 200), (150, 150, 150), (100, 100, 100)] # add more colors if necessary
        lanes_filtered = map(lambda lane: 
                            list(filter(lambda x: x[0] >= pos_meter - 20, lane))[:120], lanes)

        for i, lane_points in enumerate(lanes_filtered):
            if not len(lane_points):
                continue
            
            added_outside_bottom = False
            added_outside_top = False

            for point in lane_points:
                projected_point = Point2()

                lens.project(cam.getRelativePoint(env.engine.render, Point3(point[0], point[1], 0)), projected_point)

                p_image = [
                    (projected_point[0] * (image_width / 2)) + (image_width / 2),  # X coordinate: scale and shift
                    (image_height / 2) - (projected_point[1] * (image_height / 2))  # Y coordinate: scale and shift (flip Y axis)
                ]
                if p_image[1] >= image_height/2:
                    if p_image[1] >= image_height and not added_outside_bottom or p_image[0] <= 0:
                        added_outside_bottom = True
                        lane_coordinates[i].extend(p_image)
                    elif p_image[0] >= image_width and not added_outside_top:
                        added_outside_top = True
                        lane_coordinates[i].extend(p_image)
                    else:
                        lane_coordinates[i].extend(p_image)

        # sort by mean x coordinate
        folder = f"./camera_observations/training_data/{self.settings.lane_detection_model}_seed_{env.current_seed}"
        existence = ["1" if len(l) > 1 else "0" for l in lane_coordinates]
        self.train_ground_truth.append(
            f"{self.settings.lane_detection_model}_seed_{env.current_seed}/{step_index}.exr {self.settings.lane_detection_model}_seed_{env.current_seed}/{step_index}_lanes.png {' '.join(existence)}"
        )
        
        self.io_tasks.put(lambda: pickle.dump({"lane_coordinates": lane_coordinates, "existence": existence}, open(f"{folder}/{str(step_index)}_lanes.pkl", "wb")))

        for i, l in enumerate(lane_coordinates):
            if len(l) > 1:
                cv2.polylines(image, [np.array(l).reshape((-1, 1, 2)).astype(np.int32)], 
                            False, i+1 if not debug else lane_color_debug[i], 7 if not debug else 2, lineType=cv2.LINE_AA)
                
        if model_input is not None:
            image_in = model_input.transpose((1, 2, 0))
        else:
            image_in =  (observation["image"].get()[..., -1] if env.config["image_on_cuda"] else observation["image"][..., -1]) * 255

        self.io_tasks.put(lambda: cv2.imwrite(f"{folder}/{str(step_index)}.exr", image_in))
        self.io_tasks.put(lambda: cv2.imwrite(f"{folder}/{step_index}_lanes.png", image))

        return lane_coordinates
