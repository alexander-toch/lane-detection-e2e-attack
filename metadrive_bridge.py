
import datetime
import os
import glob
from enum import Enum
from dataclasses import dataclass, field
import queue
import threading
import cv2
import numpy as np
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import TerminationState
from metadrive.constants import HELP_MESSAGE
from metadrive.component.map.base_map import BaseMap
from metadrive.component.sensors.base_camera import _cuda_enable
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock

from metadrive_policy.TopDownCamera import TopDownCamera
from metadrive_policy.lanedetection_policy_patch_e2e import LaneDetectionPolicyE2E
from metadrive_policy.lanedetection_policy_dpatch import LaneDetectionPolicy
from metadrive_policy.lane_camera import LaneCamera

from database import ExperimentDatabase

@dataclass
class AttackConfig:
    attack_at_meter: int = 6000 # set to -1 to disable patch rendering in MetaDrive
    two_pass_attack: bool = False
    place_patch_in_image_stream: bool = False
    patch_color_replace: bool = False

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
    patch_size_meters: tuple[float, float] = (1.0, 1.0) # (width, height) in meters
    patch_geneneration_iterations: int = 90
    start_with_manual_control: bool = False
    simulator_window_size: tuple = (1280, 720) # (width, height)
    policy: str = "LaneDetectionPolicyE2E"
    attack_config: AttackConfig | None = field(default_factory=AttackConfig)
    lanes_per_direction: int = 2
    use_lane_camera: bool = False
    generate_training_data: bool = False

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
            lane_detection_model=self.settings.lane_detection_model,
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
                self.run_simulation(env)
        else:
            env = MetaDriveEnv(self.config)
            self.run_simulation(env)

        self.database.finish_experiment(self.experiment_id, datetime.datetime.now())

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
            env.start_seed = current_seed
            env.start_index = current_seed
            env.num_scenarios = 1
            self.run_simulation(env)

            # ATTACK PASS 2: Drive with mounted patch
            print(f"Pass 2: Running simulation with seed {current_seed} with attack enabled.")
            env.engine.global_config["enable_dirty_road_patch_attack"] = True
            if self.settings.attack_config.place_patch_in_image_stream:
                env.engine.global_config["dirty_road_patch_attack_at_meter"] = self.settings.attack_config.attack_at_meter
            self.run_simulation(env)
            
            current_seed += 1
            env.engine.global_config["enable_dirty_road_patch_attack"] = False
            env.engine.get_policy(env.agent.name).control_object.engine.dirty_road_patch_object = None

    def get_end_reason(self, info, step_index, steps):
        if info[TerminationState.SUCCESS] or info[TerminationState.MAX_STEP] or step_index >= self.settings.max_steps:
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

    def run_simulation(self, env: MetaDriveEnv):

        env.reset(env.start_seed)
        env.current_track_agent.expert_takeover = not self.settings.start_with_manual_control
        
        for i in range(15):
            o, r, tm, tc, infos = env.step([0, 1])
        assert isinstance(o, dict)

        start_time = datetime.datetime.now()
        step_index = 0
        offsets_center_simulator = []
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
                current_car_pos_meter = env.agent.position[0]

                # Used for calculation of E2E-LD metric
                if "step_infos" in policy.__dict__ and len(policy.step_infos) > 0:
                    offsets_center_simulator.append(abs(policy.step_infos[-1]["offset_center_simulator"]))

                    if self.settings.use_lane_camera and self.settings.generate_training_data:
                        self.save_training_data(step_index, env.current_seed, policy.step_infos[-1], o)

                if tm or tc or step_index >= self.settings.max_steps: 
                    end_reason = self.get_end_reason(info, step_index, policy.step_infos)
                
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
                        step_index = 0
                        start_time = datetime.datetime.now()
                        env.current_track_agent.expert_takeover = not self.settings.start_with_manual_control
                    else:
                        break 
            except KeyboardInterrupt as e:
                raise e
            except Exception:
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

    def save_training_data(self, step_index, seed, step_info, observation):
        if "model_input" not in step_info or "lane" not in observation:
            print("Missing model input or lane observation. Skipping training data generation.")
            return
        
        # create subfolder for each seed
        folder = f"./camera_observations/training_data/seed_{seed}"
        if not os.path.exists(folder):
            os.makedirs(folder)

        height = step_info["model_input"].shape[1]
        width = step_info["model_input"].shape[2]
        
        # save model input
        self.io_tasks.put(lambda: self.save_image(step_info["model_input"], f"{folder}/{step_index}.png", (height, width)))

        # save lane detection output
        label = (observation["lane"].get()[..., -1] if self.config["image_on_cuda"] else observation["lane"][..., -1]) * 255
        # resize label to match model input if necessary
        if label.shape[0] != height or label.shape[1] != width:
            label = cv2.resize(label, (width, height))

        self.io_tasks.put(lambda: cv2.imwrite(f"{folder}/{step_index}_label.png", label))

    def save_image(self, image, path, sizes):
        image = image.transpose((1, 2, 0))
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        image = cv2.resize(image, (sizes[1], sizes[0]))
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)