
from dataclasses import dataclass
import cv2
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE
from metadrive.component.map.base_map import BaseMap
from metadrive.component.sensors.base_camera import _cuda_enable
from metadrive.component.map.pg_map import MapGenerateMethod


from metadrive_policy.lanedetection_policy_patch_e2e import LaneDetectionPolicyE2E
from metadrive_policy.lanedetection_policy_dpatch import LaneDetectionPolicy

@dataclass
class Settings:
    seed: int = 1235
    num_scenarios: int = 1
    map_config: str = "SCS"
    headless_rendering: bool = False
    save_images: bool = False
    attack_at_step: int = 6000
    max_steps: int = 5000
    start_with_manual_control: bool = False
    simulator_window_size: tuple = (1280, 720) # (width, height)
    policy: str = "LaneDetectionPolicyE2E"

class MetaDriveBridge:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.map_config = {
            "config": self.settings.map_config, # S=Straight, C=Circular/Curve
            BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
            # BaseMap.GENERATE_CONFIG: 3,
            BaseMap.LANE_WIDTH: 3.5,
            BaseMap.LANE_NUM: 2,
        }

        self.policy = LaneDetectionPolicyE2E if self.settings.policy == "LaneDetectionPolicyE2E" else LaneDetectionPolicy

        self.config  = dict(
            use_render=not self.settings.headless_rendering,
            window_size=self.settings.simulator_window_size,
            sensors={
                "rgb_camera": (RGBCamera, self.settings.simulator_window_size[0], self.settings.simulator_window_size[1]),
            },
            interface_panel=[],
            vehicle_config={
                "image_source": "rgb_camera",
            },
            agent_policy=self.policy,
            start_seed=self.settings.seed,
            image_on_cuda=True,
            image_observation=True,
            out_of_route_done=True,
            on_continuous_line_done=True,
            crash_vehicle_done=True,
            crash_object_done=True,
            crash_human_done=True,
            traffic_density=0.0,
            map_config=self.map_config,
            num_scenarios=self.settings.num_scenarios,
            decision_repeat=1,
            preload_models=False,
            manual_control=True,
            dirty_road_patch_attack_step_index=self.settings.attack_at_step,
            force_map_generation=True # disables the PG Map cache
        )

    def run(self):
        # TODO: use settings to configure attack
        # ATTACK STEP 1: Drive without attack and generate the patch
        self.config["enable_dirty_road_patch_attack"] = False

        env = MetaDriveEnv(self.config)
        self.run_simulation(env)

        # ATTACK STEP 2: Drive with mounted patch
        env.engine.global_config["enable_dirty_road_patch_attack"] = True
        
        
        self.run_simulation(env)

    def run_simulation(self, env: MetaDriveEnv):
        env.reset(self.settings.seed)
        env.current_track_agent.expert_takeover = not self.settings.start_with_manual_control
        
        for i in range(10):
            o, r, tm, tc, infos = env.step([0, 1])
        assert isinstance(o, dict)

        step_index = 0
        while True:
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
                    cv2.imwrite(
                        f"camera_observations/{str(step_index)}.jpg",
                        (
                            o["image"].get()[..., -1]
                            if env.config["image_on_cuda"]
                            else o["image"][..., -1]
                        )
                        * 255,
                    )

            if tm or tc or step_index >= self.settings.max_steps:
                print(f"Simulation ended at step {step_index}")
                if env.current_seed + 1 < self.settings.seed + self.settings.num_scenarios:
                    env.reset(env.current_seed + 1)
                else:
                    break            
            step_index += 1
