import math
import time
from collections import deque, namedtuple
from abc import ABC, abstractmethod
from typing import Optional, List, Union
import multiprocessing
import numpy as np

from setproctitle import getproctitle

vec3 = namedtuple("vec3", ["x", "y", "z"])


class Ratekeeper:
    def __init__(
        self, rate: float, print_delay_threshold: Optional[float] = 0.0
    ) -> None:
        """Rate in Hz for ratekeeping. print_delay_threshold must be nonnegative."""
        self._interval = 1.0 / rate
        self._next_frame_time = time.monotonic() + self._interval
        self._print_delay_threshold = print_delay_threshold
        self._frame = 0
        self._remaining = 0.0
        self._process_name = getproctitle()
        self._dts = deque([self._interval], maxlen=100)
        self._last_monitor_time = time.monotonic()

    @property
    def frame(self) -> int:
        return self._frame

    @property
    def remaining(self) -> float:
        return self._remaining

    @property
    def lagging(self) -> bool:
        avg_dt = sum(self._dts) / len(self._dts)
        expected_dt = self._interval * (1 / 0.9)
        return avg_dt > expected_dt

    # Maintain loop rate by calling this at the end of each loop
    def keep_time(self) -> bool:
        lagged = self.monitor_time()
        if self._remaining > 0:
            time.sleep(self._remaining)
        return lagged

    # Monitors the cumulative lag, but does not enforce a rate
    def monitor_time(self) -> bool:
        prev = self._last_monitor_time
        self._last_monitor_time = time.monotonic()
        self._dts.append(self._last_monitor_time - prev)

        lagged = False
        remaining = self._next_frame_time - time.monotonic()
        self._next_frame_time += self._interval
        if (
            self._print_delay_threshold is not None
            and remaining < -self._print_delay_threshold
        ):
            print(f"{self._process_name} lagging by {-remaining * 1000:.2f} ms")
            lagged = True
        self._frame += 1
        self._remaining = remaining
        return lagged


class IMUState:
    def __init__(self):
        self.accelerometer: vec3 = vec3(0, 0, 0)
        self.gyroscope: vec3 = vec3(0, 0, 0)
        self.bearing: float = 0


class SimulatorState:
    def __init__(self):
        self.valid = False
        self.is_engaged = False
        self.ignition = True

        self.velocity: vec3 = None
        self.bearing: float = 0
        self.imu = IMUState()

        self.steering_angle: float = 0

        self.user_gas: float = 0
        self.user_brake: float = 0
        self.user_torque: float = 0

        self.cruise_button = 0

        self.left_blinker = False
        self.right_blinker = False

    @property
    def speed(self):
        return math.sqrt(self.velocity.x**2 + self.velocity.y**2 + self.velocity.z**2)


class World(ABC):
    def __init__(self, dual_camera):
        self.dual_camera = dual_camera

        self.image_lock = multiprocessing.Semaphore(value=0)
        self.road_image = np.zeros((H, W, 3), dtype=np.uint8)
        self.wide_road_image = np.zeros((H, W, 3), dtype=np.uint8)

    @abstractmethod
    def apply_controls(self, steer_sim, throttle_out, brake_out):
        pass

    @abstractmethod
    def tick(self):
        pass

    @abstractmethod
    def read_sensors(self, simulator_state: SimulatorState):
        pass

    @abstractmethod
    def read_cameras(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def reset(self):
        pass
