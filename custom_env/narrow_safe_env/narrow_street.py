"""
This script is only to create road network, not the main environment. The narrow-safe-v0 imports NarrowStreet from this file and hence is required to be present within the same folder.
Narrow Street Env — custom wrapper around highway-env.
Two straight lanes, right lane intermittently narrowed by parked cars.
Ego starts in right lane, may swerve or merge left if gaps permit. Left lane
has same-direction traffic. Reward prioritizes safety first, then efficient progress.

Register id: 'narrow-street-v0'
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from highway_env.envs.highway_env import HighwayEnv
from highway_env.envs.common.action import DiscreteMetaAction
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle

# Try to import a static obstacle. If unavailable, emulate with a stopped IDMVehicle.
try:
    from highway_env.vehicle.objects import Obstacle  # newer versions
except Exception:
    try:
        from highway_env.road.objects import Obstacle  # older versions
    except Exception:
        Obstacle = None


class NarrowStreetEnv(HighwayEnv):
    """Custom 2-lane straight-road environment with parked cars in right lane."""

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                # Observation: simple kinematics, no normalization
                "observation": {
                    "type": "Kinematics",
                    "features": ["x", "y", "vx", "vy", "heading"],
                    "normalize": False,
                    "vehicles_count": 10,
                },
                "action": {"type": "DiscreteMetaAction"},

                "lanes_count": 2,
                "vehicles_count": 6,
                "duration": 40,              # seconds
                "policy_frequency": 15,      # Hz
                "simulation_frequency": 15,

                "screen_width": 1200,
                "screen_height": 300,
                "offscreen_rendering": False,

                # Rewards & penalties
                "collision_reward": -10.0,
                "offroad_reward": -0.7,

                "lane_change_cost": -0.02,   # discourage weaving

                # Headway shaping
                "headway_safe": 1.6,                # seconds
                "headway_penalty": -0.30,
                "headway_strong_penalty": -2.0,
                "brake_bonus": 0.18,
                "accelerate_penalty_factor": -1.0,

                # Progress & time
                "progress_reward": 0.02,
                "parked_clear_bonus": 0.07,
                "goal_reward": 4.0,
                "time_penalty": -0.004,

                # Speed shaping
                "speed_limit": 25.0,               # m/s
                "overspeed_penalty": -0.25,

                # Early lane-change shaping near parked cars
                "approach_lcl_bonus": 0.30,
                "approach_keep_penalty": -0.15,
                "approach_window_before": 50.0,    # meters before parked start
                "approach_window_after": 20.0,     # meters before parked start
                "unsafe_lcl_bonus": 0.25,

                # Mild preference for right lane when safe
                "right_lane_pref": 0.02,

                # Scenario controls
                "road_length": 600.0,
                "parked_count": 4,
                "parked_intrusion": 0.45,          # fraction of lane width
                "left_lane_traffic": 5,

                "seed": 42,
            }
        )
        return cfg

    # ----------------------
    # Scene construction
    # ----------------------
    def _create_road(self) -> None:
        net = RoadNetwork()
        width = 4.0
        length = float(self.config["road_length"])

        # Two straight lanes: lane index 0 = right, 1 = left.
        for i in range(2):
            y = i * width
            lane = StraightLane(
                start=[0.0, y],
                end=[length, y],
                line_types=(
                    LineType.CONTINUOUS if i == 0 else LineType.STRIPED,
                    LineType.STRIPED,
                ),
                width=width,
            )
            net.add_lane("a", "b", lane)

        self.road = Road(network=net, np_random=self.np_random, record_history=True)

        # Place parked cars partially intruding into the right lane (lane 0)
        self._parked_segments = []  # (x_start, x_end)
        self._parked_objects = []
        n_parked = int(self.config.get("parked_count", 0))
        intrusion = float(self.config["parked_intrusion"]) * width

        xs = np.linspace(80.0, length - 80.0, max(n_parked, 1))[:n_parked]
        lane_right = self.road.network.get_lane(("a", "b", 0))

        for x in xs:
            # Lateral offset: intrude from the right edge but still leave passing room.
            lateral = -0.5 * lane_right.width + intrusion * 0.6
            px, py = lane_right.position(x, lateral)

            if Obstacle is not None:
                obj = Obstacle(self.road, [px, py], heading=lane_right.heading_at(x))
                obj.LENGTH = 4.5
                obj.WIDTH = 1.9
                self.road.objects.append(obj)
                self._parked_objects.append(obj)
            else:
                # Fallback: stopped vehicle
                parked = IDMVehicle.make_on_lane(self.road, ("a", "b", 0), longitudinal=x, speed=0.0)
                parked.target_velocity = 0.0
                parked.velocity_index = 0
                parked.color = (50, 50, 50)
                self.road.vehicles.append(parked)
                self._parked_objects.append(parked)

            self._parked_segments.append((x - 5.0, x + 5.0))

    def _create_vehicles(self) -> None:
        # Ego: right lane, straight road, mid-speed
        lane0 = self.road.network.get_lane(("a", "b", 0))
        ego_pos = lane0.position(20.0, 0.0)
        ego_heading = lane0.heading_at(20.0)
        ego = MDPVehicle(
            road=self.road,
            position=np.array(ego_pos, dtype=float),
            heading=float(ego_heading),
            speed=20.0,
        )
        ego.COLOR = (0, 0, 255)
        self.road.vehicles.append(ego)
        self.vehicle = ego

        # Left-lane traffic (IDM vehicles)
        n_left = int(self.config.get("left_lane_traffic", 0))
        lane1 = self.road.network.get_lane(("a", "b", 1))
        for _ in range(n_left):
            lon = float(
                60.0 + (self.config.get("road_length", 600.0) - 120.0) * self.np_random.random()
            )
            pos = lane1.position(lon, 0.0)
            heading = lane1.heading_at(lon)
            v = IDMVehicle(
                self.road,
                position=np.array(pos, dtype=float),
                heading=float(heading),
                speed=float(18 + 6 * self.np_random.random()),
            )
            v.color = (140, 140, 255)
            self.road.vehicles.append(v)

        # Optional slow lead car in right lane far ahead
        if self.np_random.random() < 0.4:
            lon = float(120.0 + 0.4 * self.config.get("road_length", 600.0))
            pos = lane0.position(lon, 0.0)
            heading = lane0.heading_at(lon)
            lead = IDMVehicle(
                self.road,
                position=np.array(pos, dtype=float),
                heading=float(heading),
                speed=13.0,
            )
            lead.color = (200, 200, 0)
            self.road.vehicles.append(lead)

    # ----------------------
    # Reward & termination
    # ----------------------
    def _reward(self, action: int) -> float:
        cfg = self.config
        ego = self.vehicle
        r = 0.0

        # Time penalty
        r += float(cfg.get("time_penalty", 0.0))

        # Hard safety
        if ego.crashed:
            return float(cfg["collision_reward"])
        if self._is_offroad(ego):
            r += cfg["offroad_reward"]

        # Convenience
        x = float(ego.position[0])
        try:
            lane_idx = ego.target_lane_index[2]
        except Exception:
            lane_idx = 0

        # Headway shaping vs lead vehicle in current lane
        lead, _ = ego.road.neighbour_vehicles(ego, ego.target_lane_index)
        t_headway = np.inf
        if lead is not None:
            try:
                gap = max(0.1, lead.distance_to(ego))
            except Exception:
                gap = max(
                    0.1,
                    float(
                        np.linalg.norm(
                            np.array(getattr(lead, "position", [0, 0]))
                            - np.array(ego.position)
                        )
                    ),
                )
            t_headway = gap / max(1e-3, ego.speed)

            headway_safe = float(cfg.get("headway_safe", 1.6))
            base_pen = float(cfg.get("headway_penalty", -0.30))
            strong_pen = float(cfg.get("headway_strong_penalty", -2.0))
            brake_bonus = float(cfg.get("brake_bonus", 0.18))
            accel_pen_factor = float(cfg.get("accelerate_penalty_factor", -1.0))

            if t_headway < headway_safe:
                deficit = headway_safe - t_headway
                r += base_pen * deficit
                if t_headway < 0.8:
                    r += strong_pen * (0.8 - t_headway)

                if action == 4:  # SLOWER
                    r += brake_bonus * deficit
                if action == 3:  # FASTER
                    r += accel_pen_factor * deficit

        # Early lane-change shaping near parked segments (right lane only)
        approach_L = float(cfg.get("approach_window_before", 50.0))
        approach_l = float(cfg.get("approach_window_after", 20.0))
        if lane_idx == 0 and getattr(self, "_parked_segments", None):
            for (x0, x1) in self._parked_segments:
                if (x0 - approach_L) <= x <= (x0 - approach_l):
                    if action == 1:  # LCL
                        r += float(cfg.get("approach_lcl_bonus", 0.3))
                    elif action in (0, 3):  # KEEP or FASTER right into blockage
                        r += float(cfg.get("approach_keep_penalty", -0.15))
                    break

        # Extra bonus for LCL if already in unsafe headway (right lane)
        headway_safe = float(cfg.get("headway_safe", 1.6))
        if lane_idx == 0 and t_headway < headway_safe and action == 1:
            r += float(cfg.get("unsafe_lcl_bonus", 0.25))

        # Progress / speed reward, downweighted when in unsafe headway
        speed_norm = np.clip(ego.speed / 30.0, 0.0, 1.0)
        progress_w = 1.0 if t_headway >= headway_safe else 0.3
        r += cfg["progress_reward"] * progress_w * speed_norm

        # Overspeed penalty
        speed_limit = float(cfg.get("speed_limit", 25.0))
        if ego.speed > speed_limit:
            overshoot = (ego.speed - speed_limit) / max(1.0, speed_limit)
            r += float(cfg.get("overspeed_penalty", -0.25)) * overshoot

        # Mild right-lane preference when safe
        right_lane_pref = float(cfg.get("right_lane_pref", 0.02))
        if t_headway >= headway_safe:
            if lane_idx == 0:
                r += right_lane_pref
            elif lane_idx == 1:
                r -= 0.5 * right_lane_pref

        # Lane-change cost
        if action is not None and action in (1, 2):
            r += float(cfg.get("lane_change_cost", -0.02))

        # Bonus for clearing a parked segment
        for (x0, x1) in getattr(self, "_parked_segments", []):
            if x1 <= x <= x1 + 0.5 and not ego.crashed:
                r += cfg["parked_clear_bonus"]

        # Goal reward near end of road
        goal_x = float(cfg.get("road_length", 600.0)) - 2.0
        if x >= goal_x and not ego.crashed:
            r += cfg["goal_reward"]

        return float(r)

    def step(self, action):
        """Use base dynamics but force termination when reaching goal distance."""
        obs, reward, terminated, truncated, info = super().step(action)

        try:
            x = float(self.vehicle.position[0])
            goal_x = float(self.config.get("road_length", 600.0)) - 2.0
        except Exception:
            x = 0.0
            goal_x = float(self.config.get("road_length", 600.0)) - 2.0

        if not terminated and (x >= goal_x) and not getattr(self.vehicle, "crashed", False):
            terminated = True
            info = dict(info)
            info["is_success"] = True

        return obs, reward, terminated, truncated, info

    def _is_terminal(self) -> bool:
        """Terminate on crash OR when ego reaches the end of the road."""
        if self.vehicle and getattr(self.vehicle, "crashed", False):
            return True
        try:
            x = float(self.vehicle.position[0])
            goal_x = float(self.config.get("road_length", 600.0)) - 2.0
            if x >= goal_x:
                return True
        except Exception:
            pass
        return False

    def _is_offroad(self, veh) -> bool:
        """Simple offroad heuristic using lane width."""
        try:
            y = veh.position[1]
            lane0 = self.road.network.get_lane(("a", "b", 0))
            total_width = 2 * lane0.width
            return (y < -0.6 * lane0.width) or (y > total_width - 0.4 * lane0.width)
        except Exception:
            return False


# ---- Gym registration ----
from gymnasium.envs.registration import register

try:
    cfg = NarrowStreetEnv.default_config()
    max_steps = int(cfg["duration"] * cfg["policy_frequency"])
    register(
        id="narrow-street-v0",
        entry_point=__name__ + ":NarrowStreetEnv",
        max_episode_steps=max_steps,
    )
except gym.error.Error:
    # Already registered
    pass


# ---- Quick manual demo ----
if __name__ == "__main__":
    import time

    env = gym.make("narrow-street-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=env.unwrapped.config.get("seed", None))

    try:
        print("Action meanings:", DiscreteMetaAction.ACTIONS)
    except AttributeError:
        print("Action class:", DiscreteMetaAction)

    ep_r = 0.0
    try:
        for t in range(300):
            action = 0  # KEEP as a dumb baseline
            obs, r, term, trunc, info = env.step(action)
            ep_r += float(r)
            env.render()
            time.sleep(1 / 30)
            if term or trunc:
                print(f"Demo episode ended. Return={ep_r:.2f}")
                break
    finally:
        env.close()
