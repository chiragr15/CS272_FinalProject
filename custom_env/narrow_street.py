"""
Narrow Street Env — minimal custom wrapper around highway-env.
Two lanes, right lane narrowed intermittently by parked cars. Ego starts in
right lane, may swerve within lane or merge left if gaps permit. Left lane has
light same-direction traffic. Reward prioritizes safety first, then efficient
progress.

Dependencies: highway-env >= 1.8, gymnasium.
Register id: 'narrow-street-v0'

Files:
- this file: narrow_street_env_min.py
- demo script: demo_narrow_street_min.py
"""
from __future__ import annotations

import math
import numpy as np
import gymnasium as gym

from highway_env.envs.highway_env import HighwayEnv
from highway_env.envs.common.action import DiscreteMetaAction
from highway_env.road.road import Road
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle

# Try to import a static obstacle. If unavailable, we will emulate with a zero-speed IDMVehicle.
try:
    from highway_env.vehicle.objects import Obstacle  # new location
except Exception:
    try:
        from highway_env.road.objects import Obstacle  # older location
    except Exception:
        Obstacle = None  # fallback handled later


class NarrowStreetEnv(HighwayEnv):
    """A lighter custom env built on top of HighwayEnv.

    Customizations kept minimal:
      - custom road (2 straight lanes)
      - add a few parked obstacles intruding into the right lane
      - light left-lane traffic
      - tailored reward combining safety + efficiency
    """

    @classmethod
    def default_config(cls) -> dict:
        """Return the default configuration for this env.

        highway-env uses a class-level default_config() style in many custom envs.
        Defining this as a classmethod ensures the base class can build self.config
        correctly when the environment is instantiated/registered.
        """
        cfg = super().default_config()
        cfg.update(
            {
                "observation": {"type": "Kinematics", "features": ["x", "y", "vx", "vy", "heading"], "normalize": True, "vehicles_count": 8},
                "action": {"type": "DiscreteMetaAction"},
                "lanes_count": 2,
                "vehicles_count": 4,  # only free traffic; ego + few others
                "duration": 40,  # seconds at 15 Hz -> ~600 steps
                "policy_frequency": 15,
                "simulation_frequency": 15,
                "screen_width": 1200,
                "screen_height": 300,
                "offscreen_rendering": False,
                # rewards (weights used in _reward)
                "collision_reward": -1.0,
                "offroad_reward": -0.5,
                "lane_change_cost": -0.02,
                "headway_penalty": -0.1,  # if time headway < 1.0s
                "progress_reward": 0.04,  # small per-step speed-scaled progress
                "parked_clear_bonus": 0.06,  # bonus for clearing a parked car zone without collision
                # scenario controls
                "road_length": 600.0,
                "parked_count": 3,  # 2-3 parked intrusions, not too many
                "parked_intrusion": 0.5,  # fraction of lane width the parked car intrudes
                "left_lane_traffic": 4,  # light traffic vehicles
                "seed": 42,
            }
        )
        return cfg

    def _create_road(self) -> None:
        net = RoadNetwork()
        width = 4.0  # lane width
        length = float(self.config["road_length"])  # meters
        # Two parallel straight lanes: lane 0 (right), lane 1 (left)
        for i in range(2):
            y = i * width
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [0.0, y],
                    [length, y],
                    line_types=(LineType.CONTINUOUS if i == 0 else LineType.STRIPED, LineType.STRIPED),
                    width=width,
                ),
            )
        self.road = Road(network=net, np_random=self.np_random, record_history=True)

        # Place parked cars partially intruding into right lane at a few x-positions
        self._parked_segments = []  # (x_start, x_end) for reward bonus
        self._parked_objects = []
        n_parked = int(self.config["parked_count"]) if self.config["parked_count"] else 0
        intrusion = float(self.config["parked_intrusion"]) * width

        # choose a few longitudinal positions (avoid the first/last 40m)
        xs = np.linspace(80.0, length - 80.0, max(n_parked, 1))[:n_parked]
        for x in xs:
            # center of the right lane is y=0; we intrude from the right boundary towards center.
            lane: StraightLane = self.road.network.get_lane(("a", "b", 0))
            # parked object near the right boundary: lateral offset negative -> towards road edge
            lateral = -0.5 * lane.width + intrusion * 0.6  # intrude ~60% of configured intrusion
            px, py = lane.position(x, lateral)

            if Obstacle is not None:
                obj = Obstacle(self.road, [px, py], heading=0.0)
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

            # reward zone ~vehicle length range
            self._parked_segments.append((x - 5.0, x + 5.0))

    def _create_vehicles(self) -> None:
        # Ego in right lane, mid-speed. Create directly to avoid possible
        # mismatches with create_random / lane id formats across highway-env versions.
        lane0 = self.road.network.get_lane(("a", "b", 0))
        ego_pos = lane0.position(20.0, 0.0)
        ego_heading = lane0.heading_at(20.0)
        ego = MDPVehicle(road=self.road, position=np.array(ego_pos, dtype=float), heading=float(ego_heading), speed=20.0)
        ego.COLOR = (0, 0, 255)
        self.road.vehicles.append(ego)
        self.vehicle = ego

        # Light traffic in left lane, same direction — place a few IDMVehicles at random positions
        n_left = int(self.config["left_lane_traffic"]) if self.config["left_lane_traffic"] else 0
        lane1 = self.road.network.get_lane(("a", "b", 1))
        for _ in range(n_left):
            # choose a longitudinal position not too close to ego
            lon = float(60.0 + (self.config.get("road_length", 600.0) - 120.0) * self.np_random.random())
            pos = lane1.position(lon, 0.0)
            heading = lane1.heading_at(lon)
            v = IDMVehicle(self.road, position=np.array(pos, dtype=float), heading=float(heading), speed=float(18 + 6 * self.np_random.random()))
            v.color = (140, 140, 255)
            self.road.vehicles.append(v)

        # Optionally add a single slow lead car in right lane far ahead (rare)
        if self.np_random.random() < 0.3:
            lon = float(120.0 + 0.4 * self.config.get("road_length", 600.0))
            pos = lane0.position(lon, 0.0)
            heading = lane0.heading_at(lon)
            lead = IDMVehicle(self.road, position=np.array(pos, dtype=float), heading=float(heading), speed=13.0)
            lead.color = (200, 200, 0)
            self.road.vehicles.append(lead)

    # ---- Reward shaping ----
    def _reward(self, action: int) -> float:
        cfg = self.config
        ego = self.vehicle
        r = 0.0

        # 1) Hard safety
        if ego.crashed:
            return cfg["collision_reward"]
        if self._is_offroad(ego):
            r += cfg["offroad_reward"]

        # 2) Keep safe headway to lead vehicle in current lane
        lead, _ = ego.road.neighbour_vehicles(ego, ego.target_lane_index)
        if lead is not None:
            rel_v = max(0.1, ego.speed - getattr(lead, "speed", 0.0))
            try:
                gap = max(0.1, lead.distance_to(ego))
            except Exception:
                # Some road objects (e.g. static obstacles) may not provide distance_to
                gap = max(0.1, float(np.linalg.norm(np.array(getattr(lead, 'position', [0, 0])) - np.array(ego.position))))
            t_headway = gap / max(1e-3, ego.speed)
            if t_headway < 1.0:
                r += cfg["headway_penalty"] * (1.0 - t_headway)  # more penalty when <1.0s

        # 3) Encourage progress / speed (capped to 1.0x desired speed ~30m/s)
        progress = np.clip(ego.speed / 30.0, 0.0, 1.0)
        r += cfg["progress_reward"] * progress

        # 4) Small cost for lane change to prevent weaving
        if action is not None:
            try:
                # DiscreteMetaAction: 0-Keep,1-LCL,2-LCR,3-Faster,4-Slower
                if action in (1, 2):
                    r += cfg["lane_change_cost"]
            except Exception:
                pass

        # 5) Bonus for safely clearing a parked segment (if ego passes the segment without crash)
        x = ego.position[0]
        for (x0, x1) in getattr(self, "_parked_segments", []):
            # give bonus once when just passed the segment (x within a small window beyond x1)
            if x1 <= x <= x1 + 0.5 and not ego.crashed:
                r += cfg["parked_clear_bonus"]

        return float(r)

    def _is_offroad(self, veh) -> bool:
        # simple offroad heuristic: y outside both lanes by > half width
        try:
            y = veh.position[1]
            lane: StraightLane = self.road.network.get_lane(("a", "b", 0))
            total_width = 2 * lane.width
            return (y < -0.6 * lane.width) or (y > total_width - 0.4 * lane.width)
        except Exception:
            return False


# ---- Registration helper ----
from gymnasium.envs.registration import register

try:
    register(
        id="narrow-street-v0",
        entry_point=__name__ + ":NarrowStreetEnv",
        max_episode_steps=NarrowStreetEnv.default_config()["duration"],
    )
except gym.error.Error:
    # Already registered in this process
    pass


# ---- Simple demo (optional quick test) ----
if __name__ == "__main__":
    import pygame
    import time

    env = gym.make("narrow-street-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=env.unwrapped.config.get("seed", None))

    try:
        print("Action meanings:", DiscreteMetaAction.ACTIONS)
    except AttributeError:
        print("Action class:", DiscreteMetaAction)

    ep_r = 0.0
    try:
        for t in range(800):
            # tiny rule: slow down if lead vehicle too close
            action = 0
            ego = env.unwrapped.vehicle
            lead, _ = ego.road.neighbour_vehicles(ego, ego.target_lane_index)
            if lead is not None:
                try:
                    gap = max(0.1, lead.distance_to(ego))
                except Exception:
                    # Some objects (e.g. static Obstacle) may not implement distance_to
                    gap = max(0.1, float(np.linalg.norm(np.array(getattr(lead, 'position', [0,0])) - np.array(ego.position))))
                if gap < 12.0:
                    action = 4  # slower
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            env.render()
            time.sleep(1 / 30)
            if term or trunc:
                print(f"Episode ended. Return={ep_r:.2f}")
                break
    finally:
        env.close()
