# narrow_safe_env.py

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle  # noqa: F401 (kept for completeness)

# IMPORTANT: reuse your existing narrow-street env geometry
from narrow_street import NarrowStreetEnv


class NarrowSafeEnv(NarrowStreetEnv):
    """
    Safer variant of narrow-street-v0:
      - keeps the same road/parked-vehicle setup as NarrowStreetEnv
      - adds strong penalties for tailgating
      - rewards safe queuing when both lanes are blocked
      - slightly slows surrounding moving traffic
    """

    @classmethod
    def default_config(cls):
        config = super().default_config()

        # Do NOT overwrite geometry/parked-count/etc here;
        # just add a few extra keys for reward & traffic shaping.
        config.update(
            {
                # Speed settings for moving traffic (clamped after creation)
                "other_vehicles_speed_min": 15.0,  # slightly slower than ego
                "other_vehicles_speed_max": 22.0,

                # Reward parameters
                "desired_time_headway": 1.5,        # s, like safe time gap
                "headway_penalty_scale": 3.0,       # how strong to penalize low gap
                "congested_block_distance": 40.0,   # m ahead to look for blocking
                "queue_reward": 0.05,               # per step reward for safe queuing
                "success_reward": 5.0,              # bonus when reaching the end
                "collision_reward": -10.0,          # big negative on crash

                # Used only for progress term; default if not already in parent
                "speed_limit": config.get("speed_limit", 25.0),
            }
        )
        return config

    # -------------------------------
    # Vehicle / traffic creation
    # -------------------------------
    def _create_vehicles(self):
        """
        Use NarrowStreetEnv's vehicle creation (parked cars, layout),
        then clamp speeds of moving IDM vehicles to a slightly lower range.
        """
        super()._create_vehicles()

        v_min = self.config["other_vehicles_speed_min"]
        v_max = self.config["other_vehicles_speed_max"]

        ego = self.vehicle

        for v in self.road.vehicles:
            if v is ego:
                continue
            # Only adjust moving traffic; parked cars are typically plain Vehicle
            if not isinstance(v, IDMVehicle):
                continue
            v.speed = float(np.clip(v.speed, v_min, v_max))

    # -------------------------------
    # Reward shaping
    # -------------------------------
    def _reward(self, action: int) -> float:
        """
        Reward = progress term + headway safety shaping + queue reward
                 + terminal bonuses/penalties.
        """

        if not self.vehicle:
            return 0.0

        v = float(self.vehicle.speed)
        speed_limit = float(self.config.get("speed_limit", 25.0))

        # 1) Progress term: encourage moving, but not too aggressively
        progress = v / max(speed_limit, 1e-3)  # ~ in [0, 1+]

        # 2) Headway penalty (distance-based; acts like time headway)
        lead_vehicle, gap = self._get_lead_vehicle_and_gap()

        headway_penalty = 0.0
        queue_reward = 0.0

        if lead_vehicle is not None and gap is not None:
            desired_th = float(self.config["desired_time_headway"])
            # Desired distance ~ v * desired_th
            safe_gap = max(v * desired_th, 5.0)  # at least 5 m
            if gap < safe_gap:
                # Relative deficit (0 when at safe gap, 1 when nose-to-tail)
                excess = (safe_gap - gap) / safe_gap
                headway_penalty = -self.config["headway_penalty_scale"] * excess

            # 3) Queue reward: if both lanes are blocked and ego is safely stopped
            if (
                gap < self.config["congested_block_distance"]  # close to a queue
                and v < 1.0                                    # basically stopped
                and self._all_lanes_blocked_ahead()
            ):
                queue_reward = self.config["queue_reward"]

        # 4) Base reward
        reward = progress + headway_penalty + queue_reward

        # 5) Terminal bonuses / penalties
        if self.vehicle.crashed:
            reward += self.config["collision_reward"]

        if self._is_success():
            reward += self.config["success_reward"]

        return float(reward)

    # -------------------------------
    # Helpers
    # -------------------------------
    def _get_lead_vehicle_and_gap(self):
        """
        Return (lead_vehicle, gap) for ego's current lane.
        gap is the longitudinal distance from ego to the lead vehicle center.
        """
        ego = self.vehicle
        if ego is None:
            return None, None

        lead = None
        min_dx = float("inf")

        for v in self.road.vehicles:
            if v is ego:
                continue
            if v.lane_index != ego.lane_index:
                continue

            dx = v.position[0] - ego.position[0]
            if 0 < dx < min_dx:
                min_dx = dx
                lead = v

        if lead is None:
            return None, None
        return lead, min_dx

    def _all_lanes_blocked_ahead(self) -> bool:
        """
        Check if, within some distance ahead, *every* lane has at least one
        vehicle in front of ego within look-ahead distance.
        """
        ego = self.vehicle
        if ego is None:
            return False

        look_ahead = float(self.config["congested_block_distance"])

        try:
            lanes_count = int(self.config["lanes_count"])
        except KeyError:
            # Fallback if not in config for some reason
            lanes_count = 2

        lanes = range(lanes_count)
        blocked = {lane: False for lane in lanes}

        for v in self.road.vehicles:
            if v is ego:
                continue

            lane = v.lane_index[2]  # assuming ("a", "b", lane_id)
            if lane not in blocked:
                continue

            dx = v.position[0] - ego.position[0]
            if 0 < dx < look_ahead:
                blocked[lane] = True

        return all(blocked.values())

    def _is_success(self) -> bool:
        """
        Define success as reaching the end of the narrow street without crashing.
        Uses road_length from NarrowStreetEnv config if available.
        """
        ego = self.vehicle
        if ego is None:
            return False

        road_len = float(self.config.get("road_length", 600.0))
        return (ego.position[0] >= road_len - 5.0) and (not ego.crashed)

    # Gymnasium compatibility: used by some wrappers / vec envs
    def compute_reward(self, obs, actions, next_obs, dones, infos):
        if isinstance(actions, np.ndarray) and actions.ndim > 0:
            return np.array([self._reward(a) for a in actions])
        else:
            return np.array(self._reward(actions))


def make_narrow_safe_env(**kwargs):
    """Factory function used by gymnasium.make."""
    return NarrowSafeEnv(**kwargs)

# Register using a callable entry_point (no module import by name needed)
from gymnasium.envs.registration import register

register(
    id="narrow-safe-v0",
    entry_point=make_narrow_safe_env,
)
