# narrow_street_env.py
import math
import numpy as np
import gymnasium as gym

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import DiscreteMetaAction
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.dynamics import Vehicle

# (Optional) Static obstacle class:
# In recent highway-env versions, Obstacle lives under highway_env.vehicle.objects or road.objects.
# We'll try both and fallback to a zero-speed Vehicle if unavailable.
ObstacleClass = None
try:
    from highway_env.road.objects import Obstacle as ObstacleClass  # newer
except Exception:
    try:
        from highway_env.vehicle.objects import Obstacle as ObstacleClass  # older
    except Exception:
        ObstacleClass = None


class NarrowStreetEnv(AbstractEnv):
    """
    Two-lane road:
      - Left lane: moving traffic
      - Right lane: ego + parked vehicles encroaching into the lane (static obstacles)
    Goal: make forward progress without side-swiping obstacles or colliding with traffic.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 8,       # N nearest (kept small to be learnable)
                "absolute": False,
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "normalize": True,
                "see_behind": True,
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            # --- Scene parameters ---
            "road_length": 350.0,
            "lane_width": 3.3,
            "speed_limit": 25.0,          # m/s 
            "right_lane_narrow_factor": 0.9,  # effective usable width fraction due to encroachment
            "parked_start": 40.0,         # start placing parked cars after x
            "parked_spacing": 22.0,       # longitudinal gap between parked cars
            "parked_depth": 0.7,          # how much they bite into the right lane (meters)
            "traffic_density": 0.12,      # moving vehicles per meter (left lane)
            "seed": 42,

            # --- Rewards (tune freely) ---
            "collision_reward": -10.0,
            "progress_reward": 0.04,      # per meter forward
            "time_penalty": -0.01,        # per step
            "lateral_offset_penalty": -0.02,   # per meter of |offset| (stronger on right lane)
            "unsafe_gap_penalty": -0.5,        # for too-close headway
            "target_headway": 1.5,        # seconds

            # --- Termination ---
            "duration": 80,               # steps
            "offroad_termination": False,

            # --- Rendering ---
            "screen_width": 1280,
            "screen_height": 240,
            "show_trajectories": False,
            "render_agent": True,
        })
        return cfg

    def _reset(self) -> None:
        #super()._reset()
        #self.np_random, _ = gym.utils.seeding.np_random(self.config["seed"])
        self._create_road()
        self._create_vehicles()

    # -----------------------------
    # Road & lane layout
    # -----------------------------
    def _create_road(self) -> None:
        net = RoadNetwork()
        L = self.config["road_length"]
        w = self.config["lane_width"]

        # Two parallel straight lanes from node ('a' -> 'b')
        # y=0 is right lane centerline; y=+w is left lane centerline
        right_lane = StraightLane([0, 0.0], [L, 0.0], width=w, line_types=(LineType.CONTINUOUS, LineType.STRIPED))
        left_lane  = StraightLane([0, w],   [L, w],   width=w, line_types=(LineType.STRIPED, LineType.CONTINUOUS))

        net.add_lane("a", "b", right_lane)
        net.add_lane("a", "b", left_lane)

        self.road = Road(
            network=net,
            vehicles=[],
            np_random=self.np_random,
            record_history=True,
        )

        # Store handles for convenience
        self._right_lane = self.road.network.get_lane(("a", "b", 0))
        self._left_lane  = self.road.network.get_lane(("a", "b", 1))

        # Derive the "usable width" of right lane (visual cue via parked cars, actual penalties in reward)
        self._right_usable_halfwidth = (w * self.config["right_lane_narrow_factor"]) / 2

        # Place parked obstacles encroaching from the right boundary into the right lane
        self._spawn_parked_vehicles()

    def _spawn_parked_vehicles(self):
        L = self.config["road_length"]
        start = self.config["parked_start"]
        spacing = self.config["parked_spacing"]
        bite = self.config["parked_depth"]  # meters encroaching into the lane

        lane = self._right_lane
        y_half = lane.width / 2.0

        for s in np.arange(start, L - 10.0, spacing):
            # lateral position near the right boundary (negative side), plus slight encroachment inward
            lat = -y_half + bite
            pos = lane.position(s, lat)

            if ObstacleClass is not None:
                ob = ObstacleClass(self.road, position=pos, heading=0.0)
                # give it the approximate size of a compact car
                if hasattr(ob, "LENGTH"):
                    ob.LENGTH = 4.2
                if hasattr(ob, "WIDTH"):
                    ob.WIDTH = 1.7
                self.road.objects.append(ob)
            else:
                # Fallback: a zero-speed vehicle that acts as a parked car
                v = Vehicle(self.road, position=pos, heading=0.0, speed=0.0)
                v.LENGTH, v.WIDTH = 4.2, 1.7
                v.color = (120, 120, 120)  # grey to visualize as parked
                # Mark as non-controlled & immobile
                v.MAX_SPEED = 0.0
                self.road.vehicles.append(v)

    # -----------------------------
    # Vehicles (ego + background traffic)
    # -----------------------------
    def _create_vehicles(self) -> None:
        rng = self.np_random

        # --- Ego on right lane near x=15 ---
        ego_x = 15.0
        ego_pos = self._right_lane.position(ego_x, 0.0)
        ego_heading = self._right_lane.heading_at(ego_x)
        ego_v = 10.0  # m/s

        # Construct ego directly (avoid create_from API differences)
        ego = MDPVehicle(
            road=self.road,
            position=np.array(ego_pos, dtype=float),
            heading=float(ego_heading),
            speed=float(ego_v),
        )
        ego.SPEED_MIN = 0.0
        ego.SPEED_MAX = self.config["speed_limit"]
        ego.color = (0, 255, 0)
        self.road.vehicles.append(ego)
        self.vehicle = ego

        # --- keep at least ~7 m clear in front of ego (remove anything too close) ---
        SAFE_GAP = 7.0  # meters

        # parked objects
        if hasattr(self.road, "objects"):
            self.road.objects = [
                o for o in self.road.objects
                if (o.position[0] - ego.position[0]) > SAFE_GAP  # keep only if far enough ahead
                or (ego.position[0] - o.position[0]) >= 0        # or behind ego
            ]

        # moving/zero-speed vehicles (except ego)
        filtered = []
        for v in self.road.vehicles:
            if v is ego:
                filtered.append(v)
                continue
            dx = v.position[0] - ego.position[0]
            if dx <= SAFE_GAP and abs(v.position[1] - ego.position[1]) < self.config["lane_width"] * 0.6:
                # too close ahead in (approximately) same lane -> drop it
                continue
            filtered.append(v)
        self.road.vehicles = filtered
        
        # --- Left-lane moving traffic ---
        density = self.config["traffic_density"]
        L = self.config["road_length"]
        approx_count = max(1, int(density * L))

        xs = sorted(rng.uniform(low=20.0, high=L - 30.0, size=approx_count))
        for x in xs:
            pos = self._left_lane.position(float(x), 0.0)
            heading = self._left_lane.heading_at(float(x))
            spd = float(rng.uniform(8.0, 18.0))

            car = IDMVehicle(
                road=self.road,
                position=np.array(pos, dtype=float),
                heading=float(heading),
                speed=spd,
            )
            self.road.vehicles.append(car)
            
            
            
    def _front_vehicle_same_lane(self, veh):
        """
        Return (front_entity, dx_along_x, front_speed).
        dx is along the road x-axis for clarity; we consider 'same lane' if |Δy| < 0.7*lane_width.
        """
        lane_tol = self.config["lane_width"] * 0.7
        vx, vy = float(veh.position[0]), float(veh.position[1])
        best_dx, front, front_speed = float("inf"), None, 0.0

        # moving vehicles
        for other in self.road.vehicles:
            if other is veh:
                continue
            dy = abs(float(other.position[1]) - vy)
            dx = float(other.position[0]) - vx
            if dx > 0 and dy < lane_tol and dx < best_dx:
                best_dx, front = dx, other
                front_speed = float(getattr(other, "speed", 0.0))

        # parked objects (treat as zero-speed fronts if they sit in lane)
        objs = getattr(self.road, "objects", [])
        for obj in objs:
            if not hasattr(obj, "position"):
                continue
            dy = abs(float(obj.position[1]) - vy)
            dx = float(obj.position[0]) - vx
            if dx > 0 and dy < lane_tol and dx < best_dx:
                best_dx, front = dx, obj
                front_speed = 0.0

        if front is None:
            return None, None, None
        return front, best_dx, front_speed

    
    # Gym API: step / reward / termination
    def _reward(self, action: int) -> float:
        conf = self.config

        # Forward progress (delta x along lane frame). We approximate by ego's vx in world coordinates.
        ego = self.vehicle
        vx = ego.speed * math.cos(ego.heading - 0.0)  # road is aligned with x
        r_progress = conf["progress_reward"] * max(0.0, vx)

        # Collision penalty (if any)
        r_collision = 0.0
        if ego.crashed:
            r_collision += conf["collision_reward"]

        # Lateral offset penalty (stronger when in right lane with parked cars present)
        # Compute signed lateral offset from right lane center
        # (Positive is towards left lane; negative towards the curb/parked cars)
        lane = self._right_lane
        s, lat = lane.local_coordinates(ego.position)
        r_lat = conf["lateral_offset_penalty"] * abs(lat)


        # Unsafe headway penalty (lane-aligned dx and TTC)
        r_gap = 0.0
        front, dx, fspd = self._front_vehicle_same_lane(ego)
        if front is not None and dx is not None:
            rel_v = max(0.0, ego.speed - max(0.0, fspd if fspd is not None else 0.0))
            if rel_v > 1e-6:  # avoid div-by-zero noise
                ttc = dx / rel_v
                if ttc < conf["target_headway"]:
                    r_gap += conf["unsafe_gap_penalty"] * (conf["target_headway"] - ttc)


        # Per-step time penalty for efficiency pressure
        r_time = conf["time_penalty"]

        return float(r_progress + r_collision + r_lat + r_gap + r_time)

    def _is_terminated(self) -> bool:
        """Episode ends due to failure events (not time)."""
        # crash or going beyond usable right-lane boundary (if you want that as failure)
        terminated = bool(getattr(self.vehicle, "crashed", False))
        if self.config.get("offroad_termination", True):
            terminated = terminated or self._is_offroad()
        return terminated

    def _is_truncated(self) -> bool:
        """Episode stops due to time horizon (timeout)."""
        return bool(self.time >= self.config["duration"])


    # Use standard DiscreteMetaAction mapping (KEEP_LANE, LEFT, RIGHT, FASTER, SLOWER)
    def _action_space(self):
        return DiscreteMetaAction(self)

    # Optional constraint: treat large lateral offset in right lane as "offroad"
    def _is_offroad(self) -> bool:
        if not self.config.get("offroad_termination", True):
            return False
        lane = self._right_lane
        _, lat = lane.local_coordinates(self.vehicle.position)
        usable = self._right_usable_halfwidth
        return bool(abs(lat) > usable + 0.15)  # small tolerance


# Gym registration helper
def register_env():
    gym.register(
        id="NarrowStreetTwoLane-v0",
        entry_point="narrow_street_env:NarrowStreetEnv",
        max_episode_steps=NarrowStreetEnv.default_config()["duration"],
    )

# Allow `python narrow_street_env.py` to quick-run a demo
if __name__ == "__main__":
    import time
    register_env()

    env = gym.make("NarrowStreetTwoLane-v0", render_mode="human")

    # ---- Override config safely (do this BEFORE reset) ----
    overrides = {
        "duration": 400,          # longer episode so the window doesn't close fast
        "traffic_density": 0.03,  # slightly easier to watch
        "parked_spacing": 35.0,   # fewer parked cars
        "right_lane_narrow_factor": 0.50,
        "parked_depth": 0.5,
    }
    # Prefer configure(...) if available; otherwise update the underlying config.
    if hasattr(env, "configure"):
        env.configure(overrides)
    else:
        env.unwrapped.config.update(overrides)

    obs, info = env.reset(seed=0)

    ep_r, ep = 0.0, 1
    try:
        while True:
            # simple baseline action: KEEP_LANE (0); occasionally SLOWER (4)
            action = 0 if np.random.rand() > 0.1 else 4
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            env.render()
            time.sleep(1/30)  # ~30 FPS so you can actually see it

            if term or trunc:
                print(f"Episode {ep} reward: {ep_r:.2f}")
                obs, info = env.reset()
                ep_r, ep = 0.0, ep + 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()