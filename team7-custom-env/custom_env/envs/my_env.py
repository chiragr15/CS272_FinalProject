import numpy as np

from highway_env.envs import HighwayEnv
from highway_env.envs.common.action import DiscreteMetaAction
from highway_env.envs.highway_env import Observation
from highway_env import utils, vehicle
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle
import highway_env.vehicle.behavior

from custom_env.vehicle.SuddenBrakingVehicle import SuddenBrakingVehicle
highway_env.vehicle.behavior.SuddenBrakingVehicle = SuddenBrakingVehicle
from custom_env.vehicle import GhostVehicle, CustomVehicle
from custom_env.objects.Pothole import Pothole

# Observation = np.ndarray

class MyEnv(HighwayEnv): 
    """
    Team 7 custom environment derived from highway-env.
    Environment includes potholes, sudden braking vehicles and ghost lidar observtaions.
    
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
                "observation": {
                    "type": "LidarObservation",
                     "cells": 36,
                     "maximum_range": 100.0,
                },
                "action":{
                    "type" : "DiscreteMetaAction",
                    "vehicle_class": "custom_env.vehicle.customvehicle.CustomVehicle",
                },
                "lanes_count": 8,
                "vehicles_count": 40,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40, #[s]
                "ego_spacing": 2,
                "vehicles_density": 3,
                "collision_reward": -1, # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1, # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "lane_change_reward": 0, # The reward received at each lane change action.
                "high_speed_reward": 0.4, # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "speed_limit": 50,
                "reward_speed_range": [25,50],
                "normalize_reward": True,
                "offroad_terminal": False,
                "other_vehicles_type": "highway_env.vehicle.behavior.SuddenBrakingVehicle",
                #"other_vehicles_type": "custom_env.vehicle.SuddenBrakingVehicle"
                "anomaly_interval": 3, # Exhibit GhostVehicle anomalies every N agent steps. Note: Every N agent steps, NOT simulation steps.
                "potholes": {
                    "enabled": True,
                    "count": 20,
                    "spawn_ahead_min": 20.0,
                    "spawn_ahead_max": 1000.0,
                }
            }
        )
        return config

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
        self._make_potholes()

    def _make_road(self) -> None:
        """Create a road composed of striaght adjacent lines"""
        self.road = Road(
            network = RoadNetwork.straight_road_network(
                lanes = self.config["lanes_count"], 
                speed_limit = self.config["speed_limit"]
            ),
            np_random = self.np_random,
            record_history = self.config["show_trajectories"],
        )

    def _make_vehicles(self) -> None:
        """Create some new rnadom vehicles of a given type, and add them on the road"""
        other_vehicles_type = utils.class_from_path(
            self.config["other_vehicles_type"]
        )
        other_per_controlled = utils.near_split(
            self.config["vehicles_count"], 
            num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:            

            # Create controlled Vehicle
            vehicle = CustomVehicle.create_random(
                self.road,
                speed=25.0,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            # Create Ghost Vehicle
            anomaly_interval = self.config["anomaly_interval"] * (self.config["simulation_frequency"] // self.config["policy_frequency"])
            ghost_vehicle = GhostVehicle.create_random(self.road, target_vehicle = vehicle, anomaly_interval = anomaly_interval) # Will use this method to create ghost vehicle
            self.road.vehicles.append(ghost_vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                        self.road, spacing=2 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
  
    def _make_potholes(self) -> None:
        """Create random potholes on random lanes."""
        p_conf = self.config.get("potholes", {})
        if not p_conf.get("enabled", False):
            return

        count = int(p_conf.get("count", 0))
        if count <= 0:
            return

        for _ in range(count):
            pothole = Pothole.create_random(
                self.road,
                x_min=p_conf.get("spawn_ahead_min", 20.0),
                x_max=p_conf.get("spawn_ahead_max", 160.0),
            )
            self.road.objects.append(pothole)

            

    def _reward(self, action):
        """Use default reward for now."""
        return super()._reward(action)


    def _is_terminated(self):
        """Use default termination condition."""        
        return super()._is_terminated() 
    