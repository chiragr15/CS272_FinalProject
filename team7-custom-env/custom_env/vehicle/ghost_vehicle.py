from __future__ import annotations

import highway_env
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.road import Road
from highway_env.utils import Vector

import numpy as np

from custom_env.vehicle.CustomVehicle import CustomVehicle

class GhostVehicle(Vehicle):
    """
    A ghost vehicle near the target vehicle
    A Vehicle-like object without physical motion or control logic.

    """

    DEFAULT_INITIAL_DEGREE = [0, 180]
    """ Range for random initial position around target vehicle [degree] """
    DEFAULT_INITIAL_DISTANCE = [2, 3]
    """ Range for random initial distance from target vehicle [units]. 1 unit = 1 car length and 1 car width """
    MAX_SPEED = 50
    """ Maximum apparent speed of the ghost vehicle [m/s] """
    MIN_SPEED = -50
    """ Minimum apparent speed of the ghost vehicle [m/s] """
    ANOMALY_TYPES = ["TELEPORT", "FLICKER", "SPEED"]
    """ Types of anomalies exhibited by ghost vehicle """
    COLOR = (195, 200, 215, 100)
    """ RGBA values for color of ghost vehicle"""


    def __init__(self, road, target_vehicle: CustomVehicle, anomaly_interval = 45, heading = 0, speed = 0 , degree = 65.0, distance = 2):
        
        # Vehicle around which ghost vehicle will appear
        self.target_vehicle = target_vehicle         
        self.position = self.target_vehicle.position
        super().__init__(road, self.position, heading, speed, predition_type = "zero_steering")


        # Position relative to target vehicle [degree] (0-> right, 90-> front, 180-> left)
        self.degree = degree 
        # distance from target vehicle [m or units]
        self.distance = distance         
        # Show some anomaly every Nth AGENT step, NOT SIMUALTION step
        self.anomaly_interval = anomaly_interval 

        self.check_collisions = False # Do not check its own collisions
        self.collidable = False # Disable collision with other collidables
        self.solid = True # For Lidar Observation 
        self.color = GhostVehicle.COLOR

        self.step_counter = 0

        
    
    @classmethod    
    def create_random(
        cls, 
        road: Road,  
        target_vehicle: Vehicle,
        anomaly_interval: float = 45,
        heading: float  = 0, 
        speed: float = None,  
        degree: float = None, 
        distance: float = None,
    ) -> GhostVehicle:
        """
        Create a random ghost vehicle.
        The position and /or speed are chosen randomly.
        """
        if speed is None:
            speed = road.np_random.uniform(
                GhostVehicle.MIN_SPEED, GhostVehicle.MAX_SPEED    
            )
        if degree is None:
            degree = road.np_random.uniform(
                GhostVehicle.DEFAULT_INITIAL_DEGREE[0], GhostVehicle.DEFAULT_INITIAL_DEGREE[1]    
            )
        if distance is None:
            distance = road.np_random.uniform(
                GhostVehicle.DEFAULT_INITIAL_DISTANCE[0], GhostVehicle.DEFAULT_INITIAL_DISTANCE[1]    
            )
        gv = cls(road, target_vehicle, anomaly_interval, heading, speed, degree, distance)
        return gv


    def step(self, dt):
        """ 
        Update position to stay near the target vehicle and/or exhibit anomalies
        """
        self.step_counter += 1
        self.solid = True
        self.color = GhostVehicle.COLOR

        if self.target_vehicle:

            # Follow target_vehicle
            target_pos = self.target_vehicle.position
            offset = np.array([
                self.target_vehicle.LENGTH * self.distance * np.sin(np.deg2rad(self.degree)),    
                self.target_vehicle.WIDTH * self.distance * np.cos(np.deg2rad(self.degree)),    
            ])
            self.position = target_pos + offset
            self.heading = self.target_vehicle.heading
            # Some random but realistic speed
            self.speed = self.road.np_random.uniform(0.5,1.5) * self.target_vehicle.speed

            # Exhibit an anomaly every N steps
            if self.step_counter % self.anomaly_interval == 0:
                anomaly = self.road.np_random.choice(GhostVehicle.ANOMALY_TYPES)
                # anomaly = "FLICKER"             
                if anomaly == "TELEPORT":                                        
                    sign = self.road.np_random.choice([-1, 1])
                    distance = sign * self.road.np_random.uniform(10, 25)
                    self.position += np.array([distance, 0.0])

                elif anomaly == "FLICKER":
                    self.solid = False
                    self.color = (0,0,0,0)
                
                elif anomaly == "SPEED":
                    # Unrealistic speed
                    sign = self.road.np_random.choice([-1, 1])
                    self.speed = sign * self.road.np_random.uniform(
                        50, 1000  
                    )



