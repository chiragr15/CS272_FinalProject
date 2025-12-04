from highway_env.vehicle.objects import Obstacle, RoadObject
from highway_env.road.road import Road
import numpy as np

class Pothole(Obstacle):
    def __init__(self, road, position, heading = 0, speed = 0):
        super().__init__(road, position, heading, speed)
        self.check_collisions = False # Do not check its own collisions
        self.collidable = True # Disable collision with other collidables
        self.solid = True # For Lidar Observation
            
    @classmethod
    def create_random(
        cls,
        road: Road,
        x_min: float = 20.0,
        x_max: float = 160.0,
    ) -> "Pothole":
        """
        Create a pothole randomly placed on a random lane.

        :param road: the Road object
        :param x_min: minimum longitudinal distance along the lane
        :param x_max: maximum longitudinal distance along the lane
        """
        
        lanes = road.network.lanes_list()
        if not lanes:
            raise ValueError("No lanes available in road network to spawn potholes.")
        lane = road.np_random.choice(lanes)
     
        L = getattr(lane, "length", 1000.0)
        longitudinal = float(road.np_random.uniform(x_min, min(x_max, L)))

        position = lane.position(longitudinal, lateral=0.0)
        heading = lane.heading_at(longitudinal)

        return cls(road, position, heading)

