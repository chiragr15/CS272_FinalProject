from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import MDPVehicle

from highway_env.vehicle.objects import RoadObject, Obstacle
from custom_env.objects import Pothole

class CustomVehicle(MDPVehicle):
    def __init__(self, road, position, heading=0, speed=0):
        super().__init__(road, position, heading, speed)

    
    def handle_collisions(self, other: RoadObject, dt: float = 0) -> None:
        """
        Check for collision with another vehicle.

        :param other: the other vehicle or object
        :param dt: timestep to check for future collisions (at constant velocity)
        """
        if other is self or not (self.check_collisions or other.check_collisions):
            return
        if not (self.collidable and other.collidable):
            return
        intersecting, will_intersect, transition = self._is_colliding(other, dt)
        
        if isinstance(other, Pothole):  
            if will_intersect:
                if self.solid and other.solid:
                    if isinstance(other, Obstacle):
                        self.impact = None
                    elif isinstance(self, Obstacle):
                        other.impact = transition
                    else:
                        self.impact = None
                        other.impact = -transition/2
            if intersecting:
                
                self.speed -= 1.5

                if self.solid and other.solid:
                    self.crashed = False
                    other.crashed = True
                if not self.solid:
                    self.hit = True
                if not other.solid:
                    other.hit = True
        else: 
            if will_intersect:
                if self.solid and other.solid:
                    if isinstance(other, Obstacle):
                        self.impact = transition
                    elif isinstance(self, Obstacle):
                        other.impact = transition
                    else:
                        self.impact = transition / 2
                        other.impact = -transition / 2
            if intersecting:
                if self.solid and other.solid:
                    self.crashed = True
                    other.crashed = True
                if not self.solid:
                    self.hit = True
                if not other.solid:
                    other.hit = True
