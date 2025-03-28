from typing import List, Tuple, Union

import numpy as np

class CircularObstacle:
    def __init__(self, position: Tuple, velocity: Tuple, radius: float):
        self.position = position
        self.velocity = velocity
        self.radius = radius

class LidarCluster:
    def __init__(self, points: np.ndarray, centroid: np.ndarray = None, min_dist: float = None):
        self.points = points
        self.centroid = centroid
        self.min_dist = min_dist
        self.velocity = (0, 0)
    
    def set_velocity(self, velocity: Tuple):
        self.velocity = velocity
        
class PerceptionInfo:
    def __init__(
        self, 
        type: str, 
        state: Tuple, 
        velocity: Tuple, 
        radius: float,
        arrive: bool,
        collision: bool, 
        desired_velocity: Tuple, 
        obstacle_list: Union[List[CircularObstacle], List[LidarCluster]]
    ):
        self.type = type
        self.state = state
        self.velocity = velocity
        self.radius = radius
        self.arrive = arrive
        self.collision = collision
        self.desired_velocity = desired_velocity
        assert type == "ground_truth" or type == "lidar"
        if type == "ground_truth":
            assert all(isinstance(item, CircularObstacle) for item in obstacle_list)
        else:
            assert all(isinstance(item, LidarCluster) for item in obstacle_list)
        self.obstacle_list = obstacle_list