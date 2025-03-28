from typing import List, Tuple

import numpy as np

class CircularObstacle:
    def __init__(self, position: Tuple, velocity: Tuple, radius: float):
        super(CircularObstacle, self).__init__(type="ground_truth")
        self.position = position
        self.velocity = velocity
        self.radius = radius

class LidarCluster:
    def __init__(self, points: np.ndarray):
        super(LidarCluster, self).__init__(type="lidar")
        self.points = points
        self.velocity = None
    
    def set_velocity(self, velocity: Tuple):
        self.velocity = velocity
        
class PerceptionInfo:
    def __init__(self, type: str, state: Tuple, velocity: Tuple, radius: float, desired_velocity: Tuple, obstacle_list: List[CircularObstacle] | List[LidarCluster]):
        self.type = type
        self.state = state
        self.velocity = velocity
        self.radius = radius
        self.desired_velocity = desired_velocity
        assert type == "ground_truth" or type == "lidar"
        if type == "ground_truth":
            assert all(isinstance(item, CircularObstacle) for item in obstacle_list)
        else:
            assert all(isinstance(item, LidarCluster) for item in obstacle_list)
        self.obstacle_list = obstacle_list