from typing import Tuple


class PerceptionInfo:
    def __init__(self, type: str):
        self.type = type

class CircularObstacle(PerceptionInfo):
    def __init__(self, position: Tuple, velocity: Tuple, radius: float):
        super(CircularObstacle, self).__init__(type="circular_obstacle")
        self.position = position
        self.velocity = velocity
        self.radius = radius