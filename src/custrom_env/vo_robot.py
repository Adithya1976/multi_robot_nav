from typing import List
from custrom_env.perception_info import PerceptionInfo
from irsim.world.object_base import ObjectBase
from irsim.world.robots.robot_diff import RobotDiff


class VORobot:
    def __init__(self, robot: RobotDiff, external_objects: List[ObjectBase], mode: str):
        self.robot = robot
        self.external_objects = external_objects
        self.mode = mode
    
    def get_perception_info(self) -> List[PerceptionInfo]:
        pass