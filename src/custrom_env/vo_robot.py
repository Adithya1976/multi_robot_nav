from ast import Tuple
import math
from typing import List
from custrom_env.perception_info import CircularObstacle, PerceptionInfo
from irsim.world.object_base import ObjectBase
from irsim.world.robots.robot_diff import RobotDiff


class VORobot:
    def __init__(self, robot: RobotDiff, external_objects: List[ObjectBase], mode: str):
        self.robot = robot
        self.external_objects = external_objects
        self.mode = mode
    
    def get_perception_info(self, previous_perception_info: List[PerceptionInfo], current_perception_info: List[PerceptionInfo]) -> PerceptionInfo:
        if self.mode == "ground_truth":
            state = self.robot.state[0, 0], self.robot.state[1, 0], self.robot.state[2, 0]
            velocity = self.robot.velocity_xy[0, 0], self.robot.velocity_xy[1, 0]
            radius = self.robot.radius
            desired_velocity = self.calculate_desired_velocity()
            
            obstacle_list = [
                CircularObstacle(
                    position = (obj.state[0, 0], obj.state[1, 0]),
                    velocity = (obj.velocity_xy[0, 0], obj.velocity_xy[1, 0]),
                    radius = obj.radius
                )
                for obj in self.external_objects
            ]

            return PerceptionInfo(
                type="ground_truth",
                state=state,
                velocity=velocity,
                radius=radius,
                desired_velocity=desired_velocity,
                obstacle_list=obstacle_list 
            )
    
    def calculate_desired_velocity(self) -> Tuple:
        if self.arrive:
            return 0, 0
        x, y = self.state[0, 0], self.state[1, 0]
        goal_x, goal_y = self.goal[0], self.goal[1]

        desired_angle = math.atan2(goal_y - y, goal_x - x)
        desired_vx = self.robot.vel_max[0, 0] * math.cos(desired_angle)
        desired_vy = self.robot.vel_max[0, 0] * math.sin(desired_angle)
        return desired_vx, desired_vy
        

    @property
    def arrive(self) -> bool:
        return self.robot.arrive

    @property
    def collision(self) -> bool:
        return self.robot.collision
    
    @property
    def state(self) -> List:
        return self.robot.state