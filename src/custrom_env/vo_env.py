import copy
from math import atan2, cos, pi
import math
from typing import List
from gym import spaces
import numpy as np

from custrom_env import vo_robot
from custrom_env.perception_info import PerceptionInfo
from custrom_env.rvo_controller import VOController
from custrom_env.vo_robot import VORobot
from irsim.env import EnvBase
from irsim.world.robots.robot_diff import RobotDiff


class VOEnv(EnvBase):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
    action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
    def __init__(
        self, 
        world_name, 
        obs_mode, 
        reward_parameters=(0.3, 1.0, 0.3, 1.2, 0.2, 3.6, 0, 0), 
        neighbor_region=4, 
        neighbor_num=5,
        collision_time_threshold=5,
        safety_distance=0.2,
        lidar_resolution=0.2,
        **kwargs
    ):
        super(VOEnv, self).__init__(world_name, **kwargs)
        self.obs_mode = obs_mode

        self.neighbor_region = neighbor_region
        self.neighbor_num = neighbor_num
        self.collision_time_threshold = collision_time_threshold

        self.lidar_resolution = lidar_resolution

        # initialise VO robots
        self.vo_robots = self.init_vo_robots()
        # initialise RVO Controller
        self.rvo_controller = VOController(
            obs_mode=obs_mode,
            reward_parameters=reward_parameters,
            robot_num=self.robot_number,
            neighbor_region=neighbor_region,
            neighbor_num=neighbor_num,
            collision_time_threshold=collision_time_threshold,
            safety_distance=safety_distance,
        )
        # initialise perception info
        self.prev_perceptions: List[PerceptionInfo] = [None for _ in range(self.robot_number)]
        self.current_perceptions: List[PerceptionInfo] = [vo_robot.get_perception_info() for vo_robot in self.vo_robots]
    
        self.recieved_goal_rewards = [False for _ in range(self.robot_number)]

    def init_vo_robots(self) -> List[VORobot]:
        vo_robots = []
        for i in range(self.robot_number):
            robot = self.robot_list[i]
            external_objects = self.objects[:i] + self.objects[i+1:]
            vo_robot = VORobot(robot, external_objects, self.neighbor_region, self.obs_mode, self.lidar_resolution)
            vo_robots.append(vo_robot)
        return vo_robots
    
    def step(self, action_list):

        # convert actions from omni to diff
        diff_action_list = []
        for i, action in enumerate(action_list):
            vo_robot = self.vo_robots[i]
            if vo_robot.arrive or vo_robot.collision:
                diff_action = np.zeros((2, 1))
            else:
                diff_action = self.omni2diff(vo_robot.state, action, vo_robot.robot.vel_max[0,0], vo_robot.robot.vel_max[1, 0])
            diff_action_list.append(diff_action)
        
        # step the environment
        super(VOEnv, self).step(diff_action_list)
        
        # update perception info
        self.prev_perceptions = copy.deepcopy(self.current_perceptions)
        self.current_perceptions = [vo_robot.get_perception_info() for vo_robot in self.vo_robots]

        # get observation and reward
        reward_list = self.rvo_controller.get_rewards(
            perception_info=self.prev_perceptions,
            recieved_goal_rewards=self.recieved_goal_rewards,
            action=action_list
            )
        observation_list = self.rvo_controller.get_observations(self.prev_perceptions, self.current_perceptions)
        
        # get collision and arrive flags
        collision_list_flag = [vo_robot.collision for vo_robot in self.vo_robots]
        arrive_list_flag = [vo_robot.arrive for vo_robot in self.vo_robots]

        return observation_list, reward_list, collision_list_flag, arrive_list_flag

    def omni2diff(self, state, action, vmax, wmax, tolerance=0.1, min_speed=0.02):
        vx = action[0]
        vy = action[1]
        t = self.step_time
        x = state[0, 0]
        y = state[1, 0]
        robot_radians = state[2, 0]

        speed = np.sqrt(vx**2 + vy**2)
        vel_radians = math.atan2(vy, vx)

        diff_radians = robot_radians - vel_radians

        if speed > vmax:
            speed = vmax

        if diff_radians > pi:
            diff_radians = diff_radians - 2*pi
        elif diff_radians < -pi:
            diff_radians = diff_radians + 2*pi
        

        if diff_radians < tolerance and diff_radians > -tolerance:
            w = 0
        else:
            w = -diff_radians/t
            if w > wmax:
                w = wmax
            elif w < -wmax:
                w = -wmax
        
        v = speed * math.cos(diff_radians)
        if v < 0:
            v = 0

        if speed < min_speed:
            v = 0
            w = 0
        
        vel_diff = np.array([[v], [w]])
        
        return vel_diff

    def reset(self, id=None, random_ori=False):
        if id is None:
            # reset all the bots
            super(VOEnv, self).reset()

            self.recieved_goal_rewards = [False for _ in range(self.robot_number)]
            
            if random_ori:
                for robot in self.robot_list:
                    robot : RobotDiff
                    # get a random orientation
                    theta = np.random.uniform(-pi, pi)
                    new_state = [robot.state[0, 0], robot.state[1, 0], theta]
                    robot.set_state(new_state)
            self.prev_perceptions = [None for _ in range(self.robot_number)]
        else:
            # reset a single bot
            self.robot_list[id].reset()

            self.recieved_goal_rewards[id] = False

            if random_ori:
                robot : RobotDiff = self.robot_list[id]
                # get a random orientation
                theta = np.random.uniform(-pi, pi)
                new_state = [robot.state[0, 0], robot.state[1, 0], theta]
                robot.set_state(new_state)

            self.prev_perceptions[id] = None
            
            return None
        
        self.current_perceptions = [vo_robot.get_perception_info() for vo_robot in self.vo_robots]
        return self.rvo_controller.get_observations(self.prev_perceptions, self.current_perceptions)

    # use this function only when absolutely necessary as it is expensive to call
    def get_observation(self):
        return self.rvo_controller.get_observations(self.prev_perceptions, self.current_perceptions)