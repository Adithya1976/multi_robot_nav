from ast import Not
from calendar import c
from typing import List

import numpy as np
from custrom_env.perception_info import PerceptionInfo


class VOController:
    def __init__(self, obs_mode, robot_num, reward_parameters, neighbor_region, neighbor_num):
        self.obs_mode = obs_mode
        self.reward_parameters = reward_parameters
        self.neighbor_region = neighbor_region
        self.neighbor_num = neighbor_num
        self.robot_num = robot_num
    
    def get_observation_and_reward(self, 
                                   prev_perception_info: List[PerceptionInfo], 
                                   current_perception_info: List[PerceptionInfo], 
                                   action_list):
        for i in range(self.robot_num):
            prev_perception = prev_perception_info[i]
            current_perception = current_perception_info[i]
            action = action_list[i]

            pro_obs = np.array([current_perception.velocity[0], current_perception.velocity[1],
                                current_perception.state[2],
                                current_perception.desired_velocity[0], current_perception.desired_velocity[1],
                                current_perception.radius])
            
            print(pro_obs)

            if self.obs_mode == "ground_truth":
                pass
            elif self.obs_mode == "lidar":
                pass
            else:
                raise ValueError("Invalid obs_mode")
        
        return None, None
    
    def get_rewards(self, perception_info: PerceptionInfo, action):
        return None

    def get_observations(self, previous_perceptions: List[PerceptionInfo], current_perceptions: List[PerceptionInfo]):
        if self.obs_mode == "ground_truth":
            for i in range(self.robot_num):
                current_perception = current_perceptions[i]
                pro_obs = np.array([current_perception.velocity[0], current_perception.velocity[1],
                                    current_perception.state[2],
                                    current_perception.desired_velocity[0], current_perception.desired_velocity[1],
                                    current_perception.radius])
                
                ext_obs = []
                for obstacle in current_perception.obstacle_list:
                    pass
                
        elif self.obs_mode == "lidar":
            raise NotImplementedError
        else:
            raise ValueError("Invalid obs_mode")