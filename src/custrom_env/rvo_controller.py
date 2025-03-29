from math import asin, atan2, cos, inf, pi, sin, sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np
from custrom_env.perception_info import CircularObstacle, LidarCluster, PerceptionInfo
import concurrent.futures


class VOController:
    def __init__(self, obs_mode, robot_num, reward_parameters, neighbor_region, neighbor_num, collision_time_threshold, safety_distance):
        self.obs_mode = obs_mode
        self.reward_parameters = reward_parameters
        self.neighbor_region = neighbor_region
        self.neighbor_num = neighbor_num
        self.robot_num = robot_num
        self.ctime_threshold = collision_time_threshold
        self.safety_distance = safety_distance
    
    def get_rewards(self, perception_info: List[PerceptionInfo], recieved_goal_rewards: List[bool], action):
        reward_list = []
        for i in range(self.robot_num):
            current_perception = perception_info[i]
            
            # desired velocity
            desired_velocity = current_perception.desired_velocity
            dis_des = sqrt((action[i][0] - desired_velocity[0])**2 + (action[i][1] - desired_velocity[1])**2)

            # min_exp_time
            max_inverse_exp_time = 0
            for obstacle in current_perception.obstacle_list:
                if self.obs_mode == "ground_truth":
                    inverse_exp_time = self.get_vo_representation_ground_truth(
                        current_perception.state,
                        current_perception.velocity,
                        current_perception.radius,
                        obstacle,
                        action[i]
                    )[-1]
                    max_inverse_exp_time = max(max_inverse_exp_time, inverse_exp_time)
                elif self.obs_mode == "lidar":
                    inverse_exp_time = self.get_vo_representation_lidar(
                        current_perception.state,
                        current_perception.velocity,
                        current_perception.radius,
                        obstacle,
                        action[i]
                    )[-1]
                    max_inverse_exp_time = max(max_inverse_exp_time, inverse_exp_time)
                else:
                    raise ValueError("Invalid obs_mode")
            
            min_collision_time = 1/max_inverse_exp_time - 0.2 if max_inverse_exp_time > 0 else inf
            
            p1, p2, p3, p4, p5, p6, p7, p8 = self.reward_parameters

            goal_reward = 0
            collision_reward = 0
            rvo_reward = 0
            if current_perception.arrive and not recieved_goal_rewards[i]:
                goal_reward = p8
                recieved_goal_rewards[i] = True
            elif current_perception.collision:
                collision_reward = p7
            elif not current_perception.arrive:
                if min_collision_time > self.ctime_threshold:
                    rvo_reward = p1 - p2 * dis_des
                elif min_collision_time > 0.1:
                    rvo_reward = p3 - p4 * (1/(min_collision_time + p5))
                else:
                    rvo_reward = -p6*(1/(min_collision_time + p5))
            reward = goal_reward + collision_reward + rvo_reward
            reward_list.append(reward)
        return reward_list

    def get_observations(self, previous_perceptions: List[PerceptionInfo], current_perceptions: List[PerceptionInfo]):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Map preserves the order of the input arguments
            observation_list = list(executor.map(
                lambda args: self.get_robot_observation(*args),
                zip(previous_perceptions, current_perceptions)
            ))
        return observation_list
    
    def get_robot_observation(self, prev_perception, current_perception):
        pro_obs = np.array([current_perception.velocity[0], current_perception.velocity[1],
                            current_perception.desired_velocity[0], current_perception.desired_velocity[1],
                            current_perception.state[2],
                            current_perception.radius])
        
        ext_obs_list = []

        if self.obs_mode == "ground_truth":
            for obstacle in current_perception.obstacle_list:
                ext_obs = self.get_vo_representation_ground_truth(
                    current_perception.state,
                    current_perception.velocity,
                    current_perception.radius,
                    obstacle
                    )
                ext_obs_list.append(ext_obs)
        elif self.obs_mode == "lidar":
            if prev_perception is None:
                obstacle_list = current_perception.obstacle_list
            else:
                # create mapping between current and previous perceptions
                prev_to_current_mapping = self.create_prev_to_current_mapping(current_perception, prev_perception)
                # set velocity for each cluster
                self.estimate_and_set_cluster_velocity(prev_to_current_mapping)

                obstacle_list = list(prev_to_current_mapping.values())

            for obstacle in obstacle_list:
                ext_obs = self.get_vo_representation_lidar(
                    current_perception.state,
                    current_perception.velocity,
                    current_perception.radius,
                    obstacle
                )
                ext_obs_list.append(ext_obs)

        ext_obs_list.sort(key=lambda x: (-x[-1], x[-2]), reverse=True)           

        # filter by neighbor_num
        if len(ext_obs_list) > self.neighbor_num:
            ext_obs_list = ext_obs_list[-self.neighbor_num:]

        if len(ext_obs_list) == 0:
            exter_obs = np.zeros((8,))
        else:
            exter_obs = np.concatenate(ext_obs_list)
        
        observation = np.round(np.concatenate((pro_obs, exter_obs)), 2)
        return observation
            
    
    def get_vo_representation_ground_truth(self, 
                                robot_state: Tuple,
                                robot_velocity: Tuple,
                                robot_radius: float,
                                obstacle: CircularObstacle,
                                action: Optional[Tuple] = None,
                                mode: str="rvo"
        ) -> np.ndarray:
        result_dict = {
            "vo_representation": None,
            "minimum_collision_time": None
        }
        if action is None:
            action = robot_velocity

        x, y, _ = robot_state
        vx, vy = robot_velocity
        r = robot_radius + self.safety_distance
        mx, my = obstacle.position
        mvx, mvy = obstacle.velocity
        mr = obstacle.radius + self.safety_distance

        if mvx == 0 and mvy == 0:
            mode = "vo"

        rel_x = x - mx
        rel_y = y - my

        dis_mr = sqrt(rel_y**2 + rel_x**2)
        angle_mr = atan2(my - y, mx - x)

        real_dis_mr = np.sqrt(rel_y**2 + rel_x**2)
        
        ratio = (r + mr) / dis_mr
        half_angle = asin(ratio)
        line_left_ori = self.wraptopi(angle_mr + half_angle)
        line_right_ori = self.wraptopi(angle_mr - half_angle)

        if mode == "vo":
            vo = [mvx, mvy, line_left_ori, line_right_ori]
            rel_vx = action[0] - mvx
            rel_vy = action[1] - mvy
        elif mode == "rvo":
            vo = [(vx + mvx) / 2, (vy + mvy) / 2, line_left_ori, line_right_ori]
            rel_vx = 2 * action[0] - mvx - vx
            rel_vy = 2 * action[1] - mvy - vy
        
        exp_time = inf

        if not self.vo_out_jud_vector(action[0], action[1], vo):
            exp_time = self.cal_exp_time(rel_x, rel_y, rel_vx, rel_vy, r + mr)
            if exp_time >= self.ctime_threshold:
                exp_time = inf
        
        inverse_exp_time = 1 / (exp_time + 0.2)
        min_dis = real_dis_mr - mr

        vo_representation = [vo[0], vo[1], cos(vo[2]), sin(vo[2]), cos(vo[3]), sin(vo[3]), min_dis, inverse_exp_time]
            
        return vo_representation
    
    def create_prev_to_current_mapping(self, 
                        current_perception: PerceptionInfo, 
                        prev_perceptions: PerceptionInfo
                    ) -> Dict[LidarCluster, LidarCluster]:
        current_clusters = current_perception.obstacle_list
        prev_clusters = prev_perceptions.obstacle_list

        prev_to_current_mapping = {}
        current_to_prev_mapping = {}
        current_to_prev_distances = {}

        for current_cluster in current_clusters:
            # find the closest cluster in the previous perception
            closest_distance = inf
            closest_prev_cluster = None
            for prev_cluster in prev_clusters:
                distance = sqrt(
                    (current_cluster.centroid[0] - prev_cluster.centroid[0])**2 +
                    (current_cluster.centroid[1] - prev_cluster.centroid[1])**2
                )
                if distance < closest_distance:
                    closest_distance = distance
                    closest_prev_cluster = prev_cluster

            if closest_prev_cluster is not None:
                current_to_prev_mapping[current_cluster] = closest_prev_cluster
                current_to_prev_distances[current_cluster] = closest_distance

        # Reverse mapping: find the best current cluster for each previous cluster
        for current_cluster, prev_cluster in current_to_prev_mapping.items():
            if prev_cluster not in prev_to_current_mapping or current_to_prev_distances[current_cluster] < current_to_prev_distances[prev_to_current_mapping[prev_cluster]]:
                prev_to_current_mapping[prev_cluster] = current_cluster

        return prev_to_current_mapping

    def estimate_and_set_cluster_velocity(self, prev_to_current_mapping: Dict[LidarCluster, LidarCluster]):
        for prev_cluster, current_cluster in prev_to_current_mapping.items():
            current_cluster.set_velocity((current_cluster.centroid[0] - prev_cluster.centroid[0], current_cluster.centroid[1] - prev_cluster.centroid[1]))
    
    def get_vo_representation_lidar(self, 
                                robot_state: Tuple,
                                robot_velocity: Tuple,
                                robot_radius: float,
                                obstacle: LidarCluster,
                                action: Optional[Tuple] = None,
                                mode: str="rvo",
        ) -> np.ndarray:
        if action is None:
            action = robot_velocity
        
        assert obstacle.velocity is not None, "Cluster velocity is not set"
        x, y, _ = robot_state

        vx, vy = robot_velocity
        r = robot_radius + self.safety_distance
        mvx, mvy = obstacle.velocity

        left_point = obstacle.points[0]
        right_point = obstacle.points[-1]

        left_dis = sqrt((left_point[0] - x)**2 + (left_point[1] - y)**2)
        right_dis = sqrt((right_point[0] - x)**2 + (right_point[1] - y)**2)

        left_half_angle = asin(r / left_dis) if r<left_dis else pi/2
        right_half_angle = asin(r / right_dis) if r<right_dis else pi/2

        left_angle = atan2(left_point[1] - y, left_point[0] - x)
        right_angle = atan2(right_point[1] - y, right_point[0] - x)

        line_left_ori = self.wraptopi(left_angle + left_half_angle)
        line_right_ori = self.wraptopi(right_angle - right_half_angle)

        if mvx == 0 and mvy == 0:
            mode = "vo"
        
        if mode == "vo":
            vo = [mvx, mvy, line_left_ori, line_right_ori]
            rel_vx = action[0] - mvx
            rel_vy = action[1] - mvy
        elif mode == "rvo":
            vo = [(vx + mvx) / 2, (vy + mvy) / 2, line_left_ori, line_right_ori]
            rel_vx = 2 * action[0] - mvx - vx
            rel_vy = 2 * action[1] - mvy - vy
        
        exp_time = inf
        if not self.vo_out_jud_vector(action[0], action[1], vo):
            exp_time = self.cal_exp_time_to_lidar_cluster(x, y, obstacle, rel_vx, rel_vy, r)
            if exp_time >= self.ctime_threshold:
                exp_time = inf

        inverse_exp_time = 1 / (exp_time + 0.2)
        min_dis = obstacle.min_dist

        vo_representation = [vo[0], vo[1], cos(vo[2]), sin(vo[2]), cos(vo[3]), sin(vo[3]), min_dis, inverse_exp_time]
    
        return vo_representation

    def cal_exp_time_to_lidar_cluster(self, x, y, cluster: LidarCluster, rel_vx, rel_vy, combined_r):
        points = cluster.points
        exp_time = inf
        for i in range(points.shape[0]):
            px = points[i, 0]
            py = points[i, 1]
            rel_x = x - px
            rel_y = y - py
            t = self.cal_exp_time(rel_x, rel_y, rel_vx, rel_vy, combined_r + self.safety_distance)
            exp_time = min(exp_time, t)
        return exp_time


    @staticmethod
    def cal_exp_time(rel_x, rel_y, rel_vx, rel_vy, combined_r):
        # rel_x: xa - xb
        # rel_y: ya - yb

        # (vx2 + vy2)*t2 + (2x*vx + 2*y*vy)*t+x2+y2-(r+mr)2 = 0

        a = rel_vx ** 2 + rel_vy ** 2
        b = 2* rel_x * rel_vx + 2* rel_y * rel_vy
        c = rel_x ** 2 + rel_y ** 2 - combined_r ** 2

        if c <= 0:
            return 0

        temp = b ** 2 - 4 * a * c

        if temp <= 0:
            t = inf
        else:
            t1 = ( -b + sqrt(temp) ) / (2 * a)
            t2 = ( -b - sqrt(temp) ) / (2 * a)

            t3 = t1 if t1 >= 0 else inf
            t4 = t2 if t2 >= 0 else inf
        
            t = min(t3, t4)

        return t
        

    @staticmethod
    def wraptopi(theta):

        if theta > pi:
            theta = theta - 2*pi
        
        if theta < -pi:
            theta = theta + 2*pi

        return theta

    @staticmethod
    def vo_out_jud_vector(vx, vy, vo):
        
        rel_vector = [vx - vo[0], vy - vo[1]]
        line_left_vector = [cos(vo[2]), sin(vo[2]) ]
        line_right_vector = [cos(vo[3]), sin(vo[3]) ]
        
        if VOController.determinant(line_left_vector, rel_vector) <= 0 and VOController.determinant(line_right_vector, rel_vector) >= 0:
            return False
        else:
            return True
    
    @staticmethod
    def determinant(v1, v2):
        return float(v1[0] * v2[1] - v1[1] * v2[0])
