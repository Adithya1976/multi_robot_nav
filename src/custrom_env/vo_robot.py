from ast import Tuple
import math
from typing import Dict, List

import numpy as np
from sympy import Li
from custrom_env.perception_info import CircularObstacle, LidarCluster, PerceptionInfo
from custrom_env.simulated_lidar import SimulatedLidar2D
from irsim.world.object_base import ObjectBase
from irsim.world.robots.robot_diff import RobotDiff


class VORobot:
    def __init__(self, robot: RobotDiff, external_objects: List[ObjectBase], neighbor_region: int, mode: str, lidar_config_dict: Dict):
        self.robot = robot
        self.external_objects = external_objects
        self.mode = mode
        self.neighbor_region = neighbor_region
        self.lidar_range_max = lidar_config_dict["range_limit"]
        self.lidar_jump_threshold = lidar_config_dict["jump_threshold"]
        self.lidar_number = lidar_config_dict["number"]
        self.lidar_angle_min = lidar_config_dict["angle_min"]
        self.lidar_angle_max = lidar_config_dict["angle_max"]
        self.lidar_angles = np.linspace(self.lidar_angle_min, self.lidar_angle_max, self.lidar_number)
        self.lidar = SimulatedLidar2D(
            range_max=self.lidar_range_max,
            angle_range=self.lidar_angle_max - self.lidar_angle_min,
            number=self.lidar_number,
            external_objects=external_objects
        )
    
    def get_perception_info(self, init_vel: bool = False) -> PerceptionInfo:
        state = self.robot.state[0, 0], self.robot.state[1, 0], self.robot.state[2, 0]
        velocity = self.robot.velocity_xy[0, 0], self.robot.velocity_xy[1, 0]
        radius = self.robot.radius
        desired_velocity = self.calculate_desired_velocity()
        if self.mode == "ground_truth":
            obstacle_list = [
                CircularObstacle(
                    position = (obj.state[0, 0], obj.state[1, 0]),
                    velocity = (obj.velocity_xy[0, 0], obj.velocity_xy[1, 0]),
                    radius = obj.radius
                )
                for obj in self.external_objects 
                if self.cal_dist(
                    self.state[0, 0], self.state[1, 0], obj.state[0, 0], obj.state[1, 0]
                ) < self.neighbor_region
            ]

            return PerceptionInfo(
                type="ground_truth",
                state=state,
                velocity=velocity,
                radius=radius,
                desired_velocity=desired_velocity,
                arrive=self.arrive,
                collision=self.collision,
                obstacle_list=obstacle_list 
            )
        elif self.mode == "lidar":
            # get lidar scan
            ranges = self.lidar.get_scan(self.robot.state.copy())

            # generate clusters
            obstacles = self.generate_clusters(ranges, self.robot.state.copy())
            
            return PerceptionInfo(
                type="lidar",
                state=state,
                velocity=velocity,
                radius=radius,
                desired_velocity=desired_velocity,
                arrive=self.arrive,
                collision=self.collision,
                obstacle_list=obstacles
            )
        else:
            raise ValueError("Invalid mode")
    
    def calculate_desired_velocity(self) -> Tuple:
        if self.arrive:
            return 0, 0
        x, y = self.state[0, 0], self.state[1, 0]
        goal_x, goal_y = self.robot.goal[0,  0], self.robot.goal[1, 0]

        desired_angle = math.atan2(goal_y - y, goal_x - x)
        desired_vx = self.robot.vel_max[0, 0] * math.cos(desired_angle)
        desired_vy = self.robot.vel_max[0, 0] * math.sin(desired_angle)
        return desired_vx, desired_vy

    def generate_clusters(self, lidar_scan, robot_state):
        labelled_scan = [0] * self.lidar_number
        current_cluster_id = -1

        # Initial clustering of valid readings
        for i, distance in enumerate(lidar_scan):
            # if reading equals (or exceeds) the range limit, mark as -1 (no obstacle)
            if distance > self.lidar_range_max - 0.01:
                labelled_scan[i] = -1
                continue

            # For the first valid reading or when the previous was invalid, start a new cluster
            if i == 0 or lidar_scan[i-1] > self.lidar_range_max - 0.01:
                current_cluster_id += 1
                labelled_scan[i] = current_cluster_id
            else:
                # If the previous reading was valid, check the difference.
                prev_distance = lidar_scan[i-1]
                # If the difference is small relative to the previous distance,
                # consider it part of the same obstacle.
                if abs(distance - prev_distance) < self.lidar_jump_threshold:
                    labelled_scan[i] = current_cluster_id
                else:
                    # Otherwise, start a new obstacle cluster.
                    current_cluster_id += 1
                    labelled_scan[i] = current_cluster_id

        has_split = False
        # Handle cyclic continuity by merging first and last clusters if appropriate.
        if labelled_scan[0] != -1 and labelled_scan[-1] != -1:
            if abs(lidar_scan[0] - lidar_scan[-1]) < self.lidar_jump_threshold:
                i = self.lidar_number - 1
                last_cluster_id = labelled_scan[-1]
                has_split = True
                while i >= 0 and labelled_scan[i] == last_cluster_id:
                    labelled_scan[i] = labelled_scan[0]
                    i -= 1
                current_cluster_id -= 1


        # Create a mapping from cluster id to the list of indices belonging to that cluster.
        clusters = {}
        temp_list = []
        for i, label in enumerate(labelled_scan):
            if label == 0 and i > self.lidar_number/2 and has_split:
                temp_list.append(i)
            elif label != -1:
                clusters.setdefault(label, []).append(i)

        if len(clusters) == 0:
            return {}

        clusters[0] = temp_list + clusters[0]

        lidar_clusters: List[LidarCluster] = []

        for _, indices in clusters.items():

            lidar_cluster = self.create_lidar_cluster(
                indices=indices,
                lidar_scan=lidar_scan,
                robot_x=robot_state[0, 0],
                robot_y=robot_state[1, 0],
                robot_theta=robot_state[2, 0]
            )

            if self.cal_dist(lidar_cluster.centroid[0], lidar_cluster.centroid[1], robot_state[0, 0], robot_state[1, 0]) < self.neighbor_region:
                lidar_clusters.append(lidar_cluster)

        return lidar_clusters
    
    def create_lidar_cluster(self, indices, lidar_scan, robot_x, robot_y, robot_theta) -> LidarCluster:
        sum_x = 0
        sum_y = 0
        points = []
        dist_from_robot = np.inf
        for i in indices:
            x = lidar_scan[i] * math.cos(robot_theta + self.lidar_angles[i])
            y = lidar_scan[i] * math.sin(robot_theta + self.lidar_angles[i])
            sum_x += x
            sum_y += y

            point = np.array([robot_x + x, robot_y + y])
            points.append(point)

            dist_from_robot = min(dist_from_robot, lidar_scan[i])
        
        points = np.array(points)

        centroid = np.array([robot_x + sum_x/len(indices), robot_y + sum_y/len(indices)])
        lidar_cluster = LidarCluster(
            centroid=centroid,
            points=points,
            min_dist=dist_from_robot
        )
        return lidar_cluster

    @staticmethod
    def cal_dist(x1, y1, x2, y2) -> float:
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        

    @property
    def arrive(self) -> bool:
        return self.robot.arrive

    @property
    def collision(self) -> bool:
        return self.robot.collision
    
    @property
    def state(self) -> List:
        return self.robot.state