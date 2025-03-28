from custrom_env.rvo_controller import VOController
from custrom_env.vo_robot import VORobot
from irsim.env import EnvBase


class VOEnv(EnvBase):
    def __init__(self, world_name, obs_mode, reward_parameters = (0.3, 1.0, 0.3, 1.2, 0.2, 3.6, 0, 0), neighbor_region=4, neighbor_num=5, **kwargs):
        super(VOEnv, self).__init__(world_name, **kwargs)
        self.obs_mode = obs_mode
        # initialise VO robots
        self.vo_robots = self.init_vo_robots()
        # initialise RVO Controller
        self.rvo_controller = VOController(obs_mode, reward_parameters, neighbor_region, neighbor_num)
    
    def init_vo_robots(self):
        for i in range(self.robot_number):
            robot = self.robot_list[i]
            external_objects = self.object_list[:i] + self.object_list[i+1:]
            vo_robot = VORobot(robot, external_objects, self.obs_mode)
            self.vo_robots.append(vo_robot)
    
    def step(self, action_list):
        pass

    def reset(self, id=None, random_ori=False):
        pass