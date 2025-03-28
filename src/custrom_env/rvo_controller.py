class VOController:
    def __init__(self, obs_mode, reward_parameters, neighbor_region, neighbor_num):
        self.obs_mode = obs_mode
        self.reward_parameters = reward_parameters
        self.neighbor_region = neighbor_region
        self.neighbor_num = neighbor_num