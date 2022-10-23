import gym
import torch

class PytorchObservation(gym.ObservationWrapper):

    def __init__(self,env):
        super().__init__(env)
        self.observation_space = env.observation_space
    
    def observation(self, obs):
        return torch.from_numpy(obs).type(torch.float32)