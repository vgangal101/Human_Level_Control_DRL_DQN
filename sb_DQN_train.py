import torch 
import stable_baselines3

from stable_baselines3 import DQN

from preprocessing_wrappers import make_atari, wrap_deepmind

from train import make_env

env_mapper = {'Pong':'PongNoFrameskip-v4',
               'Breakout': 'BreakoutNoFrameskip-v4',
               'Atlantis':'AtlantisNoFrameskip-v4'}

env = wrap_deepmind(make_atari(env_mapper['Pong']))

model = DQN("CnnPolicy",env,verbose=1,tensorboard_log='./DQN_Pong/')
model.learn(total_timesteps=10,log_interval=1)
model.save('DQN_Pong')

