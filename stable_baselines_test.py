import stable_baselines3

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env1 = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
env2 = VecFrameStack(env1,n_stack=4)



