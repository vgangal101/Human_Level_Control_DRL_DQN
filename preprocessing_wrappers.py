import gym
from collections import deque
import cv2
import numpy as np
from gym import spaces

"""
Contains preprocessing OpenAI gym wrappers
"""

"""
Notes:
Apply FrameskippingAndMax first then FrameStacking, not the otherway around

Get the EpisodicLifeEnv wrapper from stable baselines if need be
"""

"""
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""

"""
def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)
"""

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]



# Use this FireReset if agent does not learn to push FIRE to start
class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
          self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
          self.env.reset()
        return obs

class ResizeTo84by84(gym.ObservationWrapper):
    def __init__(self,env):
        super().__init__(env)
        self.dsize = (84,84)
        self.observation_space = gym.spaces.Box(low=0,high=255,shape=(1,84,84))

    def observation(self, obs):
        new_obs = cv2.resize(obs,self.dsize)
        return new_obs


class FrameskippingAndMax(gym.Wrapper):
    def __init__(self,env,skip=4):
        super().__init__(env)
        self._obs_buffer = deque([],maxlen=2)
        self.skip = skip

    def step(self,action):
        total_reward = 0
        done = None
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_obs = np.maximum(self._obs_buffer[0],self._obs_buffer[1])
        return max_obs, total_reward, done, info



class FrameStacking(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return np.array(self.frames)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return np.array(self.frames), reward, done, info

    # def _get_ob(self):
    #     assert len(self.frames) == self.k
    #     return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0 
    



# https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
class NoOpReset(gym.Wrapper):
    def __init__(self,env,no_op_max=30):
        super().__init__(env)
        self.no_op_max = no_op_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self,**kwargs):
        self.env.reset()
        for _ in range(self.no_op_max):
            obs , reward , done , info = self.env.step(0)

            # print('obs=',obs)
            # print('reward=',reward)
            # print('done=',done)
            # print('info=',info)
            if done:
                obs = self.env.reset()
        return obs


class ClipReward(gym.RewardWrapper):
    def __init__(self,env):
        super().__init__(env)

    def reward(self,reward):
        return np.sign(reward)
