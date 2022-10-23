import argparse
from collections import deque
import random
from gym.utils.play import play
import gym
import matplotlib.pyplot as plt
import numpy as np

def capture_statistics(data):
    mean = np.max(np.array(data))
    max = np.max(np.array(data))
    min = np.min(np.array(data))
    print('stats: ')
    print('Mean steps=', mean)
    print('max steps=', max)
    print('Min steps=', min)



def graph_episode_length(data):
    plt.plot(data)
    plt.savefig('episode_length_viz.png')


def regular_main():
    env = gym.make('Pong-v0',render_mode='human')
    env.action_space.seed(42)

    obs, info = env.reset(return_info=True)
    
    episode_length_track = []

    for times in range(10):
        for t in range(2000):

            observation, reward, done, info = env.step(env.action_space.sample())

            #print('info=',info)

            if done:
                episode_length_track.append(t)
                observation, info = env.reset(return_info=True)
            
            if t == 999:
                episode_length_track.append(t)

    env.close()

    capture_statistics(episode_length_track)
    graph_episode_length(episode_length_track)
    


def play_main():
    play(gym.make('Pong-v0'))




if __name__ == '__main__':
    regular_main()
    #play_main()
