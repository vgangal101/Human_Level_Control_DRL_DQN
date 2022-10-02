import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np

def graph_training_ep_len(ep_len_tracked): 
    x_vals = [i for i in range(len(ep_len_tracked))]
    fig, ax = plt.subplots()
    ax.plot(x_vals,ep_len_tracked)
    #plt.plot(ep_len_tracked)
    plt.savefig('episode_length_graph.png')

def graph_training_rewards(rewards_tracked): 
    x_vals = [i for i in range(len(rewards_tracked))]
    #plt.plot(rewards_tracked)
    #y_vals = np.array(rewards_tracked)
    fig, ax = plt.subplots()
    ax.plot(x_vals,rewards_tracked)
    #plt.plot(x_vals,rewards_tracked)
    plt.savefig('rewards_graph.png')
