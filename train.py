from preprocessing_wrappers import make_atari, wrap_deepmind
from model_arch import DQN_CNN
import argparse
import torch 
import numpy as np 


# Hyperparameters from paper 
minibatch_size = 32

replay_memory_size = 1000000

agent_history_length = 4

C = 10000 # target_network_update_frequency

discount_factor_gamma = 0.99 # discount factor used in the Q-learning update
action_repeat = 4
update_frequency = 4
lr = 0.00025 # learning rate used by RMSProp
grad_momentum = 0.95 # gradient momentum used by RMSProp
squared_gradient_momentum = 0.95 # squared
min_squared_gradient = 0.01 # constant to squared gradient in denom of RMSProp Update


initial_exploration = 1 # inital vaue of epsilon in epsilon greedy exploration
final_exploration = 0.1 # final value of epsilon in epsilon-greedy exploration
final_exploration_frame = 1000000

replay_start_size = 50000 # uniform random policy is run for this num of frames before learning starts & resulting experience is used to populate replay ReplayMemory

no_op_max = 30 # max number of 'do nothing' actions to be performed  by the agent at the start of an episode

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name',default='PongNoFrameskip-v4')
    args = parser.parse_args()
    



def main():
    args = get_args()
    train(args.env)

def train(env_name):
    env = wrap_deepmind(make_atari(env_name))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    behavior_net = DQN_CNN(env.action_space.n)
    target_net = DQN_CNN(env.action_space.n)
    target_net.load_state_dict(behavior_net.state_dict())
    target_net.eval()

    optimizer = torch.opti



if __name__ == '__main__':
    main()
