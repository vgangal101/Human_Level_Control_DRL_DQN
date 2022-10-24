#from math import gamma
#from argparse import Action
import jsonargparse
from evaluate import evaluate_perf
from atari_preprocessing_wrappers import make_atari, wrap_deepmind, ImageToPyTorch
from general_wrappers import PytorchObservation
from model_arch import NatureCNN, BasicMLP
import torch 
import numpy as np 
import random
from train_utils import LinearSchedule, ReplayMemory
import torch.nn.functional as F
from graphing_utils import graph_ep_len, graph_q_vals, graph_rewards
from torch.utils.tensorboard import SummaryWriter

import gym


#import paper_hyperparam


def get_cfg():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument('--cfg',action=jsonargparse.ActionConfigFile)
    parser.add_argument('--env_name',type=str,default='PongNoFrameskip-v4')
    parser.add_argument('--network_arch',type=str) # can be either BasicMLP or NatureCNN
    parser.add_argument('--minibatch_size',type=int)
    parser.add_argument('--replay_memory_size',type=int)
    parser.add_argument('--agent_history_length',type=int)
    parser.add_argument('--target_network_update_frequency',type=int)
    parser.add_argument('--discount_factor',type=float)
    parser.add_argument('--action_repeat',type=int)
    parser.add_argument('--update_frequency',type=int)
    parser.add_argument('--learning_rate',type=float)
    parser.add_argument('--gradient_momentum_alpha',type=float)
    parser.add_argument('--min_squared_gradient_eps',type=float)
    parser.add_argument('--initial_exploration',type=float)
    parser.add_argument('--final_exploration',type=float)
    parser.add_argument('--final_exploration_frame',type=float)
    parser.add_argument('--replay_start_size',type=int)
    parser.add_argument('--no_op_max',type=int)
    parser.add_argument('--tb_log_dir',type=str)
    parser.add_argument('--num_episodes',type=int)
    cfg = parser.parse_args()
    return cfg



def main():
    cfg = get_cfg()
    train(cfg)


# def convert_obs(obs): 
#     state = np.array(obs)
#     state = state.transpose((2,0,1))
#     state = torch.from_numpy(state)
#     return state.unsqueeze(0)

def populate_replay_memory(env,replay_memory,replay_start_size):
    state = env.reset()
    for i in range(replay_start_size):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)

        replay_memory.store(state.numpy(),action,reward, next_state.numpy(), done)

        if done: 
            state = env.reset()
        else:
            state = next_state 


def train(cfg):

    # Hyperparameters
    minibatch_size = cfg.minibatch_size 
    replay_memory_size = cfg.replay_memory_size 
    target_network_update_frequency = cfg.target_network_update_frequency
    agent_history_length = cfg.agent_history_length
    discount_factor = cfg.discount_factor
    action_repeat = cfg.action_repeat
    update_frequency = cfg.update_frequency
    learning_rate = cfg.learning_rate
    gradient_momentum_alpha = cfg.gradient_momentum_alpha 
    min_squared_gradient_eps = cfg.min_squared_gradient_eps
    initial_exploration = cfg.initial_exploration
    final_exploration = cfg.final_exploration 
    final_exploration_frame = cfg.final_exploration_frame 
    replay_start_size = cfg.replay_start_size
    no_op_max = cfg.no_op_max

    # utilities configuration 

    tb_log_dir = cfg.tb_log_dir
    tb_writer = SummaryWriter(tb_log_dir,flush_secs=10)

    torch.set_default_dtype(torch.float32)

    # training relevent 
    num_episodes = cfg.num_episodes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'CartPole' in cfg.env_name:
        env = gym.make(cfg.env_name)
        env = PytorchObservation(env)
    else: # atari setup
        env = ImageToPyTorch(wrap_deepmind(make_atari(cfg.env_name)))
    

    if cfg.network_arch == 'NatureCNN':
        policy_net = NatureCNN(env.action_space.n).to(device).float()
        target_net = NatureCNN(env.action_space.n).to(device).float()
    elif cfg.network_arch == 'BasicMLP':
        policy_net = BasicMLP(env.observation_space.shape[0],env.action_space.n).to(device).float()
        target_net = BasicMLP(env.observation_space.shape[0],env.action_space.n).to(device).float()
        

    state_dim = env.observation_space.shape
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()


    # remove this once implem is good
    # got this setup from : https://github.com/transedward/pytorch-dqn/blob/1ffda6f3724b3bb37c3195b09b651b1682d4d4fd/ram.py#L16
    # customize choice of lr, 
    optimizer = torch.optim.RMSprop(policy_net.parameters(),lr=learning_rate,alpha=gradient_momentum_alpha,eps=min_squared_gradient_eps)

    # create replay buffer 
    replay_mem = ReplayMemory(replay_memory_size,state_dim)

    # fill replay_mem here 
    
    # fill Replay memory to start size 
    populate_replay_memory(env,replay_mem,replay_start_size)

    epsilon_schedule = LinearSchedule(final_exploration_frame,initial_exploration,final_exploration) # tested is ok 
    timesteps_count = 0

    train_reward_tracker = []
    train_ep_len_tracker = []
    train_q_val_tracker = []
    train_eps_vals_tracker = []

    eval_reward_tracker = []
    eval_ep_len_tracker = []
    

    #num_episodes = 2000
    #num_episodes = 5000
    #num_episodes = 20
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = []
        for t in range(10000): # should the max time, per episode be set to 108,000 ? 
            sample = random.random()
            current_epsilon = epsilon_schedule.get_epsilon(timesteps_count)
            tb_writer.add_scalar('Epsilon/train', current_epsilon, timesteps_count)
            train_eps_vals_tracker.append(current_epsilon)
            if sample > current_epsilon:
                with torch.no_grad():
                    action = policy_net(state.unsqueeze(0).to(device)).max(1)[1].view(1,1)
            else: 
                action = torch.tensor([random.randrange(env.action_space.n)], device=device, dtype=torch.long)

            next_state, reward, done, info = env.step(action.item())
            timesteps_count += 1 

            replay_mem.store(state.numpy(),action.item(),reward,next_state.numpy(),done)

            if not done: 
                state = next_state
            else: 
                state = None  

            

            if  timesteps_count % update_frequency == 0:  #and len(replay_mem) > replay_start_size: 
                # do learning 
                
                # THIS BLOCK NEEDS TO BE REDONE DEPENDANT ON REPLAY MEMORY DATA STRUCTURE
                ##################################
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_mem.sample(minibatch_size)
                
                
                #################################

                # state_action_values. what will the policy do
                state_action_values =  policy_net(state_batch.to(device)).cpu().gather(1,action_batch.unsqueeze(1))


                # compute next state values , i.e. the targets 
                # next_state_values = torch.zeros(minibatch_size, device=device)
                # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

                # expected_state_action_values = (next_state_values * discount_factor) + reward_batch
                expected_state_action_values = reward_batch + (1-done_batch) * discount_factor * target_net(next_state_batch.to(device)).max(1)[0].detach().cpu()


                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                train_q_val_tracker.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                # LOOK AT STABLE BASELINES 3 , DO THE CLIPPING NEEDED 
                #for param in policy_net.parameters():
                #    param.grad.data.clamp_(-1,1)
                max_grad_norm = 10
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(),max_grad_norm)
                optimizer.step()

            if timesteps_count % target_network_update_frequency == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            if done:
                train_reward_tracker.append(sum(episode_reward))
                train_ep_len_tracker.append(len(episode_reward))
                tb_writer.add_scalar('Rewards/train', train_reward_tracker[-1],episode)
                tb_writer.add_scalar('Episode_Length/train',train_ep_len_tracker[-1],episode)            
                print(f'Training - Total steps: {timesteps_count} \t Episode: {episode} \t Total reward: {train_reward_tracker[-1]}') 
                break
            else: 
                episode_reward.append(reward) 
        
        # evaluate performance
        if episode % 10 == 0: 
            evaluate_perf(env,policy_net,tb_writer,eval_reward_tracker,eval_ep_len_tracker,episode)

    # save the model 
    policy_net_file_save = f'{cfg.env_name}_DQN_policy_net.pth'
    target_net_file_save = f'{cfg.env_name}_DQN_target_net.pth'

    torch.save(policy_net,policy_net_file_save)
    torch.save(target_net,target_net_file_save)

    # graph the training curves 
    graph_rewards(train_reward_tracker,phase='train')
    graph_ep_len(train_ep_len_tracker,phase='train')
    graph_q_vals(train_q_val_tracker,phase='train')
    
    graph_rewards(eval_reward_tracker,phase='eval')
    graph_ep_len(eval_ep_len_tracker,phase='eval')



if __name__ == '__main__':
    main()