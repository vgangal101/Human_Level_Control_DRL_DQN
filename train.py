#from math import gamma
from preprocessing_wrappers import make_atari, wrap_deepmind, ImageToPyTorch
from model_arch import NatureCNN, BasicMLP
import argparse
import torch 
import numpy as np 
import random
from train_utils import LinearSchedule, ReplayMemoryData
import torch.nn.functional as F
from graphing_utils import graph_training_ep_len, graph_training_rewards
import gym
import paper_hyperparam


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name',type=str,default='PongNoFrameskip-v4')
    parser.add_argument('--network_arch',type=str,default=None) # can be either BasicMLP or NatureCNN
    parser.add_argument('--replay_memory_size',type=int,default=None)
    parser.add_argument('--final_exploration_frame',type=int,default=None)
    parser.add_argument('--eps_start',type=int,default=None)
    parser.add_argument('--eps_end',type=float, default=None)
    parser.add_argument('--replay_mem_start',type=int,default=None)
    
    #parser.add_argument('--param_config_paper',default=True)
    args = parser.parse_args()
    return args



def main():
    args = get_args()
    train(args,args.env_name)


# def convert_obs(obs): 
#     state = np.array(obs)
#     state = state.transpose((2,0,1))
#     state = torch.from_numpy(state)
#     return state.unsqueeze(0)


def train(args,env_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'Cartpole' in args.env_name:
        env = gym.make(args.env_name)
        policy_net = BasicMLP(env.observation_space.shape[0].n,env.action_space.n).to(device)
        target_net = BasicMLP(env.observation_space.shape[0].n,env.action_space.n).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        
    else: # atari setup
        env = wrap_deepmind(make_atari(args.env_name))
        state_dim = (4,84,84)
        policy_net = NatureCNN(env.action_space.n).to(device)
        target_net = NatureCNN(env.action_space.n).to(device)
        
    state_dim = env.observation_space.shape
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()


    # remove this once implem is good
    # got this setup from : https://github.com/transedward/pytorch-dqn/blob/1ffda6f3724b3bb37c3195b09b651b1682d4d4fd/ram.py#L16
    # customize choice of lr, 
    optimizer = torch.optim.RMSprop(policy_net.parameters(),lr=paper_hyperparam.lr,alpha=paper_hyperparam.squared_gradient_momentum,eps=paper_hyperparam.min_squared_gradient)

    # create replay buffer 
    replay_mem = ReplayMemoryData(args.replay_memory_size,state_dim)

    epsilon_schedule = LinearSchedule(args.final_exploration_frame,args.eps_start,args.eps_ends) # tested is ok 
    timesteps_count = 0

    reward_tracker = []
    ep_len_tracker = []
    q_val_tracker = []

    # fill Replay memory to start size 


    #num_episodes = 2000
    num_episodes = 5000
    #num_episodes = 20
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = []
        for t in range(10000):
            sample = random.random()
            current_epsilon = epsilon_schedule.get_epsilon(timesteps_count)
            if sample > current_epsilon:
                with torch.no_grad():
                    action = policy_net(state.to(device)).max(1)[1].view(1,1)
            else: 
                action = torch.tensor([random.randrange(env.action_space.n)], device=device, dtype=torch.long)

            next_state, reward, done, info = env.step(action.item())
            timesteps_count += 1 

            if not done: 
                state = next_state
            else: 
                state = None  

            replay_mem.push(state,action.item(),next_state,reward)

            if  timesteps_count % update_frequency == 0 and len(replay_mem) > replay_start_size: 
                # do learning 
                transitions = replay_mem.sample(minibatch_size)
                batch = Transition(*zip(*transitions))

                actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action))) 
                rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward))) 

                non_final_mask = torch.tensor(
                    tuple(map(lambda s: s is not None, batch.next_state)),
                    device=device, dtype=torch.uint8)
    
                non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to('cuda')
                
                state_batch = torch.cat(batch.state).to(device)
                action_batch = torch.cat(actions)
                reward_batch = torch.cat(rewards)
                

                # state_action_values. what will the policy do
                state_action_values =  policy_net(state_batch).gather(1,action_batch)


                # compute next state values , i.e. the targets 
                next_state_values = torch.zeros(minibatch_size, device=device)
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

                expected_state_action_values = (next_state_values * 0.99) + reward_batch

                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                #for param in policy_net.parameters():
                #    param.grad.data.clamp_(-1,1)
                optimizer.step()

            if timesteps_count % C == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            if done:
                reward_tracker.append(sum(episode_reward))
                ep_len_tracker.append(len(episode_reward))
                print(f'Total steps: {timesteps_count} \t Episode: {episode} \t Total reward: {reward_tracker[-1]}') 
                break
            else: 
                episode_reward.append(reward) 
    
    # save the model 
    policy_net_file_save = f'{env_name}_DQN_policy_net.pth'
    target_net_file_save = f'{env_name}_DQN_target_net.pth'

    torch.save(policy_net,policy_net_file_save)
    torch.save(target_net,target_net_file_save)

    # graph the training curves 
    graph_training_rewards(reward_tracker)
    graph_training_ep_len(ep_len_tracker)



if __name__ == '__main__':
    main()
