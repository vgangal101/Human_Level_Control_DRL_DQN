#from math import gamma
#from argparse import Action
import jsonargparse
from preprocessing_wrappers import make_atari, wrap_deepmind, ImageToPyTorch
from model_arch import NatureCNN, BasicMLP
import torch 
import numpy as np 
import random
from train_utils import LinearSchedule, ReplayMemoryData
import torch.nn.functional as F
from graphing_utils import graph_training_ep_len, graph_training_rewards
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
    parser.add_argument('--min_squared_gradient',type=float)
    parser.add_argument('--initial_exploration',type=float)
    parser.add_argument('--final_exploration',type=float)
    parser.add_argument('--final_exploration_frame',type=float)
    parser.add_argument('--replay_start_size',type=int)
    parser.add_argument('--no_op_max',type=int)
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

        replay_memory.store(state,action,reward, next_state, done)

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

    # training relevent 
    num_episodes = cfg.num_episodes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'Cartpole' in cfg.env_name:
        env = gym.make(cfg.env_name)
        
    else: # atari setup
        env = ImageToPyTorch(wrap_deepmind(make_atari(cfg.env_name)))
    

    if cfg.network_arch == 'NatureCNN':
        policy_net = NatureCNN(env.action_space.n).to(device)
        target_net = NatureCNN(env.action_space.n).to(device)
    elif cfg.network == 'BasicMLP':
        policy_net = BasicMLP(env.observation_space.shape[0].n,env.action_space.n).to(device)
        target_net = BasicMLP(env.observation_space.shape[0].n,env.action_space.n).to(device)
        

    state_dim = env.observation_space.shape
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()


    # remove this once implem is good
    # got this setup from : https://github.com/transedward/pytorch-dqn/blob/1ffda6f3724b3bb37c3195b09b651b1682d4d4fd/ram.py#L16
    # customize choice of lr, 
    optimizer = torch.optim.RMSprop(policy_net.parameters(),lr=learning_rate,alpha=gradient_momentum_alpha,eps=min_squared_gradient_eps)

    # create replay buffer 
    replay_mem = ReplayMemoryData(replay_memory_size,state_dim)

    # fill replay_mem here 
    
    # fill Replay memory to start size 
    populate_replay_memory(env,replay_memory,replay_start_size)

    epsilon_schedule = LinearSchedule(final_exploration_frame,initial_exploration,final_exploration) # tested is ok 
    timesteps_count = 0

    reward_tracker = []
    ep_len_tracker = []
    q_val_tracker = []



    #num_episodes = 2000
    #num_episodes = 5000
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

            if  timesteps_count % update_frequency == 0:  #and len(replay_mem) > replay_start_size: 
                # do learning 
                
                # THIS BLOCK NEEDS TO BE REDONE DEPENDANT ON REPLAY MEMORY DATA STRUCTURE
                ##################################
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
                #################################

                # state_action_values. what will the policy do
                state_action_values =  policy_net(state_batch).gather(1,action_batch)


                # compute next state values , i.e. the targets 
                next_state_values = torch.zeros(minibatch_size, device=device)
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

                expected_state_action_values = (next_state_values * discount_factor) + reward_batch

                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                q_vals.append(loss.item)
                # LOOK AT STABLE BASELINES 3 , DO THE CLIPPING NEEDED 
                #for param in policy_net.parameters():
                #    param.grad.data.clamp_(-1,1)
                optimizer.step()

            if timesteps_count % target_network_update_frequency == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            if done:
                reward_tracker.append(sum(episode_reward))
                ep_len_tracker.append(len(episode_reward))
                print(f'Total steps: {timesteps_count} \t Episode: {episode} \t Total reward: {reward_tracker[-1]}') 
                break
            else: 
                episode_reward.append(reward) 
        # CODE TO EVALUATE PERFROMANCE OVER 10 EPISODES , AFTER A CERTAIN AMOUNT OF EPISODES ELAPSE
    # save the model 
    policy_net_file_save = f'{cfg.env_name}_DQN_policy_net.pth'
    target_net_file_save = f'{cfg.env_name}_DQN_target_net.pth'

    torch.save(policy_net,policy_net_file_save)
    torch.save(target_net,target_net_file_save)

    # graph the training curves 
    graph_training_rewards(reward_tracker)
    graph_training_ep_len(ep_len_tracker)



if __name__ == '__main__':
    main()
