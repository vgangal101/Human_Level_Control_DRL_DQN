from math import gamma
from preprocessing_wrappers import make_atari, wrap_deepmind
from model_arch import DQN_CNN
import argparse
import torch 
import numpy as np 
import random
from train_utils import LinearSchedule, ReplayMemory, Transition
import torch.nn.functional as F
from graphing_utils import graph_training_ep_len, graph_training_rewards

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
    return args



def main():
    args = get_args()
    train(args.env)




def convert_obs(obs): 
    state = np.array(obs)
    state = state.transpose((2,0,1))
    state = torch.from_numpy()
    return state


def train(env_name):
    env = wrap_deepmind(make_atari(env_name))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN_CNN(env.action_space.n)
    target_net = DQN_CNN(env.action_space.n)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # remove this once implem is good
    # got this setup from : https://github.com/transedward/pytorch-dqn/blob/1ffda6f3724b3bb37c3195b09b651b1682d4d4fd/ram.py#L16
    optimizer = torch.optim.RMSprop(policy_net.parameters(),alpha=0.95,eps=0.01)

    # create replay buffer 
    replay_mem = ReplayMemory(replay_memory_size)

    epsilon_schedule = LinearSchedule(final_exploration_frame,initial_exploration,final_exploration)
    timesteps_count = 0

    reward_tracker = []
    ep_len_tracker = []

    #num_episodes = 10000
    num_episodes = 20
    for episode in range(num_episodes):
        state = env.reset()
        state = convert_obs(state)
        rewards = []
        for t in range(10000):
            sample = random.random()
            current_epsilon = epsilon_schedule.get_epsilon(timesteps_count)
            if sample > current_epsilon:
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].view(1,1)
            else: 
                action = torch.tensor([random.randrange(env.action_space.n)], device=device, type=torch.long)

            next_state, reward, done, info = env.step(action.item())
            timesteps_count += 1 

            if not done: 
                next_state = convert_obs(next_state)
            else: 
                next_state = None  

            replay_mem.push(Transition(state,action.item(),reward,next_state))

            if  len(replay_mem) > replay_start_size: 
                # do learning 
                transitions = replay_mem.sample(minibatch_size)
                batch = Transition(*zip(*transitions))

                actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action))) 
                rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward))) 

                non_final_mask = torch.tensor(
                    tuple(map(lambda s: s is not None, batch.next_state)),
                    device=device, dtype=torch.uint8)
    
                non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to('cuda')
                
                state_batch = torch.cat(batch.state).to('cuda')
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
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1,1)
                optimizer.step()

                if timesteps_count % C == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            
            if done:
                reward_tracker.append(sum(rewards))
                ep_len_tracker.append(len(rewards))
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(timesteps_count, episode, reward_tracker[-1])) 
                break
            else: 
                rewards.append(reward) 
    # save the model 
    policy_net_file_save = f'{}_DQN_policy_net.pth'.format(env_name)
    target_net_file_save = f'{}_DQN_target_net.pth'.format(env_name)

    torch.save(policy_net,policy_net_file_save)
    torch.save(target_net,target_net_file_save)

    # graph the training curves 
    graph_training_rewards(reward_tracker)
    graph_training_ep_len(ep_len_tracker)



if __name__ == '__main__':
    main()
