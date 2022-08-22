import argparse
from collections import deque
import random
from preprocessing_wrappers import ClipReward, FrameskippingAndMax, FrameStacking, NoOpReset, ResizeTo84by84
import gym
#from gym.wrappers import
import torch
import numpy as np
from model_arch import DQN_Agent
from torch.optim import RMSprop
import matplotlib.pyplot as plt

# would you rather create a list which if the size exceeded max_size, we wrap around,
# obviously maintain a counter

class ReplayMemory:
    def __init__(self,max_size):
        self.max_size = max_size
        self.memory = deque([],maxlen=max_size)

    def store(self,transition_tuple):
        self.memory.append(transition_tuple)

    def sample(self,minibatch_size):
        # grab minibatch_size random number of elements from memory/deque object
        return random.sample(self.memory, minibatch_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()

def convert_to_tensor(sample): 
    prev_state_tensor = []
    action_t = []
    reward_t = []
    next_state_tensor = []
    done_t = []

    for s in sample: 
        prev_state = s[0]
        action = s[1]
        reward = s[2]
        next_state = s[3]
        done = s[4]
        
        prev_state_tensor.append(prev_state)
        action_t.append(action)
        reward_t.append(reward)
        next_state_tensor.append(next_state)
        done_t.append(done)

    prev_state_tensor = torch.tensor(np.array(prev_state_tensor),dtype=torch.float32)
    action_t = torch.tensor(action_t,dtype=torch.int64)
    reward_t = torch.tensor(reward_t, dtype=torch.float32)
    next_state_tensor = torch.tensor(np.array(next_state_tensor),dtype=torch.float32)
    done_t = torch.tensor(done_t,dtype=torch.int64)



    return prev_state_tensor, action_t, reward_t, next_state_tensor, done_t 


def evaluation_process_env(env):
    env = ResizeTo84by84(env)
    env = ClipReward(env)
    env = FrameskippingAndMax(env,skip=4)
    env = FrameStacking(env,4)
    env = NoOpReset(env)

    return env


def evaluate(env, policy_q_network):
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else: 
        device = torch.device('cpu')

    def select_action(state):
        probability = np.random.uniform()

        if probability < 0.05:
            return env.action_space.sample()
        else:
            state = torch.tensor(state)
            state.to(device)
            with torch.no_grad():
                action = torch.argmax(policy_q_network(state))
            return action

    timesteps_total = 0
    episode_length_tracker = []
    reward_tracker = []

    # SHOULD RUN FOR 30 episodes not 1 !!!, 1 is for debugging
    for ep_num in range(1):
        eps_end = 0
        done = False 
        obs = env.reset()
        episode_total_reward = 0 
        obs = torch.tensor(obs,dtype=torch.float32).unsqueeze(0).to(device)
        #obs = obs.unsqueeze(0)
        while not done:

            action = select_action(obs)
            obs, reward, done, info = env.step(action)
            obs = torch.tensor(obs,dtype=torch.float32).unsqueeze(0).to(device)
            episode_total_reward += reward

            timesteps_total += 1 

            if done: 
                episode_length_tracker.append(eps_end)
                reward_tracker.append(episode_total_reward)
                eps_end = 0 

            eps_end += 1 

    mean_episode_length = np.mean(episode_length_tracker)
    mean_reward = np.mean(reward_tracker)

    with open('evaluation_results.txt','w') as f:
        f.write(f'evaluation results\n')
        f.write(f'mean_episode_length={mean_episode_length} \n')
        f.write(f'mean_reward={mean_reward} \n')

    return 
            


def train_process_env(env):
    env = ResizeTo84by84(env)
    env = ClipReward(env)
    env = FrameskippingAndMax(env,skip=4)
    env = FrameStacking(env,4)
    env = NoOpReset(env)

    return env

def graph_episode_length(data,eval=False):
    plt.plot(data)
    plt.xlabel('Episodes')
    plt.ylabel('Episode Length(in timesteps)')

    header = None
    if eval: 
        header = 'Evaluation'
    else: 
        header = 'Training'

    
    plt.savefig(f'{header}--ep_num-vs-episode_length.png')

def graph_reward(data,eval=False):
    plt.plot(data)
    plt.xlabel('timesteps')
    plt.ylabel('Reward')
    
    header = None
    if eval: 
        header = 'Evaluation'
    else: 
        header = 'Training'

    plt.savefig(f'{header}--timesteps_vs_reward.png') 


# input env is already processed
def train(env):

    # do environment preprocessing

    # Hyperparameters
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

    # implement the first bit first , then write remainder

    if torch.cuda.is_available():
        print('cuda available=',torch.cuda.is_available())
        device = torch.device('cuda:0')
    else: 
        device = torch.device('cpu')

    print(device)

    def annealed_epsilon(step):
        return initial_exploration + (final_exploration - initial_exploration) * min(1, step / final_exploration_frame)



    def get_action(state, epsilon):
        probability = np.random.uniform()

        if probability < epsilon:
            return env.action_space.sample()
        else:

            state = torch.tensor(state)
            state.to(device)
            with torch.no_grad():
                action = torch.argmax(policy_q_network(state))
            return action


    def compute_targets(prev_state, action, reward, next_state, done, e):

        number_of_samples = prev_state.shape[0]
        target_values = []


        for sample_num in range(number_of_samples):
            current_prev_state = prev_state[sample_num]
            current_action = action[sample_num]
            current_reward = reward[sample_num]
            current_next_state = next_state[sample_num]
            current_done = done[sample_num]

            if current_done:
                target_values.append(current_reward)
            else:
                current_next_state = torch.unsqueeze(current_next_state,0)
                current_next_state = current_next_state.to(device)
                target_val = (current_reward + discount_factor_gamma * torch.max(target_q_network(current_next_state))).item()
                target_values.append(target_val)

        target_values = torch.tensor(np.array(target_values))

        return target_values

    # initialize replay memory to capacity N
    replay_mem = ReplayMemory(replay_memory_size)

    print('Initialized replay mem ')

    print("Storing data in Replay Memory")


    
    obs  = env.reset()
    prev_state = obs

    #for i in range(replay_start_size):
    for i in range(64):

        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        next_state = obs

        replay_mem.store((prev_state, action, reward, next_state, done))

        if done:
            obs = env.reset()
            prev_state = obs
        else:
            prev_state = next_state

        if i % 10000: 
            print('completed replay storage iter=',i)

    #prev_state, action, reward, next_state, done = convert_to_tensor(replay_mem.sample(32))
    
    # information for graphing related material 

    # track episode length , i.e episode number to steps in episode 
    episode_length_tracker = []
    # track rewards over time steps , timesteps to reward value 
    reward_tracker = []
    
    # some ways to handle : https://discuss.pytorch.org/t/copying-weights-from-one-net-to-another/1492
    # alternate approach - net1.load_state_dict(net2.state_dict()), helpful later in the loop

    # initialize action-value function Q with random weights (theta)
    policy_q_network = DQN_Agent(env.action_space.n)
    policy_q_network.to(device)

    # intialize target action-value function Q^ with weights equal to NN above
    target_q_network = DQN_Agent(env.action_space.n)
    target_q_network.load_state_dict(policy_q_network.state_dict())
    target_q_network.eval()
    target_q_network.to(device)
    
    # Use HuberLoss 
    loss_fn = torch.nn.SmoothL1Loss()

    optimizer = RMSprop(policy_q_network.parameters(),lr=lr,momentum=grad_momentum,eps=min_squared_gradient)

    # For episode =1 , M do

    timesteps_total = 0

    #for _ in range(5000):
    for ep_num in range(1):
    # Here need to prime the sequence
        done = False 
        obs = env.reset()
        timestep_start = timesteps_total
        timestep_end = timesteps_total
        while not done: 
            current_epsilon = annealed_epsilon(timesteps_total)

            # With probability epsilon select a random action At,
            # otherwise select At = argmax a Q(preproc(St),a,theta)
            action = get_action(obs, current_epsilon)
            
            # execute action At in emulator and observe reward Rt and image Xt+1
            next_obs, reward, done, info = env.step(action)

            reward_tracker.append(reward)
            
            experience = (obs, action, reward, next_obs, done)

            replay_mem.store(experience)


            # sample data from experience replay 
            sample_prev_state, sample_action, sample_reward, sample_next_state, sample_done = convert_to_tensor(replay_mem.sample(minibatch_size))

            
            # place data on GPU
            sample_prev_state = sample_prev_state.to(device)
            sample_action = sample_action.to(device)
            
            
            Yj = compute_targets(sample_prev_state, sample_action, sample_reward, sample_next_state, sample_done, device)
            Yj = Yj.to(device)

            # converting to a 32x1 tensor s.t. gather operation works as intended 
            sample_action_index = sample_action.unsqueeze(0)
            
            policy_predictions = (policy_q_network(sample_prev_state).gather(1,sample_action_index)).squeeze()
            
            loss = loss_fn(Yj,policy_predictions)


            optimizer.zero_grad()
            loss.backward()
            # Need to clip gradients between -1,1 

            for param in policy_q_network.parameters():
                param.grad.data.clamp_(-1,1)
            
            optimizer.step()
            
            # very last step before loop exit , 
            timesteps_total += 1
            timestep_end += 1 
        
            if timesteps_total % C == 0:
                target_q_network.load_state_dict(policy_q_network.state_dict()) 

        episode_length_tracker.append(timestep_end - timestep_start)
        print(f'completed train episode {ep_num}')    




    # save policy network and 
    torch.save(policy_q_network,'q_network.pt')

    graph_reward(reward_tracker,eval=False)
    graph_episode_length(episode_length_tracker,eval=False)

    return policy_q_network



def main():

    #args = get_args()

    env = gym.make('Pong-v0',obs_type='grayscale')
    env = env.unwrapped
    print(env)
    train_env = train_process_env(env)
    #print(env)
    agent_network = train(train_env)
    eval_env = evaluation_process_env(env)

    evaluate(eval_env, agent_network)



if __name__ == '__main__':
    main()
