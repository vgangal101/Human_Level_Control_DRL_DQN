from collections import namedtuple, deque 
import random
import numpy as np
import torch 


class LinearSchedule():
    def __init__(self,final_exploration_frame,eps_start,eps_final):
        self.final_exploration_frame = final_exploration_frame
        self.eps_start = eps_start
        self.eps_final = eps_final 

    def get_epsilon(self, timestep):
        fraction  = min(float(timestep) / self.final_exploration_frame, 1.0)
        return self.eps_start + fraction * (self.eps_final - self.eps_start)


# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
# class ReplayMemory(object):

#     def __init__(self, capacity):
#         self.memory = deque([],maxlen=capacity)

#     def push(self, *args):
#         """Save a transition"""
#         self.memory.append(Transition(*args))

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)



class ReplayMemory():
    def __init__(self,capacity,state_dims):
        self.capacity = capacity
        self.state = np.empty((capacity,*state_dims))
        self.action = np.empty((capacity))
        self.reward = np.empty((capacity))
        self.next_state = np.empty((capacity,*state_dims))
        self.done = np.empty((capacity))
        self.counter = 0
        self.is_full = False 
    
    def store(self,state,action,reward,next_state,done):
        self.state[self.counter] = state
        self.action[self.counter] = action
        self.reward[self.counter] = reward
        self.next_state[self.counter] = next_state 
        self.done[self.counter] = done
        
        if self.counter + 1 > self.capacity - 1:
            self.counter = 0 
            self.is_full = True 
        else: 
            self.counter += 1 
    
    def sample(self,minibatch_size):
        """
        Return the data as correctly formated pytorch tensors, 
        """

        # get contigous index positions 

        if self.is_full: 
            # can sample from anywhere 
            all_avail_indices = list(range(0,self.capacity))
        else:
            # get contigous index positions
            all_avail_indices = list(range(0,self.counter+1))
            
        select_indices = random.sample(all_avail_indices,minibatch_size)    
        state_data = torch.from_numpy(self.state[[select_indices]])
        action_data = torch.from_numpy(self.action[[select_indices]])
        reward_data = torch.from_numpy(self.reward[[select_indices]])
        next_state_data = torch.from_numpy(self.next_state[[select_indices]])
        done_data = torch.from_numpy(self.done[[select_indices]])

        return state_data.squeeze(), action_data.squeeze(), reward_data.squeeze(), next_state_data.squeeze(), done_data.squeeze() 
         