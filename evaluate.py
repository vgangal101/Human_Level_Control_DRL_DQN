import random
import torch 
import numpy as np


def evaluate_perf(env,policy_net,tb_writer,eval_rew_tracker,eval_ep_len_tracker):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_episodes = 10
    rew_tracker = []
    ep_len_tracker = []

    for episode in range(eval_episodes):
        state = env.reset()
        ep_len = 0
        ep_reward = 0
        for t in range(10000):
            sample = random.random()
            if sample > 0.05:
                with torch.no_grad():
                    action = policy_net(state.to(device)).max(1)[1].view(1,1)
            else: 
                action = torch.tensor([random.randrange(env.action_space.n)], device=device, dtype=torch.long)

            next_state, reward, done, info = env.step(action.item())
            ep_len += 1 
            ep_reward += reward

            if done: 
                rew_tracker.append(ep_reward)
                ep_len_tracker.append(ep_len)
                break 
            else: 
                state = next_state   

    mean_ep_len = np.mean(ep_len_tracker)
    mean_reward = np.mean(rew_tracker)

    eval_rew_tracker.append(mean_reward)
    eval_ep_len_tracker.append(mean_ep_len)
    
    # COLLECT TENSORBOARD STATISTICS FOR LOGGING
     
