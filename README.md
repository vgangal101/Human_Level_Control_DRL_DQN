# Human_Level_Control_DRL_DQN

This repo is an implementation of the paper: 'Human-level control through deep reinforcement learning'

Link to the paper: https://arxiv.org/pdf/1709.06009.pdf 


## Results 
In this repo we train on the atari games of Pong and Breakout. The training and evaluation graphs are as follows: 

### Pong 


![Pong_Episode_Rewards_train](results/Pong_successful_train/graph_pics/rewards_train.png)

![Pong_Episode_Rewards_Eval](results/Pong_successful_train/graph_pics/rewards_eval.png)


![Pong_episode_length_train](results/Pong_successful_train/graph_pics/pong_episode_length_train.png)

![Pong_episode_length_eval](results/Pong_successful_train/graph_pics/Pong_episode_length_eval.png)


### Breakout

![Breakout_Episode_Rewards_train](results/Breakout_successful_train/graph_pics/rewards_train.png)


![Breakout_Episode_Rewards_eval](results/Breakout_successful_train/graph_pics/rewards_eval.png)

![Breakout_Episode_Length_train](results/Breakout_successful_train/graph_pics/episode_length_train.png)

![Breakout_Episode_Length_eval](results/Breakout_successful_train/graph_pics/episode_length_eval.png)




Resources used:

https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/

https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py

