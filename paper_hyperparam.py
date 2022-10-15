
# Hyperparameters from paper 
minibatch_size = 32

#replay_memory_size = 1000000
# do not use replay_memory_size of 1 mil frames 
replay_memory_size = 100000

history_length = 4

C_target_net_update_freq = 10000 # target_network_update_frequency

gamma = 0.99 # discount factor used in the Q-learning update
action_repeat = 4
update_frequency = 4
lr = 0.00025 # learning rate used by RMSProp
grad_momentum = 0.95 # gradient momentum used by RMSProp
squared_gradient_momentum = 0.95 # squared [ alpha]
min_squared_gradient = 0.01 # constant to squared gradient in denom of RMSProp Update [ eps]


initial_exploration = 1 # inital vaue of epsilon in epsilon greedy exploration
final_exploration = 0.1 # final value of epsilon in epsilon-greedy exploration
final_exploration_frame = 1000000

replay_start_size = 50000 # uniform random policy is run for this num of frames before learning starts & resulting experience is used to populate replay ReplayMemory

no_op_max = 30 # max number of 'do nothing' actions to be performed  by the agent at the start of an episode


