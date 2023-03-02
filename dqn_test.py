import gym, torch, numpy as np, torch.nn as nn
import tianshou as ts
# from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
# from tianshou.utils.net.discrete import Actor, Critic

############
## Define parameters
task = 'MyCartPole-v0'
env_params =  {'masscart': 1.0, 'masspole': 0.1, 'length': 0.5}
lr, epoch, batch_size = 1e-3, 10, 64
train_num, test_num = 10, 100
gamma, n_step, target_freq = 0.9, 3, 320
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05
step_per_epoch, step_per_collect = 10000, 10


############
## Make environment
env = gym.make(task, render_mode="human", opts=env_params)
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n


############
## Define network
from tianshou.utils.net.common import Net
# model
# dqn
net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])  
optim = torch.optim.Adam(net.parameters(), lr=1e-3)


#########
## Setup policy dqn
policy = ts.policy.DQNPolicy(net,optim, gamma, n_step, target_update_freq=target_freq)

###########
## Load model
policy.load_state_dict(torch.load(r'E:\Character Motion\CartPole\tianshou_cartpole\dqn.pth'))      # debug didn't support relative path
print("Model's state_dict:")
for param_tensor in policy.state_dict():
    print(param_tensor, "\t", policy.state_dict()[param_tensor].size())


########
## Watch performance
policy.eval()
policy.set_eps(eps_test)   # dqn
collector = ts.data.Collector(policy, env, exploration_noise=True)
result = collector.collect(n_episode=1, render=1 / 35)
rews, lens = result["rews"], result["lens"]
print(f"Final reward: {rews.mean()}, length: {lens.mean()}")