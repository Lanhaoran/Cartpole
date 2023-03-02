import gym, torch, numpy as np, torch.nn as nn
import tianshou as ts
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.policy import PPOPolicy
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from torch.utils.tensorboard import SummaryWriter

import os
from utils import *

############
## Define parameters
task = 'MyCartPole-v0'
env_params =  {'masscart': 1.0, 'masspole': 0.1, 'length': 1.0}
#lr, epoch, batch_size = 1e-3, 10, 64
#train_num, test_num = 10, 100
gamma = 0.99
#n_step, target_freq = 3, 320
#buffer_size = 20000
#eps_train, eps_test = 0.1, 0.05
#step_per_epoch, step_per_collect = 10000, 10
seed = 1626

############
## Make environment
env = gym.make(task, render_mode="human", opts=env_params)
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
test_num = 5
env.seed([i for i in range(test_num)])
np.random.seed(seed)
torch.manual_seed(seed)


#
#test_envs = DummyVectorEnv(
#        [lambda: gym.make(task, opts = env_params) for _ in range(test_num)]
#    )
#test_envs.seed(seed)

#############
## Define Network
net = Net(state_shape, hidden_sizes=[16,16], device='cuda') 
actor = DataParallelNet(
            Actor(net, action_shape, device=None).to('cuda')
        )
critic = DataParallelNet(Critic(net, device=None).to('cuda'))
actor_critic = ActorCritic(actor, critic)
for m in actor_critic.modules():
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.zeros_(m.bias)
optim = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)
dist = torch.distributions.Categorical


#########
## Setup policy dqn
# PPO policy
policy = PPOPolicy(
    actor,
    critic,
    optim,
    dist,
    discount_factor=gamma,
    max_grad_norm=0.5,
    eps_clip=0.2,
    vf_coef=0.5,
    ent_coef=0.0,
    gae_lambda=0.95,
    reward_normalization=None,
    dual_clip=None,
    value_clip=0,
    action_space=env.action_space,
    deterministic_eval=True,
    advantage_normalization=0,
    recompute_advantage=0
)


##################################################
## Load learned model
path = r'E:\Character Motion\CartPole\tianshou_cartpole\models_7\MyCartPole-v0-{}_{}_{}\ppo\0\policy.pth'.format(env_params['masscart'], env_params['masspole'], env_params['length'])
# path = r'E:\Character Motion\CartPole\tianshou_cartpole\models\MyCartPole-v0-0.05_2.5_0.75\ppo\5\policy.pth'
dict = torch.load(path)
policy.load_state_dict(dict)      # debug didn't support relative path
#print("Model's state_dict: ")
#print(policy.state_dict().keys())
'''
path1 = r'E:\Character Motion\CartPole\tianshou_cartpole\models_7\MyCartPole-v0-{}_{}_{}\ppo\1\policy.pth'.format(env_params['masscart'], env_params['masspole'], env_params['length'])
dict1 = torch.load(path1)
for k,v in dict.items():
    if v.equal(dict1[k]): print(k)

    
#################################################
## Test
policy.eval()
collector = Collector(policy, env)
result = collector.collect(n_episode=test_num, render=0.)
rews, lens = result["rews"], result["lens"]
print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
'''


##################################################
## Random change actor net
#policy = RandomActorWeight(dict,policy)



###################################
##  Load the diffusion result
path_d =  r'E:\Character Motion\Diffusion\DenoisingDiffusionProbabilityModel-ddpm-\SampledPolicys\SampledGuidencePolicy_17_1.0-0.1-1.0_ckpt_500_.npy'
policy = LoadActorWeight_first_layer_16(dict, policy, path_d, 5)
policy.eval()
collector = Collector(policy, env)
result = collector.collect(n_episode=test_num, render=0.)
rews, lens = result["rews"], result["lens"]
print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
'''

###################################
## Add noise to actor net (layer based)
layers = ['actor.net.module.preprocess.model.model.0','actor.net.module.preprocess.model.model.2','actor.net.module.last.model']
for layer in layers:
    log_path = r"E:\Character Motion\CartPole\tianshou_cartpole\log\layer_log\2\{}".format(layer)
    writer = SummaryWriter(log_path)
    #sigma = 0.01
    sigmas = np.linspace(0.0, 0.25, 100)
    for sigma in sigmas:
        policy, total_noise = AddNoise(dict, policy, sigma, layer)

        #############################
        ## Test
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=test_num, render=0.)
        rews, lens = result["rews"], result["lens"]
        print("sigma: {}".format(sigma))
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
        writer.add_scalar('reward', rews.mean(), sigma*1000)
        writer.add_scalar('total loss', total_noise, sigma*1000)
'''

'''
#######################################
## Test the rubust of each layer under different env (size [64,64])
select_config = []
root_path = r'E:\Character Motion\CartPole\tianshou_cartpole\models'
for folder in os.listdir(root_path):
    config = folder.split('-')[-1]
    cart, pole, len = config.split('_')
    if 0.05 <= float(cart) <= 1.0 and 0.1<= float(pole) <= 0.5 and 0.75 <= float(len) <= 2.0:
        select_config.append(folder)

print(folder)

## load the policy
for config in select_config:
    fix_policy_path = os.path.join(root_path, config, 'ppo\0\policy.pth')
'''