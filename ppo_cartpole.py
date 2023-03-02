import argparse
import os
import sys
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger, LazyLogger
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.utils.net.discrete import Actor, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='MyCartPole-v0')
    #########
    parser.add_argument('--masscart', type=float, default=0.1)
    parser.add_argument('--masspole', type=float, default=0.05)
    parser.add_argument('--length', type=float, default=0.1)
    #parser.add_argument('--train-index', type=int, default=1)
    #########
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=50000)
    parser.add_argument('--step-per-collect', type=int, default=2000)
    parser.add_argument('--repeat-per-collect', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[16,16])
    parser.add_argument('--training-num', type=int, default=20)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='models_6')   # generate models
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument('--norm-adv', type=int, default=0)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)
    args = parser.parse_known_args()[0]
    return args

def test_ppo(args=None, index=1, seed = 0):
    train_index = index
    args.seed = seed
    env_params =  {'masscart': args.masscart, 'masspole': args.masspole, 'length': args.length}
    env = gym.make(args.task, opts = env_params)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    if args.reward_threshold is None:
        default_reward_threshold = {"MyCartPole-v0": 475}
        args.reward_threshold = default_reward_threshold.get(
            args.task, env.spec.reward_threshold
        )
    # train_envs = gym.make(args.task)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv(
        [lambda: gym.make(args.task, opts = env_params) for _ in range(args.training_num)]
    )
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task, opts = env_params) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
  

    # model
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    if torch.cuda.is_available():
        actor = DataParallelNet(
            Actor(net, args.action_shape, device=None).to(args.device)
        )
        critic = DataParallelNet(Critic(net, device=None).to(args.device))
    else:
        actor = Actor(net, args.action_shape, device=args.device).to(args.device)
        critic = Critic(net, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    '''
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    dist = torch.distributions.Categorical
    '''

    # Load well trained policy, and only optimize part of the parameters
    path = r'E:\Character Motion\CartPole\tianshou_cartpole\models_3\MyCartPole-v0-1.0_0.1_1.0\ppo\25\policy.pth'
    dict = torch.load(path)
    state_dict = {k:v for k,v in dict.items() if k in actor_critic.state_dict().keys()}
    #actor_critic.state_dict().update(state_dict)
    actor_critic.load_state_dict(state_dict)

    # freeze the layer parameters
    for key,param in actor_critic.named_parameters():
        #print(key)
        if 'last.model' in key: # or 'preprocess.model.model.2' in key: 
            param.requires_grad=False
        else: 
            param.requires_grad=True
    params = filter(lambda p: p.requires_grad, actor_critic.parameters())
    
    optim = torch.optim.Adam(params, lr=args.lr)
    dist = torch.distributions.Categorical

    # PPO policy
    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        action_space=env.action_space,
        deterministic_eval=True,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv
    )
    # collector
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
    )
    test_collector = Collector(policy, test_envs)
    # log
    env_params_path = '_'.join([str(val) for _,val in env_params.items()])
    log_path = os.path.join(args.logdir, args.task + '-' + env_params_path, 'ppo', str(train_index))
    if os.path.exists(log_path) == False: os.makedirs(log_path)
    #writer = SummaryWriter(log_path)
    #logger = TensorboardLogger(writer)
    logger = LazyLogger()

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger
    )
    #assert stop_fn(result['best_reward'])

    if __name__ == '__main__':
        #pprint.pprint(result)
        # write into the file
        result_path = os.path.join(log_path, 'train_result.txt')
        with open(result_path, 'w') as f:
            for key, value in result.items():
                f.write(key)
                f.write(': ')
                f.write(str(value))
                f.write('\n')
        # Let's watch its performance!
        
        env = gym.make(args.task, render_mode="rgb_array", opts = env_params)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
        #######
        save_best_fn(policy)
        

if __name__ == '__main__':
    args=get_args()
    print("Env: masscart {}, masspole {}, length {}".format(args.masscart, args.masspole, args.length))
    for i in range(10):
        #print(i)
        # 1000 best policy under one environment
        print("Round {} / 10".format(i+1))
        test_ppo(args=args, index=i, seed=i)