from agent_maddpg import Args_maddpg, Agent_MADDPG
from trainer_maddpg import Trainer_MADDPG
from env_virtual import MultiEnv

import torch
torch.set_num_threads(1)

if __name__ == '__main__':
    env = MultiEnv(uav_cnt=5, car_cnt=5, car_policy='random')
    args = Args_maddpg(exp_cap=1000000,
                       gamma=0.95,
                       scale=1.0,
                       batch=64,
                       update_cnt=20,
                       update_interval=20,
                       actor_lr=1e-4,
                       critic_lr=1e-3,
                       state_dim=env.STATE_DIM,
                       action_dim=env.ACTION_DIM,
                       exp_state_dim=env.STATE_DIM,
                       agent_num=5,
                       car_num=5)
    agent = Agent_MADDPG(args)
    trainer = Trainer_MADDPG(agent, env,
                             log_dir='./logs/fivevsfive',
                             print_freq=10,
                             test_episode_count=20,
                             normalizerround=100000,
                             action_bound=5.0)
    episode = 1000000
    trainer.train(episode)
