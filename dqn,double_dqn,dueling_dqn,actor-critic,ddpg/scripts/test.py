from agent import *
from trainer import DistributedTrainer
from numeric_env import MultiEnv

import torch
torch.set_num_threads(1)


env = MultiEnv(2, 2, car_policy='wjf')
args = DDPGArgs(state_dim=env.STATE_DIM,
                action_dim=2,
                action_scale=1.0,
                )

agents = [DDPGAgent(args) for _ in range(2)]
normalizer_path = '../models/debug/norm.pkl'
save_pathlists = [['../models/debug/actor.pkl', '../models/debug/critic.pkl'] for _ in range(2)]
trainer = DistributedTrainer(agents,
                             env,
                             parameter_share=False,
                             log_dir='../logs/debug',
                             save_pathlists=None,
                             load_pathlists=None,
                             normalizer_path=None
                             )
trainer.train(-1)

# torch.set_num_threads(1)
#
#
# env = MultiEnv(2, 2, car_policy='wjf')
# action_space = [(-5 + x * 2, -5 + y * 2) for x in range(6) for y in range(6)]
#
#
# args = DuelingDQNArgs(state_dim=env.STATE_DIM,
#                       discrete_action_space=action_space
#                       )
#
# agents = [DuelingDQNAgent(args) for _ in range(2)]
# trainer = DistributedTrainer(agents,
#                              env,
#                              parameter_share=False,
#                              log_dir='../logs/duelingdqn_d')
# trainer.train(-1)
