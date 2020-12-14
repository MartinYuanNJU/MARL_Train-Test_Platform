from agent import ACAgent, ACArgs
from trainer import CentralizedTrainer
from numeric_env import MultiEnv

import torch
torch.set_num_threads(1)


env = MultiEnv(2, 2)
args = ACArgs(state_dim=env.STATE_DIM,
              action_dim=4
              )

agent = ACAgent(args)
trainer = CentralizedTrainer(agent,
                             env,
                             log_dir='../logs/ac_c')
trainer.train(1000000)
