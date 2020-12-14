from agent import DDPGAgent, DDPGArgs
from trainer import CentralizedTrainer
from numeric_env import MultiEnv

import torch
torch.set_num_threads(1)


env = MultiEnv(2, 2)
args = DDPGArgs(state_dim=env.STATE_DIM,
                action_dim=4
                )

agent = DDPGAgent(args)
trainer = CentralizedTrainer(agent,
                             env,
                             log_dir='../logs/ddpg_c')
trainer.train(1000000)
