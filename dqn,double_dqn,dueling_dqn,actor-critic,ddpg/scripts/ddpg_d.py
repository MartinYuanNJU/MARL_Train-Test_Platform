from agent import DDPGAgent, DDPGArgs
from trainer import DistributedTrainer
from numeric_env import MultiEnv

import torch
torch.set_num_threads(1)


env = MultiEnv(2, 2)
args = DDPGArgs(state_dim=env.STATE_DIM,
                action_dim=2
                )

agents = [DDPGAgent(args) for _ in range(2)]
trainer = DistributedTrainer(agents,
                             env,
                             parameter_share=False,
                             log_dir='../logs/ddpg_d')
trainer.train(1000000)
