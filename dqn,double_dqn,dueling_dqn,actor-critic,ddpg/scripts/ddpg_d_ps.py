import numpy as np

from agent import DDPGAgent, DDPGArgs
from trainer import DistributedTrainer
from numeric_env import MultiEnv

import torch
torch.set_num_threads(1)


env = MultiEnv(2, 2)
args = DDPGArgs(state_dim=env.STATE_DIM + 1,
                action_dim=2
                )

agents = [DDPGAgent(args) for _ in range(2)]
trainer = DistributedTrainer(agents,
                             env,
                             parameter_share=True,
                             state_transformer=np.append,
                             log_dir='../logs/ddpg_d_ps')
trainer.train(1000000)
