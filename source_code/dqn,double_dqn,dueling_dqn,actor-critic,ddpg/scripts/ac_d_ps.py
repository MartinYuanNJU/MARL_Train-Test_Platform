import numpy as np

from agent import ACAgent, ACArgs
from trainer import DistributedTrainer
from numeric_env import MultiEnv

import torch
torch.set_num_threads(1)


env = MultiEnv(2, 2)
args = ACArgs(state_dim=env.STATE_DIM + 1,
              action_dim=2
              )

agents = [ACAgent(args) for _ in range(2)]
trainer = DistributedTrainer(agents,
                             env,
                             parameter_share=True,
                             state_transformer=np.append,
                             log_dir='../logs/ac_d_ps')
trainer.train(1000000)
