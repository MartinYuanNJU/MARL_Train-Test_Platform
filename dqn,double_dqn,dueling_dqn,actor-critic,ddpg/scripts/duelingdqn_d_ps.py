import numpy as np

from agent import DuelingDQNAgent, DuelingDQNArgs
from trainer import DistributedTrainer
from numeric_env import MultiEnv

import torch
torch.set_num_threads(1)


env = MultiEnv(2, 2)
action_space = [(-5 + x * 2, -5 + y * 2) for x in range(6) for y in range(6)]


args = DuelingDQNArgs(state_dim=env.STATE_DIM + 1,
                      discrete_action_space=action_space
                      )

agents = [DuelingDQNAgent(args) for _ in range(2)]
trainer = DistributedTrainer(agents,
                             env,
                             parameter_share=True,
                             state_transformer=np.append,
                             log_dir='../logs/duelingdqn_d_ps')
trainer.train(1000000)
