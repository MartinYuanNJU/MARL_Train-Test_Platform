from agent import PERDQNAgent, PERDQNArgs
from trainer import DistributedTrainer
from numeric_env import MultiEnv

import torch
torch.set_num_threads(1)


env = MultiEnv(2, 2)
action_space = [(-5 + x * 2, -5 + y * 2) for x in range(6) for y in range(6)]


args = PERDQNArgs(state_dim=env.STATE_DIM,
                  discrete_action_space=action_space,
                  beta_inc=0
                  )

agents = [PERDQNAgent(args) for _ in range(2)]
trainer = DistributedTrainer(agents,
                             env,
                             parameter_share=False,
                             log_dir='../logs/perdqn_d')
trainer.train(1000000)
