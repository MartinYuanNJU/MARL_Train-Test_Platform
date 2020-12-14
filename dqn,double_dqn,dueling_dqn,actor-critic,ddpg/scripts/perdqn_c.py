from agent import PERDQNAgent, PERDQNArgs
from trainer import CentralizedTrainer
from numeric_env import MultiEnv

import torch
torch.set_num_threads(1)


env = MultiEnv(2, 2)
action_space = [(-5 + x * 2, -5 + y * 2, -5 + a * 2, -5 + b * 2)
                for x in range(6) for y in range(6) for a in range(6) for b in range(6)]

args = PERDQNArgs(state_dim=env.STATE_DIM,
                  discrete_action_space=action_space,
                  beta_inc=0
                  )

agent = PERDQNAgent(args)
trainer = CentralizedTrainer(agent,
                             env,
                             log_dir='../logs/perdqn_c')
trainer.train(1000000)
