import os

import numpy as np
import torch
import torch.optim as optim

from agent import Agent
from agent.common import ACArgs
from . import ACActor, ACCritic


class ACAgent(Agent):

    def __init__(self, args: ACArgs):
        self._args = args

        self._actor = None
        self._critic = None
        self._actor_optimizer = None
        self._critic_optimizer = None

        self._critic_loss_function = torch.nn.MSELoss()

        self._step_count = 0
        self._update_step = 0

    def choose_action_with_exploration(self, state):
        action = self.choose_action(state)
        noise = np.random.normal(0, self._args.action_scale, (self._args.action_dim, ))
        return action + noise

    def choose_action(self, state):
        with torch.no_grad():
            mu, sigma = self._actor(torch.tensor(state).float())
            dist = torch.distributions.Normal(loc=mu, scale=sigma)
            action = dist.sample().numpy()
        return action

    def _update(self, s, a, s_, r, d):
        self._update_step += 1
        s = torch.tensor(s).float()
        a = torch.tensor(a).float()
        r = torch.tensor(r).float()
        s_ = torch.tensor(s_).float()
        d = torch.tensor(d).float()

        # update critic
        q_s = self._critic(s)
        q_s_ = self._critic(s_)
        with torch.no_grad():
            adv = torch.tensor([0.]) if d else r + self._args.gamma * q_s_ - q_s
        target = torch.tensor([0.]) if d else r + self._args.gamma * q_s_
        loss = self._critic_loss_function(target, q_s)
        self._critic_optimizer.zero_grad()
        loss.backward()
        self._critic_optimizer.step()

        # update Actor
        mu, sigma = self._actor(s)
        logit = torch.distributions.Normal(mu, sigma).log_prob(torch.Tensor(a))
        target = (-logit * adv).mean()
        self._actor_optimizer.zero_grad()
        target.backward()
        self._actor_optimizer.step()

    def train_one_step(self, s, a, s_, r, d):
        self._step_count += 1

        self._args.action_scale = np.max([self._args.final_action_scale,
                                          self._args.action_scale * self._args.scale_decay_factor])

        if self._step_count % self._args.update_freq == 0 and self._step_count > self._args.steps_before_training:
            for _ in range(self._args.update_count):
                self._update(s, a, s_, r, d)

    def set_model(self, parameter_share=False, load_paths=None):
        if not parameter_share:
            if load_paths is None or not os.path.exists(load_paths[0]):
                self._actor = ACActor(self._args.state_dim, self._args.action_dim)
                self._critic = ACCritic(self._args.state_dim)
            else:
                self._load_model(load_paths)
        else:
            if load_paths is None or not os.path.exists(load_paths[0]):
                if ACAgent.networks is None:
                    # [actor, target actor, critic, target critic]
                    ACAgent.networks = []
                    ACAgent.networks.append(ACActor(self._args.state_dim, self._args.action_dim))
                    ACAgent.networks.append(ACCritic(self._args.state_dim))
            else:
                if ACAgent.networks is None:
                    # [actor, target actor, critic, target critic]
                    self._load_model(load_paths)
                    ACAgent.networks = []
                    ACAgent.networks.append(self._actor)
                    ACAgent.networks.append(self._critic)
            self._actor = ACAgent.networks[0]
            self._critic = ACAgent.networks[1]
        self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=self._args.actor_lr)
        self._critic_optimizer = optim.Adam(self._critic.parameters(), lr=self._args.critic_lr)

    def save_model(self, paths):
        raise NotImplementedError

    def _load_model(self, paths):
        raise NotImplementedError
