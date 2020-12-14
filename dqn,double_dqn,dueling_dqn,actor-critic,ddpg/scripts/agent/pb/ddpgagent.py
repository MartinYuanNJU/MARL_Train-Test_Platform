import copy
import os

import numpy as np
import torch
import torch.optim as optim

from agent import Agent
from agent.common import DDPGArgs
from . import ReplayBuffer, DDPGActor, DDPGCritic


class DDPGAgent(Agent):

    def __init__(self, args: DDPGArgs):
        self._args = args
        self._replay_buffer = ReplayBuffer(self._args.buffer_size, self._args.state_dim, self._args.action_dim)

        self._actor = None
        self._critic = None
        self._target_actor = None
        self._target_critic = None
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
            s = torch.from_numpy(state).float()
            action = self._actor(s)
        action = action.detach().numpy()
        return action

    def _soft_copy_parms(self):
        with torch.no_grad():
            for t, s in zip(self._target_actor.parameters(), self._actor.parameters()):
                t.copy_(0.95 * t.data + 0.05 * s.data)
            for t, s in zip(self._target_critic.parameters(), self._critic.parameters()):
                t.copy_(0.95 * t.data + 0.05 * s.data)

    def _update(self):
        self._update_step += 1
        s, a, r, s_, d = self._replay_buffer.sample(self._args.batch)
        s = torch.tensor(s).float()
        a = torch.tensor(a).float()
        r = torch.tensor(r).float()
        s_ = torch.tensor(s_).float()
        d = torch.tensor(d).float()

        # update critic
        with torch.no_grad():
            a_ = self._target_actor(s_)
            s__a_ = torch.cat((s_, a_), dim=1)
            q_s__a_ = self._target_critic(s__a_)
            target = r.reshape(-1, 1) + self._args.gamma * q_s__a_ * (1 - d).reshape(-1, 1)

        q_s_a = self._critic(torch.cat((s, a), dim=1))
        loss = self._critic_loss_function(q_s_a.float(), target.float())
        self._critic_optimizer.zero_grad()
        loss.backward()
        self._critic_optimizer.step()

        # update Actor
        a = self._actor(s)
        s_a = torch.cat((s, a), dim=1)
        q_s_a = self._target_critic(s_a)
        loss = -q_s_a.mean()
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()

        self._soft_copy_parms()

    def train_one_step(self, s, a, s_, r, d):
        self._step_count += 1
        self._replay_buffer.add(s, a, r, s_, d)

        self._args.action_scale = np.max([self._args.final_action_scale,
                                          self._args.action_scale * self._args.scale_decay_factor])

        if self._step_count % self._args.update_freq == 0 and self._step_count > self._args.steps_before_training:
            for _ in range(self._args.update_count):
                self._update()

    def set_model(self, parameter_share=False, load_paths=None):
        if not parameter_share:
            if load_paths is None or not os.path.exists(load_paths[0]):
                self._actor = DDPGActor(self._args.state_dim, self._args.action_dim)
                self._target_actor = copy.deepcopy(self._actor).eval()
                self._critic = DDPGCritic(self._args.state_dim, self._args.action_dim)
                self._target_critic = copy.deepcopy(self._critic).eval()
            else:
                self._load_model(load_paths)
        else:
            if load_paths is None or not os.path.exists(load_paths[0]):
                if DDPGAgent.networks is None:
                    # [actor, target actor, critic, target critic]
                    DDPGAgent.networks = []
                    DDPGAgent.networks.append(DDPGActor(self._args.state_dim, self._args.action_dim))
                    DDPGAgent.networks.append(copy.deepcopy(DDPGAgent.networks[0]).eval())
                    DDPGAgent.networks.append(DDPGCritic(self._args.state_dim, self._args.action_dim))
                    DDPGAgent.networks.append(copy.deepcopy(DDPGAgent.networks[2]).eval())
            else:
                if DDPGAgent.networks is None:
                    # [actor, target actor, critic, target critic]
                    self._load_model(load_paths)
                    DDPGAgent.networks = []
                    DDPGAgent.networks.append(self._actor)
                    DDPGAgent.networks.append(self._target_actor)
                    DDPGAgent.networks.append(self._critic)
                    DDPGAgent.networks.append(self._target_critic)
            self._actor = DDPGAgent.networks[0]
            self._target_actor = DDPGAgent.networks[1]
            self._critic = DDPGAgent.networks[2]
            self._target_critic = DDPGAgent.networks[3]
        self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=self._args.actor_lr)
        self._critic_optimizer = optim.Adam(self._critic.parameters(), lr=self._args.critic_lr)

    def save_model(self, paths):
        torch.save(self._actor.state_dict(), paths[0])
        torch.save(self._critic.state_dict(), paths[1])

    def _load_model(self, paths):
        self._actor = DDPGActor(self._args.state_dim, self._args.action_dim)
        self._actor.load_state_dict(torch.load(paths[0]))
        self._target_actor = copy.deepcopy(self._actor).eval()
        self._critic = DDPGCritic(self._args.state_dim, self._args.action_dim)
        self._critic.load_state_dict(torch.load(paths[1]))
        self._target_critic = copy.deepcopy(self._critic).eval()
