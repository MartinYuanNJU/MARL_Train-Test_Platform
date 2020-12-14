import copy
from collections import Iterable
import os

import numpy as np
import torch
import torch.optim as optim

from agent import Agent
from agent.common import DQNArgs
from . import ReplayBuffer, QNet


class DQNAgent(Agent):

    def __init__(self, args: DQNArgs):
        self._args = args
        self._action_count = len(self._args.discrete_action_space)

        self._replay_buffer = ReplayBuffer(self._args.buffer_size, self._args.state_dim)

        self._a2n = dict()
        for i, a in enumerate(self._args.discrete_action_space):
            self._a2n[tuple(a) if isinstance(a, Iterable) else a] = i

        self._epsilon_diff = (self._args.epsilon - self._args.final_epsilon) / self._args.total_trainsteps

        self._qnet = None
        self._target_qnet = None
        self._loss_function = torch.nn.MSELoss()
        self._optimizer = None

        self._step_count = 0
        self._update_step = 0

    def choose_action_with_exploration(self, state):
        with torch.no_grad():
            seed = np.random.uniform(0, 1)
            if seed < self._args.epsilon:
                return self._args.discrete_action_space[np.random.randint(0, self._action_count)]
            else:
                return self.choose_action(state)

    def choose_action(self, state):
        s = torch.from_numpy(state).float()
        return self._args.discrete_action_space[torch.argmax(self._qnet(s)).numpy().item()]

    def _update(self):
        self._update_step += 1
        s, a, r, s_, d = self._replay_buffer.sample(self._args.batch)
        s = torch.tensor(s).float()
        a = torch.tensor(a).long()
        r = torch.tensor(r).float()
        s_ = torch.tensor(s_).float()

        with torch.no_grad():
            indices = torch.argmax(self._target_qnet(s_), dim=1)
            q_s_a = self._target_qnet(s_).gather(1, indices.reshape(-1, 1)).reshape(-1)
            label = r + self._args.gamma * q_s_a * (1 - d)
            target = self._qnet(s).scatter(1, a.reshape(-1, 1), label.reshape(-1, 1).float())
        q_s = self._qnet(s)
        loss = self._loss_function(q_s, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        if self._update_step % self._args.tau == 0:
            self._target_qnet.load_state_dict(self._qnet.state_dict())

    def train_one_step(self, s, a, s_, r, d):
        self._step_count += 1
        self._replay_buffer.add(s, self._a2n[tuple(a)], r, s_, d)

        self._args.epsilon = max(self._args.final_epsilon, self._args.epsilon - self._epsilon_diff)

        if self._step_count % self._args.update_freq == 0 and self._step_count > self._args.steps_before_training:
            for _ in range(self._args.update_count):
                self._update()

    def set_model(self, parameter_share=False, load_paths=None):
        if not parameter_share:
            if load_paths is None or not os.path.exists(load_paths[0]):
                self._qnet = QNet(self._args.state_dim, self._action_count)
                self._target_qnet = copy.deepcopy(self._qnet).eval()
            else:
                self._load_model(load_paths)
        else:
            if load_paths is None or not os.path.exists(load_paths[0]):
                if DQNAgent.networks is None:
                    # [q net, target q net]
                    DQNAgent.networks = []
                    DQNAgent.networks.append(QNet(self._args.state_dim, self._action_count))
                    DQNAgent.networks.append(copy.deepcopy(DQNAgent.networks[0]).eval())
                self._qnet = DQNAgent.networks[0]
                self._target_qnet = DQNAgent.networks[1]
            else:
                if DQNAgent.networks is None:
                    # [q net, target q net]
                    self._load_model(load_paths)
                    DQNAgent.networks = []
                    DQNAgent.networks.append(self._qnet)
                    DQNAgent.networks.append(self._target_qnet)
                self._qnet = DQNAgent.networks[0]
                self._target_qnet = DQNAgent.networks[1]
        self._optimizer = optim.Adam(self._qnet.parameters(), lr=self._args.lr)

    def save_model(self, paths):
        # [q net path]
        # only save qnet's parameters
        torch.save(self._qnet.state_dict(), paths[0])

    def _load_model(self, paths):
        # [q net path]
        self._qnet = QNet(self._args.state_dim, self._action_count)
        self._qnet.load_state_dict(torch.load(paths[0]))
        self._target_qnet = copy.deepcopy(self._qnet).eval()
