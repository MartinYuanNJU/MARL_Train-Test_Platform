import pickle
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent.common.normalizer import Normalizer


class Trainer:

    def train(self, n):
        raise NotImplementedError

    def test(self, n):
        raise NotImplementedError


class CentralizedTrainer(Trainer):

    def __init__(self,
                 agent,
                 env,
                 save_freq=100,
                 save_paths=None,
                 load_paths=None,
                 normalizer_path=None,
                 log_dir='./logs/default/',
                 print_freq=20,
                 test_episode_count=20,
                 ):
        self._agent = agent
        self._env = env

        if normalizer_path is not None:
            if os.path.exists(normalizer_path):
                with open(normalizer_path, 'rb') as f:
                    self._norm = pickle.load(f)
            else:
                self._norm = Normalizer(env)
                with open(normalizer_path, 'wb') as f:
                    pickle.dump(self._norm, f)
        else:
            self._norm = Normalizer(env)

        self._save_freq = save_freq
        # [path1, path2, ...]
        self._save_paths = save_paths
        self._load_paths = load_paths
        self._sw = SummaryWriter(log_dir)
        self._print_freq = print_freq
        self._test_episode_count = test_episode_count

        self._agent.set_model(self._load_paths)

        self._episode_count = 0
        self._train_step = 0

    def train(self, n):
        rewards = []
        i = 0
        prev_best = self.test(2 * self._test_episode_count)
        while self._train_step < n:
            rewards.append(self._train_one_episode())
            if len(rewards) > self._print_freq:
                train_r, test_r = np.mean(rewards), self.test(self._test_episode_count)
                print('trainstep: {}, train: {}, test: {}'.format(
                    self._train_step, train_r, test_r))
                self._sw.add_scalar('train', train_r, self._train_step)
                self._sw.add_scalar('test', test_r, self._train_step)
                rewards = []

            if self._save_paths is not None and (i + 1) % self._save_freq == 0:
                current = self.test(2 * self._test_episode_count)
                if current > prev_best:
                    print('save models, prev best = {}, current = {}'.format(prev_best, current))
                    self._agent.save_model(self._save_paths)
                    prev_best = current
            i += 1

    def _train_one_episode(self):
        s = self._norm.transform(self._env.reset())
        d = False
        total_reward = 0
        while not d:
            a = self._agent.choose_action_with_exploration(s)
            s_, r, d, i = self._env.step(a)
            s_ = self._norm.transform(s_)
            self._agent.train_one_step(s, a, s_, r, d)
            s = s_
            total_reward += r
            self._train_step += 1
        self._episode_count += 1
        return total_reward

    def test(self, n):
        return np.mean([self._test_one_episode() for _ in range(n)])

    def _test_one_episode(self):
        s = self._norm.transform(self._env.reset())
        d = False
        total_reward = 0
        while not d:
            a = self._agent.choose_action(s)
            s_, r, d, i = self._env.step(a)
            s_ = self._norm.transform(s_)
            s = s_
            total_reward += r
        self._episode_count += 1
        return total_reward


class DistributedTrainer(Trainer):

    def __init__(self,
                 agents,
                 env,
                 parameter_share=False,
                 state_transformer=lambda x, i: x,
                 reward_distributor=lambda s, r, i: r,
                 save_freq=100,
                 save_pathlists=None,
                 load_pathlists=None,
                 normalizer_path=None,
                 log_dir='./logs',
                 print_freq=20,
                 test_episode_count=20,
                 ):
        self._agents = agents
        self._env = env

        if normalizer_path is not None:
            if os.path.exists(normalizer_path):
                with open(normalizer_path, 'rb') as f:
                    self._norm = pickle.load(f)
            else:
                self._norm = Normalizer(env)
                with open(normalizer_path, 'wb') as f:
                    pickle.dump(self._norm, f)

        else:
            self._norm = Normalizer(env)

        self._st = state_transformer
        self._rd = reward_distributor
        self._save_freq = save_freq
        # [pathlist1, pathlist2, ...]
        self._save_pathlists = save_pathlists
        self._load_pathlists = load_pathlists
        self._sw = SummaryWriter(log_dir)
        self._print_freq = print_freq
        self._test_episode_count = test_episode_count

        for i, agent in enumerate(self._agents):
            agent.set_model(parameter_share, self._load_pathlists[i] if self._load_pathlists else None)

        self._episode_count = 0
        self._train_step = 0

    def train(self, n):
        rewards = []
        i = 0
        prev_best = self.test(2 * self._test_episode_count)
        while self._train_step < n:
            rewards.append(self._train_one_episode())
            if len(rewards) > self._print_freq:
                train_r, test_r = np.mean(rewards), self.test(self._test_episode_count)
                print('trainstep: {}, train: {}, test: {}'.format(self._train_step, train_r, test_r))
                self._sw.add_scalar('train', train_r, self._train_step)
                self._sw.add_scalar('test', test_r, self._train_step)
                rewards = []

            if self._save_pathlists is not None and (i + 1) % self._save_freq == 0:
                current = self.test(2 * self._test_episode_count)
                if current > prev_best:
                    print('save models, prev best = {}, current = {}'.format(prev_best, current))
                    for i, a in enumerate(self._agents):
                        a.save_model(self._save_pathlists[i])
                    prev_best = current
            i += 1

    def _train_one_episode(self):
        s = self._norm.transform(self._env.reset())
        d = False
        total_reward = 0
        while not d:
            a = self._choose_action_with_exploration(s)
            s_, r, d, i = self._env.step(a.reshape(-1))
            s_ = self._norm.transform(s_)
            for i, agent in enumerate(self._agents):
                agent.train_one_step(self._st(s, i), a[i],  self._st(s_, i), self._rd(s_, r, i), d)
            s = s_
            total_reward += r

            self._train_step += 1
        self._episode_count += 1
        return total_reward

    def test(self, n):
        return np.mean([self._test_one_episode() for _ in range(n)])

    def _test_one_episode(self):
        s = self._norm.transform(self._env.reset())
        d = False
        total_reward = 0
        while not d:
            a = self._choose_action(s)
            s_, r, d, i = self._env.step(a.reshape(-1))
            s_ = self._norm.transform(s_)
            s = s_
            total_reward += r
        self._episode_count += 1
        return total_reward

    def _choose_action_with_exploration(self, state):
        return np.array([a.choose_action_with_exploration(self._st(state, i)) for i, a in enumerate(self._agents)])

    def _choose_action(self, state):
        return np.array([a.choose_action(self._st(state, i)) for i, a in enumerate(self._agents)])
