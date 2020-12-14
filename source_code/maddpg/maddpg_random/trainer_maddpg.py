import numpy as np
from trainer import Trainer
from normalizer_maddpg import Normalizer
from torch.utils.tensorboard import SummaryWriter


class Trainer_MADDPG(Trainer):
    def __init__(self,
                 agent,
                 env,
                 log_dir='./logs',
                 print_freq=10,
                 test_episode_count=20,
                 normalizerround=10000,
                 action_bound=5.0
                 ):
        self.agent = agent
        self.env = env
        self.log_dir = log_dir
        self.print_freq = print_freq
        self.test_episode_count = test_episode_count
        self.normalizerround = normalizerround
        self.action_bound = action_bound
        self.normalizer = Normalizer(env, self.normalizerround)
        self.sw = SummaryWriter(self.log_dir)
        self.episode_count = 0
        self.train_step = 0

    def train_one_episode(self):
        state = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            action = self.agent.choose_action_with_exploration(self.normalizer.transform(state), self.train_step)
            state_, reward, done, _ = self.env.step(action * self.action_bound)
            self.agent.train_one_step(
                self.normalizer.transform(state),
                action,
                reward,
                self.normalizer.transform(state_),
                done)
            state = state_
            total_reward += reward
            self.train_step += 1
        self.episode_count += 1
        return total_reward

    def train(self, n):
        rewards = []
        now_best_rew = 0
        stat = []
        for i in range(n):
            if self.train_step >= 1000000:
                break
            rew = self.train_one_episode()
            rewards.append(rew)
            if len(rewards) > self.print_freq:
                train_r, test_r = np.mean(rewards), self.test(self.test_episode_count)
                # print('trainstep: {}, train: {}, test: {}'.format(self.train_step, train_r, test_r))
                self.sw.add_scalar('train', train_r, self.train_step)
                self.sw.add_scalar('test', test_r, self.train_step)
                rewards = []
            if rew > now_best_rew:
                print('new best reward at episode {} and trainstep {}: reward is {}'.format(i+1, self.train_step+1, rew))
                now_best_rew = rew
            if (i+1) % 128 == 0:
                test_reward = self.test(10)
                print('episode:', (i+1), ', reward:', test_reward)
                stat.append(test_reward)
        fp = open('./logs/maddpgresult.txt', 'w')
        for element in stat:
            fp.write(str(element))
            fp.write('\n')
        fp.close()

    def test_one_episode(self):
        state = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            action = self.agent.choose_action(self.normalizer.transform(state))
            state_, reward, done, _ = self.env.step(action * self.action_bound)
            state = state_
            total_reward += reward
        self.episode_count += 1
        return total_reward

    def test(self, n=10):
        r = [self.test_one_episode() for _ in range(n)]
        r_mean = np.mean(r)
        return r_mean
