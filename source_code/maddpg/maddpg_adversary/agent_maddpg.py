import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agent_general import Agent


class Args_maddpg:

    def __init__(self,
                 exp_cap=1000000,
                 gamma=0.95,
                 scale=1.0,
                 batch=128,
                 update_cnt=20,
                 update_interval=200,
                 actor_lr=1e-5,
                 critic_lr=1e-4,
                 state_dim=0,
                 action_dim=0,
                 exp_state_dim=0,
                 agent_num=2,
                 car_num=2,
                 traditional_update=True):
        self.exp_cap = exp_cap
        self.gamma = gamma
        self.scale = scale
        self.batch = batch
        self.update_cnt = update_cnt
        self.update_interval = update_interval
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.exp_state_dim = exp_state_dim
        self.agent_num = agent_num
        self.car_num = car_num
        self.traditional_update = traditional_update


class Experience_Pool_maddpg:

    def __init__(self, cap, state_dim, action_dim, reward_dim=1):
        self._states = np.zeros((cap, state_dim))
        self._actions = np.zeros((cap, action_dim))
        self._rewards = np.zeros((cap, reward_dim))
        self._states_next = np.zeros((cap, state_dim))
        self._done = np.zeros((cap,), dtype=np.bool)
        self._last_index = 0
        self._full = False
        self._cap = cap
        self._random_state = np.random.RandomState(random.randint(10000000, 99999999))
        self._range = np.array(range(self._cap))

    def add(self, s, a, r, s_, done):
        self._states[self._last_index] = s
        self._actions[self._last_index] = a
        self._rewards[self._last_index] = r
        self._states_next[self._last_index] = s_
        self._done[self._last_index] = done
        self._last_index += 1
        if self._last_index == self._cap:
            self._full = True
            self._last_index = 0

    def sample(self, n):
        if not self._full:
            indices = self._random_state.choice(self._range[:self._last_index], (n,))
        else:
            indices = self._random_state.choice(self._range, (n,))
        return self._states[indices], self._actions[indices], self._rewards[indices], self._states_next[indices]


class ActorNet(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self._fc0 = nn.Linear(state_dim, 128)
        self._fc1 = nn.Linear(128, 64)
        self._fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self._fc0(x))
        x = F.relu(self._fc1(x))
        x = self._fc2(x)
        return F.tanh(x)


class CriticNet(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self._fc0 = nn.Linear(state_dim + action_dim, 256)
        self._fc1 = nn.Linear(256, 128)
        self._fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self._fc0(x))
        x = F.relu(self._fc1(x))
        x = self._fc2(x)
        return x


class Agent_MADDPG(Agent):

    def __init__(self, args_):
        self.args = args_
        self.agents_actor_network = [ActorNet(self.args.state_dim,
                                              self.args.action_dim // self.args.agent_num)
                                     for i in range(self.args.agent_num)]
        self.agents_target_actor_network = [ActorNet(self.args.state_dim,
                                                     self.args.action_dim // self.args.agent_num)
                                            for i in range(self.args.agent_num)]
        for i in range(self.args.agent_num):
            self.agents_target_actor_network[i].load_state_dict(self.agents_actor_network[i].state_dict())

        self.agents_actor_optim = [optim.Adam(self.agents_actor_network[i].parameters(),
                                              lr=self.args.actor_lr)
                                   for i in range(self.args.agent_num)]

        self.agents_critic_network = [CriticNet(self.args.state_dim,
                                                self.args.action_dim)
                                      for i in range(self.args.agent_num)]
        self.agents_target_critic_network = [CriticNet(self.args.state_dim,
                                                       self.args.action_dim)
                                             for i in range(self.args.agent_num)]
        for i in range(self.args.agent_num):
            self.agents_target_critic_network[i].load_state_dict(self.agents_critic_network[i].state_dict())

        self.agents_critic_optim = [optim.Adam(self.agents_critic_network[i].parameters(),
                                               lr=self.args.critic_lr)
                                    for i in range(self.args.agent_num)]

        self.agents_critic_lossfn = [torch.nn.MSELoss() for i in range(self.args.agent_num)]

        self.exp = Experience_Pool_maddpg(self.args.exp_cap, self.args.exp_state_dim, self.args.action_dim)
        self.update_cnt = 0
        self.train_ep = 0
        self.update_tick = self.args.update_interval
        self.stat = []

    def state_2_observation(self, state, index):
        # return state
        obs = np.hstack([
            state[:, 2 * index:2 * (index + 1)],
            state[:, 2 * self.args.agent_num + 2 * index:2 * self.args.agent_num + 2 * (index + 1)],
            state[:, 0:2 * index],
            state[:, 2 * (index + 1):2 * self.args.agent_num],
            state[:, 2 * self.args.agent_num:2 * self.args.agent_num + 2 * index],
            state[:, 2 * self.args.agent_num + 2 * (index + 1):4 * self.args.agent_num],
            state[:, 4 * self.args.agent_num:]])
        return obs

    def choose_action_with_exploration(self, state, train_step):
        if (train_step + 1) % 100 == 0:
            self.args.scale *= 0.9999
            if self.args.scale < 0.1:
                self.args.scale = 0.1
        action = self.choose_action(state)
        noise = np.random.normal(0, self.args.scale, (self.args.action_dim,))
        action = np.clip(action + noise, -1, 1)  # clip action between [-1, 1]
        return action

    def choose_action(self, state):
        tempactionlist = []
        with torch.no_grad():
            for i in range(self.args.agent_num):
                nowstatetemp = self.state_2_observation(np.array([state]), i)[0]
                nowstate = torch.from_numpy(nowstatetemp).float()
                nowaction = self.agents_actor_network[i](nowstate)
                tempactionlist.append(nowaction)
        actionlist = []
        for tempaction in tempactionlist:
            oneaction = tempaction.detach().numpy()
            actionlist.append(oneaction)
        action = np.array([i for i in actionlist]).reshape((-1,))
        return action

    def soft_copy_params(self):
        with torch.no_grad():
            for i in range(self.args.agent_num):
                for t, s in zip(self.agents_target_actor_network[i].parameters(),
                                self.agents_actor_network[i].parameters()):
                    t.copy_(0.95 * t.data + 0.05 * s.data)

                for t, s in zip(self.agents_target_critic_network[i].parameters(),
                                self.agents_critic_network[i].parameters()):
                    t.copy_(0.95 * t.data + 0.05 * s.data)

    def _update(self, index):
        states, actions, rewards, states_ = self.exp.sample(self.args.batch)
        # update critic
        with torch.no_grad():
            input_actions = []
            for i in range(self.args.agent_num):
                ob = self.state_2_observation(states_, i)
                a = self.agents_target_actor_network[i](torch.tensor(ob).float())
                input_actions.append(a)

            y = self.agents_target_critic_network[index](torch.cat((torch.tensor(states_), *input_actions), 1).float())
            y *= self.args.gamma
            y += rewards

        critic_output = self.agents_critic_network[index](torch.cat((torch.tensor(states), torch.tensor(actions)), 1).float())
        critic_loss = self.agents_critic_lossfn[index](critic_output, y.float())
        self.agents_critic_optim[index].zero_grad()
        critic_loss.backward()
        self.agents_critic_optim[index].step()

        # update actor
        actions_tensor = torch.tensor(np.copy(actions))
        self_observation = torch.tensor(self.state_2_observation(states, index)).float()
        actions_tensor[:, 2 * index: 2 * index + 2] = self.agents_actor_network[index](self_observation)
        q_val = self.agents_critic_network[index](torch.cat((torch.tensor(states), actions_tensor), 1).float())
        actor_loss = -q_val.mean()
        self.agents_actor_optim[index].zero_grad()
        actor_loss.backward()
        self.agents_actor_optim[index].step()

    def update(self):
        self.update_tick -= 1
        if self.update_tick != 0:
            return
        self.update_tick = self.args.update_interval
        for _ in range(self.args.update_cnt):
            for t in range(self.args.agent_num):
                self._update(t)

            self.soft_copy_params()
            self.update_cnt += 1

    def train_one_step(self, s, a, r, s_, d):
        self.exp.add(s, a, r, s_, d)
        self.update()
