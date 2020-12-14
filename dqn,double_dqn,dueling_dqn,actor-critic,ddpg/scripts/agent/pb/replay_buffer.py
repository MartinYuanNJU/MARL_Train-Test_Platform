import numpy as np


class ReplayBuffer:

    def __init__(self, cap, state_shape, action_dim):
        self._state = np.zeros((cap, state_shape))
        self._action = np.zeros((cap, action_dim), dtype=np.float)
        self._reward = np.zeros((cap,), dtype=np.float)
        self._next_state = np.zeros((cap, state_shape))
        self._done = np.zeros((cap,), dtype=np.bool)
        self._index = 0
        self._cap = cap
        self._full = False

    def add(self, s, a, r, s_, d):
        self._state[self._index] = s
        self._action[self._index] = a
        self._reward[self._index] = r
        self._next_state[self._index] = s_
        self._done[self._index] = d
        self._index += 1
        if self._index == self._cap:
            self._full = True
            self._index = 0

    def sample(self, batch):
        if self._full:
            indices = np.random.randint(0, self._cap, (batch,))
        else:
            indices = np.random.randint(0, self._index, (batch,))
        return (self._state[indices], self._action[indices], self._reward[indices],
                self._next_state[indices], self._done[indices])
