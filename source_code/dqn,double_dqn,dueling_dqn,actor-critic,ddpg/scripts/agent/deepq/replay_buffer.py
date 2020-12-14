import numpy as np


class ReplayBuffer:

    def __init__(self, cap, state_shape):
        self._state = np.zeros((cap, state_shape))
        self._action = np.zeros((cap,), dtype=np.long)
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


class SumTree:

    def __init__(self, cap):
        self.cap = cap
        self.tree = np.zeros(2 * cap - 1)

        self.data_pointer = 0

    def add(self, p):
        tree_idx = self.cap - 1 + self.data_pointer
        self.update(tree_idx, p)
        dp = self.data_pointer

        self.data_pointer += 1
        if self.data_pointer >= self.cap:
            self.data_pointer = 0
        return dp

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            lc_idx = 2 * parent_idx + 1
            rc_idx = lc_idx + 1
            if lc_idx >= len(self.tree):
                leaf_idx = parent_idx
                break

            if v <= self.tree[lc_idx]:
                parent_idx = lc_idx
            else:
                v -= self.tree[lc_idx]
                parent_idx = rc_idx

        data_index = leaf_idx - (self.cap - 1)
        return leaf_idx, self.tree[leaf_idx], data_index

    def total_p(self):
        return self.tree[0]


class PEReplayBuffer:

    def __init__(self, args, state_dim):
        self._cap = args.buffer_size
        self._alpha = args.alpha
        self._beta = args.beta
        self._beta_inc = args.beta_inc
        self._epsilon = args.exp_epsilon
        self.abs_err_upper = 1.0
        self.state_dim = state_dim

        self.tree = SumTree(self._cap)
        self.data = np.zeros(self._cap, dtype=object)
        self.size = 0

    def add(self, s, a, r, s_, done):
        max_p = np.max(self.tree.tree[self.tree.cap - 1:])
        max_p = self.abs_err_upper if max_p == 0 else max_p
        data_pointer = self.tree.add(max_p)
        self.data[data_pointer] = (s, a, r, s_, done)
        self.size = np.min([self._cap, self.size + 1])

    def sample(self, n):
        indices, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n,))
        s = np.zeros((n, self.state_dim))
        a = np.zeros((n,), dtype=np.long)
        r = np.zeros((n,), dtype=np.long)
        s_ = np.zeros((n, self.state_dim))
        d = np.zeros((n,), dtype=np.bool)
        pri_seg = self.tree.total_p() / n
        self._beta = np.min([1., self._beta + self._beta_inc])  # max = 1

        min_prob = np.min(self.tree.tree[self._cap - 1:self._cap - 1 + self.size]) / self.tree.total_p()
        for i in range(n):
            a_, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a_, b)
            idx, p, data_index = self.tree.get_leaf(v)
            prob = p / self.tree.total_p()
            ISWeights[i] = np.power(prob / min_prob, -self._beta)
            indices[i] = idx
            s[i], a[i], r[i], s_[i], d[i] = self.data[data_index]
        return indices, (s, a, r, s_, d), ISWeights

    def batch_update(self, tree_indices, td_errors):
        abs_errors = np.abs(td_errors)
        clipped_errors = np.clip(abs_errors, self._epsilon, self.abs_err_upper)
        ps = np.power(clipped_errors, self._alpha)
        for i, p in zip(tree_indices, ps):
            self.tree.update(i, p)
