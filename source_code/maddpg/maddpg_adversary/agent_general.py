class Agent:

    def choose_action(self, state):
        raise NotImplementedError

    def choose_action_with_exploration(self, state, train_step):
        raise NotImplementedError

    def train_one_step(self, s, a, r, s_, d):
        raise NotImplementedError
