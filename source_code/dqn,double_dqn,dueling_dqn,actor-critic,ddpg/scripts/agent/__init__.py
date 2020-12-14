class Agent:

    networks = None

    def choose_action(self, state):
        raise NotImplementedError

    def choose_action_with_exploration(self, state):
        raise NotImplementedError

    def train_one_step(self, s, a, s_, r, d):
        raise NotImplementedError

    def set_model(self, parameter_share=False, load_paths=None):
        raise NotImplementedError

    def save_model(self, paths):
        raise NotImplementedError


from .deepq import *
from .pb import *
from .common import *


__all__ = ['Agent',
           'DQNAgent', 'DDQNAgent', 'PERDQNAgent', 'DuelingDQNAgent',
           'DDPGAgent', 'ACAgent',
           'DQNArgs', 'DDQNArgs', 'PERDQNArgs', 'DuelingDQNArgs',
           'DDPGArgs', 'ACArgs']
