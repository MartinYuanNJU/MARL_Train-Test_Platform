
class DQNArgs:

    def __init__(self,
                 state_dim,
                 discrete_action_space,
                 lr=5e-4,
                 buffer_size=1000000,
                 total_trainsteps=500000,
                 epsilon=0.15,
                 final_epsilon=0.02,
                 update_freq=1,
                 update_count=1,
                 batch_size=50,
                 steps_before_training=1000,
                 gamma=0.95,
                 tau=100
                 ):
        self.buffer_size = buffer_size
        self.lr = lr
        self.update_freq = update_freq
        self.update_count = update_count
        self.tau = tau
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.batch = batch_size
        self.gamma = gamma
        self.steps_before_training = steps_before_training
        self.total_trainsteps = total_trainsteps
        self.discrete_action_space = discrete_action_space
        self.state_dim = state_dim


class DDQNArgs:

    def __init__(self,
                 state_dim,
                 discrete_action_space,
                 lr=5e-4,
                 buffer_size=1000000,
                 total_trainsteps=500000,
                 epsilon=0.15,
                 final_epsilon=0.02,
                 update_freq=1,
                 update_count=1,
                 batch_size=50,
                 steps_before_training=1000,
                 gamma=0.95,
                 tau=100
                 ):
        self.buffer_size = buffer_size
        self.lr = lr
        self.update_freq = update_freq
        self.update_count = update_count
        self.tau = tau
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.batch = batch_size
        self.gamma = gamma
        self.steps_before_training = steps_before_training
        self.total_trainsteps = total_trainsteps
        self.discrete_action_space = discrete_action_space
        self.state_dim = state_dim


class PERDQNArgs:

    def __init__(self,
                 state_dim,
                 discrete_action_space,
                 lr=5e-4,
                 buffer_size=1000000,
                 total_trainsteps=500000,
                 epsilon=0.15,
                 final_epsilon=0.02,
                 update_freq=1,
                 update_count=1,
                 batch_size=50,
                 steps_before_training=1000,
                 gamma=0.95,
                 tau=100,
                 alpha=1.0,
                 beta=0.0,
                 beta_inc=1e-5,
                 exp_epsilon=1e-3
                 ):
        self.buffer_size = buffer_size
        self.lr = lr
        self.update_freq = update_freq
        self.update_count = update_count
        self.tau = tau
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.batch = batch_size
        self.gamma = gamma
        self.steps_before_training = steps_before_training
        self.total_trainsteps = total_trainsteps
        self.discrete_action_space = discrete_action_space
        self.state_dim = state_dim

        self.alpha = alpha
        self.beta = beta
        self.beta_inc = beta_inc
        self.exp_epsilon = exp_epsilon


class DuelingDQNArgs:

    def __init__(self,
                 state_dim,
                 discrete_action_space,
                 lr=5e-4,
                 buffer_size=1000000,
                 total_trainsteps=500000,
                 epsilon=0.15,
                 final_epsilon=0.02,
                 update_freq=1,
                 update_count=1,
                 batch_size=50,
                 steps_before_training=1000,
                 gamma=0.95,
                 tau=100
                 ):
        self.buffer_size = buffer_size
        self.lr = lr
        self.update_freq = update_freq
        self.update_count = update_count
        self.tau = tau
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.batch = batch_size
        self.gamma = gamma
        self.steps_before_training = steps_before_training
        self.total_trainsteps = total_trainsteps
        self.discrete_action_space = discrete_action_space
        self.state_dim = state_dim


class DDPGArgs:

    def __init__(self,
                 state_dim,
                 action_dim,
                 action_scale=1.0,
                 final_action_scale=0.1,
                 scale_decay_factor=0.9999,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 buffer_size=1000000,
                 update_freq=1,
                 update_count=1,
                 batch_size=50,
                 steps_before_training=1000,
                 gamma=0.95
                 ):
        self.buffer_size = buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.update_freq = update_freq
        self.update_count = update_count
        self.batch = batch_size
        self.gamma = gamma
        self.steps_before_training = steps_before_training
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.final_action_scale = final_action_scale
        self.scale_decay_factor = scale_decay_factor


class ACArgs:

    def __init__(self,
                 state_dim,
                 action_dim,
                 action_scale=1.0,
                 final_action_scale=0.1,
                 scale_decay_factor=0.9999,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 update_freq=1,
                 update_count=1,
                 steps_before_training=1000,
                 gamma=0.95
                 ):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.update_freq = update_freq
        self.update_count = update_count
        self.gamma = gamma
        self.steps_before_training = steps_before_training
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.final_action_scale = final_action_scale
        self.scale_decay_factor = scale_decay_factor
