import numpy as np
from keras.utils.np_utils import to_categorical


class Agent(object):
    def __init__(self, model, target_model=None, update_model_every=1,
                 update_target_every=1, min_n_frames=1, frames_per_epoch=1,
                 eps=.1, eps_rate=1, min_eps=0, test_eps=0, rng=np.random):
        """
        Parameters:
        -----------
        model: keras model (agent)
        target_model: keras model (desired)
        update_target_every : number of frames skipped before syncing the target with model
        update_model_every" : number of frames skipped before gradient step
        min_n_frames: number of frames stored to memory before start training
        frames_per_epoch : how long is an epoch?
        eps: probability of random action (epsilon greedy)
        eps_rate: epsilon annealing rate
        min_eps: min epsilon
        test_eps: probability of random action during test
        rng_seed
        """
        self.rng = rng

        self.model = model
        if target_model is None:
            self.target_model = model
        else:
            self.target_model = target_model
            self.target_swap = True  # external target flag
            print("Using target model update.")
        self.update_model_every = update_model_every
        self.update_target_every = update_target_every

        # memory info
        self.min_n_frames = min_n_frames

        # epsilon greedy info
        self.eps = eps
        self.eps_rate = eps_rate
        self.min_eps = min_eps
        self.test_eps = test_eps

    @property
    def n_actions(self):
        return self.model.output_shape[-1]

    @property
    def input_shape(self):
        return self.model.input_shape

    def get_action(self, observation, train=False):
        rnd = self.rng.rand()
        rnd_action = self.rng.randint(0, self.n_actions)
        if train and rnd < self.eps:
            return rnd_action
        elif train is False and rnd < self.test_eps:
            return rnd_action
        else:
            # this code is meant to work only if doing a single action!
            q = self.model.predict(observation)
            # random tie breaker
            max_inds = np.argwhere(q == np.max(q))
            return self.rng.choice(max_inds.reshape(-1,))

    def _train_on_batch(self, state, action, reward, next_state, terminal, gamma=.99, accum_mode="mean"):
        q = self.model.predict(state)
        next_max_q = np.max(self.target_model.predict(next_state), axis=-1)
        idx = to_categorical(action, self.n_actions)  # transform actions to one-hot indices

        # target = reward + gamma * max Q(s', a')
        target = np.copy(q)  # we will use indexing next, avoid aliasing
        target[idx == 1] = reward + (1-terminal) * gamma * next_max_q  # WARN: should only adapt actions taken
        # TODO: is this faster? target = q * idx + (1-idx) * next_max_q[:, np.newaxis].repeat(self.numactions, 1)

        # gradient step
        self.model.train_on_batch(state, target, sample_weight=None)

    def update_target(self):
        if self.target_swap:
            weights = self.model.get_weights()
            self.target_model.set_weights(weights)

    def fit(env, batch_size, epochs):
        """Learn to play
        Extra desired args:
        eps, eps_min, frames_to_min_eps

        """
        pass

    def play(env, epochs):
        """Follow learned policy and play game
        """
        pass
