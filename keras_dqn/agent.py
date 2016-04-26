import numpy as np
from keras.utils.np_utils import to_categorical


class Agent(object):
    def __init__(self, model, target_model=None, target_swap=None):
        """
        Expected values in kwargs:

        "update_target_every" : number of frames skipped before syncing the target with model
        "update_model_every" : number of frames skipped before gradient step
        """
        self.model = model
        if target_model is None:
            self.target_model = model
        else:
            self.target_model = target_model
            self.target_swap = "__swap.hdf5"
            print("Using __swap.hdf5 as swap file for the target model")

    @property
    def num_actions(self):
        return self.model.output_shape[-1]

    def train_on_batch(self, state, action, reward, next_state, terminal, gamma=.99, accum_mode="mean"):
        q = self.model.predict(state)
        next_max_q = np.argmax(self.target_model.predict(next_state), axis=-1)
        idx = to_categorical(action, self.num_actions)  # transform actions to one-hot indices

        # target = reward + gamma * max Q(s', a')
        target = np.copy(q)  # we will use indexing next, avoid aliasing
        target[idx == 1] = next_max_q  # WARN: should only adapt actions taken
        # TODO: is this faster? target = q * idx + (1-idx) * next_max_q[:, np.newaxis].repeat(self.numactions, 1)

        # @ntasfi, regardless of what is target (1-terminal) zeros it out that is why I thought we don't have to store terminal states
        target = reward + (1-terminal) * gamma * target

        # gradient step
        if accum_mode == "mean":
            self.model.train_on_batch(state, target, sample_weight=None)

    def update_target(self):
        if self.target_swap is not None:
            self.model.save_weights(self.target_swap, True)
            self.target_model.load_weights(self.target_swap)

    def fit(env, batch_size, epochs):
        """Learn to play
        Extra desired args:
        frames_per_epoch
        update_target_every (in frames)
        update_model_every (in frames)
        eps, eps_min, frames_to_min_eps

        """
        pass
