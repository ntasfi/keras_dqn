import numpy as np

class Memory():

    def __init__(self, max_size, observation_shape):
        """
        max_size: int
            The max amount of samples we want to hold until we start 
            deleting the oldest sample

        observation_shape: tuple
            A tuple describing a single observations shape from the environment. 
            Does NOT include the batch size.
            This module will try to be agnostic to the size pass as long as its 
            an image.
            eg. (84, 84, 3) or (3, 84, 84) or (84,84)

        """

    def __len__(self):
        #return the length of the memory
        pass

    def _adjust_memory(self):
        """
            this will take care of adjusting the memory size so it states within the limits
            we can be lazy with this and only adjust memory when we train batches
            we would overflow a little bit not a big deal, better than calling it every time
            we add an entry.

            The max_size would be a little leaky
        """
        pass

    def add(self, observation, action, reward, terminal):
        """
            Check is observation is the same as observation_shape
            Add to the last spot in memory (as a tuple of the above)
            Adjust memory size if its over the max_size

            obs will be stored as int8
            action will be stored as int8
            reward will be stored as float32
            terminal will be stored as int8 
        """
        pass

    def get_current_state(self, state_length):
        """
            Returns the most recent state for the agent to choose an action from.
            The return shape will be equal to (state_length, observation_shape)
        """
        pass

    def random_batch(self, batch_size, state_length):
        """
            state_length is how many single observations constitue a single 'state'
            eg. DQN usually uses 4 observations to represent a single state.

            batch_size is the size of the batch we return back. Will have shape:
            (batch_size, state_length, observation_shape)
            eg. (32, 4, 84, 84) or (32, 4, 3, 16, 16) or (32, 4, 16, 16, 3) etc.

        """
        pass
