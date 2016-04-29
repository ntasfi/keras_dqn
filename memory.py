import numpy as np

class Memory():

    def __init__(self, max_size, observation_size):
        """
        max_size: int
            The max amount of samples we want to hold until we start 
            deleting the oldest sample

        observation_size: tuple
            A tuple describing a single observations size from the environment. 
            Does NOT include the batch size.
            This module will try to be agnostic to the size pass as long as its 
            an image.
            eg. (84, 84, 3) or (3, 84, 84) or (84,84)

        """

    def _adjust_memory(self):
        """
            this will take care of adjusting the memory size so it states within the limits
            we can be lazy with this and only adjust memory when we train batches
            we would overflow a little bit not a big deal.
        """
        pass

    def add(self, obs, action, reward, terminal):
        """
            Check is obs is the same as observation_size
            Add to the last spot in memory (as a tuple of the above)
            Adjust memory size if its over the max_size

            obs will be stored as int8
            action will be stored as int8
            reward will be stored as float32
            terminal will be stored as int8 or bool (@eder what do you think)
        """
        pass

    def random_batch(self, batch_size, state_size):
        """
            state_size is how many single observations constitue a single 'state'
            eg. DQN usually uses 4 observations to represent a single state.

            batch_size is the size of the batch we return back. Will have shape:
            (batch_size, state_size, observation_size)
            eg. (32, 4, 84, 84) or (32, 4, 3, 16, 16) or (32, 4, 16, 16, 3) etc.

        """
        pass
