from keras.models import Sequential
from keras.layers import Dense
from keras_dqn.agent import Agent
from keras import backend as K

import numpy as np


def test_agent_train():
    batch_size = 1
    input_dim = 5
    num_action = 10
    models = []
    for i in range(2):
        model = Sequential()
        model.add(Dense(num_action, input_dim=input_dim))
        model.compile("sgd", "mse")
        models.append(model)

    agent = Agent(model=models[0], target_model=models[1])

    actions = np.random.randint(0, num_action, batch_size)
    reward = np.random.randint(-1, 2, batch_size)
    terminal = np.random.randint(0, 2, batch_size)
    prev_state = np.random.randn(batch_size, input_dim)
    next_state = np.random.randn(batch_size, input_dim)

    # test that it can run
    agent.train_on_batch(prev_state, actions, reward, next_state, terminal)

    # test if sync
    agent.update_target()
    assert np.sum(K.get_value(models[0].layers[0].W) -
                  K.get_value(models[1].layers[0].W)) == 0


if __name__ == "__main__":
    test_agent_train()
