import logging

from mla.neuralnet import NeuralNet
from mla.neuralnet.layers import Activation, Dense
from mla.neuralnet.optimizers import Adam
from mla.rl.dqn import DQN

logging.basicConfig(level=logging.CRITICAL)


def mlp_model(n_actions, batch_size=64):
    model = NeuralNet(
        layers=[
            Dense(32),
            Activation('relu'),
            Dense(n_actions),
        ],
        loss='mse',
        optimizer=Adam(),
        metric='mse',
        batch_size=batch_size,
        max_epochs=1,
        verbose=False,

    )
    return model


model = DQN(n_episodes=2500, batch_size=64)
model.init_environment('CartPole-v0')
model.init_model(mlp_model)

try:
    # Train the model
    # It can take from 300 to 2500 episodes to solve CartPole-v0 problem due to randomness of environment.
    # You can stop training process using Ctrl+C signal
    # Read more about this problem: https://gym.openai.com/envs/CartPole-v0
    model.train(render=False)
except:
    pass
# Render trained model
model.play(episodes=100)
