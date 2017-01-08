import logging
import random

import gym
import numpy as np
from gym import wrappers
from six.moves import range

np.random.seed(9999)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

"""
References:
    Sutton, Barto (2017). Reinforcement Learning: An Introduction. MIT Press, Cambridge, MA.
"""


class DQN(object):
    def __init__(self, n_episodes=500, gamma=0.99, batch_size=32, epsilon=1., decay=0.005, min_epsilon=0.1,
                 memory_limit=500):
        """Deep Q learning implementation.

        Parameters
        ----------

        min_epsilon : float
            Minimal value for epsilon.
        epsilon : float
            ε-greedy value.
        decay : float
            Epsilon decay rate.
        memory_limit : int
            Limit of experience replay memory.

        """

        self.memory_limit = memory_limit
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.decay = decay

    def init_environment(self, name='CartPole-v0', monitor=False):
        self.env = gym.make(name)
        if monitor:
            self.env = wrappers.Monitor(self.env, name, force=True, video_callable=False)

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        # Experience replay
        self.replay = []

    def init_model(self, model):
        self.model = model(self.n_actions, self.batch_size)

    def train(self, render=False):
        max_reward = 0

        for ep in range(self.n_episodes):
            state = self.env.reset()

            total_reward = 0

            while True:
                if render:
                    self.env.render()

                if np.random.rand() <= self.epsilon:
                    # Exploration
                    action = np.random.randint(self.n_actions)
                else:
                    # Exploitation
                    action = np.argmax(self.model.predict(state[np.newaxis, :])[0])

                # Run one timestep of the environment
                new_state, reward, done, _ = self.env.step(action)
                self.replay.append([state, action, reward, new_state, done])

                # Sample batch from experience replay
                batch_size = min(len(self.replay), self.batch_size)
                batch = random.sample(self.replay, batch_size)

                X = np.zeros((batch_size, self.n_states))
                y = np.zeros((batch_size, self.n_actions))

                states = np.array([b[0] for b in batch])
                new_states = np.array([b[3] for b in batch])

                Q = self.model.predict(states)
                new_Q = self.model.predict(new_states)

                # Construct training data
                for i in range(batch_size):
                    state_r, action_r, reward_r, new_state_r, done_r = batch[i]
                    target = Q[i]

                    if done_r:
                        target[action_r] = reward_r
                    else:
                        target[action_r] = reward_r + self.gamma * np.amax(new_Q[i])

                    X[i, :] = state_r
                    y[i, :] = target

                # Train deep learning model
                self.model.fit(X, y)

                total_reward += reward
                state = new_state

                if done:
                    # Exit from current episode
                    break

            # Remove old entries from replay memory
            if len(self.replay) > self.memory_limit:
                self.replay.pop(0)

            self.epsilon = self.min_epsilon + (1.0 - self.min_epsilon) * np.exp(-self.decay * ep)

            max_reward = max(max_reward, total_reward)
            logger.info('Episode: %s, reward %s,  epsilon %s, max reward %s' % (ep, total_reward,
                                                                                self.epsilon, max_reward))
        logging.info('Training finished.')

    def play(self, episodes):
        for i in range(episodes):
            state = self.env.reset()
            total_reward = 0

            while True:
                self.env.render()
                action = np.argmax(self.model.predict(state[np.newaxis, :])[0])
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    break
            logger.info('Episode: %s, reward %s' % (i, total_reward))
