from collections import namedtuple
import numpy as np

import gym
from gym import spaces
import torch

from dral.utils import tensor_to_numpy_1d, LOG, round_to3
from dral.config import LABEL_MAPPING, CONFIG


class QueryEnv(gym.Env):
    QUERY_LABEL = 0
    DO_NOT_QUERY = 1

    def __init__(self, dm, model, config):
        super(QueryEnv, self).__init__()
        self.dm = dm
        self.model = model
        self._queries = 0
        self._counter = 0
        self.query_indicies = []

        self.IMG_SIZE = config['img_size']
        self.MAX_QUERIES = config['max_queries']
        self.REWARD_MULT = config['reward_multiplier']
        self.REWARD_THR = 0  # calculate_reward needs it
        self.MAX_REWARD = self._get_max_reward(len(CONFIG['labels']))
        self.REWARD_THR = config['reward_treshold'] * self.MAX_REWARD
        self.QUERY_PUNISH = config['query_punishment']
        self.LEFT_QUERIES_PUNISH = config['left_queries_punishment']

        LOG.info(f'max reward: {self.MAX_REWARD}, '
                 f'reward threshold: {self.REWARD_THR}')

        n_action = 2
        self.action_space = spaces.Discrete(n_action)
        self.observation_space = spaces.Box(    # binary classification
            low=0, high=1, shape=(len(CONFIG['labels']),), dtype=np.float32)

    def reset(self):
        """Reset all environment variables. Should be called at
        the beginning of the epoch.

        Returns:
            list: initial observation
        """
        self._queries = 0
        self._counter = 0
        self.entropy = 0
        self.query_indicies = []
        self.entropy_arr = []
        self.reward_arr = []
        self._state = tensor_to_numpy_1d(self._get_state_vector())
        LOG.info('reset environment')
        return self._state

    def step(self, action):
        reward = 0

        if action == self.QUERY_LABEL:
            self._queries += 1
            self.query_indicies.append(self._counter)
            reward -= self.QUERY_PUNISH
            reward += self._calculate_reward(self._state)
        elif action == self.DO_NOT_QUERY:
            pass
        else:
            self.LOG.error(f'step got action ({action}), but expected action '
                           f'with spec: {self.action_space}')
            raise ValueError(f'Received invalid action={action} which'
                             'is not part of the action space')
        self._counter += 1

        self._state = tensor_to_numpy_1d(self._get_state_vector())
        done = True if self._queries >= self.MAX_QUERIES or \
            self._counter >= len(self.dm.unl) else False

        if done and self._queries < self.MAX_QUERIES:
            reward -= self.LEFT_QUERIES_PUNISH

        info = {}
        return self._state, reward, done, info

    def _get_state_vector(self):
        """ State vector has shape (F+O,) where F indicates length of
        feature vector extracted from CNN Flatten Layer and O indicates
        length of softmax output layer"""
        try:
            image = self.dm.unl.get_x(self._counter)
            out = self.model(image.view(-1, 1, self.IMG_SIZE, self.IMG_SIZE))
            return out
        except Exception as ex:
            print('Aux model prediction error', ex, ex.__traceback__.tb_lineno)
            return torch.zeros(self.observation_space.shape)

    def _calculate_reward(self, predictions):
        """Calculate reward based on array of predictions

        Args:
            predictions (list): List of predictions, sum of all the
            elements should equal 1

        Returns:
            int: reward, proportional to prediction uncertainty
        """
        entropy = self.calculate_entropy(predictions)
        reward = entropy*self.REWARD_MULT  # !TODO
        if reward < self.REWARD_THR:
            reward = 0
        return reward

    def calculate_entropy(self, dist):
        """ Given numpy array, calculate entropy. Use e for the log base """
        return -sum([p*np.log(p) for p in dist if p > 0])

    def _get_max_reward(self, n):
        totally_random_dist = np.full(n, 1/n)
        return self._calculate_reward(totally_random_dist)

    def render(self):
        print(f"""
        Entropy: {self.entropy}
        Probabilities
        {LABEL_MAPPING[0]}: {self._state[0]}
        {LABEL_MAPPING[1]}: {self._state[1]}
        {LABEL_MAPPING[2]}: {self._state[2]}
        """)

    def get_counter(self):
        return self._counter

    def get_query_indicies(self):
        if len(self.query_indicies) < self.MAX_QUERIES:
            LOG.warning(
                f'Called get_query_indicies but number of queries '
                f'({self.query_indicies}) is less than maximum number '
                f'({self.MAX_QUERIES})')
        return self.query_indicies


class MonitorWrapper(gym.Wrapper):
    def __init__(self, env, autolog=False):
        super(MonitorWrapper, self).__init__(env)
        self.reward_arr = []
        self.Reward = namedtuple('Reward', ['idx', 'value'])
        self.observations = []
        self.autolog = True

    def reset(self):
        obs = self.env.reset()
        self.reward_arr = []
        self.observations = []
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if action == self.env.QUERY_LABEL:
            self.reward_arr.append(self.Reward(
                self.env.get_counter(),
                round_to3(reward)))

        self.observations.append(obs)

        if done:
            extras = {
                'episode_length': self.env.get_counter(),
                'total_reward': self._calculate_total_reward(),
                'rewards': self.reward_arr,
                'observations': self.observations,
            }
            info.update(extras)

            if self.autolog:
                LOG.info(f"episode length: {info['episode_length']}, "
                         f"total reward: {info['total_reward']}")

        return obs, reward, done, info

    def _calculate_total_reward(self):
        return round_to3(sum([reward.value for reward in self.reward_arr]))
