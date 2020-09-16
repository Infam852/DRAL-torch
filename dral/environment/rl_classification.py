import sys
import numpy as np
from numpy.random import default_rng
import time

import gym
from gym import spaces

from stable_baselines3 import DQN
from stable_baselines3.dqn import CnnPolicy

# from stable_baselines.common.env_checker import check_env
# from stable_baselines.deepq.policies import CnnPolicy, MlpPolicy
# from stable_baselines import DQN, ACER, ACKTR

from dral.config import CONFIG
from dral.utils import LOG, show_img, find_most_uncertain, show_grid_imgs, \
                       evaluate
from dral.data_manipulation.dataset_manager import init_dm
from dral.environment.al_env import MonitorWrapper

"""
Maybe sort images and end episode if level of uncertainity exceeds threshold
"""


def init_and_train_rl_classification_model(
        timesteps, path='data/rl_rps.pth', save=True, n=2000):
    dm, y_oracle = init_dm(CONFIG)
    env = ClassificationEnv(dm, y_oracle)
    # env = MonitorWrapper(env, autolog=True)
    model = DQN(CnnPolicy, env, verbose=1)
    idxs = list(range(n))
    dm.label_samples(idxs, y_oracle[idxs])
    model.learn(total_timesteps=timesteps)
    if save:
        model.save(path)
    env.enable_evaluating(True)
    evaluate(model, env)
    env.enable_evaluating(False)
    return model


def label_samples(dm, y_oracle, n, random=False):
    if random:
        rng = default_rng()
        idxs = rng.choice(len(dm.unl), size=n, replace=False)
    else:
        idxs = list(range(n))
    dm.label_samples(idxs, y_oracle[idxs])
    y_oracle = np.delete(y_oracle, idxs, axis=0)
    return y_oracle


class ClassificationEnv(gym.Env):
    # !TODO use config
    CLASS_ROCK = 0
    CLASS_PAPER = 1
    CLASS_SCI = 2

    def __init__(self, dm, y_oracle):
        super(ClassificationEnv, self).__init__()
        self.dm = dm
        self.storage = self.dm.train
        self.img_size = CONFIG['img_size']
        self.correct_shape = (self.img_size, self.img_size, 1)
        self._counter = 0
        self.stats = {'good': 0, 'wrong': 0}

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            shape=(self.img_size, self.img_size, 1),
            low=0, high=255, dtype=np.uint8)

        LOG.info('Environment initialized')

    def reset(self):
        self._counter = 0
        self.stats = {'good': 0, 'wrong': 0}
        self._state = self.storage.get_x(
            self._counter).view(*self.correct_shape).numpy()
        return self._state

    def step(self, action):
        reward = 0

        predicted_label = action
        true_label = np.argmax(self.storage.get_y(self._counter))

        LOG.debug(f'Predicted label: {predicted_label},'
                  f'true_label: {true_label}')

        if predicted_label == true_label:
            self.stats['good'] += 1
            reward += 1
        else:
            self.stats['wrong'] += 1

        self._counter += 1
        info = {}
        try:
            self._state = self.storage.get_x(self._counter)\
                              .view(*self.correct_shape).numpy()
            done = False
        except IndexError:
            self._state = np.zeros(self.correct_shape,
                                   dtype=np.uint8)
            done = True
            info = self.stats
            LOG.info(f'End of epoch, results: {self.stats}')

        return self._state, reward, done, info

    def enable_evaluating(self, en):  # !TODO context manager
        self.evaluating = en
        self.storage = self.dm.test if en else self.dm.train

    def get_counter(self):
        return self._counter


if __name__ == '__main__':
    new = 1
    load = 0
    test = 0
    if new + load + test != 1:
        raise Exception('Initialize new, train or load a model')

    dm, y_oracle = init_dm(CONFIG)
    print(dm)
    env = ClassificationEnv(dm, y_oracle)

    sys.path.insert(0, 'dral')
    if new:
        model = DQN(CnnPolicy, env, verbose=1, learning_rate=2e-4,
                    gamma=0.98, batch_size=32, learning_starts=3000)
    if load:
        model = DQN.load("data/rl_query_rps.pth")
    if test:
        model = init_and_train_rl_classification_model(
            timesteps=100000, path='data/rl_query_dogs_cats.pth')

    # show_grid_imgs(dm.test.get_x(list(range(9))), dm.test.get_y(list(range(9))), (3, 3))
    n_episodes = 5
    for k in range(n_episodes):

        # label images and add them to train storage
        y_oracle = label_samples(dm, y_oracle, n=100, random=True)
        dm.train.shuffle()
        print(dm)

        model.learn(total_timesteps=6000, log_interval=30)

        # evaluation
        env.enable_evaluating(True)
        evaluate(model, env)
        env.enable_evaluating(False)
