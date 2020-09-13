import sys
import numpy as np
import time

import gym
from gym import spaces

# from stable_baselines.common.env_checker import check_env
# from stable_baselines.deepq.policies import CnnPolicy, MlpPolicy
# from stable_baselines import DQN, ACER, ACKTR


from stable_baselines3 import DQN
from stable_baselines3.dqn import CnnPolicy

from dral.config import CONFIG
from dral.utils import LOG, show_img, find_most_uncertain, show_grid_imgs, evaluate
from dral.data_manipulation.dataset_manager import init_dm
from dral.environment.al_env import MonitorWrapper

"""
Maybe sort images and end episode if level of uncertainity exceeds threshold
"""


def init_and_train_rl_classification_model(
        timesteps, path='data/rl_rps.pth', save=True):
    dm, y_oracle = init_dm(CONFIG)
    env = ClassificationEnv(dm, y_oracle)
    # env = MonitorWrapper(env, autolog=True)
    model = DQN(CnnPolicy, env, verbose=1)
    idxs = list(range(2000))
    dm.label_samples(idxs, y_oracle[idxs])
    model.learn(total_timesteps=timesteps)
    if save:
        model.save(path)
    env.enable_evaluating(True)
    evaluate(model, env)
    env.enable_evaluating(False)
    return model


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
        self._state = self.storage.get_x(self._counter).view(self.img_size, self.img_size, 1).numpy()
        return self._state

    def step(self, action):
        reward = 0

        predicted_label = action

        true_label = np.argmax(self.storage.get_y(self._counter))

        LOG.debug(f'Predicted label: {predicted_label},'
                  f'true_label: {true_label}')

        if predicted_label == true_label:
            # maybe add to list index of this img and do not query in the next epoch
            self.stats['good'] += 1
            reward += 1
        else:
            self.stats['wrong'] += 1

        self._counter += 1
        info = {}
        try:
            self._state = self.storage.get_x(self._counter)\
                              .view(self.img_size, self.img_size, 1).numpy()
            done = False
        except IndexError:
            self._state = np.zeros((self.img_size, self.img_size, 1),
                                   dtype=np.uint8)
            done = True
            info = self.stats
            LOG.info(f'End of epoch, results: {self.stats}')

        return self._state, reward, done, info

    def enable_evaluating(self, en):  # !TODO context manager
        self.evaluating = en
        self.storage = self.dm.test if en else self.dm.train


if __name__ == '__main__':
    new = 1
    load = 0
    test = 0
    if new + load + test != 1:
        raise Exception('Initialize new, train or load a model')

    dm, y_oracle = init_dm(CONFIG)
    print(dm)
    env = ClassificationEnv(dm, y_oracle)
    # check_env(env, warn=True)

    # for k in range(10):
    #     show_img(obs, label=env.y_oracle[env._counter])
    #     obs, reward, done, info = env.step(2)
    #     print(reward, env.stats)
    # model = DQN(CnnPolicy, env, verbose=1)
    # dm, y_oracle = init_dm(CONFIG)
    # env = ClassificationEnv(dm, y_oracle)
    # env = MonitorWrapper(env, autolog=True)
    # model = DQN(CnnPolicy, env, verbose=1)
    # model.learn(total_timesteps=5000)

    # while True:
    #     action, _states = model.predict(obs, deterministic=False)
    #     print(model.action_probability(obs))
    #     print(action, _states)

    #     obs, reward, done, info = env.step(action)
    #     print(obs.shape)
    #     print(reward, done, info)
    #     input()
    # Load the trained agent

    sys.path.insert(0, 'dral')
    if new:
        model = DQN(CnnPolicy, env, verbose=1)
    if load:
        model = DQN.load("data/rl_query_rps.pth")
    if test:
        model = init_and_train_rl_classification_model(
            timesteps=14000, path='data/rl_query_dogs_cats.pth')
    # print(dm.test.get_x(0))
    for k in range(10):
        # most_uncertain = find_most_uncertain(model, dm.unl.get_x(), 20)
        # print(most_uncertain)

        # get indiecies of the most uncertain images
        # idxs = [el[0] for el in most_uncertain]
        # print(f'Most uncertain indicies: {idxs}')

        # label images
        idxs = list(range(50))
        dm.label_samples(idxs, y_oracle[idxs])
        y_oracle = np.delete(y_oracle, idxs, axis=0)
        print(dm)
        # to_show = list(range(k*20, k*20+16))
        # show_grid_imgs(dm.train.get_x(to_show), dm.train.get_y(to_show), (4, 4))
        model.learn(total_timesteps=len(dm.train)*10)

        # evaluation
        env.enable_evaluating(True)
        evaluate(model, env)
        env.enable_evaluating(False)

    # print(y_oracle.shape)
    # to_show = list(range(16))

    # start = time.time()
    # obs = env.reset()
    # for k in range(1001):
    #     action, _states = model.predict(obs)
    #     # print(model.action_probability(obs))
    #     # print(action, _states)

    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         print('Done')
    #         obs = env.reset()

    #     # print(reward, done, info)
    #     # input()    # obs = env.reset()
    # end = time.time()
    # print(f'Time: {end - start}')
