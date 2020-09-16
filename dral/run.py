import sys
import numpy as np

from dral.data_manipulation.dataset_manager import init_dm
from dral.data_manipulation.loader import CONFIG
from dral.environment.al_env import QueryEnv, MonitorWrapper
from dral.models import ConvNet
from dral.utils import load_model

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env


def train_rl_model(model, timesteps, path_to_save):
    model.learn(total_timesteps=timesteps)
    model.save(path_to_save)


def init_and_train_rl_model(timesteps, path='data/rl_rps.pth'):
    CONF = CONFIG
    dm, y_oracle = init_dm(CONFIG)

    cnn = load_model('data/cnn_rps_model.pt')
    env = QueryEnv(dm, cnn, CONF)
    env = MonitorWrapper(env, autolog=True)
    model = DQN(MlpPolicy, env, verbose=1)
    train_rl_model(model, timesteps, path)


def load_dqn_model(path):
    return DQN.load(path)


def main():
    dm, y_oracle = init_dm(CONFIG)
    print(dm)

    cnn = ConvNet(n_output=2)
    env = QueryEnv(dm, cnn, CONFIG)
    env = MonitorWrapper(env, autolog=True)
    check_env(env, warn=True)

    model = load_dqn_model('data/rl_rps.pth')

    n_queries = 10
    n_epochs_cnn = 8
    for k in range(n_queries):
        print(dm)
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

        query_indicies = env.get_query_indicies()
        dm.label_samples(query_indicies, y_oracle[query_indicies])
        y_oracle = np.delete(y_oracle, query_indicies, axis=0)

        cnn.fit(*dm.train.get_xy(), n_epochs_cnn, batch_size=32)
        cnn.evaluate(*dm.eval.get_xy())

    cnn.evaluate(*dm.test.get_xy())


if __name__ == '__main__':
    # unpickling the model requires access to models.py
    sys.path.insert(0, 'dral')
    # init_and_train_rl_model(10000)
    main()
