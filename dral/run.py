import sys
import numpy as np

from dral.data_manipulation.dataset_manager import init_dm
from dral.data_manipulation.loader import CONFIG
from dral.environment.rl_env import QueryEnv, MonitorWrapper
from dral.models import ConvNet
from dral.utils import load_model

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN


def train_rl_model(model, timesteps, path_to_save):
    model.learn(total_timesteps=timesteps)
    model.save(path_to_save)


def init_and_train_rl_model(timesteps):
    CONF = CONFIG
    dm, y_oracle = init_dm(
        x_path=CONF['data']['x_path'],
        y_path=CONF['data']['y_path'],
        img_size=CONF['img_size'],
        n_train=CONF['n_train'],
        n_eval=CONF['n_eval'],
        n_test=CONF['n_test']
    )
    cnn = load_model('data/cnn_rps_model.pt')
    env = QueryEnv(dm, cnn, CONF)
    env = MonitorWrapper(env, autolog=True)
    model = DQN(MlpPolicy, env, verbose=1)
    train_rl_model(model, timesteps, 'data/rl_rps.pth')


def load_dqn_model(path):
    return DQN.load(path)


def main():
    CONF = CONFIG
    dm, y_oracle = init_dm(
        x_path=CONF['data']['x_path'],
        y_path=CONF['data']['y_path'],
        img_size=CONF['img_size'],
        n_train=CONF['n_train'],
        n_eval=CONF['n_eval'],
        n_test=CONF['n_test']
    )
    print(dm)

    cnn = ConvNet()
    env = QueryEnv(dm, cnn, CONF)
    env = MonitorWrapper(env, autolog=True)

    model = load_dqn_model('data/rl_rps.pth')

    n_queries = 10
    n_epochs_cnn = 5
    for k in range(n_queries):
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

        query_indicies = env.get_query_indicies()
        dm.label_samples(query_indicies, y_oracle[query_indicies])
        y_oracle = np.delete(y_oracle, query_indicies, axis=0)
        print(dm)

        cnn.fit(*dm.train.get_xy(), n_epochs_cnn)
        cnn.evaluate(*dm.eval.get_xy())

    cnn.evaluate(*dm.test.get_xy())


if __name__ == '__main__':
    # unpickling the model requires access to models.py
    sys.path.insert(0, 'dral')
    main()
    # init_and_train_rl_model(35000)
