import sys

from dral.data_manipulation.dataset_manager import init_dm
from dral.data_manipulation.loader import CONFIG
from dral.environment.rl_env import QueryEnv, MonitorWrapper
from dral.utils import load_model

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN


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

    cnn = load_model()
    env = QueryEnv(dm, cnn, CONF)
    env = MonitorWrapper(env, autolog=True)


if __name__ == '__main__':
    # unpickling the model requires access to models.py
    sys.path.insert(0, 'dral')

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

    cnn = load_model()
    env = QueryEnv(dm, cnn, CONF)
    env = MonitorWrapper(env, autolog=True)

    model = DQN(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=20000)
    model.save('simple_dral')

    # obs = env.reset()
    # for k in range(200):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()
    # check_env(env, warn=True)

    # obs = env.reset()

    # print(env.observation_space)
    # print(env.action_space)
    # print(env.action_space.sample())

    # n_steps = 30
    # for step in range(n_steps):
    #     print("Step {}".format(step + 1))
    #     obs, reward, done, info = env.step(QueryEnv.QUERY_LABEL)
    #     print('obs=', obs, 'reward=', reward, 'done=', done)
    #     if done:
    #         print("Goal reached!", "reward=", reward)
    #         print(env.query_indicies)
    #         break
