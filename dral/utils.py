import matplotlib.pyplot as plt
import numpy as np

import torch

from dral.logger import Logger

LOG = Logger.get_logger()


def check_dtype(val, *args):
    if type(val) not in args:
        raise ValueError(f'{type(val)} is not one of {args}')


def tensor_to_numpy_1d(tensor):
    return tensor.flatten().detach().numpy().astype(np.float32)


def show_img(img, label=None, figsize=(4, 4), cmap='gray'):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap)
    plt.title(f"label: {label}" if label is not None else "")
    plt.axis("off")
    plt.show()


def show_grid_imgs(imgs, labels, grid_size):
    if len(imgs) != len(labels) or len(imgs) != grid_size[0]*grid_size[1]:
        raise ValueError(
            'Number of images, labels and grid size have to be equal.')

    rows, cols = grid_size
    f, axarr = plt.subplots(nrows=rows, ncols=cols)
    for idx, image in enumerate(imgs):
        row = idx // cols
        col = idx % cols
        axarr[row, col].axis("off")
        axarr[row, col].set_title(labels[idx])
        axarr[row, col].imshow(image, cmap='gray')

    plt.show()


def load_model(path='data/cnn_model.pt'):
    return torch.load(path)


def round_to3(val):
    return np.around(val, decimals=3)


def round_to2(val):
    return np.around(val, decimals=2)


def find_most_uncertain(model, imgs, n):
    """Find the N the most uncertain images for RL model based
    on entropy calculation.

    Args:
        model (object): stable baselines RL model
        imgs (list): list of images
        n (int): number of the most uncertain images to be returned

    Returns:
        list: list with entries (idx, entropy), sorted by
        entropy (descending order)
    """
    entropies = []
    for idx, img in enumerate(imgs):
        probabilities = model.action_probability(img.view(64, 64, 1))
        entropies.append((idx, calculate_entropy(probabilities)))
    entropies.sort(key=lambda x: x[1])
    return entropies[-n:]


def calculate_entropy(dist):
    """ Given numpy array, calculate entropy. Use e for the log base """
    return -sum([p*np.log(p) for p in dist if p > 0])


def evaluate(model, env):
    obs = env.reset()
    done = False
    rewards = []
    while not done:
        action, _states = model.predict(obs)
        # print(model.action_probability(obs))
        # print(action, _states)

        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            print('Done')
            obs = env.reset()
    acc = (sum(rewards)/len(rewards))*100
    print(f'Total reward: {sum(rewards)}, accuracy: {acc}%')
