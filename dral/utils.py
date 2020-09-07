import matplotlib.pyplot as plt
import numpy as np

import torch

from dral.logger import get_logger

LOG = get_logger()


def check_dtype(val, *args):
    if type(val) not in args:
        raise ValueError(f'{type(val)} is not one of {args}')


def tensor_to_numpy_1d(tensor):
    return tensor.flatten().detach().numpy().astype(np.float32)


def show_img(img, label=None, figsize=(4, 4)):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    plt.title(f"label: {label}" if label is not None else "")
    plt.axis("off")
    plt.show()


def show_grid_imgs(imgs, labels, grid_size):
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
