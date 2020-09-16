import os
import numpy as np

from skimage import io
from skimage.color import rgb2gray
import skimage.transform
from tqdm import tqdm

from dral.config import CONFIG
from dral.utils import LOG


class DataLoader:
    def __init__(self, config):
        self.IMG_SIZE = config.get('img_size')
        self.LABELS = config.get('labels')  # {path: label}
        self.NCLS = len(self.LABELS)
        self.balance_counter = {label: 0 for label in self.LABELS.values()}
        self.x = []
        self.y = []
        self.loaded = False

    def _reset(self):
        self.loaded = False
        self.x = []
        self.y = []
        self.balance_counter = {label: 0 for label in self.LABELS.values()}

    def load_training_data(self, enable_normalization=True):
        self._reset()
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    # as_gray converts image to float64 within [0, 1] range
                    img = io.imread(path, as_gray=True)
                    img = skimage.transform.resize(
                        img, (self.IMG_SIZE, self.IMG_SIZE))
                    if not enable_normalization:
                        img = (img*256).astype('uint8')

                    self.x.append(np.array(img))
                    # one-hot vector
                    self.y.append(np.eye(self.NCLS)[self.LABELS[label]])

                    self.balance_counter[self.LABELS[label]] += 1
                except Exception as e:
                    LOG.warning(f'Error while processing the data: {e}')

        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.loaded = True

    def shuffle(self):
        assert len(self.x) == len(self.y)
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.y = self.y[p]

    def save(self, path, name):
        np.save(f'{path}/x_{name}.npy', self.x)
        np.save(f'{path}/y_{name}.npy', self.y)

    def print_balance_counter(self):
        if self.loaded:
            for label in self.LABELS.values():
                print(f'Number of {label} imgs: {self.balance_counter[label]}')
        else:
            raise Exception('Firstly you have to load data!')

    @staticmethod
    def load(path):
        return np.load(path, allow_pickle=True)


if __name__ == '__main__':
    dl = DataLoader(CONFIG)
    dl.load_training_data(enable_normalization=True)
    dl.shuffle()
    dl.print_balance_counter()
    dl.save('data', 'cats_dogs_64_norm')
