import os
import numpy as np
from skimage import io
import skimage.transform
from tqdm import tqdm


CONFIG = {
    'img_size': 32,
    'labels': {
        'data/PetImages/Cat': 0,
        'data/PetImages/Dog': 1
    },
    'n_train': 8000,
    'n_eval': 2000,
    'n_test': 3000,
    'data': {
        'x_path': 'data/x_cats_dogs_skimage.npy',
        'y_path': 'data/y_cats_dogs_skimage.npy'
    },
    'max_reward': 5,
    'max_queries': 20,
    'query_punishment': 0.5,
    'left_queries_punishment': 5,
    'reward_treshold': 0.92,
    'reward_multiplier': 4,
}

LABEL_MAPPING = {
    0: 'Cat',
    1: 'Dog'
}


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

    def load_training_data(self):
        self._reset()
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)):

                if 'jpg' in f:
                    try:
                        path = os.path.join(label, f)
                        # as_gray converts image to float64 within [0, 1] range
                        img = io.imread(path, as_gray=True)
                        img = skimage.transform.resize(
                            img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.x.append(np.array(img))
                        # one-hot vector
                        self.y.append(np.eye(self.NCLS)[self.LABELS[label]])

                        self.balance_counter[self.LABELS[label]] += 1
                    except Exception as e:
                        print(f'Error while processing the data: {e}')
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
    dl.load_training_data()
    dl.shuffle()
    dl.print_balance_counter()
    dl.save('data', 'cats_dogs_skimage')
