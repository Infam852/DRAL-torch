import os
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from dral.config.config_manager import ConfigManager
from dral.utils import LOG


def get_number_of_files(path, recursively=True):  # !TODO optimize
    """Count number of files in a specified directory. If recursively is
    True then count also files in all of the subdirectories.
    If path is not directory path then FiliNotFoundException is thrown.

    Args:
        path (str): path to directory
        recursively (bool, optional): if True search recursively.
        Defaults to True.

    Returns:
        int: number of files in the given directory
    """
    n_files = 0
    for f in os.listdir(path):
        fpath = os.path.join(path, f)
        if os.path.isdir(fpath):
            n_files += get_number_of_files(fpath)
        else:
            n_files += 1
    return n_files


class DataLoader:
    def __init__(self, cm, max_imgs=None):
        """Initialize data loader with specification saved in CONFIG dictionary

        Args:
            cm (obj): ConfigManager instance that provide interface to
            configuration file
            max_imgs (int): max number of images per one class to be loaded,
            if unspecified then all images will be loaded
        """
        # !TODO label_format
        self.cm = cm
        self.IMG_SIZE = cm.get_img_size()
        self.MAX_IMGS = max_imgs if max_imgs \
            else get_number_of_files(cm.get_imgs_path())
        self.N_LABELS = len(cm.get_labels())
        self.DTYPE = np.float32 if cm.do_normalization() else np.uint8

        self._reset()

    def _reset(self):
        self._x = np.zeros(
            (self.MAX_IMGS, self.IMG_SIZE, self.IMG_SIZE), dtype=self.DTYPE)
        self._y = np.zeros(
            (self.MAX_IMGS, self.N_LABELS), dtype=self.DTYPE)
        self.balance_counter = {label: 0 for label in self.cm.get_labels()}
        self.n_exceptions_while_loading = 0

    def load_raw(self):
        """ Load images and apply preprocessing on them based on config file
        """
        paths = [os.path.join(self.cm.get_imgs_path(), label)
                 for label in self.cm.get_labels()]
        self._fail_if_path_is_not_dir(paths)
        self._reset()
        for label, class_path in enumerate(paths):
            LOG.info(f'Start loading images from path: {class_path}')
            for k, f in tqdm(enumerate(os.listdir(class_path))):
                if self.MAX_IMGS and k >= self.MAX_IMGS:
                    LOG.info(f'Maximum number of load attemps is reached '
                             f'({k}) for class {self.cm.get_labels()[label]}')
                    break
                try:
                    path = os.path.join(class_path, f)

                    color_mode = cv2.IMREAD_GRAYSCALE \
                        if self.cm.do_grayscale() else cv2.IMREAD_COLOR
                    img = cv2.imread(path, flags=color_mode)
                    img = self._rescale(img)

                    if self.cm.do_normalization():
                        img = cv2.normalize(img,  img, 0, 1, cv2.NORM_MINMAX,
                                            dtype=cv2.CV_32F)

                    if self.cm.do_centering():
                        img -= img.mean()

                    if self.cm.do_standarization():
                        img /= img.std()

                    self._x[k] = img
                    # one-hot vector
                    self._y[k] = np.eye(self.N_LABELS)[label]
                    self.balance_counter[self.cm.get_labels()[label]] += 1

                except Exception as e:
                    LOG.warning(
                        f'Error while loading image from path {path}: {e}')
                    self.n_exceptions_while_loading += 1

    def shuffle(self):
        """ Shuffle all of the loaded images. Does not check if
        images were loaded.
        """
        assert len(self._x) == len(self._y)
        p = np.random.permutation(len(self._x))
        self._x = self._x[p]
        self._y = self._y[p]
        LOG.info('Data was shuffled')

    def save(self, force=False):
        """Save loaded images to the specified in config path. If path does
        not exist and force is set to false then exception is raised. Otherwise
        create required directories and then save the data.

        Args:
            force (bool, optional): If set then create required directories
            if they do not exis. Defaults to False.

        Raises:
            FileNotFoundError: If force is False and any of the directories
            does not exist then this expetion will be raised.
        """
        if self.cm.do_shuffle():
            self.shuffle()

        self._clean()

        name = self.cm.get_config_name()
        path = self.cm.get_save_path()
        if not force and not os.path.isdir(path):
            raise FileNotFoundError(
                f'Path ({path}) was not found and force was set to False')
        # create directories if they do not exist
        Path(path).mkdir(parents=True, exist_ok=True)

        np.save(f'{path}/x_{name}.npy', self._x)
        np.save(f'{path}/y_{name}.npy', self._y)
        LOG.info(f'Data was saved in directory {path}')

    def print_balance_counter(self):
        if sum(self.balance_counter.values()):
            for label in self.cm.get_labels():
                print(f'Number of {label} imgs: {self.balance_counter[label]}')
        else:
            raise Exception('Firstly you have to load data!')

    def _fail_if_path_is_not_dir(self, paths):
        """ Raise exception if there is a non directory path

        Args:
            paths (list): list of paths

        Raises:
            Exception:
        """
        for path in paths:
            if not os.path.isdir(path):
                raise Exception(f'Path {path} is not directory '
                                'or even does not exist')  # !TODO change type of exception

    def _rescale(self, img):   # !TODO optimalization
        """Rescale image to the specified in config file size.
        If rescale_with_crop flag is specified then the image is firstly
        resized that the shorter side has desired size then it is cropped.

        Args:
            img (array): array representation of an image

        Returns:
            array: array representation of rescaled image
        """
        if self.cm.do_rescale_with_crop():
            (h, w) = img.shape
            if h > w:
                ratio = h / w
                # dims: (w, h)
                dims = (self.IMG_SIZE, int(self.IMG_SIZE*ratio))
                img = cv2.resize(img, dims)
                crop_idx = (dims[1] - self.IMG_SIZE) // 2
                img = img[crop_idx:self.IMG_SIZE+crop_idx, :]
            else:
                ratio = w / h
                dims = (int(self.IMG_SIZE*ratio), self.IMG_SIZE)
                img = cv2.resize(img, dims)
                crop_idx = (dims[0] - self.IMG_SIZE) // 2
                img = img[:, crop_idx:self.IMG_SIZE+crop_idx]
        else:
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        return img

    @staticmethod
    def load(path):
        """Load pickled numpy array from specified path

        Args:
            path (file-like obj): the file to read from

        Returns:
            array: loaded numpy array
        """
        return np.load(path, allow_pickle=True)

    def _clean(self):
        """Remove trailing empty elements in _x and _y arrays. They will
        occur if any image will not be loaded properly. Number of exceptions
        durning loading is tracking by n_exceptions_while_loading variable.
        """
        self._x = self._x[:-self.n_exceptions_while_loading]
        self._y = self._y[:-self.n_exceptions_while_loading]
        LOG.info(f'Remove last {self.n_exceptions_while_loading} images')


if __name__ == '__main__':
<<<<<<< HEAD
    cm = ConfigManager('cats_dogs_96')
    dl = DataLoader(cm)
    dl.load_raw()
    dl.print_balance_counter()
    dl.save(force=False)
=======
    dl = DataLoader(CONFIG)
    dl.load_training_data(enable_normalization=True)
    dl.shuffle()
    dl.print_balance_counter()
    dl.save('data', 'cats_dogs_64_norm')
>>>>>>> ad4b236802fd6e6e9a886e813a5f826ca4bba4fc
