import os
import numpy as np
from pathlib import Path
from shutil import copyfile

import cv2
from tqdm import tqdm

from dral.config.config_manager import ConfigManager
from dral.utils import LOG, check_dtype, extract_name_from_path


def one_hot_to_numeric(one_hot):
    return np.argmax(one_hot, axis=0)


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


def move_and_rename(src_path, dst_path, start_idx):
    name = start_idx
    for f in os.listdir(src_path):
        full_src_path = os.path.join(src_path, f)

        if os.path.isfile(full_src_path):
            _, ext = f.rsplit('.', 1)
            filename = f'{name}.{ext}'
            full_dst_path = os.path.join(dst_path, filename)
            copyfile(full_src_path, full_dst_path)
            name += 1
    return name


class Image:
    IMG_PATTERN = '{name}_{label}.{ext}'

    def __init__(self, x, y, path, convert_to_numpy=False, relative=True):
        """Image representation

        Args:
            x (np.ndarray): representation of an image as 3d array, where third
            dimension refers to color channels

            y (np.uint8): numeric label

            path (str): path where png or jpg representation of an image is
            stored

            name (str): last part of the path e.g. /path/to/file -> file

            convert_to_numpy (bool, optional): if set to True then it
            is possible to pass array like object and it will be converted to
            numpy. Defaults to False.

            relative (bool, optional): if set to True then path will be cut so
            that it starts with static directory. This operation is necessary
            for proper loading image in flask. E.g. ./ Defaults to True.
        """
        if convert_to_numpy:
            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.uint8)

        if relative:
            idx = path.index('static')
            path = path[idx:]

        check_dtype(x, np.ndarray)
        check_dtype(y, np.uint8)

        self.x = x
        self.y = y
        self.path = path
        self.name = extract_name_from_path(path)

    def __repr__(self):
        return f'({self.x.shape}) ({self.y}) ({self.path})'

    def set_label(self, label):
        self.y = label

    def set_path(self, path):
        self.path = path
        self.name = extract_name_from_path(path)


class DataLoader:
    def __init__(self, cm, max_imgs=50000):
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
        self.N_LABELS = len(cm.get_label_names())
        self.MAX_IMGS = max_imgs
        self._reset()

    def _reset(self):
        color_channels = 1 if self.cm.do_grayscale() else 3
        dtype = np.float32 if cm.do_normalization() else np.uint8

        self._x = np.zeros(
            (self.MAX_IMGS, self.IMG_SIZE, self.IMG_SIZE, color_channels),
            dtype=dtype)
        self._y = np.zeros(self.MAX_IMGS, dtype=np.uint8)

        self.balance_counter = \
            {label: 0 for label in self.cm.get_label_names()}
        self.n_exceptions_while_loading = 0

    def load_raw(self):
        """ Load images and apply preprocessing on them based on config file
        """
        paths = list(self.cm.get_imgs_path_to_label().keys())
        labels = list(self.cm.get_imgs_path_to_label().values())
        self._fail_if_path_is_not_dir(paths)
        self._reset()
        self.MAX_IMGS_PER_CLASS = self.MAX_IMGS // len(paths)

        k = 0
        for ctr, (class_path, label) in enumerate(zip(paths, labels)):
            LOG.info(f'Start loading images from path: {class_path}...')
            print(len(os.listdir(class_path)))
            for f in tqdm(os.listdir(class_path)):
                if k >= self.MAX_IMGS_PER_CLASS:
                    LOG.info(
                        f'Maximum number of load attemps is reached ({k})'
                        f' for class {label}')
                    break
                try:
                    path = os.path.join(class_path, f)

                    color_mode = cv2.IMREAD_GRAYSCALE \
                        if self.cm.do_grayscale() else cv2.IMREAD_COLOR
                    img = cv2.imread(path, flags=color_mode)
                    if img is None:
                        raise OSError('Failed to read an image')
                    img = self._rescale(img)

                    if self.cm.do_normalization():
                        img = cv2.normalize(img,  img, 0, 1, cv2.NORM_MINMAX,
                                            dtype=cv2.CV_32F)

                    if self.cm.do_centering():
                        img -= img.mean()

                    if self.cm.do_standarization():
                        img /= img.std()

                    self._x[k] = img
                    self._y[k] = label

                    self.balance_counter[self.cm.get_label_name(ctr)] += 1
                    k += 1

                except OSError as e:
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

    def save(self, save_path=None, force=False,
             as_png=False, png_and_npy=False):
        """Save loaded images to the specified in config path (by default as
        numpy array). If path does not exist and force is set to false then
        exception is raised. Otherwise create required directories and then
        save the data.

        Args:
            save_path (str, optional): Path to save loaded data, if unspecified
            then use path saved in config file

            force (bool, optional): If set then create required directories
            if they do not exis. Defaults to False.

            as_png (bool, optional): If set to True then save images as png
            instead of numpy array

        Raises:
            FileNotFoundError: If force is False and any of the directories
            does not exist then this expetion will be raised.
        """
        if self.cm.do_shuffle():
            self.shuffle()

        self._clean()

        name = self.cm.get_config_name()
        path = save_path if save_path is not None else self.cm.get_save_path()
        path = os.path.join(path, name)
        if not force and not os.path.isdir(path):
            raise FileNotFoundError(
                f'Path ({path}) was not found and force was set to False')
        # create directories if they do not exist
        Path(path).mkdir(parents=True, exist_ok=True)

        if as_png:
            for counter, (img, label) in enumerate(zip(self._x, self._y)):
                filename = f'{counter}_{label}.png'
                img_path = os.path.join(path, filename)
                cv2.imwrite(img_path, img)
                LOG.debug(f'save image: {img_path}')
        else:
            np.save(f'{path}/x.npy', self._x)
            np.save(f'{path}/y.npy', self._y)
        LOG.info(f'Data was saved in directory {path}')

    def print_balance_counter(self):
        if sum(self.balance_counter.values()):
            for label in self.cm.get_label_names():
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
                                'or even does not exist'.format(path))  # !TODO change type of exception

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
            (h, w) = img.shape[:2]
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
    def load_npy(prefix, sufix):
        """Load pickled numpy array from specified path

        Args:
            prefix (file-like obj): prefix of the filepath

            sufix (file-like obj): filename

        Returns:
            array: loaded numpy array
        """
        path = os.path.join(prefix, sufix)
        return np.load(path, allow_pickle=True)

    @staticmethod
    def load_images(path, ext='png'):
        x = DataLoader.load_npy(path, 'x.npy')
        y = DataLoader.load_npy(path, 'y.npy')
        images = []
        for k in range(len(x)):
            img_path = os.path.join(
                path, Image.IMG_PATTERN.format(name=k, label=y[k], ext=ext))
            if not os.path.isfile(img_path):
                raise Exception(f'Path {img_path} does not exist')
            images.append(Image(x[k], y[k], img_path))
        LOG.info(f'{len(images)} images have been loaded')
        return images

    @staticmethod
    def load_image(path, color_mode=cv2.IMREAD_COLOR, convert_to_rgb=True):
        """Load image from specified path and converrt it to rgb if necessary

        Args:
            path (file-like obj): path from where image will be loaded

            color_mode (cv2 color_mode, optional): color mode.
            Defaults to cv2.IMREAD_COLOR.

            convert_to_rgb (bool, optional): if set to Treu then convert image
            to rgb, cv2 uses bgr by default. Defaults to True.

        Returns:
            np.ndarray: numpy representation of an image
        """
        img = cv2.imread(path, flags=color_mode)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if convert_to_rgb else img

    def _clean(self):
        """Remove trailing empty elements in _x and _y arrays. They will
        occur if any image will not be loaded properly. Number of exceptions
        durning loading is tracking by n_exceptions_while_loading variable.
        """
        LOG.info(f'Remove last {self.n_exceptions_while_loading} images...')
        if not self.n_exceptions_while_loading:
            return
        self._x = self._x[:-self.n_exceptions_while_loading]
        self._y = self._y[:-self.n_exceptions_while_loading]

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y


if __name__ == '__main__':
    cm = ConfigManager('cats_dogs_128')
    dl = DataLoader(cm, max_imgs=15000)
    dl.load_raw()
    dl.print_balance_counter()
    dl.save(force=True, as_png=True)

    cm.enable_npy_preprocessing(True)

    dl.load_raw()
    dl.print_balance_counter()
    dl.save(force=True)
    # idx = move_and_rename('/home/dawid/project/DRAL-torch/data/PetImages/Dog',
    #                       '/home/dawid/project/DRAL-torch/data/PetImages/Unknown',
    #                       12501)
    # print(idx)
