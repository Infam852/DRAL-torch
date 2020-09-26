import numpy as np
import os

import cv2
import torch

from dral.data_manipulation.loader import DataLoader, Image
from dral.config.config_manager import ConfigManager
from dral.utils import check_dtype
from dral.logger import Logger
from dral.errors import fail_if_len_mismatch
from dral.utils import show_img, show_grid_imgs

LOG = Logger.get_logger()


class DatasetsManager:
    def __init__(self, cm, imgs):
        """
        Samples passed should be preprocessed, especially their
        dimension should be expanded
        """
        self.cm = cm
        self.BASEPATH = self.cm.get_dataset_path()
        self.DELIMITER = self.cm.get_name_class_delimiter()
        self.unl, self.labelled = self.split_images(imgs)
        self.train = ImagesStorage([])
        self.eval = ImagesStorage([])
        self.test = ImagesStorage([])

    def _get_imgs_with_labels(self):
        """Iterate over all files in the BASEPATH and get all paths
        of png or jpg files.

        Returns:
            list: List of images paths
        """
        all_images = []
        for filename in os.listdir(self.BASEPATH):
            name, sufix = filename.rsplit('.')
            if sufix not in ['jpg', 'png']:
                continue

            if filename.count(self.DELIMITER) != 1:
                LOG.error(f'Wrong filename ({filename}), there must'
                          f'be exactly one ({self.DELIMITER}) character')
                raise Exception('Filename must contain exactly one delimiter')
            all_images.append(filename)
        return all_images

    def split_images(self, imgs):
        unl = []
        labelled = []
        for img in imgs:
            if img.y == self.cm.get_unknown_label():
                unl.append(img)
            else:
                labelled.append(img)
        return ImagesStorage(unl), ImagesStorage(labelled)

    def label_samples(self, idxs, labels):
        fail_if_len_mismatch(idxs, labels)

        imgs = self.unl.pop(idxs)
        ImagesStorage.set_labels(imgs, labels)
        self.train.append(imgs)

    def __str__(self):
        msg = """
        Number of unlabelled samples: {}
        Number of labelled samples: {}
        Number of evaluation samples: {}
        Number of test samples: {}
        """.format(len(self.unl), len(self.train),
                   len(self.eval), len(self.test))
        return msg

    def __repr__(self):
        return '(unl: {}, train: {}, eval: {}, test: {})'.format(
            len(self.unl), len(self.train), len(self.eval), len(self.test))


class ImagesStorage:
    def __init__(self, imgs):
        if not all(isinstance(img, Image) for img in imgs):
            raise ValueError('All elements in imgs list have to be'
                             'an instance of Image class')
        self.imgs = imgs

    def __getitem__(self, idx):
        return self.imgs[idx]

    def append(self, imgs):
        self.imgs.extend(imgs)

    def remove(self, idxs):
        for idx in sorted(idxs, reverse=True):
            del self.imgs[idx]

    def pop(self, idxs):
        imgs = []
        for idx in sorted(idxs, reverse=True):
            imgs.append(self.imgs.pop(idx))
        return imgs

    @staticmethod
    def set_labels(imgs, labels):
        fail_if_len_mismatch(imgs, labels)
        for imgs, label in zip(imgs, labels):
            imgs.y = labels

    def sample(self, n):
        pass

    def get(self, idxs=-1):
        check_dtype(idxs, int, list, np.ndarray)

        if isinstance(idxs, int) and idxs < 0:
            return self.imgs

        if isinstance(idxs, int):
            return self.imgs[idxs]

        return [self.imgs[idx] for idx in idxs]

    def get_x(self, idxs=-1):
        check_dtype(idxs, int, list, np.ndarray)

        if isinstance(idxs, int) and idxs < 0:
            return self.imgs

        if isinstance(idxs, int):
            return self.imgs[idxs].x

        return [self.imgs[idx].x for idx in idxs]

    def get_y(self, idxs=-1):
        check_dtype(idxs, int, list, np.ndarray)

        if isinstance(idxs, int) and idxs < 0:
            return self.imgs

        if isinstance(idxs, int):
            return self.imgs[idxs].y

        return [self.imgs[idx].y for idx in idxs]

    def get_path(self, idxs=-1):
        check_dtype(idxs, int, list, np.ndarray)

        if isinstance(idxs, int) and idxs < 0:
            return self.imgs

        if isinstance(idxs, int):
            return self.imgs[idxs].path

        return [self.imgs[idx].path for idx in idxs]

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    cm = ConfigManager('testset')
    imgs = DataLoader.load_images(cm.get_dataset_path())

    dm = DatasetsManager(cm, imgs)
    labels = [0, 1, 1, 0]
    dm.label_samples([0, 1, 2, 3], labels)
    img = DataLoader.load_image(dm.unl.get_path(110))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # cv2 uses BGR format
    show_img(img)
