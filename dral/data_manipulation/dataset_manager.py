import numpy as np
import os
import random

import torch

from dral.data_manipulation.loader import DataLoader, Image
from dral.config.config_manager import ConfigManager
from dral.utils import check_dtype, extract_name_from_path,\
        extract_names_from_paths
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
        self.train = ImagesStorage(self.labelled)
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
        self.labelled.append(imgs)

    def label_samples_mapping(self, label_idx_mapping):
        all_idxs = []
        all_labels = []
        for label, idxs in label_idx_mapping.items():
            all_idxs.extend(idxs)
            all_labels.extend([label]*len(idxs))

        self.label_samples(all_idxs, all_labels)

    def __str__(self):
        msg = """
        Number of unlabelled samples: {}
        Number of labelled samples: {}
        Number of training samples: {}
        Number of evaluation samples: {}
        Number of test samples: {}
        """.format(len(self.unl), len(self.labelled), len(self.train),
                   len(self.eval), len(self.test))
        return msg

    def __repr__(self):
        return '(unl: {}, train: {}, eval: {}, test: {})'.format(
            len(self.unl), len(self.train), len(self.eval), len(self.test))


class ImagesStorage:  # !TODO check dimension when add new samples
    def __init__(self, imgs, return_tensors=True):
        self.imgs = imgs
        self.return_tensors = return_tensors

    # def conditional_decorator(dec, condition):
    #     def decorator(func):
    #         if not condition:
    #             # Return the function unchanged, not decorated.
    #             return func
    #         return dec(func)
    #     return decorator

    def to_tensor(func):
        def inner(*args, **kwargs):
            a = func(*args, **kwargs)
            print(a[0:2])
            return torch.Tensor(a)
        return inner

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
        return imgs[::-1]  # to preserve the order

    @staticmethod
    def set_labels(imgs, labels):
        fail_if_len_mismatch(imgs, labels)
        for img, label in zip(imgs, labels):
            img.y = label

    def sample(self, n):
        pass

    def get(self, idxs=-1, start=0, end=0):
        check_dtype(idxs, int, list, np.ndarray)

        if start and end:
            return [self.imgs[k] for k in range(start, end)]

        if isinstance(idxs, int) and idxs < 0:
            return self.imgs

        if isinstance(idxs, int):
            return self.imgs[idxs]

        return [self.imgs[idx] for idx in idxs]

    def get_x(self, idxs=-1, start=0, end=0):
        check_dtype(idxs, int, list, np.ndarray)

        if start and end:
            return [self.imgs[k].x for k in range(start, end)]

        if isinstance(idxs, int) and idxs < 0:
            return [img.x for img in self.imgs]

        if isinstance(idxs, int):
            return self.imgs[idxs].x

        return [self.imgs[idx].x for idx in idxs]

    def get_y(self, idxs=-1, start=0, end=0):
        check_dtype(idxs, int, list, np.ndarray)

        if start and end:
            return [self.imgs[k].y for k in range(start, end)]

        if isinstance(idxs, int) and idxs < 0:
            return [img.y for img in self.imgs]

        if isinstance(idxs, int):
            return self.imgs[idxs].y

        return [self.imgs[idx].y for idx in idxs]

    def shuffle(self):
        """ Shuffle all of the loaded images. Does not check if
        images were loaded.
        """
        random.shuffle(self.imgs)
        LOG.info('Data was shuffled')

    def __len__(self):
        return len(self.imgs)

    def get_shape(self):
        return self.imgs[0].x.shape if self.imgs else None


if __name__ == "__main__":
    cm = ConfigManager('testset')
    print(cm.get_dataset_path())
    imgs = DataLoader.get_images_objects(
        cm.get_dataset_path(), 'processed_x.npy',
        'processed_y.npy', to_tensor=True)

    dm = DatasetsManager(cm, imgs)
    dm.labelled.shuffle()
    print(dm)
    imgs = dm.labelled.get(list(range(0, 9)))

    show_grid_imgs(
        [img.x for img in imgs],
        [img.y for img in imgs],
        (3, 3)
    )

    # labels = [0, 1, 1, 0]
    # dm.label_samples([0, 1, 2, 3], labels)
    # img = DataLoader.load_image(dm.unl.get_path(110))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # cv2 uses BGR format
    # show_img(img)
