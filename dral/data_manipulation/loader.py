import os
import numpy as np

from skimage import io
from skimage.color import rgb2gray
import skimage.transform
from tqdm import tqdm

from dral.config.config_manager import ConfigManager
from dral.utils import LOG


class DataLoader:
    def __init__(self, cm, max_imgs):
        """Initialize data loader with specification saved in CONFIG dictionary

        Args:
            cm (obj): ConfigManager instance that provide interface to
            configuration file
            max_imgs (int): max number of images per one class to be loaded
        """
        self.cm = cm
        self.IMG_SIZE = cm.get_img_size()
        self.MAX_IMGS = max_imgs
        self.N_LABELS = len(cm.get_labels())
        self.DTYPE = np.float32 if cm.do_normalization() else np.uint8

        self._reset()

    def _reset(self):
        self._x = np.zeros(
            (self.N_LABELS, self.IMG_SIZE, self.IMG_SIZE), dtype=self.DTYPE)
        self._y = np.zeros(
            (self.N_LABELS, self.IMG_SIZE, self.IMG_SIZE), dtype=self.DTYPE)
        self.balance_counter = {label: 0 for label in self.cm.get_labels()}

    def load_raw(self):
        paths = [os.path.join(self.cm.get_imgs_path(), label)
                 for label in self.cm.get_labels()]
        self._fail_if_path_is_not_dir(paths)
        self._reset()
        for class_path in paths:
            LOG.info(f'Start loading images from path: {class_path}')
            for k, f in tqdm(enumerate(os.listdir(class_path))):
                if k >= self.MAX_IMGS:
                    LOG.info(f'Loaded maximum number ({k}) of images')
                    break
                try:
                    path = os.path.join(class_path, f)
                    # as_gray converts image to float64 within [0, 1] range
                    img = io.imread(path, as_gray=self.cm.do_grayscale())
                    img = self._rescale(img)
                    # if not enable_normalization:
                    #     img = (img*256).astype('uint8')

                    self.x.append(np.array(img))
                    # one-hot vector
                    # self.y.append(np.eye(self.NCLS)[self.LABELS[label]])

                    # self.balance_counter[self.LABELS[label]] += 1
                except Exception as e:
                    LOG.warning(
                        f'Error while loading image from path {path}: {e}')

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
        if sum(self.balance_counter.values()):
            for label in self.LABELS.values():
                print(f'Number of {label} imgs: {self.balance_counter[label]}')
        else:
            raise Exception('Firstly you have to load data!')

    def _fail_if_path_is_not_dir(self, paths):
        """ Raise exception if any of paths is not directory path

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
        if self.cm.do_rescale_with_crop():
            (h, w) = img.shape
            if h > w:
                ratio = h / w
                dims = (self.IMG_SIZE*ratio, self.IMG_SIZE)
                img = skimage.transform.rescale(img, dims)
                crop_idx = (dims[0] - self.IMG_SIZE) // 2
                img = img[crop_idx:self.IMG_SIZE+crop_idx, :]
            else:
                ratio = w / h
                dims = (self.IMG_SIZE, self.IMG_SIZE*ratio)
                img = skimage.transform.rescale(img, dims)
                crop_idx = (dims[1] - self.IMG_SIZE) // 2
                img = img[:, crop_idx:self.IMG_SIZE+crop_idx, ]
            print('crop_idx: ', crop_idx)
            print(img.shape)
        else:
            img = skimage.transform.rescale(
                img, (self.IMG_SIZE, self.IMG_SIZE))
        return img


    @staticmethod
    def load(path):
        return np.load(path, allow_pickle=True)


if __name__ == '__main__':
    cm = ConfigManager('cats_dogs_64')
    dl = DataLoader(cm, max_imgs=1)
    dl.load_raw()
    # dl.shuffle()
    # dl.print_balance_counter()
    # dl.save('data', 'cats_dogs_64_norm')
