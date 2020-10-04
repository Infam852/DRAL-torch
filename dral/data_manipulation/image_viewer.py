import os

import cv2

from dral.utils import LOG
from dral.data_manipulation.loader import DataLoader


class SavedImage:
    def __init__(self, path, idx):
        cut_idx = path.index('static')
        path = path[cut_idx:]
        self.path = path
        self.idx = idx


def save_img(img, path):
    cv2.imwrite(path, img)
    LOG.debug(f'save image: {path}')


def create_images(imgs, unl_idxs, basepath, unprocessed_path, save_dir):
    import time
    start = time.time()

    saved_images = []
    x = DataLoader.load_npy(basepath, unprocessed_path)
    for k, img in enumerate(imgs):
        filename = f'{img.idx}.png'
        path = os.path.join(basepath, save_dir, filename)
        img_to_save = x[img.idx]

        save_img(img_to_save, path)
        saved_images.append(SavedImage(path, unl_idxs[k]))

    result = time.time() - start
    print(f'[DEBUG] create images time: {result}')
    return saved_images
