import os
import numpy as np

from data_manipulation.loader import DataLoader
from utils import show_grid_imgs


DATASETS_DICT = 'data'

x_cv2 = DataLoader.load(os.path.join(DATASETS_DICT, 'x_train_cats_dogs.npy'))
y_cv2 = DataLoader.load(os.path.join(DATASETS_DICT, 'y_train_cats_dogs.npy'))

x_skimage = DataLoader.load(os.path.join(DATASETS_DICT,
                            'x_cats_dogs_skimage.npy'))
y_skimage = DataLoader.load(os.path.join(DATASETS_DICT,
                            'y_cats_dogs_skimage.npy'))

print(x_cv2[0])
print(x_skimage[0])
print(np.max(x_skimage[0]*255))


print(np.all(x_cv2[0]==x_skimage[0]))
print('cv2:', x_cv2.shape, y_cv2.shape, 'skimage:', x_skimage.shape, y_skimage.shape)

n_imgs = 9

show_grid_imgs(x_cv2[:n_imgs], y_cv2[:n_imgs], (3,3))
show_grid_imgs(x_skimage[:n_imgs], y_skimage[:n_imgs], (3,3))
