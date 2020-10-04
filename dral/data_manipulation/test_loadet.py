import numpy as np
import os
import time
from tqdm import tqdm

from skimage import transform
from skimage import io


n_load = 1
arr = np.zeros((n_load, 128, 128), dtype=np.float32)
start = time.time()
path = os.path.join('data', 'PetImages', 'Cat')
for k, f in tqdm(enumerate(os.listdir(path))):
    try:
        if k == n_load:
            break
        img_path = os.path.join(path, f)
        img = io.imread(img_path, as_gray=True)
        print(f'type: {type(img)}, shape: {img.shape[:2]}')
        img = transform.resize(img, (128, 128))

        arr[k] = img
    except Exception as e:
        print(e)
        print(f'Skip image number {k}...')

# np_arr = np.array(arr, dtype=np.float32)
# end = time.time()
# print(f'Time: {end - start}')

# print(arr.shape)

# imgs = np.load('data/x_cats_dogs_128.npy')
# imgs2 = np.load('data/x_cats_dogs_128.npy')
# print(imgs.shape)
# import psutil
# process = psutil.Process(os.getpid())
# print(process.memory_info().rss)  # in bytes 
# time.sleep(5)