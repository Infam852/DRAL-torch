import numpy as np
import pytest

from dral.data_manipulation.loader import Image
from dral.utils import extract_name_from_path
from dral.data_manipulation.dataset_manager import ImagesStorage


N_SAMPLES = 10
SEED = 1
X_DIM = (2, 2)
UNLABELLED = 255
LABELS = [UNLABELLED]*N_SAMPLES
PATH_TEMPLATE = 'path/to/file_{}'
PATHS = [PATH_TEMPLATE.format(idx) for idx in range(N_SAMPLES)]


@pytest.fixture
def img_storage():
    np.random.seed(SEED)
    xs = np.random.rand(N_SAMPLES, *X_DIM)
    ys = np.array(LABELS, dtype=np.uint8)
    imgs = [Image(x, y, path, relative=False)
            for x, y, path in zip(xs, ys, PATHS)]
    return ImagesStorage(imgs)


# !TODO move to test utils
def test_extract_name_from_path(img_storage):
    test_values = [
        ('path/to/file.txt', 'file.txt'),
        ('another_/path-/!@#$%&*_+', '!@#$%&*_+'),
        ('abc', 'abc'),
        ('', '')
    ]

    for path, expected in test_values:
        assert extract_name_from_path(path) == expected


def test_get_image_with_path(img_storage):
    img = img_storage.get(3)
    filename = 'filename_01.png'
    img.set_path(f'/random/static/path/{filename}')

    found_img = img_storage.get_image_with_path(filename)

    assert img == found_img


def test_get_images_with_paths(img_storage):
    idxs = [0, 2, 3]
    imgs = img_storage.get(idxs)
    filenames = ['filename_0{}.png'.format(idx) for idx in idxs]
    for filename, img in zip(filenames, imgs):
        img.set_path(f'/random/static/path/{filename}')

    found_imgs = img_storage.get_images_with_paths(filenames)

    assert imgs == found_imgs


def test_pop_images_with_paths(img_storage):
    idxs = [0, 2, 3, 6, 9]
    expected_length = len(img_storage) - len(idxs)
    paths = [PATHS[idx] for idx in idxs]
    expected_imgs = img_storage.get(idxs)

    popped_imgs = img_storage.pop_images_with_paths(paths)

    assert all([img in expected_imgs for img in popped_imgs])
    assert len(img_storage) == expected_length
