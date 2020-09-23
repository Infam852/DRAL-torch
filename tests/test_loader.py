import os

import confuse
import numpy as np
import pytest
import png

import dral.config.config_manager
from dral.config.config_manager import ConfigManager
from dral.data_manipulation.loader import DataLoader


""" !TODO Refactoring is obligatory """


@pytest.fixture(scope='session')
def image_factory(**kwargs):
    def _image(**kwargs):
        if kwargs.pop('random_dims', False):
            min_h = kwargs.pop('min_h', 2)
            max_h = max(kwargs.pop('max_h', 8), min_h+1)
            min_w = kwargs.pop('min_w', 2)
            max_w = max(kwargs.pop('max_w', 8), min_w+1)
            h = np.random.randint(min_h, max_h)
            w = np.random.randint(min_w, max_w)
            dims = (h, w)
        else:
            # in tests we can create smaller images
            dims = kwargs.pop('dims', (4, 4))
        if kwargs.pop('normalized', False):
            img = np.random.rand(*dims)
        else:
            min_ = kwargs.pop('min', 0)
            max_ = kwargs.pop('max', 256)
            img = np.random.randint(min_, max_, size=dims, dtype=np.uint8)
        return img
    return _image


@pytest.fixture(scope='session')
def temp_dir(tmpdir_factory):
    def _temp_dir(name):
        fh = tmpdir_factory.mktemp(name)
        return fh

    return _temp_dir


@pytest.fixture(scope='session')
def image_files_factory(tmpdir_factory, image_factory):
    counter = 0

    def _image_file(dirpath):
        nonlocal counter
        counter += 1
        img = image_factory()
        img_name = 'img_{}.png'.format(counter)
        path = os.path.join(dirpath, img_name)
        png.from_array(img, 'L').save(path)
        return path

    return _image_file


@pytest.fixture(scope='session')
def data_dir(temp_dir, image_files_factory):
    dir_path = temp_dir('test-dir')
    # fill temp dir with random images
    for label in ['a', 'b']:
        dir_path.mkdir(label)
        path = os.path.join(dir_path, label)
        for k in range(10):
            image_files_factory(path)

    return dir_path


def test_create_image(data_dir):
    # Arrange
    test_config = confuse.Configuration('DRAL', __name__)
    test_config.set_file('tests/sample_config.yml')
    path = data_dir.strpath
    test_config['sample_dataset']['paths']['imgs'] = path
    dral.config.config_manager.config = test_config
    ConfigManager.configurations = test_config.keys()
    cm = ConfigManager('sample_dataset')
    dl = DataLoader(cm)

    # Act
    dl.load_raw()

    # Assert
    assert 0
