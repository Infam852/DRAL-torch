import os

import numpy as np
import pytest
import png


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
        if kwargs.pop('normalized', True):
            img = np.random.rand(*dims)
        else:
            min_ = kwargs.pop('min', 0)
            max_ = kwargs.pop('max', 256)
            img = np.random.randint(min_, max_, size=dims, dtype=np.uint8)
        return img
    return _image


@pytest.fixture(scope='session')
def image_files_factory(tmpdir_factory, image_factory, name):
    counter = 0

    def _image_file(tmpdir_factory, image_factory, name):
        nonlocal counter
        counter += 1
        img = image_factory(normalized=False)
        img_name = 'img_{}.png'.format(counter)
        fh = tmpdir_factory.mktemp('pytest').join(img_name)
        png.from_array(img, 'L').save(fh)
        return fh
    return _image_file


def data_dir(image_files_factory):
    


def test_create_image(image_file):
    import time
    with open(image_file, 'rb') as f:
        print(f.read())
    
    assert 0

# def test_create_image(tmpdir):
#     p = tmpdir.mkdir("sub").join("hello.txt")
#     p.write("content")
#     assert p.read() == "content"
#     assert len(tmpdir.listdir()) == 1
#     assert 0