import numpy as np

import pytest


n_samples = 10
x_samples_dim = 5
y_range = 5


@pytest.fixture
def dm():
    from dral.data_manipulation.dataset_manager import DatasetsManager

    np.random.seed(1)
    unl = np.random.rand(n_samples, x_samples_dim)
    x_eval = np.random.rand(n_samples, x_samples_dim)
    y_eval = np.random.randint(y_range, size=n_samples)
    x_test = np.random.rand(n_samples, x_samples_dim)
    y_test = np.random.randint(y_range, size=n_samples)
    dm = DatasetsManager(unl, x_eval, y_eval, x_test, y_test)
    return dm


class TestDatasetManager:
    def test_initialization(self, dm):
        assert hasattr(dm, 'unl')
        assert hasattr(dm, 'eval')
        assert hasattr(dm, 'test')
        assert hasattr(dm, 'train')
        assert len(dm.unl) == n_samples
        assert len(dm.eval) == n_samples
        assert len(dm.test) == n_samples
        assert len(dm.train) == 0

    def test_label_samples(self, dm):
        n_to_sample = 5
        labels = np.random.randint(y_range, size=n_to_sample)
        idxs = np.arange(n_to_sample)
        unl_to_label = dm.unl.get_x(idxs)

        dm.label_samples(idxs, labels)

        assert len(dm.unl) == n_samples - n_to_sample
        assert len(dm.train) == n_to_sample
        assert np.all(dm.train.get_x() == unl_to_label)
