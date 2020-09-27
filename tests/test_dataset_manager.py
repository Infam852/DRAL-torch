import numpy as np
import pytest
from unittest.mock import MagicMock

from dral.data_manipulation.dataset_manager import DatasetsManager
from dral.data_manipulation.loader import Image


N_SAMPLES = 6
UNLABELLED = 255
LABELS = [UNLABELLED]*N_SAMPLES
X_DIM = (2, 2)
N_LABELS = 2
PATH = 'placeholder'
SEED = 1


@pytest.fixture
def dm():
    np.random.seed(SEED)
    xs = np.random.rand(N_SAMPLES, *X_DIM)
    ys = np.array(LABELS, dtype=np.uint8)
    imgs = [Image(x, y, PATH) for x, y in zip(xs, ys)]

    cm = MagicMock()
    cm.get_dataset_path.return_value = 'undefined'
    cm.get_unknown_label.return_value = UNLABELLED

    dm = DatasetsManager(cm, imgs)
    return dm


class TestDatasetManager:
    def test_initialization(self, dm):
        assert len(dm.unl) == 6
        assert len(dm.labelled) == 0

    def test_label_samples(self, dm):
        idxs = [0, 2, 3]
        labels = [0, 1, 0]
        imgs_to_sample = dm.unl.get(idxs)

        dm.label_samples(idxs, labels)
        sampled_imgs = dm.train.get()

        assert len(dm.unl) == 3
        assert len(dm.train) == 3
        assert len(dm.labelled) == 3
        assert imgs_to_sample == sampled_imgs
