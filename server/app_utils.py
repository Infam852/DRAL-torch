from dral.data_manipulation.loader import DataLoader
from dral.data_manipulation.dataset_manager import DatasetsManager
from dral.config.config_manager import ConfigManager

from server import app


def get_cm():
    return ConfigManager('cats_dogs_128')


def get_dm():
    cm = get_cm()
    import time
    start = time.time()
    imgs = DataLoader.get_images_objects(
        cm.get_dataset_path(), 'processed_x.npy', 'processed_y.npy')
    end = time.time()
    print(end-start)
    dm = DatasetsManager(cm, imgs)
    return dm


cm = ConfigManager('cats_dogs_128')
dm = get_dm()
