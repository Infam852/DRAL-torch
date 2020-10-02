from dral.data_manipulation.loader import DataLoader
from dral.data_manipulation.dataset_manager import DatasetsManager
from dral.config.config_manager import ConfigManager

from server import app


def get_cm():
    return ConfigManager(app.app.config['CM_CONFIG'])


def get_dm():
    cm = get_cm()
    imgs = DataLoader.load_images(cm.get_dataset_path())
    dm = DatasetsManager(cm, imgs)
    return dm
