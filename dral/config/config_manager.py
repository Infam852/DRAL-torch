import os

import numpy as np
import confuse

CONFIG_PATH = 'dral/config/config.yml'
UNKNOWN_LABEL = 'Unknown'

config = confuse.Configuration('DRAL', __name__)
config.set_file(CONFIG_PATH)  # !TODO change location of file path


class ConfigManager:
    configurations = config.keys()

    def __init__(self, config_name):
        if config_name not in self.configurations:
            raise ValueError(f'({config_name}) is not defined in config file. '
                             f'Defined configurations: {self.configurations}')

        self.config = config[config_name]
        self.config_name = config_name
        self.preprocessing = 'preprocessing'

    def get_config(self):
        return self.config.get()

    def get_dataset_name(self):
        return self.config['dataset'].get(str)

    def get_img_size(self):
        return self.config['general']['image_size'].get(int)

    def get_label_mapping(self, idx=None):
        return self.config['general']['label_mapping'].get(dict)

    def get_label_names(self):
        return list(self.get_label_mapping().keys())

    def get_label_name(self, idx):
        return self.get_label_names()[idx]

    def get_name_class_delimiter(self):
        return self.config['general']['name_class_delimiter'].get(str)

    def get_number_of_labels(self):
        return len(self.get_label_names())

    def get_imgs_path(self):
        return self.config['paths']['imgs'].get()

    def get_save_path(self):
        return self.config['paths']['save'].get(str)

    def get_dataset_path(self):
        return os.path.join(self.get_save_path(), self.get_config_name())

    def do_shuffle(self):
        return self.config['loader']['shuffle'].get(bool)

    def get_label_format(self):
        return self.config['loader']['label_format'].get(str)

    def do_grayscale(self):
        return self.config[self.preprocessing]['grayscale'].get(bool)

    def do_rescale_with_crop(self):
        return self.config[self.preprocessing]['rescale_with_crop'].get(bool)

    def do_normalization(self):
        return self.config[self.preprocessing]['normalization'].get(bool)

    def do_centering(self):
        return self.config[self.preprocessing]['centering'].get(bool)

    def do_standarization(self):
        return self.config[self.preprocessing]['standarization'].get(bool)

    def do_strict_balance(self):
        return self.config[self.preprocessing]['strict_balance'].get(bool)

    def get_config_name(self):
        return self.config_name

    def get_unknown_label(self):
        if UNKNOWN_LABEL not in self.get_label_names():
            raise ValueError(f'There is no {UNKNOWN_LABEL} in labels names')
        return self.get_label_mapping()[UNKNOWN_LABEL]

    def get_class_paths(self):
        return [os.path.join(self.get_imgs_path(), label)
                for label in self.get_label_names()]

    def get_one_hot_labels(self):
        labels = []
        for k, label in enumerate(self.get_label_names()):
            label = np.eye(self.get_number_of_labels())[k]
            labels.append(label)
        return labels

    def get_numeric_labels(self):
        return list(self.get_label_mapping().values())

    def enable_npy_preprocessing(self, enable):
        """Switch config between numpy config and pure images config

        Args:
            enable (bool): if set to True then use numpy config
            for preprocessing
        """
        self.preprocessing = 'preprocessing_npy' if enable else 'preprocessing'
