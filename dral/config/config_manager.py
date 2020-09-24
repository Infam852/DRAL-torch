import confuse

CONFIG_PATH = 'dral/config/config.yml'

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

    def get_config(self):
        return self.config.get()

    def get_dataset_name(self):
        return self.config['dataset'].get(str)

    def get_img_size(self):
        return self.config['general']['image_size'].get(int)

    def get_labels(self):
        return self.config['general']['labels'].get(list)

    def get_imgs_path(self):
        return self.config['paths']['imgs'].get()

    def get_save_path(self):
        return self.config['paths']['save'].get(str)

    def do_shuffle(self):
        return self.config['loader']['shuffle'].get(bool)

    def get_label_format(self):
        return self.config['loader']['label_format'].get(str)

    def do_grayscale(self):
        return self.config['preprocessing']['grayscale'].get(bool)

    def do_rescale_with_crop(self):
        return self.config['preprocessing']['rescale_with_crop'].get(bool)

    def do_normalization(self):
        return self.config['preprocessing']['normalization'].get(bool)

    def do_centering(self):
        return self.config['preprocessing']['centering'].get(bool)

    def do_standarization(self):
        return self.config['preprocessing']['standarization'].get(bool)

    def do_strict_balance(self):
        return self.config['preprocessing']['strict_balance'].get(bool)

    def get_config_name(self):
        return self.config_name
