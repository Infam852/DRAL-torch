from flask.views import MethodView

from dral.config.config_manager import ConfigManager


class ConfigsView(MethodView):
    def get(self, name):
        config = self._get_config(name)
        if config is None:
            return f'Configuration with name ({name}) not found', 404

        return config, 200

    def search(self):
        return self._get_all_configs(), 200

    def _get_all_configs(self):
        configs = []
        for config_name in ConfigManager.configurations:
            configs.append(self._get_config(config_name))
        return configs

    def _get_config(self, name):
        if name not in ConfigManager.configurations:
            return

        cm = ConfigManager(name)
        config = cm.get_config()
        config['name'] = name
        return config
