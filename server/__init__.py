import os

import connexion
from connexion.resolver import MethodViewResolver
from dral.config.config_manager import ConfigManager


cm = ConfigManager('testset')
app = connexion.FlaskApp(__name__, port=5000, specification_dir='openapi/')
app.add_api('openapi.yml', strict_validation=True, validate_responses=True,
            resolver=MethodViewResolver('views'))
app.app.config['IMGS_DIR'] = os.path.join('static', 'imgs')
app.app.config['cm'] = cm
