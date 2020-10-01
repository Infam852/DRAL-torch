# import connexion
# from flask import Flask
# # from routes.config import ConfigView
from server import app

from dral.data_manipulation.loader import DataLoader
from dral.data_manipulation.dataset_manager import DatasetsManager


# def compose_app():
#     app = connexion.FlaskApp(__name__, specification_dir='./')
#     # app.add_url_rule('/configs', view_func=ConfigView.as_view(
#     #                  ConfigView.__name__))
#     app.add_api('openapi.yml')
#     app = Flask(__name__)
#     return app


def get_dm():
    imgs = DataLoader.load_images(app.app.config['cm'].get_dataset_path())
    dm = DatasetsManager(app.app.config['cm'], imgs)
    return dm
