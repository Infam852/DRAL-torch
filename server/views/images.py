import os

from flask.views import MethodView
from flask import render_template, send_file, url_for
from server import app

from dral.data_manipulation.dataset_manager import DatasetsManager
from dral.data_manipulation.loader import DataLoader


class ImagesView(MethodView):

    def search(self):
        imgs = DataLoader.load_images(app.app.config['cm'].get_dataset_path())
        dm = DatasetsManager(app.app.config['cm'], imgs)
        paths = dm.unl.get_path()
        return render_template("predictions.html.j2", image_names=paths), 200

    def post(self):
        pass
