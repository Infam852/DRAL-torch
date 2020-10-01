from flask.views import MethodView
from flask import render_template, send_file, url_for, g
from server import app

from server.app_utils import get_dm


class PredictionsView(MethodView):

    def search(self):
        import time
        start = time.time()
        dm = get_dm()
        paths = dm.unl.get_path()
        end = time.time()
        print('time: ', end-start)
        return render_template("predictions.html.j2", image_names=paths), 200

    def post(self):
        pass
