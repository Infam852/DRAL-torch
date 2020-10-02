from flask import request
from flask.views import MethodView

from server.app import get_dm


class TrainView(MethodView):
    def post(self):
        imgs_paths= request.json
        # imgs = 