from flask.views import MethodView
from flask import render_template


class TrainImagesView(MethodView):

    def get(self):
        return render_template('images.html', imgpath='server/static/cats_dogs/0_0.png')
