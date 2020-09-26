from flask.views import MethodView
from flask import render_template, send_file, url_for
from server import app
import os

IMG_FOLDER = os.path.join('static', 'cats_dogs_96')


class ImagesView(MethodView):

    def search(self):
        PEOPLE_FOLDER = os.path.join('static', 'imgs')
        full_filename = os.path.join(PEOPLE_FOLDER, '1_0.png')

        return render_template("images.html", user_image=full_filename)

    def post(self):
        pass
