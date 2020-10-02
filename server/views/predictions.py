from flask.views import MethodView
from flask import render_template, url_for, request, redirect

from server.app_utils import get_dm, get_cm


class PredictionsView(MethodView):

    def search(self):
        import time
        start = time.time()
        idxs = list(range(10))
        dm = get_dm()
        paths = dm.unl.get_path(idxs)
        end = time.time()
        print('time: ', end-start)
        return render_template("predictions.html.j2", image_names=paths), 200

    def post(self):
        class1_paths = request.json['class1']
        class2_paths = request.json['class2']
        cm = get_cm()
        labels = cm.get_numeric_labels()
        dm = get_dm()
        imgs1 = dm.unl.get_indicies_from_paths(class1_paths)
        imgs2 = dm.unl.get_indicies_from_paths(class2_paths)
        # add to training data
        dm.label_samples_with_specified_label(imgs1, labels[0])
        dm.label_samples_with_specified_label(imgs1, labels[1])
        print(dm)
        # do training and pass imgs

        print(imgs1)
        print(imgs2)
        return redirect(url_for('.views_PredictionsView_search'))
