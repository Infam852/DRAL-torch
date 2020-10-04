from flask.views import MethodView
from flask import render_template, url_for, request, redirect

from server.app_utils import cm, dm
from dral.data_manipulation.loader import DataLoader
from dral.data_manipulation.image_viewer import create_images
from dral.utils import show_grid_imgs


class PredictionsView(MethodView):

    def search(self):
        import time
        start = time.time()
        self._get_predictions()
        # mock predictions
        idxs = [1, 2, 50, 100, 160, 180, 11, 232]
        imgs = dm.unl.get(idxs)
        images = create_images(imgs, idxs,
                               cm.get_dataset_path(),
                               cm.get_unprocessed_x_name(),
                               cm.get_tmp_dir())
        end = time.time()
        print('time: ', end-start)
        return render_template("predictions.html.j2", images=images), 200

    def post(self):
        class1_idxs = [int(idx) for idx in request.json['class1']]
        class2_idxs = [int(idx) for idx in request.json['class2']]
        labels = cm.get_numeric_labels()
        print(dm)
        print("class1 idxs:", class1_idxs)
        print("class2 idxs:", class2_idxs)
        # add to training data
        dm.label_samples_mapping({labels[0]: class1_idxs,
                                  labels[1]: class2_idxs})
        print(dm)
        # do training and pass imgs
        imgs = dm.train.get_x(list(range(4)))
        labels = dm.train.get_y(list(range(4)))
        show_grid_imgs(imgs, labels, (2, 2))

        return redirect(url_for('.views_PredictionsView_search'))

    def _get_predictions(self):
        pass
