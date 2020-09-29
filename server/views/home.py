from flask.views import MethodView
from flask import render_template
from flask import request, redirect, url_for


class HomeView(MethodView):
    def search(self):
        return render_template('home.html.j2')

    def post(self):
        print("I am inside post")
        post_body = request.json
        print(post_body)
        return redirect(url_for('.views_HomeView_search'))
