from flask import Flask, jsonify
from flask_cors import CORS
import tensorflow as tf

from config import *

from views import user, forum, auto

# instantiate the profile
app = Flask(__name__)
app.config.from_object(Test)

app.register_blueprint(forum.forum_bp, url_prefix='/forum')
app.register_blueprint(auto.auto_bp, url_prefix='/auto')
app.register_blueprint(user.user_bp, url_prefix='/user')

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    return jsonify('hello world!')


if __name__ == '__main__':
    app.run()
