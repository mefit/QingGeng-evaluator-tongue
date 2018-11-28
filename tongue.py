from flask import Flask
from flask import request
from tongue_tasks import evaluate

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def listen():
    evaluate.delay(
        request.form['user_id'],
        request.form['id'],
        request.form['tongue_image'] if 'tongue_image' in request.form else None)
    return "ok", 200
