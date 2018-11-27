from flask import Flask
from flask import request
from tongue_tasks import evaluate

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def listen():
    evaluate.delay(
        request.form['id'],
        request.form['user_id'],
        request.form['tongue_image'])
    return "ok", 200
