from flask import Flask
from flask import request

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def evaluate():
    return "ok", 200
