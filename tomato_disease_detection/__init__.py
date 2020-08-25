from flask import Flask

app = Flask(__name__)

from tomato_disease_detection import routes