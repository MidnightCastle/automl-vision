from flask import jsonify, request, json
from tomato_disease_detection import app
from tomato_disease_detection.model import DiseaseDetection

@app.route('/')
def hello():
    return('Hello Automl vision, welcome')


@app.route('/detect_disease', methods=['POST'])
def get_disease():
    if request.get_json() is None:
        return 'request data not detected'
    else:
        request_data = request.get_json()
        img_b64 = request_data['img_b64']
        result = DiseaseDetection().model_predict(img_b64)
        if len(result)!=0:
            return jsonify(result)
        else:
            return 'No matching disease found'



