from flask import Flask, request, jsonify

from utils import evaluate_image
import torch
import json

import numpy as np
import cv2

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from transformers import OwlViTProcessor, OwlViTForObjectDetection, OwlViTTextConfig


MAX_PREDICTIONS=20

app = Flask(__name__)

# loading the model
print("Loading the model..")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = model.to(device)
model.eval()
print("Model loaded!")

@app.route('/detect', methods=['POST'])
def detect_objects():
    if request.method == 'POST':
        # Receive the JSON data from the local machine
        received_json = json.loads(request.data.decode('utf-8'))
        
        # Deserialize the JSON data to a numpy array
        frame = np.array(received_json['frame'])
        score_thresh = received_json['score_thresh']
        vocabulary = received_json['vocabulary']
        
        results = evaluate_image(model, processor, frame, vocabulary, MAX_PREDICTIONS, score_thresh=score_thresh, nms=True, print_time=False)

        return jsonify(results), 200

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000)
