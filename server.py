from flask import Flask, request, jsonify

from utils import evaluate_image

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from transformers import OwlViTProcessor, OwlViTForObjectDetection, OwlViTTextConfig

import torch

MAX_PREDICTIONS=100

app = Flask(__name__)

# loading the model
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = model.to(device)
model.eval()


@app.route('/detect', methods=['POST'])
def detect_objects():
    if request.method == 'POST':
        # Receive the image frame from the local machine
        frame = request.data  

        # Perform object detection
        vocabulary = ["person"]
        results = evaluate_image(model, processor, frame, vocabulary, MAX_PREDICTIONS, nms=True)

        return jsonify(results)

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000)
