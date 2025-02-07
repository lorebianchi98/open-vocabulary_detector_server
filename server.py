import argparse
import dataclasses
import io
import time

from flask import Flask, request, jsonify
from PIL import Image

from model import ServedModel

app = Flask(__name__)
model = None

@app.route('/vocabulary', methods=['GET'])
def get_vocabulary():
    labelmap = model.tree.get_label_map()
    depthmap = model.tree.get_label_depth_map()
    thresholdmap = model.tree.get_label_threshold_map()
    return jsonify({
        'message': 'success',
        'prompt': model.last_prompt,
        'labelmap': labelmap,
        'depthmap': depthmap,
        'thresholdmap': thresholdmap,
    }), 200


@app.route('/vocabulary', methods=['POST', 'PUT'])
def set_vocabulary():
    if 'prompt' not in request.form:
        return jsonify({'message': 'No prompt found in the request'}), 400

    prompt = request.form['prompt']
    threshold = float(request.form.get('default_threshold', 0.05))

    try:
        model.set_vocabulary(prompt, threshold)
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 400

    return get_vocabulary()


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in the request'}), 400

    elapsed = -time.time()
    image = Image.open(request.files['image'].stream).convert('RGB')
    # flip image vertically
    # image = image.transpose(Image.FLIP_TOP_BOTTOM)

    if 'crop' in request.form:
        # Crop the center of the image
        width, height = image.size
        new_width = new_height = min(width, height)
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        image = image.crop((left, top, right, bottom))

    output = model.predict(image)
    elapsed += time.time()

    if True: # 'debug' in request.form:
        debug_image = model.draw(image, output)
        debug_image.save('debug.png')

    labelmap = model.tree.get_label_map()
    detections = [dataclasses.asdict(d) for d in output.detections]
    for d in detections:
        d['label_names'] = [labelmap[i] for i in d['labels']]

    return jsonify({'detections': detections, 'time_s': elapsed}), 200


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch16")
    parser.add_argument("--image-encoder-engine", type=str, default="nanoowl/data/owl_image_encoder_patch16.engine")
    parser.add_argument("--vocabulary", type=str, default="[a person [a head {0.08} (with a helmet, without a helmet), a glove, a high visibility vest {0.1}]]")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    elapsed = -time.time()
    model = ServedModel(args.model, args.image_encoder_engine)
    model.set_vocabulary(args.vocabulary, args.threshold)
    # warmup model with random image
    model.predict(Image.new('RGB', (768, 768), color=None))
    elapsed += time.time()
    print(f'model setup: {elapsed}s')

    app.run(host="0.0.0.0", port=5000, debug=True)
