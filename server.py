from flask import Flask, request, jsonify
from PIL import Image

from detector import OpenVocabularyDetector


detector = OpenVocabularyDetector()
app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in the request'}), 400

    # get the image
    image = Image.open(request.files['image'].stream)

    print(request.form)

    # optional parameters
    vocabulary = request.form.getlist('vocabulary', None)
    prompt = request.form.get('prompt', None)
    max_predictions = request.form.get('max_predictions', None)
    max_predictions = int(max_predictions) if max_predictions is not None else None
    score_thresh = request.form.get('score_thresh', None)
    score_thresh = float(score_thresh) if score_thresh is not None else None
    nms_thresh = request.form.get('nms_thresh', None)
    nms_thresh = float(nms_thresh) if nms_thresh is not None else None

    print(f"Received request with:")
    print(f"  {vocabulary=}")
    print(f"  {prompt=}")
    print(f"  {max_predictions=}")
    print(f"  {score_thresh=}")
    print(f"  {nms_thresh=}")

    # run detection
    results = detector.detect(
        image,
        vocabulary=vocabulary,
        max_predictions=max_predictions,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        prompt=prompt,
    )

    return jsonify(results), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)
