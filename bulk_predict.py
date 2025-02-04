import argparse
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from detector import OpenVocabularyDetector


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bulk predict image folder.')
    parser.add_argument('image_folder', type=Path, help='Path to the folder containing images to predict.')
    parser.add_argument('-o', '--output', default=None, type=Path, help='Path to the folder to save the predictions. If not provided, the predictions will be saved in the same folder as the images.')
    parser.add_argument('-v', '--vocabulary', nargs='+', default=None, help='List of classes to predict.')
    parser.add_argument('-p', '--prompt', default=None, help='Prompt to use for the predictions.')
    parser.add_argument('-m', '--max_predictions', type=int, default=None, help='Maximum number of predictions to return.')
    parser.add_argument('-t', '--score-thresh', type=float, default=None, help='Threshold for the predictions.')
    parser.add_argument('-n', '--nms-thresh', type=float, default=None, help='Threshold for non-maximum suppression.')
    args = parser.parse_args()

    args.output = args.output or args.image_folder
    args.output.mkdir(parents=True, exist_ok=True)
    # get list of images
    images = sorted(p for p in args.image_folder.iterdir() if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png'])
    if not images:
        print(f"No images found in {args.image_folder}")
        exit(1)

    # create the detector
    detector = OpenVocabularyDetector()

    # run the detection
    for image_path in tqdm(images):
        image = Image.open(image_path)
        results = detector.detect(
            image,
            vocabulary=args.vocabulary,
            max_predictions=args.max_predictions,
            score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh,
            prompt=args.prompt,
        )
        output_path = args.output / f"{image_path.stem}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f)

