import argparse
import json
from PIL import Image, ImageDraw, ImageFont


def draw(image, output):
    # draw on the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(image.height // 20)
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta", "lime", "pink"]
    for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
        # pick a color based on the label
        color = colors[label % len(colors)]
        # unnormalize the box
        x0, y0, x1, y1 = box[0] * image.width, box[1] * image.height, box[2] * image.width, box[3] * image.height
        # draw the box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        # draw the label
        draw.text((x0, y0), f"{output['vocabulary'][label]}: {score:.2f}", fill=color, font=font)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw bounding boxes on an image.')
    parser.add_argument('image', help='The image file to draw the boxes on.')
    parser.add_argument('detections', help='The JSON file containing the detections.')
    parser.add_argument('-o', '--output-file', default='output.jpg', help='The file to save the image with the boxes drawn on it.')
    args = parser.parse_args()

    image = Image.open(args.image)

    with open(args.detections, 'r') as f:
        output = json.load(f)

    draw(image, output)

    image.save(args.output_file)
    print(f'Saved: {args.output_file}')
