import argparse
import json
import itertools

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont



def get_colors(count: int):
    cmap = plt.cm.get_cmap("rainbow", count)
    colors = []
    for i in range(count):
        color = cmap(i)
        color = [int(255 * value) for value in color]
        colors.append(tuple(color))
    return colors


def draw(image, output, draw_text=True):
    # breakpoint()
    detections = output['detections']
    vocab = set(itertools.chain.from_iterable(d['label_names'] for d in detections))
    num_colors = len(vocab)

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default((image.width + image.height) // 70)
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta", "lime", "pink"]
    color_map = dict(zip(vocab, colors[:num_colors]))

    for detection in detections:
        if detection['parent_id'] == -1:  # skip 'image'
            continue

        box = detection['box']
        label_names = detection['label_names']
        scores = detection['scores']

        main_label = label_names[0]
        draw.rectangle(box, outline=color_map[main_label], width=3)
        if draw_text:
            offset_y = .75 * font.size
            offset_x = .75 * font.size
            for label_text, score in zip(label_names, scores):
                draw.text(
                    (box[0] + offset_x, box[1] + offset_y),
                    f'({score:.2f}) {label_text}',
                    fill=color_map[label_text],
                    font=font
                )
                offset_y += font.size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw bounding boxes on an image.')
    parser.add_argument('image', help='The image file to draw the boxes on.')
    parser.add_argument('detections', help='The JSON file containing the detections.')
    parser.add_argument('-o', '--output-file', default='output.jpg', help='The file to save the image with the boxes drawn on it.')
    args = parser.parse_args()

    image = Image.open(args.image).convert('RGB')

    with open(args.detections, 'r') as f:
        output = json.load(f)

    draw(image, output)

    image.save(args.output_file)
    print(f'Saved: {args.output_file}')