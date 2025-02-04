import argparse
import json
from PIL import Image, ImageDraw, ImageFont


def draw(image, output, threshold=0, icons=None):
    # draw on the image
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta", "lime", "pink"]

    icon_map = {}
    if icons:
        with open(f'{icons}/map.json', 'r') as f:
            icon_map = json.load(f)

    for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
        if score < threshold:
            continue
        # unnormalize the box
        x0, y0, x1, y1 = box[0] * image.width, box[1] * image.height, box[2] * image.width, box[3] * image.height

        class_name = output['vocabulary'][label]
        draw = ImageDraw.Draw(image, "RGBA")
        # draw the icon
        if class_name in icon_map:
            draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=(255, 255, 255, 75))

            icon = Image.open(f'{icons}/{icon_map[class_name]}')
            ix0 = int(x0 + x1 - icon.width) // 2
            iy0 = int(y0 + y1 - icon.height) // 2
            image.paste(icon, (ix0, iy0), icon)

        # draw the box
        else:
            # pick a color based on the label
            color = colors[label % len(colors)]
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

            # draw the label
            font = ImageFont.load_default(image.height // 20)
            draw.text((x0, y0), f"{class_name}: {score:.2f}", fill=color, font=font)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw bounding boxes on an image.')
    parser.add_argument('image', help='The image file to draw the boxes on.')
    parser.add_argument('detections', help='The JSON file containing the detections.')
    parser.add_argument('-o', '--output-file', default='output.jpg', help='The file to save the image with the boxes drawn on it.')
    parser.add_argument('-t', '--threshold', type=float, default=0, help='The threshold for the detections.')
    parser.add_argument('-i', '--icons', default='icons/', help='The folder containing the icons for the classes.')
    args = parser.parse_args()

    image = Image.open(args.image)

    with open(args.detections, 'r') as f:
        detections = json.load(f)

    draw(image, detections, threshold=args.threshold, icons=args.icons)

    image.save(args.output_file)
    print(f'Saved: {args.output_file}')
