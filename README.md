# SUN XR: "Open-Vocabulary Object Detection" Component

This component is a part of the [SUN XR](https://sun-xr-project.eu) project.

## Requirements

- torch & torchvision
- stuff in `requirements.txt`

Tested with:
 - Python 3.10.12
 - torch==2.0.1+cu117
 - torchvision==0.15.2+cu117
 - transformers==4.38.2
 - Flask==3.0.2

## Getting Started

Start the server (default port is 5000):
```bash
python server.py
```

The API endpoint is `/detect`. It accepts a POST request with the following parameters:
- `image`: a file with the image to process
- `vocabulary`: a list of strings with the vocabulary to search for in the image

Optional parameters:
- `prompt`: (string) the prompt to use for the image, defaults to "A photo of a"
- `max_predictions`: (integer) the maximum number of predictions to return, defaults to 10
- `score_thresh`: (float) the minimum score for a prediction to be returned, defaults to 0.05
- `nms_tresh`: (float) the IoU threshold for non-maximum suppression, defaults to 0.5


Test it:
```bash
# Download a test image
wget http://lenna.org/len_std.jpg

# Send a request, save results to json file
curl \
    -X POST \
    -F "image=@len_std.jpg" \
    -F "vocabulary=eye" \
    -F "vocabulary=hat" \
    -F "vocabulary=scarf" \
    -o results.json \
    http://localhost:5000/detect

# visualize the results (output.jpg)
python visualize.py len_std.jpg results.json
```

