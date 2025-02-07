# SUN XR: "Open-Vocabulary Object Detection" Component

This component is a part of the [SUN XR](https://sun-xr-project.eu) project.

## Requirements

- stuff in `requirements.txt`

Tested with:
 - Python 3.10.12
 - torch==2.4.0
 - torchvision==0.19.0
 - pillow==10.4.0
 - transformers==4.45.2
 - timm==1.0.9
 - accelerate==1.0.0
 - matplotlib==3.7.3
 - imageio[ffmpeg]==2.36.0
 - Flask==3.1.0
 - git+https://github.com/NVIDIA-AI-IOT/torch2trt.git@4e820ae31b4e35d59685935223b05b2e11d47b03
 - git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1


## Getting Started

### Compile the model and start the server
Compile the model to TorchScript and TensorRT:
```bash
cd nanoowl
mkdir -p data
python3 -m nanoowl.build_image_encoder_engine --model_name google/owlvit-base-patch16 data/owl_image_encoder_patch16.engine
cd ..
```

Start the server (default port is 5000):
```bash
python server.py
```

### Perform detection
The main API endpoint is `/predict`. It accepts a POST request with an `image` file parameter:

```bash
# Download a test image
wget https://www.milwaukeetool.com.au/on/demandware.static/-/Sites-mwt-master-catalog/default/dw67e80874/Safety/protective-work-wear/48735041/48735041_APP_10.png -O test.png

# Send a request, save results to json file
curl -X POST -F "image=@test.png" -o results.json http://localhost:5000/predict

# visualize the results (output.jpg)
python visualize.py test.png results.json
```

### Set the detection vocabulary
You can use the `/vocabulary` endpoint to get (HTTP GET) and set (HTTP POST or PUT with a `prompt` parameter and an optional `default_threshold` parameter) the vocabulary.

```bash
# get current vocabulary
curl http://localhost:5000/vocabulary

# set new vocabulary
curl -X POST \
  -F prompt="[a person [a head {0.08} [glasses, a mask], an arm {0.01}]]" \
  -F default_threshold=0.32 \
  http://localhost:5000/vocabulary
```

Both requests will return the current vocabulary in the following format:
```json
TODO
```

#### Prompt format
The prompt format supports hierarchical detections and attribute classification, with the following syntax:

 - `[class1, class2, ..., classN]` - objects to search for in the current level of the hierarchy of detected objects, where the first level is the whole image.
 For example, the prompt `[a hat, a scarf, a face [an eye, a nose]]` will be interpreted as: search for `hat`s, `scarf`s, and `face`s in the image; if `face`s are detected, search for `eye`s and `nose`s in the detected faces.

 - `... class (attribute1, attribute2, ..., attributeN)` - if `class` is detected, assign one of the given attributes (in an exclusive manner) to it. E.g., `[a head (with a hat, without hats)]` will be interpreted as: if `head` is detected, assign either `with a hat` or `without hats` to it.

 - `class {threshold}` - set the detection threshold for the class. E.g., `[a person {0.5}]` will be interpreted as: use a score threshold of 0.5 for class `a person`. Detections with scores below the threshold will be filtered out from the returned results.

 ### Example prompts
  - `[a person {0.5}]` - search for a person in the image with a score threshold of 0.5.

  - `[a cart or container (full | empty)]` - search for a cart or container in the image; if a cart or container is detected, classify it as either `full` or `empty`.

  - `[a person [a head (with a helmet, without a helmet), a glove, a high visibility vest]]` - search for a person in the image; if a person is detected, search for a head, a glove, and a high visibility vest in the detected person; if a head is detected, assign either `with a helmet` or `without a helmet` to it.

  - `[a face [an eye, a nose](happy, sad)]` - search for a face in the image; if a face is detected, assign either `happy` or `sad` to it and search for an eye and a nose in the detected face.

  - `(indoors, outdoors)` - classify the image as either `indoors` or `outdoors`.


## Results format

Bounding boxes are in the format `[x0, y0, x1, y1]`, where `(x0, y0)` is the top-left corner and `(x1, y1)` is the bottom-right corner of the bounding box.
The `id` and `parent_id` fields are used to represent the hierarchy of detected objects. The `parent_id` of `-1` represents the whole image and is always present in the results.
The `label_names` and `labels` fields contain the detected class names and their corresponding indices in the vocabulary, respectively.
The `scores` field contains the detection or classification scores for each reported class.

```json
{
  "detections": [
    {
      "box": [0.0, 0.0, 1000.0, 1000.0],
      "id": 0,
      "label_names": ["image"],
      "labels": [0],
      "parent_id": -1,
      "scores": [1.0]
    },
    {
      "box": [264.6484375, 40.52734375, 716.30859375, 996.58203125],
      "id": 1,
      "label_names": ["a person"],
      "labels": [1],
      "parent_id": 0,
      "scores": [0.4062928259372711]
    },
    {
      "box": [394.5462646484375, 44.261932373046875, 580.575439453125, 287.24359130859375],
      "id": 2,
      "label_names": ["a head", "with a helmet"],
      "labels": [2, 3],
      "parent_id": 1,
      "scores": [0.08526547998189926, 0.861328125]
    },
    {
      "box": [260.10107421875, 771.5730590820312, 347.3970947265625, 926.5584716796875],
      "id": 4,
      "label_names": ["a glove"],
      "labels": [5],
      "parent_id": 1,
      "scores": [0.402551531791687]
    },
    {
      "box": [611.3858032226562, 766.9048461914062, 713.1533813476562, 926.5584716796875],
      "id": 5,
      "label_names": ["a glove"],
      "labels": [5],
      "parent_id": 1,
      "scores": [0.4932815432548523]
    },
    {
      "box": [310.98486328125, 310.1179504394531, 685.6107788085938, 803.3170776367188],
      "id": 3,
      "label_names": ["a high visibility vest"],
      "labels": [6],
      "parent_id": 1,
      "scores": [0.17654767632484436]
    }
  ],
  "time_s": 0.1469132900238037
}

