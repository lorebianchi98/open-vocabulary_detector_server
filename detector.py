import time

import torch
from torchvision.ops import batched_nms
from transformers import OwlViTProcessor, OwlViTForObjectDetection


class OpenVocabularyDetector:

    DEFAULT_PROMPT = "A photo of a "
    DEFAULT_VOCABULARY = [
        "dog",
        "cat",
        "car",
        "bicycle",
        "person",
        "tree",
        "flower",
        "building",
        "mountain",
        "river",
    ]
    DEFAULT_SCORE_THRESH = 0.05
    DEFAULT_NMS_THRESH = 0.50
    DEFAULT_MAX_PREDICTIONS = 50


    def __init__(self, device=None):
        # Use GPU if available
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # loading the model
        print(f"Loading the model (device: {device}) ... ", end="", flush=True)
        elapsed = -time.time()
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = self.model.to(self.device)
        self.model.eval()
        elapsed += time.time()
        print(f"done ({elapsed:.3f}s)")

    def postprocess(self, outputs, score_thresh, nms_thresh, max_predictions):
        # score thresholding and ccwh to xyxy conversion
        outputs = self.processor.post_process_object_detection(outputs=outputs, threshold=score_thresh)

        # nms
        if nms_thresh:
            for output in outputs:
                keep = batched_nms(output['boxes'], output['scores'], output['labels'], nms_thresh)
                output['boxes' ] = output['boxes' ][keep]
                output['scores'] = output['scores'][keep]
                output['labels'] = output['labels'][keep]

        # keep only the top-k predictions
        if max_predictions:
            for output in outputs:
                keep = torch.argsort(output['scores'], descending=True)[:max_predictions]
                output['boxes' ] = output['boxes' ][keep]
                output['scores'] = output['scores'][keep]
                output['labels'] = output['labels'][keep]

        return outputs

    @torch.no_grad()
    def detect(
        self,
        image,
        vocabulary=None,
        prompt=None,
        score_thresh=None,
        max_predictions=None,
        nms_thresh=None,
    ):
        elapsed = -time.time()
        vocabulary = vocabulary if vocabulary is not None else self.DEFAULT_VOCABULARY
        prompt = prompt if prompt is not None else self.DEFAULT_PROMPT
        score_thresh = score_thresh if score_thresh is not None else self.DEFAULT_SCORE_THRESH
        nms_thresh = nms_thresh if nms_thresh is not None else self.DEFAULT_NMS_THRESH
        max_predictions = max_predictions if max_predictions is not None else self.DEFAULT_MAX_PREDICTIONS

        # inserting the prompt
        original_vocabulary = vocabulary
        if prompt:
            vocabulary = [prompt + word for word in vocabulary]

        # preparing the inputs
        inputs = self.processor(text=vocabulary, images=image, return_tensors="pt", padding=True).to(self.device)
        assert inputs['input_ids'].shape[1] <= 16, "The tokens length is above 16, the model can't handle them"

        outputs = self.model(**inputs)  # get predictions
        outputs = self.postprocess(outputs, score_thresh, nms_thresh, max_predictions)  # post-processing
        elapsed += time.time()

        # we assume that the batch size is 1
        output = outputs[0]
        output = {k: v.tolist() for k, v in output.items()}  # tensors to native python types
        output['inference_time_s'] = elapsed
        output['vocabulary'] = original_vocabulary
        return output


if __name__ == "__main__":
    import timeit
    from PIL import Image
    from visualize import draw

    detector = OpenVocabularyDetector()
    vocabulary = ["hat", "face", "scarf", "eye"]

    image = "len_std.jpg"
    image = Image.open(image)
    print("Image size:", image.size, "(width x height)")

    output = detector.detect(
        image,
        vocabulary=vocabulary,
        max_predictions=0,
        score_thresh=0.05,
        nms_thresh=0.50,
        prompt="A photo of a "
    )

    draw(image, output)
    image.save("output.jpg")

    # benchmarking the inference time and memory usage
    torch.cuda.reset_peak_memory_stats()
    elapsed = timeit.timeit("detector.detect(image, vocabulary=vocabulary)", globals=globals(), number=100)
    print(f"Average inference time: {elapsed / 100:.3f}s (FPS: {100 / elapsed:.2f})")
    print(f"Max GPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
