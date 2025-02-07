from nanoowl.owl_predictor import OwlPredictor
from nanoowl.tree_predictor import TreePredictor, Tree
from nanoowl.tree_drawing import draw_tree_output


class ServedModel:
    def __init__(self, model_name, image_encoder_engine):
        self.last_prompt = None
        self.tree = None
        self.clip_text_encodings = None
        self.owl_text_encodings = None
        self.predictor = TreePredictor(
            owl_predictor=OwlPredictor(
                model_name=model_name,
                image_encoder_engine=image_encoder_engine
            )
        )

    def set_vocabulary(self, prompt, threshold):
        new_tree = Tree.from_prompt(prompt, threshold)  # may raise RuntimeError

        self.last_prompt = prompt
        self.tree = new_tree
        self.clip_text_encodings = self.predictor.encode_clip_text(self.tree)
        self.owl_text_encodings = self.predictor.encode_owl_text(self.tree)

    def predict(self, image):
        output = self.predictor.predict(
            image=image,
            tree=self.tree,
            clip_text_encodings=self.clip_text_encodings,
            owl_text_encodings=self.owl_text_encodings,
        )
        return output

    def draw(self, image, output):
        return draw_tree_output(image, output, self.tree)