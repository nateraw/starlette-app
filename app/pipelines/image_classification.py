from app.pipelines.base import Pipeline


class ImageClassificationPipeline(Pipeline):
    def __init__(self, model_id):
        self.model_id = model_id

    def __call__(self, inputs):
        return [{"XXX": 0.90}, {"YYY": 0.10}]
