from allennlp.predictors.predictor import Predictor


@Predictor.register("vanilla_predictor")
class VanillaPredictor(Predictor):
    pass
