from typing import List

from microcosm.api import binding, defaults

from microcosm_sagemaker.bundle import Bundle
from microcosm_sagemaker.frameworks.allennlp.bundle import AllenNLPBundle


@binding("simple_allennlp_bundle")
@defaults(
    allennlp_parameters=dict(
        dataset_reader="text_classification_json",
        model=dict(
            type="basic_classifier",
            text_field_embedder=dict(
                token_embedders=dict(
                    tokens="bag_of_word_counts",
                ),
            ),
            seq2vec_encoder=dict(
                type="boe",
                embedding_dim=4,
            ),
        ),
        train_data_path=f"train/dataset.jsonl",
        validation_data_path=f"develop/dataset.jsonl",
        iterator="basic",
        trainer=dict(
            optimizer="adam",
            num_epochs=1,
            cuda_device=-1,
        ),
    )
)
class SimpleAllenNLPBundle(AllenNLPBundle):
    def __init__(self, graph):
        config = graph.config.simple_allennlp_bundle

        self.allennlp_parameters = config.allennlp_parameters

    @property
    def dependencies(self) -> List[Bundle]:
        return []

    def predict(self, text: str) -> dict:
        instance = self.predictor._dataset_reader.text_to_instance(text)

        return self.predictor.predict_instance(instance)
