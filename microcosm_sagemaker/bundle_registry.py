from microcosm.api import binding

from microcosm_sagemaker.artifact import InputArtifact, OutputArtifact
from microcosm_sagemaker.bundle import Bundle
from microcosm_sagemaker.input_data import InputData


@binding("bundles")
class BundleRegistry:
    """
    The bundle registry is a place to register bundles that need to be
    fit, saved, and loaded.  Note that the fit, save and load commands for each
    bundle will be called in the order the bundles were registered.

    """
    def __init__(self, graph):
        self.graph = graph
        self.bundles = {}

    def register(self, name: str, bundle: Bundle):
        self.bundles[name] = bundle

    def fit(self, input_data: InputData):
        """
        Call `fit` on all registered bundles with the given `input_data`

        """
        for name, bundle in self.bundles.items():
            bundle.fit(input_data)

    def save(self, output_artifact: OutputArtifact):
        """
        Call `save` on all registered bundles, passing each its own nested
        output_artifact

        """
        for name, bundle in self.bundles.items():
            child_output_artifact = output_artifact / name
            child_output_artifact.init()
            bundle.save(child_output_artifact)

    def load(self, input_artifact: InputArtifact):
        """
        Call `load` on all registered bundles, passing each its own nested
        input_artifact

        """
        for name, bundle in self.bundles.items():
            bundle.load(input_artifact / name)
