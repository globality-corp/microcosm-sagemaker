
from microcosm.api import defaults
from microcosm.object_graph import ObjectGraph

from microcosm_sagemaker.artifact import InputArtifact, OutputArtifact
from microcosm_sagemaker.input_data import InputData


@defaults(
    bundle_orchestrator="single_threaded_bundle_orchestrator",
)
class ActiveBundleAndDependenciesLoader:
    def __init__(self, graph: ObjectGraph):
        self.bundle_orchestrator = getattr(
            graph,
            graph.config.load_active_bundle_and_dependencies.bundle_orchestrator,
        )
        self.active_bundle = graph.active_bundle
        self.graph = graph

    def __call__(self, input_artifact: InputArtifact):
        def load(bundle):
            name = _get_component_name(self.graph, bundle)
            bundle.load(input_artifact / name)

        self.bundle_orchestrator(self.active_bundle, load)


@defaults(
    bundle_orchestrator="single_threaded_bundle_orchestrator",
)
class ActiveBundleAndDependenciesTrainer:
    def __init__(self, graph: ObjectGraph):
        self.bundle_orchestrator = getattr(
            graph,
            graph.config.train_active_bundle_and_dependencies.bundle_orchestrator,
        )
        self.active_bundle = graph.active_bundle
        self.graph = graph

    def __call__(self, input_data: InputData, output_artifact: OutputArtifact):
        def train(bundle):
            name = _get_component_name(self.graph, bundle)
            nested_output_artifact = output_artifact / name
            nested_output_artifact.init()

            bundle.fit(input_data)
            bundle.save(nested_output_artifact)

        self.bundle_orchestrator(self.active_bundle, train)


def _get_component_name(graph, component):
    return next(
        key
        for key, possible_component in graph.items()
        if possible_component == component
    )
