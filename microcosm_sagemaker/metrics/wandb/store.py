from os import environ

from microcosm_logging.decorators import logger

from microcosm_sagemaker.decorators import metrics_observer, training_initializer


try:
    import wandb
except ImportError:
    pass


@training_initializer()
@metrics_observer()
@logger
class WeightsAndBiases:
    def __init__(self, graph):
        self.graph_config = graph.config
        self.testing = graph.metadata.testing
        self.project_name = graph.metadata.name.replace("_", "-")

    def init(self):
        # Only initialize wandb if it is not a testing
        if not self.testing:
            # TODO: Remove this line if devops come up with a solution
            environ["WANDB_API_KEY"] = "e9698d764c80c1076a1ff017fa9f722b3135624a"

            wandb.init(
                project=self.project_name,
                config=self.graph_config,
            )
            self.logger.info("`weights & biases` was registered as a metric logger.")

    def log_time_series(self, *args, **kwargs):
        wandb.log(*args, **kwargs)
        return None

    def log_static(self, **kwargs):
        wandb.run.summary.update(kwargs)
        return None
