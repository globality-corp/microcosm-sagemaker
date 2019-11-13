from os import environ

from microcosm_logging.decorators import logger

from microcosm_sagemaker.decorators import metrics_observer, training_initializer
from microcosm_sagemaker.hyperparameters import get_hyperparameters


try:
    import wandb
except ImportError:
    pass


@training_initializer()
@metrics_observer()
@logger
class WeightsAndBiases:
    def __init__(self, graph):
        self.graph = graph
        self.testing = graph.metadata.testing
        self.project_name = graph.metadata.name.replace("_", "-")

    def init(self):
        # Only initialize wandb if it is not a testing
        if not self.testing:
            # TODO: Remove this line if devops come up with a solution
            # https://globality.atlassian.net/browse/DEVOPS-635
            environ["WANDB_API_KEY"] = self.graph.config.wandb.api_key
            wandb.init(
                project=self.project_name,
                config={
                    flattened_hyperparam: value
                    for flattened_hyperparam, value in get_hyperparameters(self.graph)
                }
            )
            self.logger.info("`weights & biases` was registered as a metric observer.")

    def log_timeseries(self, **kwargs):
        step = kwargs.pop("step")
        wandb.log(row=kwargs, step=step)
        return None

    def log_static(self, **kwargs):
        wandb.run.summary.update(kwargs)
        return None
