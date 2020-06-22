from os import environ, getenv

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

        self.run_path = getattr(graph.config, "wandb_run_path", None)

    def init(self):
        # Only initialize wandb if it is not a testing
        if self.testing:
            return

        if self.run_path:
            # Pushing into an existing wandb experiment
            self.wandb_run = wandb.Api().run(path=self.run_path)

        else:
            # Creating a new wandb experiment
            self.wandb_run = wandb.init(
                project=self.project_name,
                config={
                    flattened_hyperparam: value
                    for flattened_hyperparam, value in get_hyperparameters(self.graph)
                }
            )

            # Injecting the wandb run path into the config
            self.graph.config.wandb_run_path = self.wandb_run.path

            self.logger.info("`weights & biases` was registered as a metric observer.")

    def log_timeseries(self, **kwargs):
        step = kwargs.pop("step")
        self.wandb_run.log(row=kwargs, step=step)
        return None

    def log_static(self, **kwargs):
        self.wandb_run.summary.update(kwargs)
        return None
