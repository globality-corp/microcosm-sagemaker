import os

from microcosm_sagemaker.decorators import metrics_observer, training_initializer


try:
    import wandb
except ImportError:
    pass


@training_initializer()
@metrics_observer()
class WeightsAndBiases:
    def __init__(self, graph):
        self.graph_config = graph.config
        self.testing = graph.metadata.testing
        self.name = graph.metadata.name

    def init(self):
        # os.environ["WANDB_API_KEY"] = "XXXXXXXX"
        if self.testing:
            os.environ["WANDB_MODE"] = "dryrun"
        wandb.init(
            project=self.name.replace("_", "-"),
            config=self.graph_config
        )

    def log_metric(self, log_dict, *args, **kwargs):
        wandb.log(log_dict, *args, **kwargs)

    def run_summary(self, summary_dict, *args, **kwargs):
        wandb.run.summary.update(summary_dict, *args, **kwargs)
