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
        # TODO: Remove this line if devops come up with a solution
        # os.environ["WANDB_API_KEY"] = "XXXXXXXX"
        if not self.testing:
            # Only initialize wandb if it is not a testing
            wandb.init(
                project=self.name.replace("_", "-"),
                config=self.graph_config
            )

    def log_time_series(self, *args, **kwargs):
        wandb.log(*args, **kwargs)
        return None

    def log_static(self, **kwargs):
        wandb.run.summary.update(kwargs)
        return None
