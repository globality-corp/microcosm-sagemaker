from unittest.mock import patch

from microcosm.api import create_object_graph, load_from_dict


class TestWandbNewRun():
    def setup(self):
        self.graph = create_object_graph(
            name="test-project",
            loader=load_from_dict(
                dict(
                    wandb=dict(api_key="API_KEY")
                )
            ),
            testing=False,
        )
        self.graph.use("training_initializers", "wandb", "bundle_with_metric")
        self.graph.lock()

    def test_init(self):
        with patch("wandb.init") as wandb_init:
            self.graph.training_initializers.init()
            wandb_init.assert_called_with(
                project="test-project",
                config={
                    "bundle_with_metric__hyperparam": 2
                }
            )

    def test_log_static_metric(self):
        with patch("wandb.init"):
            self.graph.training_initializers.init()

            self.graph.bundle_with_metric.log_static_metric()
            self.graph.wandb.wandb_run.summary.update.assert_called_with(
                {"static_metric": 3}
            )

    def test_log_timeseries_metric(self):
        with patch("wandb.init"):
            self.graph.training_initializers.init()

            self.graph.bundle_with_metric.log_timeseries_metric()
            self.graph.wandb.wandb_run.log.assert_called_with(
                row={"timeseries_metric": 1},
                step=0,
            )
