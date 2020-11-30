from unittest.mock import patch

from microcosm.api import create_object_graph, load_from_dict

import wandb


class TestWandbExistingRun():
    """
    Ensures that both log_static and log_timerseries methods work
    when an existing wandb run is loaded.

    """

    def setup(self):
        self.graph = create_object_graph(
            name="test-project",
            loader=load_from_dict(
                dict(
                    active_bundle="simple_bundle_with_metric",
                    wandb=dict(
                        run_path="WANDB_RUN_PATH"
                    ),
                )
            ),
            testing=False,
        )
        self.graph.use("training_initializers", "wandb", "simple_bundle_with_metric")
        self.graph.lock()

    def test_init(self):
        with patch.object(wandb.Api, "run") as wandb_api_run:
            self.graph.training_initializers.init()

            wandb_api_run.assert_called_with(
                path="WANDB_RUN_PATH"
            )

    def test_log_static_metric(self):
        with patch.object(wandb.Api, "run"):
            self.graph.training_initializers.init()

            self.graph.simple_bundle_with_metric.log_static_metric()
            self.graph.wandb.wandb_run.summary.update.assert_called_with(
                {"static_metric": 3}
            )

    def test_log_timeseries_metric(self):
        with patch.object(wandb.Api, "run"):
            self.graph.training_initializers.init()

            self.graph.simple_bundle_with_metric.log_timeseries_metric()
            self.graph.wandb.wandb_run.log.assert_called_with(
                {"timeseries_metric": 1},
                step=0,
            )
