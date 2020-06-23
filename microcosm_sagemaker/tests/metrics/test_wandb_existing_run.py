from unittest.mock import patch

from microcosm.api import create_object_graph, load_from_dict


class TestWandb():
    def setup(self):
        self.graph = create_object_graph(
            name="test-project",
            loader=load_from_dict(
                dict(
                    wandb=dict(api_key="API_KEY"),
                    wandb_run_path="WANDB_RUN_PATH",
                )
            ),
            testing=False,
        )
        self.graph.use("training_initializers", "wandb", "bundle_with_metric")
        self.graph.lock()

    def test_init(self):
        with patch("wandb.Api().run") as wandb_api:
            self.graph.training_initializers.init()
            wandb_api.assert_called_with(
                path="WANDB_RUN_PATH",
            )

    # TODO: How can we do this?!

    # def test_log_static_metric(self):
    #     with patch("wandb.run") as wandb_run:
    #         self.graph.bundle_with_metric.log_static_metric()
    #         wandb_run.summary.update.assert_called_with(
    #             {"static_metric": 3}
    #         )

    # def test_log_timeseries_metric(self):
    #     with patch("wandb.log") as wandb_log:
    #         self.graph.bundle_with_metric.log_timeseries_metric()
    #         wandb_log.assert_called_with(
    #             row={"timeseries_metric": 1},
    #             step=0,
    #         )


my_test = TestWandb()
my_test.setup()
my_test.test_init()