from unittest.mock import patch

from hamcrest import (
    assert_that,
    equal_to,
    has_entries,
    is_,
)
from microcosm.api import create_object_graph, load_from_dict


class TestWandbNewRun():
    """
    Ensures that both log_static and log_timerseries methods work
    when a new wandb run is created.

    """

    def setup(self):
        self.graph = create_object_graph(
            name="test-project",
            loader=load_from_dict(
                dict(
                    active_bundle="simple_bundle_with_metric",
                )
            ),
            testing=False,
        )
        self.graph.use("training_initializers", "wandb", "simple_bundle_with_metric")
        self.graph.lock()

    def test_init(self):
        with patch("wandb.init") as wandb_init:
            self.graph.training_initializers.init()
            _, call_kw_args = wandb_init.call_args
            assert_that(
                call_kw_args,
                has_entries(
                    project=is_(equal_to("test-project")),
                    config=is_(equal_to(
                        dict(
                            simple_bundle_with_metric=dict(
                                param1=1,
                                param2=2,
                            )
                        )
                    ))
                )
            )

    def test_log_static_metric(self):
        with patch("wandb.init"):
            self.graph.training_initializers.init()

            self.graph.simple_bundle_with_metric.log_static_metric()
            self.graph.wandb.wandb_run.summary.update.assert_called_with(
                {"static_metric": 3}
            )

    def test_log_timeseries_metric(self):
        with patch("wandb.init"):
            self.graph.training_initializers.init()

            self.graph.simple_bundle_with_metric.log_timeseries_metric()
            self.graph.wandb.wandb_run.log.assert_called_with(
                row={"timeseries_metric": 1},
                step=0,
            )


if __name__ == "__main__":
    my_test = TestWandbNewRun()
    my_test.setup()
    my_test.test_init()
