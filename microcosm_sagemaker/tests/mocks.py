from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from microcosm_sagemaker.constants import EVALUATE_APP_HOOK, SERVE_APP_HOOK, TRAIN_APP_HOOK
from microcosm_sagemaker.tests.app_hooks.evaluate.app import create_app as create_evaluate_app
from microcosm_sagemaker.tests.app_hooks.serve.app import create_app as create_serve_app
from microcosm_sagemaker.tests.app_hooks.train.app import create_app as create_train_app


@contextmanager
def mock_app_hooks():
    """
    Mock all app hooks to those defined in microcosm_sagemaker.tests.app_hooks

    """
    with patch(
        "microcosm_sagemaker.app_hooks.iter_entry_points",
        return_value=[
            _create_app_hook_mock(
                name=TRAIN_APP_HOOK,
                load=MagicMock(return_value=create_train_app),
            ),
            _create_app_hook_mock(
                name=SERVE_APP_HOOK,
                load=MagicMock(return_value=create_serve_app),
            ),
            _create_app_hook_mock(
                name=EVALUATE_APP_HOOK,
                load=MagicMock(return_value=create_evaluate_app),
            ),
        ]
    ):
        yield


def _create_app_hook_mock(name, load):
    """
    We can't just instantiate a MagicMock with the right kwargs above because
    the name parameter is expected by MagicMock itself.

    """
    mock = MagicMock(name=name)
    mock.configure_mock(
        name=name,
        load=load,
    )
    return mock
