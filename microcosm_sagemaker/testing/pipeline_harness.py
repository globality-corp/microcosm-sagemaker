from abc import ABC, abstractmethod
from inspect import signature


class PipelineHarness(ABC):

    """
    Base class to support feature tests, by specifying the steps/pipes
    of a pipeline
    """

    @property
    @abstractmethod
    def _steps(self):
        pass

    def test_steps(self):
        store = {}
        for fn in self._steps:
            fn_parameters = signature(fn).parameters

            # Pass through only the parameters that the function requires
            current_output = fn(**{key: store[key] for key in fn_parameters})

            # update store if any outputs are returned
            store.update(current_output or {})
