from abc import ABC, abstractmethod


class PipelineHarness(ABC):

    @property
    @abstractmethod
    def _steps(self):
        pass

    def test_steps(self):
        step_input = {}

        for fn in self._steps:
            fn_arguments = fn.__code__.co_varnames

            # Pass through all arguments that the function requires, from what we're currently
            # caching as the outputs from previous runs
            try:
                current_output = fn(
                    **{
                        key: value
                        for key, value in step_input.items()
                        if key in fn_arguments
                    },
                )
            except Exception as e:
                raise e

            step_input = {
                **step_input,
                **(current_output or {}),
            }
