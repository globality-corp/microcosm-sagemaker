# microcosm-sagemaker
Opinionated machine learning with SageMaker

## Usage
For best practices, see
[`cookiecutter-microcosm-sagemaker`](https://github.com/globality-corp/cookiecutter-microcosm-sagemaker).

## Profiling
Make sure `pyinstrument` is installed, either using `pip install pyinstrument` or by installing `microcosm-sagemaker` with `profiling` extra dependencies:

```
pip install -e '.[profiling]'
```

To enable profiling of the app, use the `--profile` flag with `runserver`:

```
runserver --profile
```

The service will log that it is in profiling mode and announce the directory to which it is exporting. Each call to the endpoint will be profiled and its results with be stored in a time-tagged html file in the profiling directory.

## Experiment Tracking
To use `Weights and Biases`, install `microcosm-sagemaker` with `wandb` extra depdency:

```
pip install -e '.[wandb]'
```

To enable experiment tracking in an ML repository:

* Choose the experiment tracking stores for your ML model. It is recommended to store metrics in both `wandb` and Amazon `cloudwatch`. To do so, add `wandb` and `cloudwatch` to `graph.use()` in `app_hooks/train/app.py` and `app_hooks/evaluate/app.py`.

* Add the API key for `wandb` to your ML model's `config.json` file:

```
{
    "wandb": {
        "api_key": "XXXXXX"
    }
}
```

* To define hyperparameters for your model:

```
from microcosm.api import defaults, binding
from microcosm_sagemaker.bundle import Bundle
from microcosm_sagemaker.hyperparameters import hyperparameter

@binding("my_classifier")
@defaults(
    param = 10,
    hyperparam = hyperparameter(20),
)
class MyClassifier(Bundle):
    ...
```

That automatically adds the hyperparameters to your experiment, which simplifies hyperparameter optimization and tuning.

* To report a static metric:

```
class MyClassifier(Bundle):
    ...

    def fit(self, input_data):
        ...
        self.experiment_metrics.log_static(<metric_name>=<metric_value>)
```

* To report a time-series metric:

```
class MyClassifier(Bundle):
    ...

    def fit(self, input_data):
        ...
        self.experiment_metrics.log_timeseries(
            <metric_name>=<metric_value>,
            step=<step_number>
        )
```

Note that the `step` keyword argument must be provided for logging time-series.
