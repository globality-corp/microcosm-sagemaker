import tempfile
from pathlib import Path
from typing import Mapping

from hamcrest.core.base_matcher import BaseMatcher

from microcosm_sagemaker.commands import train
from microcosm_sagemaker.testing.cli_test_case import CliTestCase
from microcosm_sagemaker.testing.directory_comparison import directory_comparison


class TrainCliTestCase(CliTestCase):
    def test_train(self,
                   input_data_path: Path,
                   gold_output_artifact_path: Path,
                   output_artifact_matchers: Mapping[Path, BaseMatcher]):
        with tempfile.TemporaryDirectory() as output_artifact_path:
            self.run_and_check(
                command_name="train",
                command=train.main,
                args=[
                    "--input-data",
                    str(input_data_path),
                    "--output-artifact",
                    output_artifact_path,
                ],
            )

            directory_comparison(
                gold_dir=gold_output_artifact_path,
                actual_dir=Path(output_artifact_path),
                matchers=output_artifact_matchers,
            )
