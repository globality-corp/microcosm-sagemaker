import logging
import sys
import tempfile
from pathlib import Path
from traceback import print_exception
from typing import Mapping

from click.testing import CliRunner
from hamcrest import assert_that, equal_to, is_
from hamcrest.core.base_matcher import BaseMatcher

from microcosm_sagemaker.commands.train import main
from microcosm_sagemaker.testing.directory_comparison import directory_comparison


class TrainCliTestCase:
    def setup(self):
        self.runner = CliRunner()

    def run_and_check(self, *args):
        logging.info(f"Running command: train {' '.join(args)}")

        result = self.runner.invoke(main, args)

        if result.exit_code != 0:
            sys.stdout.write(result.output)
            if result.exc_info is not None:
                print_exception(*result.exc_info)

        assert_that(result.exit_code, is_(equal_to(0)))

    def test_train(self,
                   input_data_path: Path,
                   gold_output_artifact_path: Path,
                   output_artifact_matchers: Mapping[Path, BaseMatcher]):
        with tempfile.TemporaryDirectory() as output_artifact_path:
            self.run_and_check(
                "--input-data",
                str(input_data_path),
                "--output-artifact",
                output_artifact_path,
                "--no-auto-evaluate",
            )

            directory_comparison(
                gold_dir=gold_output_artifact_path,
                actual_dir=Path(output_artifact_path),
                matchers=output_artifact_matchers,
            )
