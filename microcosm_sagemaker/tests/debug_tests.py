# from microcosm_sagemaker.tests.fixtures.directory_comparison import DIRECTORY_COMPARISON_TEST_CASES
# from microcosm_sagemaker.tests.test_directory_comparison import test_directory_comparison


# # my_test = TestTrainCli()
# # my_test.setup()
# # my_test.test_train()

# test_directory_comparison(*DIRECTORY_COMPARISON_TEST_CASES[0])

from microcosm_sagemaker.tests.test_loaders import TestLoaders


my_test = TestLoaders()
my_test.setUp()
my_test.test_train_conventions_loader_order()
