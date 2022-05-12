"""
PyTest configuration script
"""

def pytest_addoption(parser):
    """
    Register argparse-style options.
    """
    parser.addoption(
        "--model-params",
        dest="model_params",
        action="append",
        default=[],
        help="Model's parameters",
    )

def pytest_generate_tests(metafunc):
    """
    Defines the custom parametrization schema for each test. It is called each
    time a test is collected.
    """
    if "model_params" in metafunc.fixturenames:
        metafunc.parametrize("model_params", metafunc.config.getoption("model_params"))
