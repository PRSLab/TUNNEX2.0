# --- conftest for tests/end_to_end_tests/ ---
import pytest  # type: ignore


def pytest_addoption(parser):
    " Updates the reference files "
    parser.addoption("--update-golden", action="store_true", default=False,
    help="write the current outputs of the end-to-end tests as the new reference (golden) files")


@pytest.fixture
def update_golden(request) -> bool:
    """ True when the golden files should be (re)written instead of being compared """
    return request.config.getoption("--update-golden")