# --- conftest for tests/end_to_end_tests/ ---
import shutil
import sys

import pytest  # type: ignore


def pytest_addoption(parser):
    " Updates the reference files "
    parser.addoption("--update-golden", action="store_true", default=False,
    help="write the current outputs of the end-to-end tests as the new reference (golden) files")


@pytest.fixture
def update_golden(request) -> bool:
    """ True when the golden files should be (re)written instead of being compared """
    return request.config.getoption("--update-golden")


@pytest.fixture(scope="module")
def tunnex_cmd() -> list[str]:
    """ The installed console command """
    exe = shutil.which("tunnex_2")
    if exe is not None:
        return [exe]
    return [sys.executable, "-c", "from tunnex_2.cli import cli; cli()"]
