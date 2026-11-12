# --- conftest for the whole tunnex_2 test suite ---
import logging

import pytest  # type: ignore


@pytest.fixture(autouse=True)
def _reset_package_logger():
    """
    Restore the tunnex_2 package logger after every test. A test which calls setup_logger()
    (directly or through main) leaves file handlers and propagate=False behind, which would
    hide the records from caplog in the tests that run later
    """
    yield
    logger = logging.getLogger("tunnex_2")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)
