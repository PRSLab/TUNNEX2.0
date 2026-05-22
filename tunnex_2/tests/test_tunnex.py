"""
Unit and regression test for the tunnex package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import tunnex


def test_tunnex_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "tunnex" in sys.modules
