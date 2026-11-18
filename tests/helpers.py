# --- Shared constants and helpers for the tunnex_2 test suite ---
import re
from pathlib import Path

import numpy as np


# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The folder where pyproject.toml is located
CH3COH_DATA_DIR = PROJECT_ROOT / "tests" / "data" / "ch3coh"
# The recorded Gaussian and ORCA calculations for CH3COH, one subfolder per program,
# shared by the integration and the end-to-end tests


# --- End-to-end settings ---
OUTPUT_SUFFIX = "_tunnex_2.out"
# The default output suffix for the files
TIMEOUT = 120
# The timeout time for subprocesses
GOLDEN_RTOL = 1e-6
# Relative tolerance of the comparison with the reference outputs

SKIPPED_LINES = re.compile(r"version|date|file|path|directory", re.IGNORECASE)
# The lines which are not compared
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")
# The number pattern used by a program


def numbers(text: str) -> np.ndarray:
    """ All the numbers of the text (the lines matching SKIPPED_LINES are dropped) """
    values = []
    for line in text.splitlines():
        if SKIPPED_LINES.search(line):
            continue
        values.extend(float(x.replace("D", "e").replace("d", "e")) for x in NUMBER.findall(line))
    return np.array(values)
