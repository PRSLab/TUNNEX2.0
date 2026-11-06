# --- Test: test_qmt_mode_e2e. End-to-end tests for tunnex_2 --qmt-only
#
# Run all tests with: pytest tests/end_to_end_tests/test_qmt_mode_e2e.py -v ---
# Run only e2e tests: pytest tests/end_to_end_tests/test_qmt_mode_e2e.py.py -v -m e2e ---
#
# Run only the reference-output tests:
#   pytest tests/end_to_end_tests/test_qmt_mode_e2e.py -v -k TestReferenceOutputs
#
# Update the golden (reference) output files:
#   pytest tests/end_to_end_tests/test_qmt_mode_e2e.py -v --update-golden
#
# Without --update-golden, the generated numerical results are compared
# against the existing reference files using GOLDEN_RTOL.
#
# With --update-golden, the reference files are overwritten with the
# outputs of successful runs. Review the changes before committing:
#   git diff -- tests/end_to_end_tests/golden/
#
# The --update-golden option is defined in tests/end-to-end/conftest.py
# and is available to tests through the update_golden fixture.
# All the computation tests including the run of Gaussian or ORCA were done manually on the clusters.


# --- Modules ---
import pytest  # type: ignore

import numpy as np
import re
import shutil
import subprocess
import sys
from pathlib import Path


# --- Helpers and shared data ---
pytestmark = pytest.mark.e2e

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The folder where pyproject.toml is located
DATA_DIR = PROJECT_ROOT / "tests/end_to_end_tests/data/tunnex_2/"
# The folder with the example input files
GOLDEN_DIR = PROJECT_ROOT / "tests/end_to_end_tests/golden/tunnex_2/"
# The folder with the reference outputs

INPUT_NAMES = ["ch3coh_backward_gaussian.txt",
               "ch3coh_backward_orca.txt"]
BASELINE_INPUT = DATA_DIR / INPUT_NAMES[0]
# The valid input used as the starting point of the broken-input tests

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


# The broken inputs
BROKEN_INPUTS = [
    ("missing_key", lambda t: _drop_key(t, "T"), "Missing required keys"),
    ("unknown_key", lambda t: _append_to_header(t, 0, "foo=1"), "Unrecognized keys"),
    ("duplicate_key", lambda t: _append_to_header(t, 0, "freq=1000"), "duplicate key"),
    ("not_a_number", lambda t: _set_key(t, "freq", "abc"), "Invalid key=value format"),
    ("negative_freq", lambda t: _set_key(t, "freq", -5), "freq must be > 0"),
    ("fractional_N", lambda t: _set_key(t, "N", 7.5), "N must be an integer"),
    ("unknown_averaging_mode", lambda t: _set_key(t, "tunn_prob_aver", 7), "temperature averaging mode key"),
    ("T_min_not_below_T_max", lambda t: _set_key(t, "T_arrhenius_min", 2000), "cannot be >="),
    ("too_few_header_lines", lambda t: _keep_first_lines(t, 2), "at least 3 header lines"),
    ("irc_not_increasing", lambda t: _swap_first_two_rows(t, "IRC; E", "END; E"), "strictly increasing"),
    ("nan_in_irc", lambda t: _replace_first_row(t, "IRC; E", "END; E", "-2.296280; nan"), "Invalid IRC row"),
    ("zpve_start_mismatch", lambda t: _replace_first_row(t, "IRC; ZPVE", "END; ZPVE", "-2.2; 0.05"), "must match"),
    ("zpve_block_missing", lambda t: _drop_line(t, "IRC; ZPVE"), "empty or missing")]


@pytest.fixture(scope="module")
def tunnex_cmd() -> list[str]:
    """ The installed console command """
    exe = shutil.which("tunnex_2")
    if exe is not None:
        return [exe]
    return [sys.executable, "-c", "from tunnex_2.cli import cli; cli()"]

def _run(cmd, args, cwd) -> subprocess.CompletedProcess:
    """ Run the command in a separate process in the given working directory """
    return subprocess.run([*cmd, *args], cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT)

def _output_for(inp: Path) -> Path:
    """ Defines the output files for tunnex_2 """
    return inp.with_name(inp.stem + OUTPUT_SUFFIX)

def _numbers(text: str) -> np.ndarray:
    """ Search all the numbers of the text """
    values = []
    for line in text.splitlines():
        if SKIPPED_LINES.search(line):
            continue
        values.extend(float(x.replace("D", "e").replace("d", "e")) for x in NUMBER.findall(line))
    return np.array(values)

def _created_files(directory: Path, inp: Path) -> set[str]:
    """ The names of the files created in the directory except the input file """
    return {path.name for path in directory.iterdir()} - {inp.name}

def _set_key(text: str, key: str, value) -> str:
    """ Set key=value in the first 3 header lines """
    lines = text.splitlines()
    pattern = re.compile(rf"(?<![\w-]){re.escape(key)}\s*=\s*[^;\n]*")
    for i in range(3):
        new, count = pattern.subn(f"{key}={value}", lines[i], count=1)
        if count:
            lines[i] = new
            return "\n".join(lines) + "\n"
    raise AssertionError(f"key {key} was not found in the header")

def _drop_key(text: str, key: str) -> str:
    """ Remove key=value from the first 3 header lines """
    lines = text.splitlines()
    for i in range(3):
        items = [item for item in lines[i].split(";") if item.strip()]
        kept = [item for item in items if item.split("=")[0].strip() != key]
        if len(kept) != len(items):
            lines[i] = "; ".join(item.strip() for item in kept)
            return "\n".join(lines) + "\n"
    raise AssertionError(f"key {key} was not found in the header")

def _append_to_header(text: str, line_index: int, extra: str) -> str:
    """ Append extra content to a specified header line """
    lines = text.splitlines()
    lines[line_index] = lines[line_index].rstrip().rstrip(";") + f"; {extra}"
    return "\n".join(lines) + "\n"

def _keep_first_lines(text: str, n_lines: int) -> str:
    """ Keep only the first n lines of the text """
    return "\n".join(text.splitlines()[:n_lines]) + "\n"

def _block_rows(lines: list[str], start_tag: str, end_tag: str) -> list[int]:
    """ Return indices of non-empty rows between the specified tags """
    start = next(i for i, line in enumerate(lines) if line.strip() == start_tag)
    end = next(i for i, line in enumerate(lines) if line.strip() == end_tag)
    return [i for i in range(start + 1, end) if lines[i].strip()]

def _swap_first_two_rows(text: str, start_tag: str, end_tag: str) -> str:
    """ Swap the first two non-empty rows between the specified tags """
    lines = text.splitlines()
    first, second = _block_rows(lines, start_tag, end_tag)[:2]
    lines[first], lines[second] = lines[second], lines[first]
    return "\n".join(lines) + "\n"

def _replace_first_row(text: str, start_tag: str, end_tag: str, new_row: str) -> str:
    """ Replace the first non-empty row between the specified tags """
    lines = text.splitlines()
    lines[_block_rows(lines, start_tag, end_tag)[0]] = new_row
    return "\n".join(lines) + "\n"

def _drop_line(text: str, tag: str) -> str:
    """ Remove the line matching the specified tag """
    return "\n".join(line for line in text.splitlines() if line.strip() != tag) + "\n"

def _assert_rejected(done: subprocess.CompletedProcess, tmp_path: Path, message: str) -> None:
    """ The run is rejected by the parser: exit code 2, the message and the hint, no output files """
    assert done.returncode == 2, \
    f"expected exit code {2}, got {done.returncode}; stderr: {done.stderr[-500:]}; test #1"
    assert message in done.stderr, \
    f"expected {message!r} in {done.stderr[-500:]}; test #2"
    assert "Use --help for more information." in done.stderr, \
    f"expected the hint in {done.stderr[-500:]}; test #3"
    assert not list(tmp_path.glob("*_tunnex_2.out")), \
    "expected no output files; test #4"


# ===========================================================================
# --- tunnex_2 - command line ---
# ===========================================================================
class TestCommandLine:

    def test_help_exits_with_zero(self, tmp_path, tunnex_cmd):
        """ --help prints the usage and exits with the zero code """
        done = _run(tunnex_cmd, ["--help"], tmp_path)
        assert done.returncode == 0, \
        f"expected exit code {0}, got {done.returncode}; stderr: {done.stderr[-500:]}; test #1"
        assert "usage: tunnex_2" in done.stdout, \
        f"expected {'usage: tunnex_2'} in {done.stdout[:200]}; test #2"

    def test_qmt_only_with_two_files_exits_with_two(self, tmp_path, tunnex_cmd):
        """ --qmt_only with two files is rejected with the exit code 2 and a hint; nothing is created """
        done = _run(tunnex_cmd, ["--qmt_only", "a.txt", "b.txt"], tmp_path)
        assert done.returncode == 2, \
        f"expected exit code {2}, got {done.returncode}; test #1"
        assert "Use --help for more information." in done.stderr, \
        f"expected the hint in {done.stderr}; test #2"
        assert not list(tmp_path.glob("*_tunnex_2.out")), \
        "expected no output files; test #3"

    def test_qmt_only_with_irc_option_exits_with_two(self, tmp_path, tunnex_cmd):
        """ --qmt_only cannot be combined with the IRC calculation options """
        done = _run(tunnex_cmd, ["--qmt_only", "--orca", "a.txt"], tmp_path)
        _assert_rejected(done, tmp_path, "--qmt_only cannot be combined with IRC calculation options")
        # it should raise an error

    def test_qmt_only_with_no_qmt_exits_with_two(self, tmp_path, tunnex_cmd):
        """ --qmt_only and --no_qmt contradict each other """
        done = _run(tunnex_cmd, ["--qmt_only", "--no_qmt", "a.txt"], tmp_path)
        _assert_rejected(done, tmp_path, "--qmt_only cannot be combined with --no_qmt")
        # it should raise an error

    def test_qmt_only_without_files_exits_with_two(self, tmp_path, tunnex_cmd):
        """ --qmt_only without an input file is rejected and the number of files is reported """
        done = _run(tunnex_cmd, ["--qmt_only"], tmp_path)
        _assert_rejected(done, tmp_path, "requires exactly 1 input file, got 0")
        # it should raise an error

    def test_eckart_with_hybrid_exits_with_two(self, tmp_path, tunnex_cmd):
        """ --hybrid cannot be used with --eckart """
        done = _run(tunnex_cmd, ["--eckart", "--hybrid", "ts.txt", "react.txt", "prod.txt"], tmp_path)
        _assert_rejected(done, tmp_path, "--hybrid cannot be used with --eckart")
        # it should raise an error

    def test_orca_with_calcfc_exits_with_two(self, tmp_path, tunnex_cmd):
        """ --calcfc works only for Gaussian, so it is rejected together with --orca """
        done = _run(tunnex_cmd, ["--orca", "--calcfc", "ts.txt"], tmp_path)
        _assert_rejected(done, tmp_path, "--calcfc cannot be used with --orca (Gaussian only)")
        # it should raise an error

    def test_comp_with_wrong_number_of_files_exits_with_two(self, tmp_path, tunnex_cmd):
        """ --comp accepts only 2, 3 or 5 files; 4 files (without --eckart) are rejected """
        done = _run(tunnex_cmd, ["--comp", "a.txt", "b.txt", "c.txt", "d.txt"], tmp_path)
        _assert_rejected(done, tmp_path, "--comp requires 2, 3 or 5 input files, got 4")
        # it should raise an error


# ===========================================================================
# --- tunnex_2 --qmt_only - a valid run ---
# ===========================================================================
class TestQmtOnlyRun:

    @pytest.fixture
    def baseline(self, tmp_path):
        """ Copy the baseline input file to a temporary directory, or skip if it is missing """
        if not BASELINE_INPUT.is_file():
            pytest.skip(f"{BASELINE_INPUT} not found")
        inp = tmp_path / BASELINE_INPUT.name
        shutil.copy(BASELINE_INPUT, inp)
        return inp

    def test_run_succeeds_and_creates_output(self, baseline, tunnex_cmd):
        """
        The run exits with the zero code, there is no traceback,
        and the output file is created next to the input
        """
        done = _run(tunnex_cmd, ["--qmt_only", baseline.name], baseline.parent)
        out = _output_for(baseline)
        assert done.returncode == 0, \
        f"expected exit code {0}, got {done.returncode}; stderr: {done.stderr[-500:]}; test #1"
        assert "Traceback" not in done.stderr, \
        f"expected no traceback, got {done.stderr[-500:]}; test #2"
        assert out.is_file() and out.stat().st_size > 0, \
        f"expected a non-empty {out}; test #3"
        created = _created_files(baseline.parent, baseline)
        assert all(name == out.name or name.endswith(".log") for name in created), \
        f"expected only the output (and log) files, got {created}; test #4"

    def test_input_is_not_modified(self, baseline, tunnex_cmd):
        """ The input file stays byte-identical """
        before = baseline.read_bytes()
        _run(tunnex_cmd, ["-q", baseline.name], baseline.parent)
        assert baseline.read_bytes() == before, "expected the input file to stay unchanged"

    def test_repeated_runs_give_identical_output(self, baseline, tunnex_cmd):
        """ Two runs give byte-identical output files """
        _run(tunnex_cmd, ["-q", baseline.name], baseline.parent)
        first = _output_for(baseline).read_bytes()
        _run(tunnex_cmd, ["-q", baseline.name], baseline.parent)
        assert _output_for(baseline).read_bytes() == first, "expected identical output files"

    def test_absolute_path_from_another_directory(self, baseline, tmp_path_factory, tunnex_cmd):
        """
        The input given by an absolute path from another directory:
        the output is next to the input
        """
        elsewhere = tmp_path_factory.mktemp("elsewhere")
        done = _run(tunnex_cmd, ["-q", str(baseline)], elsewhere)
        assert done.returncode == 0, \
        f"expected exit code {0}, got {done.returncode}; stderr: {done.stderr[-500:]}; test #1"
        assert _output_for(baseline).is_file(), \
        f"expected the output {_output_for(baseline)} next to the input; test #2"
        assert not list(elsewhere.glob("*_tunnex_2.out")), \
        "expected no output in the working directory; test #3"

    def test_missing_input_fails_without_output(self, tmp_path, tunnex_cmd):
        """
        A missing input file gives a non-zero exit code
        and a clear message; nothing is written
        """
        done = _run(tunnex_cmd, ["-q", "no_such_file.txt"], tmp_path)
        assert done.returncode != 0, f"expected a non-zero exit code, got {done.returncode}; test #1"
        assert "File not found" in done.stderr, f"expected {'File not found'} in {done.stderr[-500:]}; test #2"
        assert not list(tmp_path.glob("*_tunnex_2.out")), "expected no output files; test #3"


# ===========================================================================
# --- tunnex_2 --qmt_only - the reference outputs ---
# ===========================================================================
class TestReferenceOutputs:

    @pytest.mark.parametrize("mode", [1, 2], ids=["finite_sum", "infinite_sum"])
    # skipping the "integral" mode because it requires the heavy computations,
    # it was tested manually
    @pytest.mark.parametrize("input_name", INPUT_NAMES, ids=["gaussian", "orca"])
    def test_output_matches_reference(self, tmp_path, tunnex_cmd, input_name, mode, update_golden):
        """ Compare output with the reference, or update the reference """
        source = DATA_DIR / input_name
        if mode == 1:   # mode == 1, "finite_sum"
            reference = GOLDEN_DIR / f"{Path(input_name).stem}_fin_tunnex_2.out"
        else:  # mode == 2, "infinite_sum"
            reference = GOLDEN_DIR / f"{Path(input_name).stem}_inf_tunnex_2.out"
        if not source.is_file():
            pytest.skip(f"{source} not found")
        if not update_golden and not reference.is_file():
            pytest.skip(f"Reference output {reference} not found")
        inp = tmp_path / input_name
        inp.write_text(_set_key(source.read_text(encoding="utf-8"), "tunn_prob_aver", mode), encoding="utf-8")
        done = _run(tunnex_cmd, ["-q", inp.name], tmp_path)
        assert done.returncode == 0, \
        f"expected exit code 0, got {done.returncode}; stderr: {done.stderr[-500:]}; test #1"
        output = _output_for(inp)
        assert output.is_file(), f"Output file {output} was not created; test #2"
        got_text = output.read_text(encoding="utf-8")
        if update_golden:
            reference.write_text(got_text, encoding="utf-8")
            return  # None
        expected_text = reference.read_text(encoding="utf-8")
        got = _numbers(got_text)
        expected = _numbers(expected_text)
        assert got.shape == expected.shape, \
        f"expected {expected.shape[0]} numbers, got {got.shape[0]}; test #3"
        assert np.allclose(got, expected, rtol=GOLDEN_RTOL, atol=1e-12), \
        f"max relative deviation {np.max(np.abs(got - expected) /
        np.maximum(np.abs(expected), 1e-300))}; test #4"


# ===========================================================================
# --- tunnex_2 --qmt_only - broken input files ---
# ===========================================================================
class TestBrokenInputs:

    @pytest.mark.parametrize("make_broken, message", [(b[1], b[2]) for b in BROKEN_INPUTS],
    ids=[b[0] for b in BROKEN_INPUTS])
    def test_broken_input_fails_without_output(self, tmp_path, tunnex_cmd, make_broken, message):
        """
        A broken input gives a non-zero exit code and the explaining message;
        no output file is written
        """
        if not BASELINE_INPUT.is_file():
            pytest.skip(f"{BASELINE_INPUT} not found")
        inp = tmp_path / "broken_input.txt"
        inp.write_text(make_broken(BASELINE_INPUT.read_text(encoding="utf-8")))
        done = _run(tunnex_cmd, ["-q", inp.name], tmp_path)
        assert done.returncode != 0, f"expected a non-zero exit code, got {done.returncode}; test #1"
        assert message in done.stderr, f"expected {message!r} in {done.stderr[-500:]}; test #2"
        assert not _output_for(inp).exists(), f"expected no output file, but {_output_for(inp)} exists; test #3"