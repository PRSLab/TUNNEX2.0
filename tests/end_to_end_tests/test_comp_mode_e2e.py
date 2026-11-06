# --- Test: test_comp_mode_e2e. End-to-end tests for tunnex_2 --comp
# (no Gaussian or ORCA is launched; the launch is checked with trap scripts)
#
# Run all tests with: pytest tests/end_to_end_tests/test_comp_mode_e2e.py -v ---
# Run only e2e tests: pytest tests/end_to_end_tests/test_comp_mode_e2e.py -v -m e2e ---
#
# Update the golden (reference) output files:
#   pytest tests/end_to_end_tests/test_comp_mode_e2e.py -v --update-golden
#
# Without --update-golden, the generated numerical results are compared
# against the existing reference files using GOLDEN_RTOL.
#
# With --update-golden, the reference files are overwritten with the
# outputs of successful runs. Review the changes before committing:
#   git diff -- tests/end_to_end_tests/golden/
#
# The --update-golden option is defined in tests/end_to_end_tests/conftest.py
# and is available to tests through the update_golden fixture.
# All the computation tests including the run of Gaussian or ORCA were done manually on the clusters.


# --- Modules ---
import pytest  # type: ignore

import numpy as np
import os
import re
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from tunnex_2.qmt_computations.data_reader_writer.reader import read_input  # type: ignore


# --- Helpers and shared data ---
pytestmark = pytest.mark.e2e

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The folder where pyproject.toml is located
DATA_DIR = PROJECT_ROOT / "tests/end_to_end_tests/data/"
# The folder with the example input files and the reference outputs
GOLDEN_DIR = PROJECT_ROOT / "tests/end_to_end_tests/golden/"
# The folder with the reference outputs

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

PROGRAMS = {
    "gaussian": dict(
        directory="gaussian", flags=[],
        files=dict(ts_input="ch3coh.gjf", ts="ch3coh_ts.out", irc="ch3coh_irc.out",
                   react="ch3coh_react.out", prod="ch3coh_prod.out"),
        required=["ch3coh.gjf", "ch3coh_ts.out", "ch3coh_irc.out", "ch3coh_react.out", "ch3coh_prod.out"],
        terminated=["ch3coh_ts.out", "ch3coh_irc.out", "ch3coh_react.out", "ch3coh_prod.out"],
        marker="Normal termination"),
    "orca": dict(
        directory="orca", flags=["-o"],
        irc_flags=["-z"],  # adding -z to avoid the hessian file parsing (incompatible with -e)
        files=dict(ts_input="ch3coh.inp", ts="ch3coh_ts.out", irc="ch3coh_irc_IRC_Full_trj.xyz",
                   react="ch3coh_react.out", prod="ch3coh_prod.out"),
        required=["ch3coh.inp", "ch3coh_ts.out", "ch3coh_ts.hess", "ch3coh_irc_IRC_Full_trj.xyz",
                  "ch3coh_irc_IRC_B.xyz", "ch3coh_irc_IRC_F.xyz", "ch3coh_irc.out",
                  "ch3coh_react.out", "ch3coh_react.hess", "ch3coh_prod.out", "ch3coh_prod.hess"],
        terminated=["ch3coh_ts.out", "ch3coh_irc.out", "ch3coh_react.out", "ch3coh_prod.out"],
        marker="ORCA TERMINATED NORMALLY")}
        # The program reference data


@pytest.fixture(scope="module")
def tunnex_cmd() -> list[str]:
    """ The installed console command """
    exe = shutil.which("tunnex_2")
    if exe is not None:
        return [exe]
    return [sys.executable, "-c", "from tunnex_2.cli import cli; cli()"]

def _snapshot(directory: Path) -> dict[str, str]:
    """ File name -> SHA-256 of the content for all the files of the directory """
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.iterdir() if path.is_file()}

def _numbers(text: str) -> np.ndarray:
    """ All the numbers of the text (the lines matching SKIPPED_LINES are dropped) """
    values = []
    for line in text.splitlines():
        if SKIPPED_LINES.search(line):
            continue
        values.extend(float(x.replace("D", "e").replace("d", "e")) for x in NUMBER.findall(line))
    return np.array(values)

def _outputs(work: Path) -> list[Path]:
    """ Should create the output files """
    return sorted(work.glob(f"*{OUTPUT_SUFFIX}"))

def _tunnex_input_for(work: Path, output: Path, original: dict) -> Path:
    """ Find the Tunnex 2.0 input file (created by the IRC stage) which the output was computed from """
    base = output.name[:-len(OUTPUT_SUFFIX)]
    now = _snapshot(work)
    found = [work / n for n, d in now.items()
            if Path(n).stem == base and n != output.name and original.get(n) != d]
    assert len(found) == 1, f"expected one Tunnex 2.0 input for {output.name}, got {[p.name for p in found]}"
    return found[0]

def _check_golden(case_id: str, output: Path, update: bool):
    """ Compare the numbers of the output with the reference file, or (re)write the reference file """
    golden = GOLDEN_DIR / case_id / output.name
    if update:
        golden.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(output, golden)
        return
    if not golden.is_file():
        pytest.skip(f"no reference file {golden}; run pytest with --update-golden, check the result and commit it")
    got, expected = _numbers(output.read_text(encoding="utf-8")), _numbers(golden.read_text(encoding="utf-8"))
    assert got.shape == expected.shape, \
    f"{output.name}: expected {expected.shape[0]} numbers, got {got.shape[0]}; test #1"
    deviation = np.abs(got - expected) / np.maximum(np.abs(expected), 1e-300)
    assert np.allclose(got, expected, rtol=GOLDEN_RTOL, atol=1e-12), \
    f"{output.name}: max relative deviation {np.max(deviation)}; test #2"


@pytest.fixture(params=["gaussian", "orca"])
def case(request, tmp_path, tunnex_cmd):
    """ A copy of the recorded files, a trap for the launch of Gaussian or ORCA and the command lines """
    program = request.param
    config = PROGRAMS[program]
    source = DATA_DIR / config["directory"]
    missing = [name for name in config["required"] if not (source / name).is_file()]
    if missing:
        pytest.skip(f"the recorded files {missing} are not found in {source}")
    work = tmp_path / "work"
    shutil.copytree(source, work)

    # --- Fake Gaussian or ORCA executables which fail loudly and leave a trace ---
    bin_dir = tmp_path / "trap_bin"
    bin_dir.mkdir()
    trap_log = tmp_path / "trap_calls.txt"
    if sys.platform == "win32":
        for name in ("g16", "g09", "g03", "orca"):
            script = bin_dir / f"{name}.cmd"
            script.write_text(
                f'@echo off\r\n'
                f'echo {name} %*>>"{trap_log}"\r\n'
                f'exit /b 99\r\n')
    else:  # Pylance greys this out on Windows
        for name in ("g16", "g09", "g03", "orca"):
            script = bin_dir / name
            script.write_text(
                f'#!/bin/sh\n'
                f'echo "{name} $@" >> "{trap_log}"\n'
                f'exit 99\n')
            script.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    if sys.platform == "win32":
        env["PATHEXT"] = ".COM;.EXE;.BAT;.CMD"
    files = config["files"]
    flags = config["flags"]
    irc_flags = config.get("irc_flags", [])
    return SimpleNamespace(
        program=program,
        config=config,
        source=source,
        work=work,
        tmp=tmp_path,
        env=env,
        trap_log=trap_log,
        cmd=tunnex_cmd,
        original=_snapshot(work),
        full_argv=flags + irc_flags + ["-c", files["ts_input"], files["ts"],
                                       files["irc"], files["react"], files["prod"]],
        eckart_argv=flags + ["-e", "-c", files["ts_input"],
                             files["ts"], files["react"], files["prod"]])

def _run(case, argv, cwd=None) -> subprocess.CompletedProcess:
    """ Run tunnex_2 in a separate process (the trap is in the PATH) """
    return subprocess.run([*case.cmd, *argv], cwd=cwd or case.work, env=case.env,
                          capture_output=True, text=True, timeout=TIMEOUT)

def _copy_recorded(case, name: str) -> Path:
    """ Make one more clean copy of the recorded files """
    destination = case.tmp / name
    shutil.copytree(case.source, destination)
    return destination

def _assert_clean_run(done, case):
    assert done.returncode == 0, \
    f"expected exit code {0}, got {done.returncode}; stderr: {done.stderr[-800:]}; test #1"
    assert "Traceback" not in done.stderr, \
    f"expected no traceback, got {done.stderr[-800:]}; test #2"
    assert not case.trap_log.exists(), \
    f"expected no launch of Gaussian/ORCA, got {case.trap_log.read_text(encoding="utf-8")}; test #3"


# ===========================================================================
# --- recorded data - sanity check ---
# ===========================================================================
class TestRecordedData:

    def test_calculations_terminated_normally(self, case):
        """
        The recorded outputs are complete: the program has terminated normally
        (the recording is not corrupted)
        """
        for name in case.config["terminated"]:
            path = case.source / name
            if not path.is_file():
                continue
            assert case.config["marker"] in path.read_text(errors="ignore", encoding="utf-8"), \
            f"{name}: expected {case.config['marker']!r} in the file"


# =====================================================================================
# --- tunnex_2 --comp - TS guess, TS, IRC, reactant and product are given (5 files) ---
# =====================================================================================
class TestCompFullRun:

    def test_run_succeeds_and_creates_two_outputs(self, case):
        """
        The run succeeds without launching Gaussian or ORCA;
        the forward and backward outputs are created
        """
        done = _run(case, case.full_argv)
        _assert_clean_run(done, case)
        outputs = _outputs(case.work)
        assert len(outputs) == 2, \
        f"expected {2} outputs, got {[p.name for p in outputs]}; test #1"
        assert all(path.stat().st_size > 0 for path in outputs), \
        "expected non-empty outputs; test #2"

    def test_generated_tunnex_inputs_are_valid(self, case):
        """
        The Tunnex 2.0 input files made by the IRC stage are read
        by the real reader and contain a barrier
        """
        _assert_clean_run(_run(case, case.full_argv), case)
        for output in _outputs(case.work):
            tunnex_input = _tunnex_input_for(case.work, output, case.original)
            params, irc = read_input(tunnex_input)
            assert params.freq > 0 and params.T > 0, \
            f"{tunnex_input.name}: expected positive freq and T; test #1"
            assert params.number_of_levels >= 0, \
            f"{tunnex_input.name}: expected number_of_levels >= 0; test #2"
            top = int(np.argmax(irc.E_y))
            assert 0 < top < len(irc.E_y) - 1, \
            (f"{tunnex_input.name}: expected the energy maximum inside the IRC, "
            f"got the index {top} of {len(irc.E_y)}; test #3")

    def test_outputs_match_reference(self, case, update_golden):
        """ The numbers of the outputs equal the numbers of the reference outputs """
        _assert_clean_run(_run(case, case.full_argv), case)
        for output in _outputs(case.work):
            _check_golden(f"{case.program}", output, update_golden)

    def test_recorded_files_are_not_modified(self, case):
        """ The recorded input and output files stay byte-identical """
        _assert_clean_run(_run(case, case.full_argv), case)
        after = _snapshot(case.work)
        changed = [name for name, digest in case.original.items() if after.get(name) != digest]
        assert changed == [], f"expected the recorded files to stay unchanged, changed: {changed}"

    def test_repeated_runs_give_identical_outputs(self, case):
        """ Two runs in two clean copies give byte-identical outputs """
        _assert_clean_run(_run(case, case.full_argv), case)
        second = _copy_recorded(case, "work_second")
        _assert_clean_run(_run(case, case.full_argv, cwd=second), case)
        first_outputs = {path.name: path.read_bytes() for path in _outputs(case.work)}
        second_outputs = {path.name: path.read_bytes() for path in _outputs(second)}
        assert first_outputs == second_outputs, "expected identical outputs for the repeated runs"


# =================================================================================
# --- tunnex_2 --comp --no_qmt, then tunnex_2 --qmt_only on the generated input ---
# =================================================================================
class TestCompIRCStageThenQmtOnly:

    def test_no_qmt_makes_tunnex_inputs_without_outputs(self, case):
        """ --no_qmt creates the four input files and does not write QMT outputs """
        argv = [*case.full_argv, "--no_qmt"]
        done = _run(case, argv)
        _assert_clean_run(done, case)
        if case.program == "gaussian":
            expected_inputs = ['ch3coh_backward.txt', 'ch3coh_forward.txt',
            'ch3coh_prod_corr.txt', 'ch3coh_react_corr.txt', 'gaussian_atom_masses.out']
        else:
            expected_inputs = ['ch3coh_backward.txt', 'ch3coh_forward.txt',
            'ch3coh_prod_corr.txt', 'ch3coh_react_corr.txt', 'orca_atom_masses.out']
        missing = [name for name in expected_inputs if not (case.work / name).is_file()]
        assert not missing, (f"expected all four Tunnex input files, missing: {missing}; test #1")
        assert _outputs(case.work) == [], \
        f"expected no outputs, got {[p.name for p in _outputs(case.work)]}; test #2"

    def test_qmt_only_on_generated_inputs_reproduces_full_run(self, case):
        """
        The outputs of the full run equal the outputs
        of 'tunnex_2 --qmt_only' on the Tunnex inputs of the --no_qmt run
        """
        _assert_clean_run(_run(case, case.full_argv), case)
        full_outputs = {path.name: path.read_bytes() for path in _outputs(case.work)}
        only_irc = _copy_recorded(case, "work_irc_only")
        original = _snapshot(only_irc)
        _assert_clean_run(_run(case, [*case.full_argv, "--no_qmt"], cwd=only_irc), case)
        created = [n for n in _snapshot(only_irc) if n not in original and not n.endswith(".log")]
        assert len(created) >= 2, f"expected at least 2 new files, got {created}; test #1"
        for name in full_outputs:
            base = name[:-len(OUTPUT_SUFFIX)]
            candidates = [n for n in created if Path(n).stem == base]
            assert len(candidates) == 1, \
                f"expected one Tunnex input for {name}, got {candidates}; test #2"
            done = _run(case, ["--qmt_only", candidates[0]], cwd=only_irc)
            assert done.returncode == 0, \
                f"expected exit code 0, got {done.returncode}; stderr: {done.stderr[-800:]}; test #3"
            assert (only_irc / name).read_bytes() == full_outputs[name], \
                f"{name}: expected the same output; test #4"


# =========================================================================================
# --- tunnex_2 --eckart --comp - TS guess, TS, reactant and product are given (4 files) ---
# =========================================================================================
class TestEckartCompRun:

    def test_run_succeeds_and_creates_outputs(self, case):
        """
        The Eckart run on the recorded files succeeds without launching
        Gaussian or ORCA and writes outputs
        """
        done = _run(case, case.eckart_argv)
        _assert_clean_run(done, case)
        outputs = _outputs(case.work)
        assert len(outputs) >= 1, \
        f"expected at least {1} output, got {[p.name for p in outputs]}; test #1"
        assert all(path.stat().st_size > 0 for path in outputs), "expected non-empty outputs; test #2"

    def test_outputs_match_reference(self, case, update_golden):
        """ The numbers of the Eckart outputs equal the numbers of the reference outputs """
        done = _run(case, case.eckart_argv)
        _assert_clean_run(done, case)
        for output in _outputs(case.work):
            _check_golden(f"{case.program}_eckart", output, update_golden)