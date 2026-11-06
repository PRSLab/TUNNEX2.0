# --- Test: test_main_int. Integration tests for main_int.py
# Run with: pytest test_main_int.py -v ---

# --- Modules ---
import pytest  # type: ignore
from contextlib import ExitStack
from unittest.mock import MagicMock, patch
pytestmark = pytest.mark.integration  # type: ignore

import tunnex_2  # type: ignore

import numpy as np
import os
import runpy
import subprocess
import sys
from pathlib import Path
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace


from tunnex_2.constants_and_dataclasses.dataclasses import FindIRCConfig  # type: ignore
from tunnex_2.irc_computations.ancillary_functions.interface import log_run_start  # type: ignore
from tunnex_2.qmt_computations.core.potential import Potential  # type: ignore
from tunnex_2.qmt_computations.data_reader_writer.writer import build_report  # type: ignore
from tunnex_2.qmt_computations.module_qmt_computations import qmt_computations_main  # type: ignore
from tunnex_2.qmt_computations.pipeline.qmt_computations_pipeline import qmt_computations_pipeline  # type: ignore


# --- Module to test ---
MAIN_MODULE = "tunnex_2.main"
QMT_MODULE = "tunnex_2.qmt_computations.module_qmt_computations"
from tunnex_2.main import main  # type: ignore


# --- Helpers and shared data ---
E0, ZPVE0 = -153.15, 0.02
# The reference electronic energy and ZPVE (Hartree) of the synthetic IRC

DATA_SPEC = {"forward": dict(B=0.375, T=300.0),
             "backward": dict(B=0.600, T=350.0),
             "input": dict(B=0.450, T=450.0)}
# Different barriers and temperatures, so the results of the files are distinguishable

OUTPUT_SUFFIX = "_tunnex_2.out"
# The default output suffix for the files 

PROJECT_ROOT = Path(tunnex_2.__file__).resolve().parent.parent
# The folder where pyproject.toml is located
ENTRY_FILE = PROJECT_ROOT / "tunnex_2" / "__main__.py"
# The entry file for tunnex_2

IRC_MAIN = "tunnex_2.irc_computations.module_irc_computations.irc_computations_main"
QMT_MAIN = "tunnex_2.qmt_computations.module_qmt_computations.qmt_computations_main"
# Defines where irc_main and qmt_main are located

TIMEOUT = 120
# The timeout time for subprocesses


@pytest.fixture(autouse=True)
def _work_in_tmp_dir(tmp_path, monkeypatch):
    """
    Run every test inside tmp_path, so the real logger
    can not write anything into the project folder
    """
    monkeypatch.chdir(tmp_path)


def _eckart(x, A=-0.085, B=0.375, L=3.275):
    """
    Asymmetric Eckart potential (Hartree), reactant on the left (V -> 0),
    product on the right (V -> A)
    """
    y = - np.exp(2 * np.pi * np.asarray(x, dtype=float) / L)
    result = - A * y / (1 - y) - B * y / (1 - y) ** 2
    return result

def _make_data(B, T, n_points=401, half_range=8.0):
    """ Return the synthetic (params, irc) pair """
    x = np.linspace(-half_range, half_range, n_points)
    irc = SimpleNamespace(E_x=x, E_y=E0 + _eckart(x, B=B),
                          ZPVE_x=x.copy(), ZPVE_y=np.full_like(x, ZPVE0))
    params = SimpleNamespace(E0=E0, ZPVE0=ZPVE0, potential_scaling_factor=1.0, T=T,
                             number_of_levels=3, freq=1000.0, prob_aver_mode_qmt="finite_sum",
                             T_arrhenius_min=200.0, T_arrhenius_max=400.0, T_step=50.0)
    return params, irc

def _expected_report(data) -> str:
    """ Return the report obtained by the independent manual chain of the real stages """
    params, irc = data
    potential = Potential(irc.E_x, irc.E_y, irc.ZPVE_x, irc.ZPVE_y,
                          params.E0, params.ZPVE0, params.potential_scaling_factor)
    interpolated_irc, levels, aver, arrhenius = qmt_computations_pipeline(params, potential)
    return build_report(levels, aver, params.T, arrhenius, interpolated_irc, potential.offset)

def _output_for(path) -> Path:
    """ Return the expected output path for the input path """
    path = Path(path).resolve()
    return path.with_name(path.stem + OUTPUT_SUFFIX)

def _read(path) -> str:
    """ Return the text of the file """
    return Path(path).read_text(encoding="utf-8")

def _run_module(cwd) -> subprocess.CompletedProcess:
    """ Run 'python -m tunnex_2' in a separate process in the given working directory """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), env.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, "-m", "tunnex_2"], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=TIMEOUT)

# --- Runs the function with the real stages ---
class _Harness:
    """ Prepares the files, the synthetic data and the config, and runs main with the real stages """
    def __init__(self, directory, with_tunnex_input=False, qmt_module=True,
                 irc_return=None, reader_override=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.ts_in, self.fwd, self.bwd, self.tun = (
            self._touch(directory / name) for name in (
            "mol.gjf", "tunnex_2_forward.txt", "tunnex_2_backward.txt", "tunnex_2_input.txt"))
        self.data = {"forward": _make_data(**DATA_SPEC["forward"]),
                     "backward": _make_data(**DATA_SPEC["backward"]),
                     "input": _make_data(**DATA_SPEC["input"])}
        self.by_path = {self.fwd.resolve(): self.data["forward"],
                        self.bwd.resolve(): self.data["backward"],
                        self.tun.resolve(): self.data["input"]}
        self.irc_return = (self.fwd, self.bwd) if irc_return is None else irc_return
        self.reader_override = reader_override
        self.cfg = FindIRCConfig(qmt_module=qmt_module,
                                 ts_input_file=self.ts_in,
                                 tunnex_input=self.tun if with_tunnex_input else None)
        self.manager = None
        self.spies = {}

    @staticmethod
    def _touch(path) -> Path:
        path.write_text("dummy", encoding="utf-8")
        return path

    def _default_reader(self, path):
        return self.by_path[Path(path).resolve()]

    def _reader(self, path):
        if self.reader_override is not None:
            return self.reader_override(path, self._default_reader)
        return self._default_reader(path)

    @property
    def out_fwd(self) -> Path:
        return _output_for(self.fwd)

    @property
    def out_bwd(self) -> Path:
        return _output_for(self.bwd)

    @property
    def out_tun(self) -> Path:
        return _output_for(self.tun)

    def run(self):
        self.manager = MagicMock()
        irc_kwargs = ({"side_effect": self.irc_return} if isinstance(self.irc_return, Exception)
                      else {"return_value": self.irc_return})
        with ExitStack() as stack:
            self.spies = {
                "log_run_start": stack.enter_context(patch(f"{MAIN_MODULE}.log_run_start", wraps=log_run_start)),
                "irc_computations_main": stack.enter_context(
                patch(f"{MAIN_MODULE}.irc_computations_main", **irc_kwargs)),
                "qmt_computations_main": stack.enter_context(
                patch(f"{MAIN_MODULE}.qmt_computations_main", wraps=qmt_computations_main)),
                "read_input": stack.enter_context(
                patch(f"{QMT_MODULE}.read_input", side_effect=self._reader))}
            for name in ("log_run_start", "irc_computations_main", "qmt_computations_main"):
                self.manager.attach_mock(self.spies[name], name)
            main(self.cfg)
        return self


# ===========================================================================
# --- main - full run with the IRC stage (tunnex_input is None) ---
# ===========================================================================
class TestFullRunWithIRCStage:

    def test_both_outputs_are_created(self, tmp_path):
        """ The forward and backward output files are created next to the input files """
        h = _Harness(tmp_path).run()
        assert h.out_fwd.is_file(), f"expected the output file {h.out_fwd} to exist; test #1"
        assert h.out_bwd.is_file(), f"expected the output file {h.out_bwd} to exist; test #2"
        assert h.out_fwd.name == "tunnex_2_forward_tunnex_2.out", \
        f"expected {'tunnex_2_forward_tunnex_2.out'}, got {h.out_fwd.name}; test #3"
        assert h.out_bwd.name == "tunnex_2_backward_tunnex_2.out", \
        f"expected {'tunnex_2_backward_tunnex_2.out'}, got {h.out_bwd.name}; test #4"

    def test_outputs_match_independent_chain(self, tmp_path):
        """ Each output file equals the report built by the manual chain from its own data """
        h = _Harness(tmp_path).run()
        assert _read(h.out_fwd).strip() == _expected_report(h.data["forward"]).strip(), \
        "the forward output differs from the independently built report; test #1"
        assert _read(h.out_bwd).strip() == _expected_report(h.data["backward"]).strip(), \
        "the backward output differs from the independently built report; test #2"
        assert _read(h.out_fwd) != _read(h.out_bwd), \
        "expected different reports for different IRC data; test #3"

    def test_stages_called_in_order(self, tmp_path):
        """ The run is logged first, then the IRC stage runs, then QMT runs for forward and for backward """
        h = _Harness(tmp_path).run()
        names = [c[0] for c in h.manager.mock_calls]
        expected = ["log_run_start", "irc_computations_main", "qmt_computations_main", "qmt_computations_main"]
        assert names == expected, f"expected {expected}, got {names}; test #1"
        qmt_args = [c.args[0] for c in h.spies["qmt_computations_main"].call_args_list]
        assert qmt_args == [h.fwd, h.bwd], f"expected {[h.fwd, h.bwd]}, got {qmt_args}; test #2"

    def test_irc_stage_receives_config_and_run_is_logged_with_ts_input(self, tmp_path):
        """ The IRC stage receives the config object; the run is logged with the transition state input """
        h = _Harness(tmp_path).run()
        h.spies["irc_computations_main"].assert_called_once_with(h.cfg)  # it should be called once
        h.spies["log_run_start"].assert_called_once()  # it should be called once
        logged = h.spies["log_run_start"].call_args.args[1]
        assert logged == h.ts_in, f"expected {h.ts_in}, got {logged}"

    def test_qmt_module_disabled_creates_no_outputs(self, tmp_path):
        """
        With the QMT module disabled the IRC stage runs,
        but nothing is read and no output is written
        """
        h = _Harness(tmp_path, qmt_module=False).run()
        h.spies["irc_computations_main"].assert_called_once()  # it should be called once
        h.spies["qmt_computations_main"].assert_not_called()  # it should not be called
        h.spies["read_input"].assert_not_called()  # it should not be called
        assert not h.out_fwd.exists(), f"expected no output file, but {h.out_fwd} exists; test #1"
        assert not h.out_bwd.exists(), f"expected no output file, but {h.out_bwd} exists; test #2"


# ===========================================================================
# --- main - full run from the Tunnex input (tunnex_input is not None) ---
# ===========================================================================
class TestFullRunFromTunnexInput:

    def test_irc_stage_is_skipped(self, tmp_path):
        """ The IRC stage is not run when the Tunnex input is provided """
        h = _Harness(tmp_path, with_tunnex_input=True).run()
        h.spies["irc_computations_main"].assert_not_called()  # it should not be called

    def test_output_is_created_and_matches_independent_chain(self, tmp_path):
        """ One output file is created next to the Tunnex input and equals the independently built report """
        h = _Harness(tmp_path, with_tunnex_input=True).run()
        assert h.out_tun.is_file(), f"expected the output file {h.out_tun} to exist; test #1"
        assert h.out_tun.name == "tunnex_2_input_tunnex_2.out", \
        f"expected {'tunnex_2_input_tunnex_2.out'}, got {h.out_tun.name}; test #2"
        assert _read(h.out_tun).strip() == _expected_report(h.data["input"]).strip(), \
        "the output differs from the independently built report; test #3"
        assert not h.out_fwd.exists() and not h.out_bwd.exists(), \
        "expected no forward/backward outputs in this branch; test #4"

    def test_qmt_called_once_and_run_is_logged_with_tunnex_input(self, tmp_path):
        """ QMT is called once with the Tunnex input; the run is logged with the same file """
        h = _Harness(tmp_path, with_tunnex_input=True).run()
        h.spies["qmt_computations_main"].assert_called_once_with(h.tun)  # it should be called once
        logged = h.spies["log_run_start"].call_args.args[1]
        assert logged == h.tun, f"expected {h.tun}, got {logged}"

    def test_qmt_module_flag_is_ignored(self, tmp_path):
        """ The output is written even if qmt_module is False, when the Tunnex input is provided """
        h = _Harness(tmp_path, with_tunnex_input=True, qmt_module=False).run()
        assert h.out_tun.is_file(), f"expected the output file {h.out_tun} to exist"


# ===========================================================================
# --- main - reproducibility and side effects ---
# ===========================================================================
class TestReproducibilityAndSideEffects:

    def test_repeated_runs_give_identical_outputs(self, tmp_path):
        """ Two runs with the same inputs give identical output files """
        h = _Harness(tmp_path).run()
        first = (_read(h.out_fwd), _read(h.out_bwd))
        h.run()
        second = (_read(h.out_fwd), _read(h.out_bwd))
        assert first == second, \
        "expected identical output files for the repeated runs"

    def test_config_is_not_modified(self, tmp_path):
        """ The config object is not modified by the run """
        h = _Harness(tmp_path)
        before = deepcopy(h.cfg)
        h.run()
        assert h.cfg == before, \
        f"expected {before}, got {h.cfg}"

    def test_input_files_are_not_modified(self, tmp_path):
        """ The input files are not modified by the run """
        h = _Harness(tmp_path).run()
        for path in (h.ts_in, h.fwd, h.bwd):
            assert _read(path) == "dummy", \
            f"expected {path.name} to stay {'dummy'}, got {_read(path)}"


# ===========================================================================
# --- main - input validation ---
# ===========================================================================
class TestFailures:

    def test_irc_stage_failure_stops_qmt(self, tmp_path):
        """ An error of the IRC stage propagates, and QMT is not run. Expecting a RuntimeError """
        h = _Harness(tmp_path, irc_return=RuntimeError("irc failed"))
        with pytest.raises(RuntimeError, match="irc failed"):
            h.run()
        h.spies["qmt_computations_main"].assert_not_called()  # it should not be called
        assert not h.out_fwd.exists() and not h.out_bwd.exists(), "expected no output files"

    def test_missing_forward_file_stops_before_backward(self, tmp_path):
        """
        A missing forward file raises an error
        and the backward file is not processed. Expecting a FileNotFoundError
        """
        missing = tmp_path / "no_such_forward.txt"
        h = _Harness(tmp_path)
        h.irc_return = (missing, h.bwd)
        with pytest.raises(FileNotFoundError, match="File not found"):
            h.run()
        assert h.spies["qmt_computations_main"].call_count == 1, \
        f"expected {1} call, got {h.spies['qmt_computations_main'].call_count}; test #1"
        h.spies["read_input"].assert_not_called()  # it should not be called
        assert not h.out_bwd.exists(), f"expected no output file, but {h.out_bwd} exists; test #2"

    def test_forward_failure_stops_before_backward(self, tmp_path):
        """
        An error while processing the forward file propagates,
        and the backward file is not processed. Expecting a ValueError
        """
        def failing_forward(path, default):
            if Path(path).name == "tunnex_2_forward.txt":
                raise ValueError("bad forward")
            return default(path)
        h = _Harness(tmp_path, reader_override=failing_forward)
        with pytest.raises(ValueError, match="bad forward"):
            h.run()
        assert h.spies["qmt_computations_main"].call_count == 1, \
        f"expected {1} call, got {h.spies['qmt_computations_main'].call_count}; test #1"
        assert not h.out_fwd.exists() and not h.out_bwd.exists(), \
        "expected no output files; test #2"

    def test_backward_failure_keeps_forward_output(self, tmp_path):
        """
        An error while processing the backward file propagates;
        the forward output stays complete. Expecting a ValueError
        """
        def failing_backward(path, default):
            if Path(path).name == "tunnex_2_backward.txt":
                raise ValueError("bad backward")
            return default(path)
        h = _Harness(tmp_path, reader_override=failing_backward)
        with pytest.raises(ValueError, match="bad backward"):
            h.run()
        assert h.out_fwd.is_file(), f"expected the forward output {h.out_fwd} to exist; test #1"
        assert _read(h.out_fwd).strip() == _expected_report(h.data["forward"]).strip(), \
        "expected the forward output to be complete and correct; test #2"
        assert not h.out_bwd.exists(), f"expected no output file, but {h.out_bwd} exists; test #3"

    def test_missing_tunnex_input_raises_and_skips_irc(self, tmp_path):
        """
        A missing Tunnex input raises an error and the IRC stage is not run.
        Expecting a FileNotFoundError
        """
        h = _Harness(tmp_path, with_tunnex_input=True)
        h.tun.unlink()
        with pytest.raises(FileNotFoundError, match="File not found"):
            h.run()
        h.spies["irc_computations_main"].assert_not_called()  # it should not be called
        h.spies["read_input"].assert_not_called()  # it should not be called
        assert not h.out_tun.exists(), f"expected no output file, but {h.out_tun} exists"


# ===========================================================================
# --- __main__.py - normal behavior and input validation ---
# ===========================================================================
class TestEntryMain:

    def test_entry_file_runs_without_errors(self, monkeypatch, tmp_path):
        """ __main__.py runs main.py as __main__ without errors from any working directory """
        monkeypatch.chdir(PROJECT_ROOT)
        result = runpy.run_path(str(ENTRY_FILE), run_name="__main__")
        assert isinstance(result, dict), f"expected {dict}, got {type(result)}; test #1"
        completed = _run_module(PROJECT_ROOT)
        assert completed.returncode == 0, \
        f"expected exit code {0}, got {completed.returncode}; stderr: {completed.stderr[-500:]}; test #2"
        completed = _run_module(tmp_path)
        assert completed.returncode == 0, \
        f"expected exit code {0}, got {completed.returncode}; stderr: {completed.stderr[-500:]}; test #3"

    def test_entry_file_does_not_run_computations(self, monkeypatch):
        """ Running main.py as __main__ does not start any computation """
        monkeypatch.chdir(PROJECT_ROOT)
        with patch(IRC_MAIN) as irc_mock, patch(QMT_MAIN) as qmt_mock:
             runpy.run_path(str(ENTRY_FILE), run_name="__main__")
        irc_mock.assert_not_called()  # it should not be called
        qmt_mock.assert_not_called()  # it should not be called

    def test_main_py_is_executed_as_main(self, monkeypatch):
        """ main.py is executed with __name__ == '__main__' and defines the main function """
        monkeypatch.chdir(PROJECT_ROOT)
        result = runpy.run_path(str(PROJECT_ROOT / "tunnex_2" / "main.py"), run_name="__main__")
        assert result["__name__"] == "__main__", \
        f"expected {'__main__'}, got {result['__name__']}; test #1"
        assert callable(result["main"]), \
        f"expected a callable main, got {result.get('main')}; test #2"