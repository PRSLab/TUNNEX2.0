# --- Test: test_cli_int. Integration tests for cli.py
# Run with: pytest test_cli_int.py -v ---

# --- Modules ---
import pytest  # type: ignore
from contextlib import ExitStack, contextmanager
from unittest.mock import patch
pytestmark = pytest.mark.integration  # type: ignore

import tunnex_2  # type: ignore

import numpy as np
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from tunnex_2.qmt_computations.core.potential import Potential  # type: ignore
from tunnex_2.qmt_computations.data_reader_writer.writer import build_report  # type: ignore
from tunnex_2.qmt_computations.module_qmt_computations import qmt_computations_main  # type: ignore
from tunnex_2.qmt_computations.pipeline.qmt_computations_pipeline import qmt_computations_pipeline  # type: ignore
from tunnex_2.main import main  # type: ignore


# --- Module to test ---
CLI_MODULE = "tunnex_2.cli"
MAIN_MODULE = "tunnex_2.main"
QMT_MODULE = "tunnex_2.qmt_computations.module_qmt_computations"
from tunnex_2.cli import cli  # type: ignore


# --- Helpers and shared data ---
PROJECT_ROOT = Path(tunnex_2.__file__).resolve().parent.parent
# The folder where pyproject.toml is located
HELP_HINT = "Use --help for more information."
# The tunnex_2 help hint
OUTPUT_SUFFIX = "_tunnex_2.out"
# The default output suffix for the files
TIMEOUT = 120
# The timeout time for subprocesses

E0, ZPVE0 = -153.15, 0.02
# The reference electronic energy and ZPVE (Hartree) of the synthetic IRC

DATA_SPEC = {"forward": dict(B=0.375, T=300.0),
             "backward": dict(B=0.600, T=350.0),
             "input": dict(B=0.450, T=450.0)}
# Different barriers and temperatures, so the results of the files are distinguishable

# The expected config fields reaching the IRC stage: (argv, {field: expected value})
CONFIG_CASES = [
    (["mol.gjf"],
     dict(prog_mode="gaussian", proj_freq=True, calc_all=True, hybrid_mode=False, eckart=False,
          hess_parsing_flag=False, qmt_module=True, ts_input_file="mol.gjf", tunnex_input=None)),
    (["-o", "mol.inp"],
     dict(prog_mode="orca", proj_freq=True, ts_input_file="mol.inp")),
    (["-z", "-f", "-y", "-i", "mol.gjf"],
     dict(proj_freq=False, calc_all=False, hybrid_mode=True, hess_parsing_flag=True, ts_input_file="mol.gjf")),
    (["-p", "mol.gjf"],
     dict(qmt_module=False, ts_input_file="mol.gjf")),
    (["-e", "ts.gjf", "r.gjf", "p.gjf"],
     dict(eckart=True, ts_input_file="ts.gjf", react_input_file="r.gjf", prod_input_file="p.gjf")),
    (["-c", "ts.gjf", "ts.out"],
     dict(ts_input_file="ts.gjf", ts_computed_file="ts.out", irc_computed_file=None,
          react_computed_file=None, prod_computed_file=None)),
    (["-c", "ts.gjf", "ts.out", "irc.out", "r.out", "p.out"],
     dict(ts_input_file="ts.gjf", ts_computed_file="ts.out", irc_computed_file="irc.out",
          react_computed_file="r.out", prod_computed_file="p.out")),
    (["-e", "-c", "ts.gjf", "ts.out", "r.out", "p.out"],
     dict(eckart=True, ts_input_file="ts.gjf", ts_computed_file="ts.out", react_input_file="r.out",
          react_computed_file="r.out", prod_input_file="p.out", prod_computed_file="p.out"))]

# The invalid combinations of the options and files: every one should end with SystemExit 2
INVALID_ARGV = [
    [],
    ["mol.gjf", "extra.gjf"],
    ["-o", "-f", "mol.inp"],
    ["-e", "-z", "ts.gjf", "r.gjf", "p.gjf"],
    ["-e", "-f", "ts.gjf", "r.gjf", "p.gjf"],
    ["-e", "-y", "ts.gjf", "r.gjf", "p.gjf"],
    ["-e", "ts.gjf", "r.gjf"],
    ["-c", "ts.gjf"],
    ["-c", "ts.gjf", "ts.out", "irc.out", "r.out"],
    ["-e", "-c", "ts.gjf", "ts.out", "r.out"],
    ["-q", "tunnex_2_input.txt", "extra.txt"],
    ["-q"],
    ["-q", "-o", "tunnex_2_input.txt"],
    ["-q", "-p", "tunnex_2_input.txt"],
    ["-q", "-c", "tunnex_2_input.txt"]]


@contextmanager
def _cwd(path):
    """ Change the working directory temporarily """
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)

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

def _value(config, name):
    """ Return the config field in a comparable form: bool and None as they are, paths as strings """
    value = getattr(config, name)
    return value if isinstance(value, (bool, type(None))) else str(value)

def _run_process(argv, cwd) -> subprocess.CompletedProcess:
    """ Run cli() in a separate process with the given command line arguments """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), env.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, "-c", "from tunnex_2.cli import cli; cli()", *argv],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=TIMEOUT)


# --- Runs the function with the real stages ---
class _CliHarness:
    """ Runs cli() with the real argparse, FindIRCConfig, main and QMT stages """
    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fwd, self.bwd, self.tun = (
            self._touch(self.dir / name)
            for name in ("tunnex_2_forward.txt", "tunnex_2_backward.txt", "tunnex_2_input.txt"))
        self.data = {"forward": _make_data(**DATA_SPEC["forward"]),
                     "backward": _make_data(**DATA_SPEC["backward"]),
                     "input": _make_data(**DATA_SPEC["input"])}
        self.by_path = {self.fwd.resolve(): self.data["forward"],
                        self.bwd.resolve(): self.data["backward"],
                        self.tun.resolve(): self.data["input"]}
        self.spies = {}

    @staticmethod
    def _touch(path) -> Path:
        path.write_text("dummy", encoding="utf-8")
        return path

    def _reader(self, path):
        return self.by_path[Path(path).resolve()]

    def run(self, argv, create_files=True):
        if create_files:
            for arg in argv:
                path = self.dir / arg
                if not arg.startswith("-") and not path.exists():
                    self._touch(path)
        with ExitStack() as stack:
            self.spies = {
                "cli_main": stack.enter_context(patch(f"{CLI_MODULE}.main", wraps=main)),
                "irc_computations_main": stack.enter_context(
                    patch(f"{MAIN_MODULE}.irc_computations_main", return_value=(self.fwd, self.bwd))),
                "qmt_computations_main": stack.enter_context(
                    patch(f"{MAIN_MODULE}.qmt_computations_main", wraps=qmt_computations_main)),
                "read_input": stack.enter_context(patch(f"{QMT_MODULE}.read_input", side_effect=self._reader))}
            stack.enter_context(patch("sys.argv", ["tunnex_2"] + argv))
            stack.enter_context(_cwd(self.dir))
            cli()
        return self

    def assert_nothing_started(self):
        """ Nothing is run after a rejected command line """
        for name in ("cli_main", "irc_computations_main", "qmt_computations_main", "read_input"):
            self.spies[name].assert_not_called()  # it should not be called


# ===========================================================================
# --- cli - QMT only mode, end to end ---
# ===========================================================================
class TestQmtOnlyEndToEnd:

    def test_output_is_created_and_matches_independent_chain(self, tmp_path):
        """
        --qmt_only writes the output file next to the Tunnex input;
        it equals the independently built report
        """
        h = _CliHarness(tmp_path).run(["--qmt_only", "tunnex_2_input.txt"])
        out = _output_for(h.tun)
        assert out.is_file(), f"expected the output file {out} to exist; test #1"
        assert out.name == "tunnex_2_input_tunnex_2.out", \
        f"expected {'tunnex_2_input_tunnex_2.out'}, got {out.name}; test #2"
        assert _read(out).strip() == _expected_report(h.data["input"]).strip(), \
        "the output differs from the independently built report; test #3"

    @pytest.mark.parametrize("flag", ["-q", "--qmt_only", "--qmt-only"])
    def test_all_spellings_give_the_same_output(self, tmp_path, flag):
        """ The short flag and both long spellings of the QMT-only option work in the same way """
        h = _CliHarness(tmp_path / flag.strip("-")).run([flag, "tunnex_2_input.txt"])
        assert _read(_output_for(h.tun)).strip() == _expected_report(h.data["input"]).strip(), \
        f"{flag}: the output differs from the independently built report"

    def test_irc_stage_is_skipped_and_str_path_reaches_qmt(self, tmp_path):
        """ The IRC stage is not run; QMT receives the Tunnex input from the command line once """
        h = _CliHarness(tmp_path).run(["-q", "tunnex_2_input.txt"])
        h.spies["irc_computations_main"].assert_not_called()  # it should not be called
        h.spies["qmt_computations_main"].assert_called_once_with("tunnex_2_input.txt")
        # it should be called once
        assert not _output_for(h.fwd).exists() and not _output_for(h.bwd).exists(), \
        "expected no forward and backward outputs in this mode"

    def test_missing_tunnex_input_raises_and_writes_nothing(self, tmp_path):
        """
        A missing Tunnex input passes the command line check and raises
        an error in the QMT stage. Expecting a FileNotFoundError
        """
        h = _CliHarness(tmp_path)
        with pytest.raises(FileNotFoundError, match="File not found"):
            h.run(["-q", "no_such_file.txt"], create_files=False)
        h.spies["read_input"].assert_not_called()  # it should not be called
        assert not _output_for(tmp_path / "no_such_file.txt").exists(), "expected no output file"


# ===========================================================================
# --- cli - default mode, end to end ---
# ===========================================================================
class TestDefaultModeEndToEnd:

    def test_both_outputs_are_created_and_match_independent_chain(self, tmp_path):
        """ The default mode writes the forward and backward outputs, each built from its own data """
        h = _CliHarness(tmp_path).run(["mol.gjf"])
        assert _read(_output_for(h.fwd)).strip() == _expected_report(h.data["forward"]).strip(), \
        "the forward output differs from the independently built report; test #1"
        assert _read(_output_for(h.bwd)).strip() == _expected_report(h.data["backward"]).strip(), \
        "the backward output differs from the independently built report; test #2"

    def test_qmt_called_for_forward_then_backward(self, tmp_path):
        """ QMT is called for the forward file and then for the backward file """
        h = _CliHarness(tmp_path).run(["mol.gjf"])
        called = [c.args[0] for c in h.spies["qmt_computations_main"].call_args_list]
        assert called == [h.fwd, h.bwd], f"expected {[h.fwd, h.bwd]}, got {called}"

    def test_no_qmt_runs_only_irc_stage(self, tmp_path):
        """ --no_qmt runs the IRC stage only: nothing is read and no output is written """
        h = _CliHarness(tmp_path).run(["--no_qmt", "mol.gjf"])
        h.spies["irc_computations_main"].assert_called_once()  # it should be called once
        h.spies["qmt_computations_main"].assert_not_called()  # it should not be called
        h.spies["read_input"].assert_not_called()  # it should not be called
        assert not _output_for(h.fwd).exists() and not _output_for(h.bwd).exists(), \
        "expected no output files"

    def test_cli_config_is_the_one_reaching_the_irc_stage(self, tmp_path):
        """
        main is called with the configs keyword argument;
        the same object reaches the IRC stage
        """
        h = _CliHarness(tmp_path).run(["mol.gjf"])
        h.spies["cli_main"].assert_called_once()  # it should be called once
        cfg = h.spies["cli_main"].call_args.kwargs["configs"]
        assert h.spies["irc_computations_main"].call_args.args[0] is cfg, \
        "expected the IRC stage to receive the config object built by cli"


# ===========================================================================
# --- cli - config built from the command line reaches the IRC stage ---
# ===========================================================================
class TestConfigReachesIRCStage:

    @pytest.mark.parametrize("argv, expected", CONFIG_CASES, ids=[" ".join(c[0]) for c in CONFIG_CASES])
    def test_config_fields(self, tmp_path, argv, expected):
        """ The real FindIRCConfig built by cli has the expected fields when it reaches the IRC stage """
        h = _CliHarness(tmp_path).run(argv)
        cfg = h.spies["irc_computations_main"].call_args.args[0]
        for name, value in expected.items():
            assert _value(cfg, name) == value, f"{name}: expected {value}, got {_value(cfg, name)}"

    @pytest.mark.parametrize("variants", [["-z", "--no_zpve", "--no-zpve"],
                                          ["-p", "--no_qmt", "--no-qmt"]])
    def test_equivalent_spellings_give_equal_configs(self, tmp_path, variants):
        """ The short and both long spellings of an option give equal configs """
        configs = []
        for i, flag in enumerate(variants):
            h = _CliHarness(tmp_path / f"run_{i}").run([flag, "mol.gjf"])
            configs.append(asdict(h.spies["irc_computations_main"].call_args.args[0]))
        assert all(cfg == configs[0] for cfg in configs), f"expected equal configs, got {configs}"


# ===========================================================================
# --- cli - rejected command lines ---
# ===========================================================================
class TestRejectedCommandLines:

    @pytest.mark.parametrize("argv", INVALID_ARGV, ids=[" ".join(a) or "no arguments" for a in INVALID_ARGV])
    def test_invalid_command_line_starts_nothing(self, tmp_path, capsys, argv):
        """ An invalid command line ends with SystemExit 2 and a hint; no stage is started """
        h = _CliHarness(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            h.run(argv)
        assert exc_info.value.code == 2, f"expected {2}, got {exc_info.value.code}; test #1"
        assert HELP_HINT in capsys.readouterr().err, f"expected {HELP_HINT} in stderr; test #2"
        h.assert_nothing_started()
        assert not _output_for(h.tun).exists() and not _output_for(h.fwd).exists(), \
        "expected no output files; test #3"

    def test_help_exits_with_zero_and_starts_nothing(self, tmp_path, capsys):
        """ --help prints the usage and ends with SystemExit 0; no stage is started """
        h = _CliHarness(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            h.run(["--help"])
        assert exc_info.value.code == 0, f"expected {0}, got {exc_info.value.code}; test #1"
        assert "usage: tunnex_2" in capsys.readouterr().out, \
        f"expected {'usage: tunnex_2'} in stdout; test #2"
        h.assert_nothing_started()

    def test_qmt_only_with_eckart_is_rejected(self, tmp_path, capsys):
        """ --qmt-only cannot be combined with IRC calculation options """
        h = _CliHarness(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            h.run(["--qmt-only", "--eckart", "tunnex_2_input.txt"])
        assert exc_info.value.code == 2, f"expected {2}, got {exc_info.value.code}; test #1"
        assert HELP_HINT in capsys.readouterr().err, \
        f"expected {HELP_HINT} in {capsys.readouterr().err}; test #2"
        h.assert_nothing_started()

    def test_calcfc_with_eckart_is_rejected(self, tmp_path, capsys):
        """ --сalcfc cannot be combined with Eckart mode """
        h = _CliHarness(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            h.run(["--eckart", "--calcfc", "ts.gjf", "r.gjf", "p.gjf"])
        assert exc_info.value.code == 2, f"expected {2}, got {exc_info.value.code}; test #1"
        assert HELP_HINT in capsys.readouterr().err, \
        f"expected {HELP_HINT} in {capsys.readouterr().err}; test #2"
        h.assert_nothing_started()


# ===========================================================================
# --- cli - separate process ---
# ===========================================================================
class TestProcessLevel:

    def test_help_exits_with_zero(self, tmp_path):
        """ cli() started in a separate process prints the usage with the zero exit code """
        completed = _run_process(["--help"], tmp_path)
        assert completed.returncode == 0, \
        f"expected exit code {0}, got {completed.returncode}; stderr: {completed.stderr[-500:]}; test #1"
        assert "usage: tunnex_2" in completed.stdout, \
        f"expected {'usage: tunnex_2'} in {completed.stdout}; test #2"

    def test_invalid_combination_exits_with_two(self, tmp_path):
        """ An invalid combination of the options ends the separate process with the exit code 2 and a hint """
        completed = _run_process(["--orca", "--calcfc", "mol.inp"], tmp_path)
        assert completed.returncode == 2, \
        f"expected exit code {2}, got {completed.returncode}; stderr: {completed.stderr[-500:]}; test #1"
        assert HELP_HINT in completed.stderr, f"expected {HELP_HINT} in {completed.stderr}; test #2"