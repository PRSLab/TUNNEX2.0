# --- Test: test_module_qmt_computations_int. Integration tests for module_qmt_computations.py
# Run with: pytest test_module_qmt_computations_int.py -v ---

# --- Modules ---
import pytest  # type: ignore
from contextlib import ExitStack
from unittest.mock import MagicMock, patch
pytestmark = pytest.mark.integration  # type: ignore

import numpy as np
import re
from dataclasses import astuple
from pathlib import Path
from types import SimpleNamespace

from tunnex_2.irc_computations.ancillary_functions.attempt_freq_calc import detect_outliers_local  # type: ignore
from tunnex_2.qmt_computations.core.potential import Potential  # type: ignore
from tunnex_2.qmt_computations.data_reader_writer.writer import write_output, build_report  # type: ignore
from tunnex_2.qmt_computations.pipeline.qmt_computations_pipeline import qmt_computations_pipeline  # type: ignore


# --- Module to test ---
FUNC_PATH = "tunnex_2.qmt_computations.module_qmt_computations"
from tunnex_2.qmt_computations.module_qmt_computations import qmt_computations_main  # type: ignore


# --- Helpers and shared data ---
E0, ZPVE0 = -153.15, 0.02
# The reference electronic energy and ZPVE (Hartree) of the synthetic IRC

AVR_PROB, AVR_RATE, AVR_HL_S = range(3)
# Positions of the fields in the (positional) TemperatureAverResult dataclass

PIPE_GRID, PIPE_LEVELS, PIPE_AVER, PIPE_ARRHENIUS = range(4)
# Positions of the items returned by qmt_computations_pipeline

STAGES = ("read_input", "detect_outliers_local", "Potential",
          "qmt_computations_pipeline", "build_report", "write_output")
# The stages of qmt_computations_main in the expected order of the calls

OUTPUT_SUFFIX = "_tunnex_2.out"
# The default output suffix for the files


def _eckart(x, A=-0.085, B=0.375, L=3.275):
    """
    Asymmetric Eckart potential (Hartree),
    reactant on the left (V -> 0), product on the right (V -> A)
    """
    y = - np.exp(2 * np.pi * np.asarray(x, dtype=float) / L)
    result = - A * y / (1 - y) - B * y / (1 - y) ** 2
    return result

def _make_irc(B=0.375, n_points=401, half_range=8.0):
    """ Return the synthetic IRC data (absolute energies; constant ZPVE along the IRC) """
    x = np.linspace(-half_range, half_range, n_points)
    return SimpleNamespace(E_x=x, E_y=E0 + _eckart(x, B=B),
                           ZPVE_x=x.copy(), ZPVE_y=np.full_like(x, ZPVE0))

def _make_params(**overrides) -> SimpleNamespace:
    """ Return the synthetic parameters """
    params = dict(E0=E0,
                  ZPVE0=ZPVE0,
                  potential_scaling_factor=1.0,
                  T=300.0,
                  number_of_levels=3,
                  freq=1000.0,
                  prob_aver_mode_qmt="finite_sum",
                  T_arrhenius_min=200.0,
                  T_arrhenius_max=400.0,
                  T_step=50.0)
    params.update(overrides)
    return SimpleNamespace(**params)

def _write_input(directory, name="mol_forward.txt") -> Path:
    """ Create the (dummy) input file; read_input is patched, so only its path matters """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    inp = directory / name
    inp.write_text("dummy", encoding="utf-8")
    return inp

def _output_for(inp) -> Path:
    """ Return the expected output path for the input path """
    inp = Path(inp).resolve()
    return inp.with_name(inp.stem + OUTPUT_SUFFIX)

def _run_main(directory, params=None, irc=None, input_name="mol_forward.txt", as_str=False):
    """ Run the main function with the real stages; only read_input is replaced by the synthetic data """
    params = _make_params() if params is None else params
    irc = _make_irc() if irc is None else irc
    inp = _write_input(directory, input_name)
    with patch(f"{FUNC_PATH}.read_input", return_value=(params, irc)):
        qmt_computations_main(str(inp) if as_str else inp)
    return inp, _output_for(inp)

def _run_main_with_spies(directory, params=None, irc=None, input_name="mol_forward.txt"):
    """
    Run the main function with the real stages wrapped into spies;
    the results of the pipeline and of build_report are captured
    """
    params = _make_params() if params is None else params
    irc = _make_irc() if irc is None else irc
    inp = _write_input(directory, input_name)
    captured = {}

    def pipeline_wrapper(*args, **kwargs):
        captured["pipeline"] = qmt_computations_pipeline(*args, **kwargs)
        return captured["pipeline"]

    def report_wrapper(*args, **kwargs):
        captured["report"] = build_report(*args, **kwargs)
        return captured["report"]

    manager = MagicMock()
    with ExitStack() as stack:
        spies = {
            "read_input": stack.enter_context(patch(f"{FUNC_PATH}.read_input", return_value=(params, irc))),
            "detect_outliers_local": stack.enter_context(
            patch(f"{FUNC_PATH}.detect_outliers_local", wraps=detect_outliers_local)),
            "Potential": stack.enter_context(patch(f"{FUNC_PATH}.Potential", wraps=Potential)),
            "qmt_computations_pipeline": stack.enter_context(
            patch(f"{FUNC_PATH}.qmt_computations_pipeline", side_effect=pipeline_wrapper)),
            "build_report": stack.enter_context(patch(f"{FUNC_PATH}.build_report", side_effect=report_wrapper)),
            "write_output": stack.enter_context(patch(f"{FUNC_PATH}.write_output", wraps=write_output)),
            "logger": stack.enter_context(patch(f"{FUNC_PATH}.logger"))}
        for name in STAGES:
            manager.attach_mock(spies[name], name)
        qmt_computations_main(inp)
    return SimpleNamespace(inp=inp, out=_output_for(inp), params=params, irc=irc,
                           spies=spies, manager=manager, captured=captured)

def _expected_report(params, irc) -> str:
    """ Return the report obtained by the independent manual chain of the real stages """
    potential = Potential(irc.E_x, irc.E_y, irc.ZPVE_x, irc.ZPVE_y,
                          params.E0, params.ZPVE0, params.potential_scaling_factor)
    interpolated_irc, levels, aver, arrhenius = qmt_computations_pipeline(params, potential)
    return build_report(levels, aver, params.T, arrhenius, interpolated_irc, potential.offset)

def _read(path) -> str:
    """ Return the text of the file """
    return Path(path).read_text(encoding="utf-8")


# ===========================================================================
# --- qmt_computations_main - output file ---
# ===========================================================================
class TestOutputFile:

    def test_output_file_is_created_next_to_input(self, tmp_path):
        """ The output file is created in the directory of the input file and is not empty """
        inp, out = _run_main(tmp_path)
        assert out.is_file(), f"expected the output file {out} to exist; test #1"
        assert out.parent == inp.resolve().parent, \
        f"expected {inp.resolve().parent}, got {out.parent}; test #2"
        assert out.name == "mol_forward_tunnex_2.out", \
        f"expected {'mol_forward_tunnex_2.out'}, got {out.name}; test #3"
        assert len(_read(out).strip()) > 0, "expected a non-empty output file; test #4"

    @pytest.mark.parametrize("temperature", [100.0, 300.0, 600.0])
    def test_output_matches_independent_chain(self, tmp_path, temperature):
        """ The output file equals the report built by the manual chain of the real stages """
        params = _make_params(T=temperature)
        irc = _make_irc()
        _, out = _run_main(tmp_path, params, irc)
        expected = _expected_report(params, irc)
        assert _read(out).strip() == expected.strip(), \
        f"T={temperature}: the output file differs from the independently built report"

    def test_report_has_no_nan(self, tmp_path):
        """ The output file does not contain NaN values """
        _, out = _run_main(tmp_path)
        text = _read(out)
        assert re.search(r"\bnan\b", text, flags=re.IGNORECASE) is None, \
        f"expected no NaN in the report, got {text}"

    def test_string_and_path_inputs_give_identical_output(self, tmp_path):
        """ A str path and a Path object give the same output file """
        _, out_path = _run_main(tmp_path / "as_path", as_str=False)
        _, out_str = _run_main(tmp_path / "as_str", as_str=True)
        assert _read(out_path) == _read(out_str), \
        "expected identical output files for str and Path inputs"

    def test_existing_output_is_overwritten(self, tmp_path):
        """ An already existing output file is replaced, not appended """
        inp = _write_input(tmp_path)
        out = _output_for(inp)
        out.write_text("OLD CONTENT", encoding="utf-8")
        params, irc = _make_params(), _make_irc()
        with patch(f"{FUNC_PATH}.read_input", return_value=(params, irc)):
            qmt_computations_main(inp)
        text = _read(out)
        assert "OLD CONTENT" not in text, \
        f"expected the old content to be replaced, got {text[:100]}; test #1"
        assert text.strip() == _expected_report(params, irc).strip(), \
        "expected the new report in the output file; test #2"

    def test_multi_dot_input_name(self, tmp_path):
        """ Only the last suffix of the input filename is replaced """
        _, out = _run_main(tmp_path, input_name="mol.v2.forward.txt")
        assert out.name == "mol.v2.forward_tunnex_2.out", \
        f"expected {'mol.v2.forward_tunnex_2.out'}, got {out.name}; test #1"
        assert out.is_file(), f"expected the output file {out} to exist; test #2"


# ===========================================================================
# --- qmt_computations_main - wiring between the real stages ---
# ===========================================================================
class TestWiring:

    def test_stages_called_in_order(self, tmp_path):
        """ The stages are called in the expected order, and outlier detection is called twice """
        run = _run_main_with_spies(tmp_path)
        names = [call[0] for call in run.manager.mock_calls]
        expected = ["read_input", "detect_outliers_local", "detect_outliers_local",
                    "Potential", "qmt_computations_pipeline", "build_report", "write_output"]
        assert names == expected, f"expected {expected}, got {names}"

    def test_read_input_receives_resolved_path(self, tmp_path):
        """ read_input receives the resolved input path """
        run = _run_main_with_spies(tmp_path)
        run.spies["read_input"].assert_called_once()  # it should be called once
        called_path = run.spies["read_input"].call_args.args[0]
        assert called_path == run.inp.resolve(), f"expected {run.inp.resolve()}, got {called_path}"

    def test_outlier_detection_receives_irc_data(self, tmp_path):
        """ Outlier detection receives the electronic energy and ZPVE data of the IRC """
        run = _run_main_with_spies(tmp_path)
        calls = run.spies["detect_outliers_local"].call_args_list
        assert calls[0].args[0] is run.irc.E_x and calls[0].args[1] is run.irc.E_y, \
        "expected the electronic energy data in the first call; test #1"
        assert calls[0].args[2] == "electronic energy", \
        f"expected {'electronic energy'}, got {calls[0].args[2]}; test #2"
        assert calls[1].args[0] is run.irc.ZPVE_x and calls[1].args[1] is run.irc.ZPVE_y, \
        "expected the ZPVE data in the second call; test #3"
        assert calls[1].args[2] == "ZPVE", f"expected {'ZPVE'}, got {calls[1].args[2]}; test #4"

    def test_real_potential_flows_into_pipeline_and_report(self, tmp_path):
        """ The real Potential object goes to the pipeline; its offset goes to build_report """
        run = _run_main_with_spies(tmp_path)
        pipe_args = run.spies["qmt_computations_pipeline"].call_args.args
        report_args = run.spies["build_report"].call_args.args
        assert pipe_args[0] is run.params, f"expected {run.params}, got {pipe_args[0]}; test #1"
        assert isinstance(pipe_args[1], Potential), \
        f"expected {Potential}, got {type(pipe_args[1])}; test #2"
        assert report_args[-1] == pipe_args[1].offset, \
        f"expected offset={pipe_args[1].offset}, got {report_args[-1]}; test #3"

    def test_pipeline_outputs_flow_into_report(self, tmp_path):
        """ The exact objects returned by the pipeline are passed to build_report in the right positions """
        run = _run_main_with_spies(tmp_path)
        pipe = run.captured["pipeline"]
        args = run.spies["build_report"].call_args.args
        assert args[0] is pipe[PIPE_LEVELS], f"expected the level results at position {0}; test #1"
        assert args[1] is pipe[PIPE_AVER], f"expected the averaged results at position {1}; test #2"
        assert args[2] == run.params.T, f"expected T={run.params.T}, got {args[2]}, position {2}; test #3"
        assert args[3] is pipe[PIPE_ARRHENIUS], f"expected the Arrhenius data at position {3}; test #4"
        assert args[4] is pipe[PIPE_GRID], f"expected the interpolated IRC at position {4}; test #5"

    def test_report_flows_into_write_output(self, tmp_path):
        """ write_output receives the output path and the report built by build_report """
        run = _run_main_with_spies(tmp_path)
        wo = run.spies["write_output"]
        wo.assert_called_once()  # it should be called once
        assert wo.call_args.args[0] == run.out, f"expected {run.out}, got {wo.call_args.args[0]}; test #1"
        assert wo.call_args.args[1] is run.captured["report"], "expected the built report; test #2"

    def test_logger_reports_output_path_once(self, tmp_path):
        """ The success message with the output path is logged once, after the output file is written """
        run = _run_main_with_spies(tmp_path)
        logger = run.spies["logger"]
        logger.info.assert_called_once()  # it should be called once
        assert logger.info.call_args.args[1] == run.out, \
        f"expected {run.out}, got {logger.info.call_args.args[1]}; test #1"
        assert run.out.is_file(), f"expected the output file {run.out} to exist at the end; test #2"


# ===========================================================================
# --- qmt_computations_main - physical consistency of the whole chain ---
# ===========================================================================
class TestPhysicalConsistency:

    def test_higher_barrier_gives_longer_half_life(self, tmp_path):
        """ A higher barrier gives a smaller averaged probability and a longer averaged half-life """
        low = _run_main_with_spies(tmp_path / "low", irc=_make_irc(B=0.375))
        high = _run_main_with_spies(tmp_path / "high", irc=_make_irc(B=0.750))
        aver_low = astuple(low.captured["pipeline"][PIPE_AVER])
        aver_high = astuple(high.captured["pipeline"][PIPE_AVER])
        assert aver_high[AVR_PROB] < aver_low[AVR_PROB], \
        f"expected prob < {aver_low[AVR_PROB]}, got {aver_high[AVR_PROB]}; test #1"
        assert aver_high[AVR_HL_S] > aver_low[AVR_HL_S], \
        f"expected half-life > {aver_low[AVR_HL_S]}, got {aver_high[AVR_HL_S]}; test #2"
        assert _read(low.out) != _read(high.out), "expected different reports for different barriers; test #3"

    def test_higher_temperature_does_not_decrease_probability(self, tmp_path):
        """ The averaged probability does not decrease when the temperature in the input rises """
        probs = []
        comp_tol = 1e-8
        for i, temperature in enumerate((100.0, 300.0, 600.0)):
            run = _run_main_with_spies(tmp_path / f"T{i}", params=_make_params(T=temperature))
            probs.append(astuple(run.captured["pipeline"][PIPE_AVER])[AVR_PROB])
        assert all(p2 >= p1 * (1.0 - comp_tol) for p1, p2 in zip(probs, probs[1:])), \
        f"expected non-decreasing probabilities, got {probs}"

    @pytest.mark.parametrize("n_levels", [0, 1, 5, 7, 9, 11])
    def test_number_of_levels_in_the_input_defines_the_levels_in_the_pipeline(self, tmp_path, n_levels):
        """
        number_of_levels from the input gives number_of_levels + 1 level results,
        and the output is written
        """
        run = _run_main_with_spies(tmp_path, params=_make_params(number_of_levels=n_levels))
        levels = run.captured["pipeline"][PIPE_LEVELS]
        assert len(levels) == n_levels + 1, \
        f"expected {n_levels + 1} levels, got {len(levels)}; test #1"
        assert run.out.is_file(), \
        f"expected the output file {run.out} to exist; test #2"


# ===========================================================================
# --- qmt_computations_main - reproducibility and side effects ---
# ===========================================================================
class TestReproducibilityAndSideEffects:

    def test_repeated_runs_give_identical_output(self, tmp_path):
        """ Two runs with the same inputs give identical output files """
        _, out = _run_main(tmp_path)
        first = _read(out)
        _, out = _run_main(tmp_path)
        second = _read(out)
        assert first == second, "expected identical output files for the repeated runs"

    def test_input_file_is_not_modified(self, tmp_path):
        """ The input file is not modified by the run """
        inp, _ = _run_main(tmp_path)
        assert inp.read_text(encoding="utf-8") == "dummy", \
        f"expected the input file to stay {'dummy'}, got {inp.read_text(encoding='utf-8')}"

    def test_params_and_irc_are_not_modified(self, tmp_path):
        """ The parameters and the IRC data are not modified by the run """
        params, irc = _make_params(), _make_irc()
        params_before = dict(vars(params))
        irc_before = {name: np.array(value) for name, value in vars(irc).items()}
        _run_main(tmp_path, params, irc)
        assert vars(params) == params_before, \
        f"expected {params_before}, got {vars(params)}; test #1"
        for name, value in irc_before.items():
            assert np.array_equal(getattr(irc, name), value), \
            f"expected irc.{name} to stay unchanged; test #2"

    def test_only_the_output_file_is_created(self, tmp_path):
        """ The run creates the output file and nothing else in the directory """
        inp, out = _run_main(tmp_path)
        files = {path.name for path in tmp_path.iterdir()}
        expected = {inp.name, out.name}
        assert files == expected, f"expected {expected}, got {files}"


# ===========================================================================
# --- qmt_computations_main - input validation ---
# ===========================================================================
class TestInputValidation:

    def test_missing_file_raises_and_creates_nothing(self, tmp_path):
        """
        A missing input file raises an error; nothing is read or written.
        Expecting a FileNotFoundError
        """
        with patch(f"{FUNC_PATH}.read_input") as read_mock:
            with pytest.raises(FileNotFoundError, match="File not found"):
                qmt_computations_main(tmp_path / "nonexistent.txt")
        read_mock.assert_not_called()  # it should not be called
        assert list(tmp_path.iterdir()) == [], \
        f"expected an empty directory, got {list(tmp_path.iterdir())}"

    def test_reader_failure_propagates_and_creates_no_output(self, tmp_path):
        """ An error of read_input propagates and no output file is created. Expecting a ValueError """
        inp = _write_input(tmp_path)
        with patch(f"{FUNC_PATH}.read_input", side_effect=ValueError("bad input")):
            with pytest.raises(ValueError, match="bad input"):
                qmt_computations_main(inp)
        assert not _output_for(inp).exists(), \
        f"expected no output file, but {_output_for(inp)} exists"

    def test_pipeline_failure_creates_no_output(self, tmp_path):
        """ An error in the pipeline propagates. Expecting an AttributeError """
        params = _make_params()
        delattr(params, "freq")
        inp = _write_input(tmp_path)
        with patch(f"{FUNC_PATH}.read_input", return_value=(params, _make_irc())):
            with pytest.raises(AttributeError, match="freq"):
                qmt_computations_main(inp)
        assert not _output_for(inp).exists(), \
        f"expected no output file, but {_output_for(inp)} exists"