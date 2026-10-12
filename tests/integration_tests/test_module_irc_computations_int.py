# --- Test: test_module_irc_computations_int. Integration tests for module_irc_computations.py
# Run with: pytest test_module_irc_computations_int.py -v ---

# --- Modules ---
import pytest # type: ignore
from contextlib import ExitStack
from unittest.mock import MagicMock, patch, sentinel
pytestmark = pytest.mark.integration  # type: ignore

from pathlib import Path

from tunnex_2.constants_and_dataclasses.constants_and_settings import ( # type: ignore
    COMMAND_FILE_PATH,
    RUN_SOFTWARE_KEY_MARKER,
    CORR_ANALYSIS_BOOL,
    POTENTIAL_SCAL_FACTOR_DEFAULT,
    PICK_MAX_FREQ_FLAG)
from tunnex_2.constants_and_dataclasses.dataclasses import FindIRCConfig # type: ignore


# --- Module to test ---
from tunnex_2.irc_computations.module_irc_computations import irc_computations_main # type: ignore


# --- Helpers and shared data ---
IRC_MAIN_MODULE = "tunnex_2.irc_computations.module_irc_computations"

PIPELINE_FUNCS = ("ts_optimization",
                  "irc_computations",
                  "irc_file_parsing",
                  "react_prod_optimization",
                  "opt_file_parsing",
                  "attempt_freq_func",
                  "number_of_levels_func",
                  "potential_scal_factor_comp",
                  "tunnex_input_writing",
                  "eckart_react_prod_optimization",
                  "eckart_potential_results")

STANDARD_ORDER = ["ts_optimization", "irc_computations", "irc_file_parsing", "react_prod_optimization",
                  "attempt_freq_func", "opt_file_parsing", "number_of_levels_func", "tunnex_input_writing"]

HYBRID_ORDER = ["ts_optimization", "irc_computations", "irc_file_parsing", "react_prod_optimization",
                "attempt_freq_func", "opt_file_parsing", "potential_scal_factor_comp",
                "number_of_levels_func", "tunnex_input_writing"]

ECKART_ORDER = ["ts_optimization", "eckart_react_prod_optimization", "attempt_freq_func",
                "opt_file_parsing", "eckart_potential_results", "number_of_levels_func", "tunnex_input_writing"]


TS_OPT = Path("ts_opt.out")
IRC_OUT = Path("irc.out")
REACT_OPT = Path("react_opt.out")
PROD_OPT = Path("prod_opt.out")
ECKART_REACT_OPT = Path("eckart_react_opt.out")
ECKART_PROD_OPT = Path("eckart_prod_opt.out")
TUNNEX_FORWARD = Path("tunnex_forward.inp")
TUNNEX_BACKWARD = Path("tunnex_backward.inp")
ATT_FREQ_REACT = 1.0e13
ATT_FREQ_PROD = 2.0e13
SCAL_FACTORS = (0.9, 1.1)
NUM_LEVELS = (7, 9)
# Synthetic outputs of the mocked stages


def _target(name: str) -> str:
    """ Returns the patch target for a name imported into the module under test """
    return f"{IRC_MAIN_MODULE}.{name}"

def _same_path(a, b) -> bool:
    """ Compares two paths regardless of the relative or absolute form """
    return Path(a).resolve() == Path(b).resolve()

def _touch(tmp_path: Path, name: str) -> Path:
    """ Creates an empty file (it stands for a precomputed output file) and return its path """
    path = tmp_path / name
    path.write_text("dummy output\n")
    return path

def _make_config(tmp_path: Path, **overrides) -> FindIRCConfig:
    """
    Returns a real FindIRCConfig for the standard (non-Eckart, Gaussian) scenario;
    fields can be overridden
    """
    params = dict(ts_input_file=tmp_path / "ts.inp",
                  prog_mode="gaussian",
                  eckart=False,
                  react_input_file=None,
                  prod_input_file=None,
                  ts_computed_file=None,
                  irc_computed_file=None,
                  react_computed_file=None,
                  prod_computed_file=None,
                  proj_freq=False,
                  calc_all=False,
                  hess_parsing_flag=False,
                  hybrid_mode=False)
    params.update(overrides)
    return FindIRCConfig(**params)


class _Pipeline:
    """ Mocked pipeline stages with the synthetic return values; records the order of the calls """
    def __init__(self):
        # --- Mocking the intermal functions ---
        self.manager = MagicMock()
        self.mocks = {name: MagicMock() for name in PIPELINE_FUNCS}
        for name, mock in self.mocks.items():
            self.manager.attach_mock(mock, name)
        # --- Mocking the attempt frequency ---
        self.attempt_freq = MagicMock()
        self.attempt_freq.att_freq_react = ATT_FREQ_REACT
        self.attempt_freq.att_freq_prod = ATT_FREQ_PROD
        # --- Defining the mock output ---
        self.mocks["ts_optimization"].return_value = TS_OPT
        self.mocks["irc_computations"].return_value = IRC_OUT
        self.mocks["irc_file_parsing"].return_value = sentinel.irc_data
        self.mocks["react_prod_optimization"].return_value = (REACT_OPT, PROD_OPT)
        self.mocks["eckart_react_prod_optimization"].return_value = (ECKART_REACT_OPT, ECKART_PROD_OPT)
        self.mocks["attempt_freq_func"].return_value = self.attempt_freq
        self.mocks["opt_file_parsing"].return_value = sentinel.energy_points
        self.mocks["eckart_potential_results"].return_value = sentinel.eckart_irc_data
        self.mocks["potential_scal_factor_comp"].return_value = SCAL_FACTORS
        self.mocks["number_of_levels_func"].return_value = NUM_LEVELS
        self.mocks["tunnex_input_writing"].return_value = (TUNNEX_FORWARD, TUNNEX_BACKWARD)

    def run(self, config):
        """ Run irc_computations_main with all the pipeline stages mocked """
        with ExitStack() as stack:
            for name, mock in self.mocks.items():
                stack.enter_context(patch(_target(name), mock))
            return irc_computations_main(config)

    def call_order(self) -> list[str]:
        """ Names of the stages in the order they were called """
        return [call[0] for call in self.manager.mock_calls if "." not in call[0]]

    def args(self, name: str) -> tuple:
        """ Positional arguments of the (single) call of the stage """
        mock = self.mocks[name]
        assert mock.call_count == 1, \
        f"expected {name} to be called once, got {mock.call_count} calls"
        return mock.call_args.args

    def not_called(self) -> list[str]:
        """ Names of the stages that were never called """
        return [name for name, mock in self.mocks.items() if mock.call_count == 0]


# ======================================================================================
# --- irc_computations_main - standard flow (Gaussian or ORCA, no precomputed files) ---
# ======================================================================================
class TestStandardFlow:

    @pytest.mark.parametrize("prog_mode", ["gaussian", "orca"])
    def test_stages_called_in_expected_order(self, tmp_path, prog_mode):
        """ All stages of the standard flow are called once, in the expected order """
        pipe = _Pipeline()
        pipe.run(_make_config(tmp_path, prog_mode=prog_mode))
        assert pipe.call_order() == STANDARD_ORDER, f"expected {STANDARD_ORDER}, got {pipe.call_order()}"

    def test_eckart_and_hybrid_stages_not_called(self, tmp_path):
        """ Eckart-specific and hybrid-specific stages are skipped in the standard flow """
        pipe = _Pipeline()
        pipe.run(_make_config(tmp_path))
        for name in ("eckart_react_prod_optimization", "eckart_potential_results", "potential_scal_factor_comp"):
            assert name in pipe.not_called(), \
            f"expected {name} not to be called, got {pipe.mocks[name].call_count} calls"

    def test_returns_tunnex_input_files(self, tmp_path):
        """ The function returns exactly the pair of files produced by tunnex_input_writing """
        pipe = _Pipeline()
        result = pipe.run(_make_config(tmp_path))
        assert result == (TUNNEX_FORWARD, TUNNEX_BACKWARD), \
        f"expected {(TUNNEX_FORWARD, TUNNEX_BACKWARD)}, got {result}"

    @pytest.mark.parametrize("prog_mode", ["gaussian", "orca"])
    def test_prog_mode_reaches_every_stage(self, tmp_path, prog_mode):
        """ The selected program mode is passed to all the stages which depend on it """
        pipe = _Pipeline()
        pipe.run(_make_config(tmp_path, prog_mode=prog_mode))
        expected_positions = {"ts_optimization": 2, "irc_computations": 3, "irc_file_parsing": 4,
                              "react_prod_optimization": 3, "attempt_freq_func": 4, "opt_file_parsing": 4}
        for name, pos in expected_positions.items():
            result = pipe.args(name)[pos]
            assert result == prog_mode, f"{name}: expected prog_mode={prog_mode}, got {result}"

    def test_ts_optimization_arguments(self, tmp_path):
        """
        ts_optimization receives the TS input, the command file path,
        the program mode and the marker
        """
        pipe = _Pipeline()
        config = _make_config(tmp_path)
        pipe.run(config)
        expected = (config.ts_input_file, COMMAND_FILE_PATH, "gaussian", RUN_SOFTWARE_KEY_MARKER)
        assert pipe.args("ts_optimization") == expected, \
        f"expected {expected}, got {pipe.args('ts_optimization')}"

    def test_irc_computations_gets_ts_output(self, tmp_path):
        """
        The optimized TS file is passed on to the IRC computation
        together with proj_freq and calc_all
        """
        pipe = _Pipeline()
        config = _make_config(tmp_path, proj_freq=True, calc_all=True)
        pipe.run(config)
        expected = (config.ts_input_file, TS_OPT, COMMAND_FILE_PATH, "gaussian",
        RUN_SOFTWARE_KEY_MARKER, True, True)
        assert pipe.args("irc_computations") == expected, \
        f"expected {expected}, got {pipe.args('irc_computations')}"

    def test_irc_file_parsing_gets_irc_and_ts_outputs(self, tmp_path):
        """ IRC parsing receives the IRC output and the TS output produced by the previous stages """
        pipe = _Pipeline()
        config = _make_config(tmp_path, proj_freq=True, hess_parsing_flag=True)
        pipe.run(config)
        expected = (IRC_OUT, TS_OPT, config.ts_input_file, COMMAND_FILE_PATH, "gaussian",
        True, True, RUN_SOFTWARE_KEY_MARKER)
        assert pipe.args("irc_file_parsing") == expected, \
        f"expected {expected}, got {pipe.args('irc_file_parsing')}"

    def test_react_prod_optimization_gets_irc_output(self, tmp_path):
        """ Reactant or product optimization is built from the TS input and the IRC output """
        pipe = _Pipeline()
        config = _make_config(tmp_path)
        pipe.run(config)
        expected = (config.ts_input_file, IRC_OUT, COMMAND_FILE_PATH, "gaussian", RUN_SOFTWARE_KEY_MARKER)
        assert pipe.args("react_prod_optimization") == expected, \
        f"expected {expected}, got {pipe.args('react_prod_optimization')}"

    def test_attempt_freq_gets_all_optimized_structures(self, tmp_path):
        """ Attempt frequencies are computed from the optimized TS, reactant and product """
        pipe = _Pipeline()
        pipe.run(_make_config(tmp_path, hess_parsing_flag=True))
        expected = (TS_OPT, REACT_OPT, PROD_OPT, CORR_ANALYSIS_BOOL, "gaussian", True, PICK_MAX_FREQ_FLAG)
        assert pipe.args("attempt_freq_func") == expected, \
        f"expected {expected}, got {pipe.args('attempt_freq_func')}"

    def test_opt_file_parsing_gets_attempt_freq(self, tmp_path):
        """ Energy parsing receives the optimized structures and the attempt-frequency object """
        pipe = _Pipeline()
        pipe.run(_make_config(tmp_path, hess_parsing_flag=True))
        expected = (TS_OPT, REACT_OPT, PROD_OPT, pipe.attempt_freq, "gaussian", True)
        assert pipe.args("opt_file_parsing") == expected, \
        f"expected {expected}, got {pipe.args('opt_file_parsing')}"

    def test_number_of_levels_gets_default_scaling(self, tmp_path):
        """ Without the hybrid mode the number of levels is computed with the default scaling factors """
        pipe = _Pipeline()
        pipe.run(_make_config(tmp_path, hybrid_mode=False))
        expected = (sentinel.irc_data, sentinel.energy_points, ATT_FREQ_REACT, ATT_FREQ_PROD,
        POTENTIAL_SCAL_FACTOR_DEFAULT, POTENTIAL_SCAL_FACTOR_DEFAULT)
        assert pipe.args("number_of_levels_func") == expected, \
        f"expected {expected}, got {pipe.args('number_of_levels_func')}"

    def test_tunnex_input_writing_gets_results_of_all_stages(self, tmp_path):
        """ The final writer gets the data accumulated across the whole pipeline """
        pipe = _Pipeline()
        config = _make_config(tmp_path)
        pipe.run(config)
        expected = (config.ts_input_file, sentinel.irc_data, sentinel.energy_points, pipe.attempt_freq,
        NUM_LEVELS[0], NUM_LEVELS[1], POTENTIAL_SCAL_FACTOR_DEFAULT, POTENTIAL_SCAL_FACTOR_DEFAULT)
        assert pipe.args("tunnex_input_writing") == expected, \
            f"expected {expected}, got {pipe.args('tunnex_input_writing')}"


# ===========================================================================
# --- irc_computations_main - hybrid mode (potential scaling factors) ---
# ===========================================================================
class TestHybridMode:

    def test_stage_order_with_hybrid_mode(self, tmp_path):
        """ The scaling-factor stage is inserted between the energy parsing and the levels computation """
        pipe = _Pipeline()
        pipe.run(_make_config(tmp_path, hybrid_mode=True))
        assert pipe.call_order() == HYBRID_ORDER, f"expected {HYBRID_ORDER}, got {pipe.call_order()}"

    def test_scaling_factor_gets_irc_data_and_energy_points(self, tmp_path):
        """ potential_scal_factor_comp is fed with the parsed IRC data and the energy points """
        pipe = _Pipeline()
        pipe.run(_make_config(tmp_path, hybrid_mode=True))
        expected = (sentinel.irc_data, sentinel.energy_points)
        assert pipe.args("potential_scal_factor_comp") == expected, \
        f"expected {expected}, got {pipe.args('potential_scal_factor_comp')}"

    def test_scaling_factors_propagate_to_levels_and_writer(self, tmp_path):
        """ The computed scaling factors reach both the levels computation and the input writer """
        pipe = _Pipeline()
        pipe.run(_make_config(tmp_path, hybrid_mode=True))
        levels_args = pipe.args("number_of_levels_func")
        writer_args = pipe.args("tunnex_input_writing")
        assert levels_args[4:] == SCAL_FACTORS, \
        f"levels: expected scaling={SCAL_FACTORS}, got {levels_args[4:]}; test #1"
        assert writer_args[6:] == SCAL_FACTORS, \
        f"writer: expected scaling={SCAL_FACTORS}, got {writer_args[6:]}; test #2"

    def test_number_of_levels_propagate_to_writer(self, tmp_path):
        """ The number of levels from number_of_levels_func is passed to the input writer unchanged """
        pipe = _Pipeline()
        pipe.run(_make_config(tmp_path, hybrid_mode=True))
        writer_args = pipe.args("tunnex_input_writing")
        assert writer_args[4:6] == NUM_LEVELS, \
        f"expected levels={NUM_LEVELS}, got {writer_args[4:6]}"


# ===========================================================================
# --- irc_computations_main - Eckart potential flow ---
# ===========================================================================
class TestEckartFlow:

    def _config(self, tmp_path, **overrides):
        """ Config for the Eckart scenario with the reactant and product inputs """
        return _make_config(tmp_path, eckart=True,
                            react_input_file=tmp_path / "react.inp",
                            prod_input_file=tmp_path / "prod.inp", **overrides)
    
    @pytest.mark.parametrize("prog_mode", ["gaussian", "orca"])
    def test_stages_called_in_expected_order(self, prog_mode, tmp_path):
        """ The Eckart flow skips the IRC stages and uses the Eckart-specific ones """
        pipe = _Pipeline()
        pipe.run(self._config(tmp_path, prog_mode=prog_mode))
        assert pipe.call_order() == ECKART_ORDER, \
        f"expected {ECKART_ORDER}, got {pipe.call_order()}"

    def test_irc_stages_not_called(self, tmp_path):
        """
        No IRC computation, IRC parsing or IRC-based optimization
        is done for the Eckart potential
        """
        pipe = _Pipeline()
        pipe.run(self._config(tmp_path))
        for name in ("irc_computations", "irc_file_parsing", "react_prod_optimization"):
            assert name in pipe.not_called(), \
            f"expected {name} not to be called, got {pipe.mocks[name].call_count} calls"

    def test_eckart_optimization_gets_react_and_prod_inputs(self, tmp_path):
        """ Reactant and product inputs are passed to the Eckart optimization stage """
        pipe = _Pipeline()
        config = self._config(tmp_path)
        pipe.run(config)
        expected = (config.react_input_file, config.prod_input_file, COMMAND_FILE_PATH, "gaussian",
        RUN_SOFTWARE_KEY_MARKER)
        assert pipe.args("eckart_react_prod_optimization") == expected, \
        f"expected {expected}, got {pipe.args('eckart_react_prod_optimization')}"

    def test_eckart_optimized_structures_reach_attempt_freq(self, tmp_path):
        """ The structures optimized in the Eckart branch are used for the attempt frequencies """
        pipe = _Pipeline()
        pipe.run(self._config(tmp_path))
        args = pipe.args("attempt_freq_func")
        assert args[:3] == (TS_OPT, ECKART_REACT_OPT, ECKART_PROD_OPT), \
        f"expected {(TS_OPT, ECKART_REACT_OPT, ECKART_PROD_OPT)}, got {args[:3]}"

    def test_eckart_potential_results_gets_energy_points(self, tmp_path):
        """ eckart_potential_results receives the TS output and the parsed energy points """
        pipe = _Pipeline()
        pipe.run(self._config(tmp_path, hess_parsing_flag=True))
        expected = (TS_OPT, sentinel.energy_points, "gaussian", True)
        assert pipe.args("eckart_potential_results") == expected, \
        f"expected {expected}, got {pipe.args('eckart_potential_results')}"

    def test_eckart_irc_data_used_downstream(self, tmp_path):
        """
        The IRC data from the Eckart potential replace
        the parsed IRC data in the later stages
        """
        pipe = _Pipeline()
        pipe.run(self._config(tmp_path))
        levels_irc = pipe.args("number_of_levels_func")[0]
        writer_irc = pipe.args("tunnex_input_writing")[1]
        assert levels_irc is sentinel.eckart_irc_data, \
        f"levels: expected {sentinel.eckart_irc_data}, got {levels_irc}; test #1"
        assert writer_irc is sentinel.eckart_irc_data, \
        f"writer: expected {sentinel.eckart_irc_data}, got {writer_irc}; test #2"

    def test_hybrid_mode_ignored_for_eckart(self, tmp_path):
        """
        The hybrid mode has no effect for the Eckart potential
        and default potential scaling factors are used
        """
        pipe = _Pipeline()
        pipe.run(self._config(tmp_path, hybrid_mode=True))
        assert "potential_scal_factor_comp" in pipe.not_called(), \
        f"expected {"potential_scal_factor_comp"} not to be called; test #1"
        writer_args = pipe.args("tunnex_input_writing")
        expected = (POTENTIAL_SCAL_FACTOR_DEFAULT, POTENTIAL_SCAL_FACTOR_DEFAULT)
        assert writer_args[6:] == expected, \
        f"expected scaling={expected}, got {writer_args[6:]}; test #2"

    def test_returns_tunnex_input_files(self, tmp_path):
        """ The Eckart flow returns the files produced by the input writer """
        pipe = _Pipeline()
        result = pipe.run(self._config(tmp_path))
        assert result == (TUNNEX_FORWARD, TUNNEX_BACKWARD), \
        f"expected {(TUNNEX_FORWARD, TUNNEX_BACKWARD)}, got {result}"

    def test_irc_computed_file_is_ignored(self, tmp_path):
        """ A (non-existing) precomputed IRC file does not matter for the Eckart flow """
        pipe = _Pipeline()
        config = self._config(tmp_path, irc_computed_file=tmp_path / "does_not_exist.out")
        result = pipe.run(config)
        assert result == (TUNNEX_FORWARD, TUNNEX_BACKWARD), \
        f"expected {(TUNNEX_FORWARD, TUNNEX_BACKWARD)}, got {result}"


# ======================================================================================
# --- irc_computations_main - precomputed files ---
# ======================================================================================
class TestPrecomputedFiles:

    def test_precomputed_ts_skips_optimization(self, tmp_path):
        """ A precomputed TS file replaces the TS optimization and is used by the next stages """
        pipe = _Pipeline()
        ts_file = _touch(tmp_path, "ts_precomputed.out")
        pipe.run(_make_config(tmp_path, ts_computed_file=ts_file))
        assert "ts_optimization" in pipe.not_called(), \
        f"expected {"ts_optimization"} not to be called; test #1"
        assert _same_path(pipe.args("irc_computations")[1], ts_file), \
        f"expected TS output={ts_file}, got {pipe.args('irc_computations')[1]}; test #2"
        assert _same_path(pipe.args("attempt_freq_func")[0], ts_file), \
        f"expected TS output={ts_file}, got {pipe.args('attempt_freq_func')[0]}; test #3"

    def test_precomputed_irc_skips_irc_computation(self, tmp_path):
        """ A precomputed IRC file replaces the IRC computation but is still parsed and used afterwards """
        pipe = _Pipeline()
        irc_file = _touch(tmp_path, "irc_precomputed.out")
        pipe.run(_make_config(tmp_path, irc_computed_file=irc_file))
        assert "irc_computations" in pipe.not_called(), \
        f"expected {"irc_computations"} not to be called; test #1"
        assert _same_path(pipe.args("irc_file_parsing")[0], irc_file), \
        f"expected IRC output={irc_file}, got {pipe.args('irc_file_parsing')[0]}; test #2"
        assert _same_path(pipe.args("react_prod_optimization")[1], irc_file), \
        f"expected IRC output={irc_file}, got {pipe.args('react_prod_optimization')[1]}; test #3"

    def test_precomputed_react_and_prod_skip_optimization(self, tmp_path):
        """ Both precomputed reactant and product files replace the optimization stage """
        pipe = _Pipeline()
        react_file = _touch(tmp_path, "react_precomputed.out")
        prod_file = _touch(tmp_path, "prod_precomputed.out")
        pipe.run(_make_config(tmp_path, react_computed_file=react_file, prod_computed_file=prod_file))
        assert "react_prod_optimization" in pipe.not_called(), \
        f"expected {"react_prod_optimization"} not to be called; test #1"
        args = pipe.args("attempt_freq_func")
        assert _same_path(args[1], react_file), \
        f"expected reactant={react_file}, got {args[1]}; test #2"
        assert _same_path(args[2], prod_file), \
        f"expected product={prod_file}, got {args[2]}; test #3"

    def test_only_one_of_react_prod_precomputed_triggers_optimization(self, tmp_path):
        """
        With only one of the two precomputed files both structures are optimized
        and the file is not used
        """
        pipe = _Pipeline()
        react_file = _touch(tmp_path, "react_precomputed.out")
        pipe.run(_make_config(tmp_path, react_computed_file=react_file))
        args = pipe.args("attempt_freq_func")
        assert args[1:3] == (REACT_OPT, PROD_OPT), f"expected {(REACT_OPT, PROD_OPT)}, got {args[1:3]}"

    def test_eckart_precomputed_react_and_prod_skip_optimization(self, tmp_path):
        """ Precomputed reactant and product files also replace the optimization in the Eckart flow """
        pipe = _Pipeline()
        react_file = _touch(tmp_path, "react_precomputed.out")
        prod_file = _touch(tmp_path, "prod_precomputed.out")
        config = _make_config(tmp_path, eckart=True,
                              react_input_file=tmp_path / "react.inp",
                              prod_input_file=tmp_path / "prod.inp",
                              react_computed_file=react_file,
                              prod_computed_file=prod_file)
        pipe.run(config)
        assert "eckart_react_prod_optimization" in pipe.not_called(), \
        f"expected {"eckart_react_prod_optimization"} not to be called; test #1"
        args = pipe.args("attempt_freq_func")
        assert _same_path(args[1], react_file), f"expected reactant={react_file}, got {args[1]}; test #2"
        assert _same_path(args[2], prod_file), f"expected product={prod_file}, got {args[2]}; test #3"

    def test_all_precomputed_runs_only_parsing_stages(self, tmp_path):
        """ With every output precomputed only the parsing/post-processing stages are executed """
        pipe = _Pipeline()
        config = _make_config(tmp_path,
                              ts_computed_file=_touch(tmp_path, "ts.out"),
                              irc_computed_file=_touch(tmp_path, "irc.out"),
                              react_computed_file=_touch(tmp_path, "react.out"),
                              prod_computed_file=_touch(tmp_path, "prod.out"))
        result = pipe.run(config)
        expected_order = ["irc_file_parsing", "attempt_freq_func", "opt_file_parsing",
                          "number_of_levels_func", "tunnex_input_writing"]
        assert pipe.call_order() == expected_order, \
        f"expected {expected_order}, got {pipe.call_order()}; test #1"
        assert result == (TUNNEX_FORWARD, TUNNEX_BACKWARD), \
        f"expected {(TUNNEX_FORWARD, TUNNEX_BACKWARD)}, got {result}; test #2"

    @pytest.mark.parametrize("missing_key", ["ts_computed_file", "irc_computed_file"])
    def test_missing_precomputed_file_raises_before_any_stage(self, tmp_path, missing_key):
        """ A non-existing precomputed file stops the run; the stages after it are not executed """
        pipe = _Pipeline()
        overrides = {"ts_computed_file": _touch(tmp_path, "ts.out"), missing_key: tmp_path / "missing.out"}
        with pytest.raises(FileNotFoundError, match="was not found"):
            pipe.run(_make_config(tmp_path, **overrides))
        for name in ("attempt_freq_func", "opt_file_parsing", "number_of_levels_func", "tunnex_input_writing"):
            assert name in pipe.not_called(), \
            f"expected {name} not to be called, got {pipe.mocks[name].call_count} calls"


# ===========================================================================
# --- irc_computations_main - input validation ---
# ===========================================================================
class TestValidation:

    def _assert_nothing_computed(self, pipe):
        """ Assert that no pipeline stage was called """
        assert pipe.call_order() == [], \
        f"expected no stages to be called, got {pipe.call_order()}"

    def test_react_prod_inputs_not_required_without_eckart(self, tmp_path):
        """ Reactant and product inputs are not required in the standard flow """
        pipe = _Pipeline()
        result = pipe.run(_make_config(tmp_path, react_input_file=None, prod_input_file=None))
        assert result == (TUNNEX_FORWARD, TUNNEX_BACKWARD), \
        f"expected {(TUNNEX_FORWARD, TUNNEX_BACKWARD)}, got {result}"
    
    @pytest.mark.parametrize("prog_mode", ["", "psi4", "ORCA", None])
    def test_unknown_prog_mode_raises(self, tmp_path, prog_mode):
        """ An unsupported program mode raises an error. Expecting a ValueError """
        pipe = _Pipeline()
        with pytest.raises(ValueError, match="Expected"):
            pipe.run(_make_config(tmp_path, prog_mode=prog_mode))
        self._assert_nothing_computed(pipe) # it should not be called

    def test_missing_ts_input_raises(self, tmp_path):
        """ No TS input file raises an error. Expecting a FileNotFoundError """
        pipe = _Pipeline()
        with pytest.raises(FileNotFoundError, match="TS input file"):
            pipe.run(_make_config(tmp_path, ts_input_file=None))
        self._assert_nothing_computed(pipe) # it should not be called
    
    def test_eckart_without_react_input_raises(self, tmp_path):
        """ Eckart mode without the reactant input raises an error. Expecting a FileNotFoundError """
        pipe = _Pipeline()
        config = _make_config(tmp_path, eckart=True, prod_input_file=tmp_path / "prod.inp")
        with pytest.raises(FileNotFoundError, match="Reactant input file"):
            pipe.run(config)
        self._assert_nothing_computed(pipe) # it should not be called

    def test_eckart_without_prod_input_raises(self, tmp_path):
        """ Eckart mode without the product input raises an error. Expecting a FileNotFoundError """
        pipe = _Pipeline()
        config = _make_config(tmp_path, eckart=True, react_input_file=tmp_path / "react.inp")
        with pytest.raises(FileNotFoundError, match="Product input file"):
            pipe.run(config)
        self._assert_nothing_computed(pipe) # it should not be called


# ===========================================================================
# --- irc_computations_main - error propagation between the stages ---
# ===========================================================================
class TestErrorPropagation:

    @pytest.mark.parametrize("failing_stage", STANDARD_ORDER)
    def test_failure_stops_the_pipeline(self, tmp_path, failing_stage):
        """ An exception in any stage is propagated and the later stages are not executed """
        pipe = _Pipeline()
        pipe.mocks[failing_stage].side_effect = RuntimeError(f"{failing_stage} failed")
        with pytest.raises(RuntimeError, match=f"{failing_stage} failed"):
            pipe.run(_make_config(tmp_path))
        expected_order = STANDARD_ORDER[:STANDARD_ORDER.index(failing_stage) + 1]
        assert pipe.call_order() == expected_order, \
        f"expected {expected_order}, got {pipe.call_order()}"

    @pytest.mark.parametrize("failing_stage", ECKART_ORDER)
    def test_failure_stops_the_eckart_pipeline(self, tmp_path, failing_stage):
        """
        An exception in any stage of the Eckart flow is propagated
        and the later stages are not executed
        """
        pipe = _Pipeline()
        pipe.mocks[failing_stage].side_effect = RuntimeError(f"{failing_stage} failed")
        config = _make_config(tmp_path, eckart=True,
                              react_input_file=tmp_path / "react.inp",
                              prod_input_file=tmp_path / "prod.inp")
        with pytest.raises(RuntimeError, match=f"{failing_stage} failed"):
            pipe.run(config)
        expected_order = ECKART_ORDER[:ECKART_ORDER.index(failing_stage) + 1]
        assert pipe.call_order() == expected_order, \
        f"expected {expected_order}, got {pipe.call_order()}"

    def test_failed_run_returns_no_files(self, tmp_path):
        """
        The input writer is the last stage: if it fails,
        no partial result is returned. Mocking an OSError
        """
        pipe = _Pipeline()
        pipe.mocks["tunnex_input_writing"].side_effect = OSError("disk is full")
        with pytest.raises(OSError, match="disk is full"):
            pipe.run(_make_config(tmp_path))