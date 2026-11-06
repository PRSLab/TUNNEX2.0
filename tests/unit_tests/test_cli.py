# --- Test: test_cli. Unit tests for cli.py
# Run with: pytest test_cli.py -v ---

# --- Modules ---
import pytest  # type: ignore
from unittest.mock import patch


# --- Module to test ---
from tunnex_2.cli import cli # type: ignore


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.cli"


def _run(argv: list[str]):
    """ Should patch sys.argv and call cli(), returning the FindIRCConfig passed to main """
    captured = {}
    def capture(configs):
        captured["configs"] = configs
    with patch("sys.argv", ["tunnex_2"] + argv), \
         patch(FUNC_PATH+".main", side_effect=capture):
         cli()
    return captured.get("configs")

def _run_expect_error(argv: list[str]):
    """ Should expect cli() to call parser.error() """
    with patch("sys.argv", ["tunnex_2"] + argv), \
         patch(FUNC_PATH+".main"):
        with pytest.raises(SystemExit) as exc_info:
            cli()
        return exc_info


# ============================================================================
# --- cli - default mode (no --eckart, no --qmt_only, no --comp; 1 file) ---
# ============================================================================
class TestDefaultMode:

    def test_single_file_sets_ts_input(self):
        """ Sets the transition state input file from the provided file """
        cfg = _run(["mol.gjf"])
        assert str(cfg.ts_input_file) == "mol.gjf", f"Expected {"mol.gjf"}, "
        f"got {str(cfg.ts_input_file)}"

    def test_default_prog_mode_is_gaussian(self):
        """ Sets Gaussian as the default program mode """
        cfg = _run(["mol.gjf"])
        assert cfg.prog_mode == "gaussian", f"Expected {"gaussian"}, got {cfg.prog_mode}"

    def test_default_proj_freq_true(self):
        """ Enables projected frequency calculation by default """
        cfg = _run(["mol.gjf"])
        assert cfg.proj_freq is True, f"Expected {True}, got {cfg.proj_freq}"

    def test_default_calc_all_true(self):
        """ Enables calculation of all frequencies by default """
        cfg = _run(["mol.gjf"])
        assert cfg.calc_all is True, f"Expected {True}, got {cfg.calc_all}"

    def test_default_hybrid_mode_false(self):
        """ Disables hybrid mode by default """
        cfg = _run(["mol.gjf"])
        assert cfg.hybrid_mode is False, f"Expected {False}, got {cfg.hybrid_mode}"

    def test_default_eckart_false(self):
        """ Disables the Eckart potential by default """
        cfg = _run(["mol.gjf"])
        assert cfg.eckart is False, f"Expected {False}, got {cfg.eckart}"
    
    def test_hess_parsing_flag_false(self):
        """ Disables Hessian parsing by default """
        cfg = _run(["mol.gjf"])
        assert cfg.hess_parsing_flag is False, f"Expected {False}, "
        f"got {cfg.hess_parsing_flag}"
   
    def test_default_qmt_module_true(self):
        """ Enables the QMT module by default """
        cfg = _run(["mol.gjf"])
        assert cfg.qmt_module is True, f"Expected {True}, got {cfg.qmt_module}"

    def test_default_prob_aver_mode_finite_sum(self):
        """ Sets finite sum as the default QMT probability averaging mode """
        cfg = _run(["mol.gjf"])
        assert cfg.prob_aver_mode_qmt == "finite_sum", f"Expected {"finite_sum"}, "
        f"got {cfg.prob_aver_mode_qmt}"

    def test_default_tunnex_input_none(self):
        """ Sets the Tunnex input file to None by default """
        cfg = _run(["mol.gjf"])
        assert cfg.tunnex_input == None, f"Expected {None}, got {cfg.tunnex_input}"

    def test_zero_files_raises(self):
        """ Raises an error when no input files are provided. Expecting a SystemExit 2 """
        result = _run_expect_error([])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"

    def test_two_files_raises(self):
        """
        Raises an error when more than one input file
        is provided in default mode. Expecting a SystemExit 2
        """
        result = _run_expect_error(["mol.gjf", "extra.gjf"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"


# ============================================================================
# --- cli - [-o] [--orca] flag
# ============================================================================
class TestOrcaFlag:

    def test_orca_sets_prog_mode(self):
        """ Tests that the ORCA flag sets the program mode to ORCA """
        cfg = _run(["-o", "--no_zpve", "mol.inp"])
        assert cfg.prog_mode == "orca", f"Expected {"orca"}, got {cfg.prog_mode}"

    def test_orca_with_calcfc_raises(self):
        """ Tests that ORCA cannot be combined with the calcfc option. Expecting a SystemExit 2 """
        result = _run_expect_error(["--orca", "--calcfc", "mol.inp"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"

    def test_orca_with_qmt_only_raises(self):
        """ Tests that ORCA cannot be combined with QMT-only mode. Expecting a SystemExit 2 """
        result = _run_expect_error(["--orca", "--qmt_only", "mol.inp"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"


# ============================================================================
# --- cli - [-z] [--no_zpve] flag
# ============================================================================
class TestNoZpveFlag:

    def test_no_zpve_sets_proj_freq_false(self):
        """ Tests that the no-ZPVE flag disables projected frequencies """
        cfg = _run(["-z", "mol.gjf"])
        assert cfg.proj_freq is False, f"Expected {False}, got {cfg.proj_freq}"

    def test_no_zpve_with_eckart_raises(self):
        """
        Tests that no-ZPVE cannot be combined with the Eckart option.
        Expecting a SystemExit 2
        """
        result = _run_expect_error(["--no_zpve", "--eckart", "mol.inp"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"

    def test_no_zpve_with_qmt_only_raises(self):
        """
        Tests that no-ZPVE cannot be combined with QMT-only mode.
        Expecting a SystemExit 2
        """
        result = _run_expect_error(["--no_zpve", "--qmt_only", "mol.inp"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"


# ============================================================================
# --- cli - [-f] [--calcfc] flag
# ============================================================================
class TestCalcfcFlag:

    def test_calcfc_sets_calc_all_false(self):
        """ Tests that the calcfc flag disables calculation of all frequencies """
        cfg = _run(["-f", "mol.gjf"])
        assert cfg.calc_all is False, f"Expected {False}, got {cfg.calc_all}"

    def test_calcfc_with_eckart_raises(self):
        """
        Tests that calcfc cannot be combined with the Eckart option.
        Expecting a SystemExit 2
        """
        result = _run_expect_error(["--calcfc", "--eckart", "mol.inp"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"

    def test_calcfc_with_qmt_only_raises(self):
        """
        Tests that calcfc cannot be combined with QMT-only mode.
        Expecting a SystemExit 2
        """
        result = _run_expect_error(["--calcfc", "--qmt_only", "mol.inp"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"


# ============================================================================
# --- cli - [-y] [--hybrid] flag
# ============================================================================
class TestHybridFlag:

    def test_hybrid_sets_hybrid_mode_true(self):
        """ Tests that the hybrid flag enables hybrid mode """
        cfg = _run(["-y", "mol.gjf"])
        assert cfg.hybrid_mode is True, f"Expected {True}, got {cfg.hybrid_mode}"

    def test_hybrid_with_eckt_raises(self):
        """
        Tests that hybrid mode cannot be combined with the Eckart option.
        Expecting a SystemExit 2
        """
        result = _run_expect_error(["--hybrid", "--eckart", "ts.gjf", "react.gjf", "prod.gjf"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"

    def test_hybrid_with_qmt_only_raises(self):
        """
        Tests that hybrid mode cannot be combined with QMT-only mode.
        Expecting a SystemExit 2
        """
        result = _run_expect_error(["--hybrid", "--qmt_only", "mol.gjf"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"


# ============================================================================
# --- cli - [-i] [--iso] flag
# ============================================================================
class TestIsoFlag:

    def test_iso_sets_hess_parsing_flag(self):
        """ Tests that the iso flag enables Hessian parsing """
        cfg = _run(["--i", "mol.gjf"])
        assert cfg.hess_parsing_flag is True, f"Expected {True}, got {cfg.hess_parsing_flag}"

    def test_iso_with_qmt_only_raises(self):
            """
            Tests that the iso flag cannot be combined with QMT-only mode.
            Expecting a SystemExit 2
            """
            result = _run_expect_error(["--iso", "--qmt_only", "mol.gjf"])
            assert result.value.code == 2, f"Expected {2}, got {result.value.code}"


# ============================================================================
# --- cli - [-p] [--no_qmt] flag
# ============================================================================
class TestNoQmtFlag:

    def test_no_qmt_sets_qmt_module_false(self):
        """ Tests that the no-QMT flag disables the QMT module """
        cfg = _run(["-p", "mol.gjf"])
        assert cfg.qmt_module is False, f"Expected {False}, got {cfg.qmt_module}"

    def test_no_qmt_with_qmt_only_raises(self):
        """
        Tests that the no-QMT flag cannot be combined with QMT-only mode.
        Expecting a SystemExit 2
        """
        result = _run_expect_error(["--no_qmt", "--qmt_only", "mol.gjf"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"


# ============================================================================
# --- cli - QMT mode (--qmt_only, no other flags; 1 file) ---
# --- cli - [-q] [--qmt_only] flag
# ============================================================================
class TestQmtOnlyMode:

    def test_qmt_only_sets_qmt_module_true(self):
        """ Tests that QMT-only mode enables the QMT module """
        cfg = _run(["-q", "tunnex.txt"])
        assert cfg.qmt_module is True, f"Expected {True}, got {cfg.qmt_module}"

    def test_qmt_only_sets_tunnex_input(self):
        """ Tests that QMT-only mode sets the Tunnex input file """
        cfg = _run(["--qmt_only", "tunnex.txt"])
        assert str(cfg.tunnex_input) == "tunnex.txt", f"Expected {"tunnex.txt"}, "
        f"got {str(cfg.tunnex_input)}"

    def test_qmt_only_requires_exactly_one_file(self):
        """
        Tests that QMT-only mode requires exactly one input file.
        Expecting a SystemExit 2
        """
        result = _run_expect_error(["--qmt_only", "a.txt", "b.txt"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"

    def test_qmt_only_with_no_files_raises(self):
        """
        Tests that QMT-only mode raises an error when
        no input file is provided. Expecting a SystemExit 2
        """
        result = _run_expect_error(["--qmt_only"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"


# ============================================================================
# --- cli - Eckart mode (--eckart, no --qmt_only, no --comp; 3 files) ---
# --- cli - [-e] [--eckart] flag
# ============================================================================
class TestEckartMode:

    def test_eckt_sets_eckart_true(self):
        """ Tests that the Eckart flag enables Eckart mode """
        cfg = _run(["--e", "ts.gjf", "react.gjf", "prod.gjf"])
        assert cfg.eckart is True, f"Expected {True}, got {cfg.eckart}"

    def test_eckt_sets_all_three_input_files(self):
        """ Tests that Eckart mode sets the transition state, reactant, and product input files """
        cfg = _run(["--eckart", "ts.gjf", "react.gjf", "prod.gjf"])
        expected = ["ts.gjf", "react.gjf", "prod.gjf"]
        result = [str(cfg.ts_input_file), str(cfg.react_input_file), str(cfg.prod_input_file)]
        for i, val in enumerate(expected):
            assert result[i] == val, f"Expected {val}, got {result[i]}; test #{i+1}"

    def test_eckt_requires_three_files(self):
        """
        Tests that Eckart mode requires exactly three input files.
        Expecting a SystemExit 2
        """
        result = _run_expect_error(["--eckart", "ts.gjf", "react.gjf"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"
    
    def test_eckart_with_qmt_only_raises(self):
        """
        Tests that the Eckart option cannot be combined with QMT-only mode.
        Expecting a SystemExit 2
        """
        result = _run_expect_error(["--eckart", "--qmt_only", "ts.gjf", "react.gjf", "prod.gjf"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"


# =========================================================================================
# --- cli - computed files mode (--comp, no --eckart, no --qmt_only; 2, 3, or 5 files) ---
# --- cli - [-c] [--comp] flag
# =========================================================================================
class TestCompMode:

    def test_comp_two_files_sets_ts_and_ts_computed(self):
        """ Tests that comp mode sets the transition state and computed transition state files """
        cfg = _run(["-c", "ts.gjf", "ts.out"])
        assert str(cfg.ts_input_file)  == "ts.gjf", f"Expected {"ts.gjf"}, "
        f"got {str(cfg.ts_input_file)}; test #1"
        assert str(cfg.ts_computed_file) == "ts.out", f"Expected {"ts.out"}, "
        f"got {str(cfg.ts_computed_file)}; test #2"
        assert cfg.irc_computed_file is None, f"Expected {None}, "
        f"got {cfg.irc_computed_file}; test #3"
        assert cfg.react_computed_file is None, f"Expected {None}, "
        f"got {cfg.react_computed_file}; test #4"
        assert cfg.prod_computed_file is None, f"Expected {None}, "
        f"got {cfg.prod_computed_file}; test #5"

    def test_comp_three_files_sets_irc(self):
        """ Tests that comp mode sets the computed IRC file when three files are provided """
        cfg = _run(["--comp", "ts.gjf", "ts.out", "irc.out"])
        assert str(cfg.irc_computed_file) == "irc.out", f"Expected {None}, "
        f"got {cfg.prod_computed_file}; test #1"
        assert cfg.react_computed_file is None, f"Expected {None}, "
        f"got {cfg.prod_computed_file}; test #2"

    def test_comp_five_files_sets_react_and_prod(self):
        """
        Tests that comp mode sets the computed reactant and product files
        when five files are provided
        """
        cfg = _run(["--comp", "ts.gjf", "ts.out", "irc.out", "react.out", "prod.out"])
        assert str(cfg.react_computed_file) == "react.out", f"Expected {"react.out"}, "
        f"got {str(cfg.react_computed_file)}; test #1"
        assert str(cfg.prod_computed_file)  == "prod.out", f"Expected {"prod.out"}, "
        f"got {str(cfg.prod_computed_file)}; test #2"

    def test_comp_wrong_file_count_raises(self):
        """
        Tests that comp mode raises an error for
        an invalid number of input files. Expecting a SystemExit 2
        """
        result = _run_expect_error(["--comp", "ts.gjf", "ts.out", "irc.out", "react.out"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"

    def test_comp_one_file_raises(self):
        """
        Tests that comp mode raises an error
        when only one input file is provided. Expecting a SystemExit 2
        """
        result = _run_expect_error(["--comp", "ts.gjf"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"


# =========================================================================================
# --- cli - Eckart computed files mode (--eckart and --comp, no --qmt_only; 4 files) ---
# --- cli - [-e] [--eckart] flag
# --- cli - [-c] [--comp] flag
# =========================================================================================
class TestEckartCompMode:

    def test_eckt_comp_four_files(self):
        """ Tests that Eckart comp mode sets the transition state, reactant, and product files """
        cfg = _run(["--eckart", "--comp", "ts.gjf", "ts.out", "react.out", "prod.out"])
        assert str(cfg.ts_input_file) == "ts.gjf", f"Expected {"ts.gjf"}, "
        f"got {str(cfg.ts_input_file)}; test #1"
        assert str(cfg.ts_computed_file) == "ts.out", f"Expected {"ts.out"}, "
        f"got {str(cfg.ts_computed_file)}; test #2"
        assert str(cfg.react_input_file) == "react.out", f"Expected {"react.out"}, "
        f"got {str(cfg.react_input_file)}; test #3"
        assert str(cfg.prod_input_file) == "prod.out", f"Expected {"prod.out"}, "
        f"got {str(cfg.prod_input_file)}; test #4"
        assert str(cfg.react_computed_file) == "react.out", f"Expected {"react.out"}, "
        f"got {str(cfg.react_computed_file)}; test #5"
        assert str(cfg.prod_computed_file) == "prod.out", f"Expected {"prod.out"}, "
        f"got {str(cfg.prod_computed_file)}; test #6"

    def test_eckt_comp_wrong_file_count_raises(self):
        """
        Tests that Eckart comp mode raises an error
        when fewer than four input files are provided. Expecting a SystemExit 2
        """
        result = _run_expect_error(["--eckart", "--comp", "ts.gjf", "ts.out", "react.out"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"

    def test_eckt_comp_five_files_raises(self):
        """
        Tests that Eckart comp mode raises an error
        when more than four input files are provided. Expecting a SystemExit 2
        """
        result = _run_expect_error(["--eckart", "--comp", "ts.gjf", "ts.out",
        "react.out", "prod.out", "extra.out"])
        assert result.value.code == 2, f"Expected {2}, got {result.value.code}"


# =========================================================================================
# --- cli - main() is always called ---
# =========================================================================================
class TestMainAlwaysCalled:

    def test_main_called_in_default_mode(self):
        """ Tests that main is called in the default mode """
        with patch("sys.argv", ["tunnex_2", "mol.gjf"]), \
             patch(FUNC_PATH+".main") as mock_main:
             cli()
        mock_main.assert_called_once() # it should be called once

    def test_main_called_in_qmt_only_mode(self):
        """ Tests that main is called in QMT-only mode """
        with patch("sys.argv", ["tunnex_2", "--qmt_only", "tunnex.txt"]), \
             patch(FUNC_PATH+".main") as mock_main:
             cli()
        mock_main.assert_called_once() # it should be called once

    def test_main_called_in_eckart_mode(self):
        """ Tests that main is called in Eckart mode """
        with patch("sys.argv", ["tunnex_2", "--eckart", "ts.inp", "r.inp", "p.inp"]), \
             patch(FUNC_PATH+".main") as mock_main:
             cli()
        mock_main.assert_called_once() # it should be called once

    def test_main_called_in_comp_mode(self):
        """ Tests that main is called in comp mode """
        with patch("sys.argv", ["tunnex_2", "--comp", "ts.inp", "ts.out"]), \
             patch(FUNC_PATH+".main") as mock_main:
             cli()
        mock_main.assert_called_once() # it should be called once

    def test_main_called_in_eckart_and_comp_mode(self):
        """ Tests that main is called in Eckart comp mode """
        with patch("sys.argv", ["tunnex_2", "--eckart", "--comp", "ts.inp", "ts.out", "r.out", "p.out"]), \
             patch(FUNC_PATH+".main") as mock_main:
             cli()
        mock_main.assert_called_once() # it should be called once

    def test_main_called_with_configs_kwarg(self):
        """ Tests that main is called with the configs keyword argument """
        with patch("sys.argv", ["tunnex_2", "mol.gjf"]), \
             patch(FUNC_PATH+".main") as mock_main:
             cli()
        assert "configs" in mock_main.call_args.kwargs, f"Expected {"configs"} "
        f"in {mock_main.call_args.kwargs}"