# --- Test: test_irc_computations_pipeline. Unit tests for irc_computations_pipeline.py
# Run with: pytest test_irc_computations_pipeline.py -v ---


# --- Comment ---
# All external I/O (file_check, run_software, readers, writers,
# error_check functions) is mocked so tests are fast and do not touch the disk.
# Only the orchestration logic of each function is verified.
# --- Comment ---


# --- Modules ---
import pytest # type: ignore
from unittest.mock import MagicMock, patch, call

import math
from pathlib import Path


# --- Module to test ---
from tunnex_2.irc_computations.pipeline.irc_computations_pipeline import ( # type: ignore
    ts_optimization,
    irc_computations,
    _heat_check_func,
    irc_file_parsing,
    react_prod_optimization,
    opt_file_parsing,
    attempt_freq_func,
    number_of_levels_func,
    potential_scal_factor_comp,
    tunnex_input_writing,
    eckart_react_prod_optimization,
    eckart_potential_results)


# --- Helpers and shared data ---
TS_IN = Path("mol.gjf")
TS_OUT = Path("mol_ts.out")
REACT_OUT = Path("mol_react.out")
PROD_OUT = Path("mol_prod.out")
IRC_OUT = Path("mol_irc.out")
CMD = Path("command_settings.txt")
FUNC_PATH = "tunnex_2.irc_computations.pipeline.irc_computations_pipeline"


def _make_irc_data(with_el_energies: bool = True, with_zpve: bool = True):
    """ Returns the dummy IRC curves """
    m = MagicMock()
    if with_el_energies:
        m.electronic_energies = [(-1.0, -0.10), (0.0, -0.05), (1.0, -0.09)]
    else:
        m.electronic_energies = None
    if with_zpve:
        m.zpve_energies_forward = [(0.0, 0.01), (1.0, 0.02)]
        m.zpve_energies_reverse = [(-1.0, 0.01), (0.0, 0.02)]
    else:
        m.zpve_energies_forward = None
        m.zpve_energies_reverse = None
    return m

def _make_energy_points(ts_el=-0.05, ts_zpve=0.05, react_el=-0.15, react_zpve=0.02, prod_el=-0.10, prod_zpve=0.02):
    """ Returns the energies of the stationary points """
    ep = MagicMock()
    ep.transition_state_el_energy = (0.0, ts_el)
    ep.transition_state_zpve = (0.0, ts_zpve)
    ep.reactant_el_energy = (0.0, react_el)
    ep.reactant_zpve = (0.0, react_zpve)
    ep.product_el_energy = (0.0, prod_el)
    ep.product_zpve = (0.0, prod_zpve)
    return ep

def _make_attempt_freq_data(freq_react=1000.0, freq_prod=900.0):
    """ Returns the attempt frequencies and number of levels """
    m = MagicMock()
    m.att_freq_react = freq_react
    m.att_freq_react_corr = [(freq_react, 0.9), (freq_react + 100, 0.1)]
    m.att_freq_prod = freq_prod
    m.att_freq_prod_corr = [(freq_prod, 0.85), (freq_prod + 100, 0.15)]
    return m


# ===========================================================================
# --- ts_optimization - normal behaviour and input validation ---
# ===========================================================================
class TestTsOptimization:

    @patch(FUNC_PATH+".gauss_error_check")
    @patch(FUNC_PATH+".gauss_out_filename", return_value=TS_OUT)
    @patch(FUNC_PATH+".run_software")
    @patch(FUNC_PATH+".create_gauss_input")
    @patch(FUNC_PATH+".file_check", return_value=TS_IN)
    def test_gaussian_returns_output_path(self, fc, cgi, rs, gof, gec):
        """ Verify 'gaussian' mode returns the expected output path """
        result = ts_optimization(TS_IN, CMD, mode="gaussian")
        assert isinstance(result, Path), f"expected {Path}, got {type(result)}; test #1"
        assert result == TS_OUT, f"expected {TS_OUT}, got {result}; test #2"

    @patch(FUNC_PATH+".orca_error_check")
    @patch(FUNC_PATH+".orca_out_filename", return_value=TS_OUT)
    @patch(FUNC_PATH+".run_software")
    @patch(FUNC_PATH+".create_orca_input")
    @patch(FUNC_PATH+".file_check", return_value=TS_IN)
    def test_orca_calls_orca_functions(self, fc, coi, rs, oof, oec):
        """ Verify 'orca' mode calls the expected ORCA functions and returns the output path """
        result = ts_optimization(TS_IN, CMD, mode="orca")
        coi.assert_called_once() # it should be called once
        oec.assert_called_once() # it should be called once
        assert isinstance(result, Path), f"expected {Path}, got {type(result)}; test #1"
        assert result == TS_OUT, f"expected {TS_OUT}, got {result}; test #2"

    @patch(FUNC_PATH+".gauss_error_check")
    @patch(FUNC_PATH+".gauss_out_filename", return_value=TS_OUT)
    @patch(FUNC_PATH+".run_software")
    @patch(FUNC_PATH+".create_gauss_input")
    @patch(FUNC_PATH+".file_check", return_value=TS_IN)
    def test_run_software_skipped_when_key_false(self, fc, cgi, rs, gof, gec):
        """ Verify the software execution is skipped when disabled """
        ts_optimization(TS_IN, CMD, mode="gaussian", run_software_key=False)
        rs.assert_not_called() # it should not be called

    @patch(FUNC_PATH+".file_check", return_value=TS_IN)
    def test_invalid_mode_raises(self, fc):
        """ Verify an invalid mode raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match="as program mode"):
            ts_optimization(TS_IN, CMD, mode="molpro")

    @patch(FUNC_PATH+".file_check", return_value=TS_IN)
    def test_invalid_mode_error_raises(self, fc):
        """ Verify an invalid mode raises the corresponding error. Expecting a ValueError """
        with pytest.raises(ValueError, match="as program mode"):
            ts_optimization(TS_IN, CMD, mode="invalid_mode")


# ===========================================================================
# --- irc_computations - normal behaviour and input validation ---
# ===========================================================================
class TestIrcComputations:

    @patch(FUNC_PATH+".gauss_error_check")
    @patch(FUNC_PATH+".gauss_out_filename", return_value=IRC_OUT)
    @patch(FUNC_PATH+".run_software")
    @patch(FUNC_PATH+".create_gauss_irc_input", return_value=Path("mol_irc.gjf"))
    @patch(FUNC_PATH+".file_check", return_value=TS_IN)
    def test_gaussian_returns_output_path(self, fc, cgi, rs, gof, gec):
        """ Verify 'gaussian' mode returns the expected IRC output path """
        result = irc_computations(TS_IN, TS_OUT, CMD, mode="gaussian")
        assert isinstance(result, Path), f"expected {Path}, got {type(result)}; test #1"
        assert result == IRC_OUT, f"expected {IRC_OUT}, got {result}; test #2"

    @patch(FUNC_PATH+".orca_error_check")
    @patch(FUNC_PATH+".orca_out_filename", return_value=IRC_OUT)
    @patch(FUNC_PATH+".run_software")
    @patch(FUNC_PATH+".create_orca_irc_input", return_value=Path("mol_irc.inp"))
    @patch(FUNC_PATH+".file_check", return_value=TS_IN)
    def test_orca_uses_xyz_suffix(self, fc, coi, rs, oof, oec):
        """ Verify 'orca' mode uses the expected .xyz output suffix and returns the expected IRC output path """
        result = irc_computations(TS_IN, TS_OUT, CMD, mode="orca")
        oof.assert_called_once_with(Path("mol_irc.inp"), ".xyz") # it should use .xyz suffix
        assert result == IRC_OUT, f"expected {IRC_OUT}, got {result}"

    @patch(FUNC_PATH+".gauss_error_check")
    @patch(FUNC_PATH+".gauss_out_filename", return_value=IRC_OUT)
    @patch(FUNC_PATH+".run_software")
    @patch(FUNC_PATH+".create_gauss_irc_input", return_value=Path("mol_irc.gjf"))
    @patch(FUNC_PATH+".file_check", return_value=TS_IN)
    def test_run_software_skipped_when_key_false(self, fc, cgi, rs, gof, gec):
        """ Verify the software execution is skipped when disabled """
        irc_computations(TS_IN, TS_OUT, CMD, mode="gaussian", run_software_key=False)
        rs.assert_not_called() # it should not be called

    @patch(FUNC_PATH+".file_check", return_value=TS_IN)
    def test_invalid_mode_raises(self, fc):
        """ Verify an invalid mode raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match="as program mode"):
            irc_computations(TS_IN, TS_OUT, CMD, mode="bad")


# ===========================================================================
# --- _heat_check_func - normal behaviour and input validation ---
# ===========================================================================
class TestHeatCheckFunc:

    @patch(FUNC_PATH+".detect_outliers_local")
    def test_logs_energy_difference(self, dol):
        """ Verify the electronic energy difference is logged """
        data = _make_irc_data()
        with patch(FUNC_PATH+".logger") as mock_logger:
            _heat_check_func(data, outlier_check=False)
        mock_logger.info.assert_called() # it should be called

    def test_logs_exothermic_reaction(self):
        """ Verify exothermic reaction information is logged """
        data = MagicMock()
        # react_energy > prod_energy implies the forward reaction to be exothermic
        energy = -0.20
        data.electronic_energies = [(-1.0, energy), (0.0, -0.05), (1.0, 2.0 * energy)]
        expected = "exothermic"
        data.zpve_energies_forward = None
        data.zpve_energies_reverse = None
        with patch(FUNC_PATH+".logger") as mock_logger:
            _heat_check_func(data, outlier_check=False)
        mock_logger.info.assert_called() # it should be called
        args, _ = mock_logger.info.call_args
        assert expected in args[0], f"expected {expected} in {args[0]}"

    def test_logs_endothermic_reaction(self):
        """ Verify endothermic reaction information is logged """
        data = MagicMock()
        # react_energy < prod_energy implies the forward reaction to be endothermic
        energy = -0.20
        data.electronic_energies = [(-1.0, energy), (0.0, -0.05), (1.0, 0.5 * energy)]
        expected = "endothermic"
        data.zpve_energies_forward = None
        data.zpve_energies_reverse = None
        with patch(FUNC_PATH+".logger") as mock_logger:
            _heat_check_func(data, outlier_check=False)
        mock_logger.info.assert_called() # it should be called
        args, _ = mock_logger.info.call_args
        assert expected in args[0], f"expected {expected} in {args[0]}"

    def test_logs_thermoneutral_reaction(self):
        """ Verify thermoneutral reaction information is logged """
        data = MagicMock()
        # react_energy = prod_energy implies the forward reaction to be thermoneutral
        energy = -0.20
        data.electronic_energies = [(-1.0, energy), (0.0, -0.05), (1.0, energy)]
        expected = "thermoneutral"
        data.zpve_energies_forward = None
        data.zpve_energies_reverse = None
        with patch(FUNC_PATH+".logger") as mock_logger:
            _heat_check_func(data, outlier_check=False)
        mock_logger.info.assert_called() # it should be called
        args, _ = mock_logger.info.call_args
        assert expected in args[0], f"expected {expected} in {args[0]}"

    @patch(FUNC_PATH+".detect_outliers_local")
    def test_outlier_check_is_not_skipped_by_default(self, dol):
        """ Verify outlier detection is skipped by default """
        data = _make_irc_data()
        _heat_check_func(data)
        dol.assert_called() # it should be called

    @patch(FUNC_PATH+".detect_outliers_local")
    def test_outlier_check_called_when_enabled(self, dol):
        """ Verify outlier detection is called when enabled """
        data = _make_irc_data()
        _heat_check_func(data, outlier_check=True)
        expected = 1
        assert dol.call_count >= expected, f"expected {expected} and more calls, got {dol.call_count} calls"

    @patch(FUNC_PATH+".detect_outliers_local")
    def test_outlier_check_called_twice_with_zpve(self, dol):
        """ Verify outlier detection is called twice when there are ZPVE data """
        data = _make_irc_data()
        _heat_check_func(data, outlier_check=True)
        expected = 2
        assert dol.call_count == expected, f"expected {expected} calls, got {dol.call_count} calls"

    @patch(FUNC_PATH+".detect_outliers_local")
    def test_outlier_check_called_once_no_zpve(self, dol):
        """ Verify outlier detection is called once when there is no ZPVE data """
        data = _make_irc_data(with_zpve=False)
        _heat_check_func(data, outlier_check=True)
        expected = 1
        assert dol.call_count == expected, f"expected {expected} calls, got {dol.call_count} calls"

    def test_invalid_electronic_energy_raises(self):
        """ Verify an invalid Electronic energy array raises an error. Expecting a ValueError """        
        irc_data = MagicMock()
        irc_data.electronic_energies = None
        irc_data.zpve_energies_forward = None
        irc_data.zpve_energies_reverse = None
        with pytest.raises(ValueError, match="Expected an array of"):
            _heat_check_func(irc_data)


# ===========================================================================
# --- irc_file_parsing - normal behaviour and input validation ---
# ===========================================================================
class TestIrcFileParsing:

    @patch(FUNC_PATH+".reading_gauss_irc", return_value=_make_irc_data())
    def test_gaussian_returns_irc_data(self, rgi):
        """ Verify 'gaussian' mode returns the parsed IRC data """
        result = irc_file_parsing(IRC_OUT, TS_OUT, TS_IN, CMD, mode="gaussian", outlier_check=False)
        rgi.assert_called_once() # it should be called once
        assert result is rgi.return_value, f"expected {result} is {rgi.return_value}"

    @patch(FUNC_PATH + "._heat_check_func")
    @patch(FUNC_PATH+".reading_gauss_irc_tunnex", return_value=_make_irc_data())
    def test_gaussian_hess_parsing_uses_tunnex_reader(self, rgit, hcf):
        """ Verify 'gaussian' IRC parsing uses the tunnex IRC reader and calls _heat_check_func """
        result = irc_file_parsing(IRC_OUT, TS_OUT, TS_IN, CMD, mode="gaussian", hess_parsing=True)
        rgit.assert_called_once() # it should be called once
        hcf.assert_called_once() # it should be called once
        assert result is rgit.return_value, f"expected {result} is {rgit.return_value}"

    @patch(FUNC_PATH+".reading_orca_irc_tunnex", return_value=_make_irc_data())
    def test_orca_uses_orca_reader(self, roit):
        """ Verify 'orca' mode returns the parsed IRC data """
        result = irc_file_parsing(IRC_OUT, TS_OUT, TS_IN, CMD, mode="orca", outlier_check=False)
        roit.assert_called_once() # it should be called once
        assert result is roit.return_value, f"expected {result} is {roit.return_value}"

    def test_invalid_mode_raises(self):
        """ Verify an invalid mode raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match="as program mode"):
            irc_file_parsing(IRC_OUT, TS_OUT, TS_IN, CMD, mode="priroda")

    @patch(FUNC_PATH+".reading_gauss_irc", return_value=_make_irc_data(with_el_energies=False))
    def test_invalid_electronic_energy_raises(self, rgi):
        """ Verify an invalid Electronic energy array raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match="Expected an array"):
            irc_file_parsing(IRC_OUT, TS_OUT, TS_IN, CMD, mode="gaussian")


# ===========================================================================
# --- react_prod_optimization - normal behaviour and input validation ---
# ===========================================================================
class TestReactProdOptimization:

    @patch(FUNC_PATH+".gauss_error_check")
    @patch(FUNC_PATH+".gauss_out_filename", side_effect=[Path("r.out"), Path("p.out")])
    @patch(FUNC_PATH+".run_software")
    @patch(FUNC_PATH+".create_gauss_react_prod_input_from_irc", return_value=(Path("r.gjf"), Path("p.gjf")))
    @patch(FUNC_PATH+".file_check", return_value=TS_IN)
    def test_gaussian_returns_two_paths(self, fc, cgr, rs, gof, gec):
        """ Verify 'gaussian' mode returns paths to the reactant and product output files """
        r, p = react_prod_optimization(TS_IN, IRC_OUT, CMD, mode="gaussian")
        result = (r, p)
        expected = [Path("r.out"), Path("p.out")]
        gec.assert_has_calls([call(Path("r.out")), call(Path("p.out"))])
        # it should be called with p.out and p.out files
        for i, val in enumerate(result):
            assert isinstance(val, Path), f"expected {Path}, got {type(result)}; test #1"
            assert val == expected[i], f"expected {expected[i]}, got {val}; test #2"

    @patch(FUNC_PATH+".orca_error_check")
    @patch(FUNC_PATH+".orca_out_filename", side_effect=[Path("r.out"), Path("p.out")])
    @patch(FUNC_PATH+".run_software")
    @patch(FUNC_PATH+".create_orca_react_prod_input_from_irc", return_value=(Path("r.gjf"), Path("p.gjf")))
    @patch(FUNC_PATH+".file_check", return_value=TS_IN)
    def test_run_software_called_twice_for_orca(self, fc, corp, rs, oof, oec):
        """ Verify 'orca' mode runs the software twice for reactant and product optimization """
        react_prod_optimization(TS_IN, IRC_OUT, CMD, mode="orca")
        expected = 2
        oec.assert_has_calls([call(Path("r.out")), call(Path("p.out"))])
        # it should be called with p.out and p.out files
        assert rs.call_count == 2, f"expected {expected}, got {rs.call_count}"

    @patch(FUNC_PATH+".gauss_error_check")
    @patch(FUNC_PATH+".gauss_out_filename", side_effect=[Path("r.out"), Path("p.out")])
    @patch(FUNC_PATH+".run_software")
    @patch(FUNC_PATH+".create_gauss_react_prod_input_from_irc", return_value=(Path("r.gjf"), Path("p.gjf")))
    @patch(FUNC_PATH+".file_check", return_value=TS_IN)
    def test_run_software_skipped_when_key_false(self, fc, cgr, rs, gof, gec):
        """ Verify software execution is skipped when the run flag is disabled """
        react_prod_optimization(TS_IN, IRC_OUT, CMD, mode="gaussian", run_software_key=False)
        rs.assert_not_called() # it should not be called

    @patch(FUNC_PATH+".file_check", return_value=TS_IN)
    def test_invalid_mode_raises(self, fc):
        """ Verify an invalid program mode raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match="as program mode"):
            react_prod_optimization(TS_IN, IRC_OUT, CMD, mode="nwchem")


# ===========================================================================
# --- opt_file_parsing - normal behaviour and input validation ---
# ===========================================================================
class TestOptFileParsing:

    @patch(FUNC_PATH+".reading_gauss_struct_file",
           return_value=((0.0, -150.0), (0.0, 0.05)))
    @patch(FUNC_PATH+".file_check", side_effect=[TS_OUT, Path("r.out"), Path("p.out")])
    def test_gaussian_returns_energy_points_data(self, fc, rgsf):
        """ Verify 'gaussian' mode runs the file parcing three rimes and return not None """
        result = opt_file_parsing(TS_OUT, Path("r.out"), Path("p.out"), _make_attempt_freq_data(), mode="gaussian")
        expected = 3
        assert rgsf.call_count == expected, f"expected {expected}, got {rgsf.call_count}; test #1"
        assert result is not None, f"expected not {None}, got {result}; test #2"

    @patch(FUNC_PATH+".reading_orca_struct_file", return_value=((0.0, -150.0), (0.0, 0.05)))
    @patch(FUNC_PATH+".file_check", side_effect=[TS_OUT, Path("r.out"), Path("p.out")])
    def test_orca_calls_orca_reader(self, fc, rosf):
        """ Verify 'orca' mode runs the file parcing three rimes and return not None """
        result = opt_file_parsing(TS_OUT, Path("r.out"), Path("p.out"), _make_attempt_freq_data(), mode="orca")
        expected = 3
        assert rosf.call_count == expected, f"expected {expected}, got {rosf.call_count}; test #1"
        assert result is not None, f"expected not {None}, got {result}; test #2"

    @patch(FUNC_PATH+".file_check", side_effect=[TS_OUT, Path("r.out"), Path("p.out")])
    def test_invalid_mode_raises(self, fc):
        """ Verify an invalid program mode raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match="as program mode"):
            opt_file_parsing(TS_OUT, Path("r.out"), Path("p.out"), _make_attempt_freq_data(), mode="psi4")


# ===========================================================================
# --- attempt_freq_func - normal behaviour and input validation ---
# ===========================================================================
class TestAttemptFreqFunc:

    def test_returns_attempt_freq_data(self):
        """ Verify the function returns the expected attempt frequency data """
        expected_freq = 1000
        expected_freq_max = 2000
        expected_corr = [(expected_freq, 0.9), (expected_freq + 100, 0.1)]
        with patch(FUNC_PATH+".attempt_freq_calc", return_value=(expected_freq,
             expected_corr, expected_freq_max)) as afc, patch(FUNC_PATH+".write_freq_corr") as wfc, \
             patch(FUNC_PATH+".file_check", side_effect=[TS_OUT, Path("r.out"), Path("p.out")]):
             result = attempt_freq_func(TS_OUT, Path("r.out"), Path("p.out"), corr_analysis=False, mode="gaussian")
        assert result.att_freq_react == expected_freq, f"expected {expected_freq}, "
        f"got {result.att_freq_react}; test #1"
        assert result.att_freq_prod == expected_freq, f"expected {expected_freq}, "
        f"got {result.att_freq_prod}; test #2"
        assert result.att_freq_react_corr == expected_corr, f"expected {expected_corr}, "
        f"got {result.att_freq_react_corr}; test #3"
        assert result.att_freq_prod_corr == expected_corr, f"expected {expected_corr}, "
        f"got {result.att_freq_prod_corr}; test #4"

    def test_write_freq_corr_called_when_corr_analysis_true(self):
        """ Verify frequency and correlations function is called when correlation analysis is enabled """
        with patch(FUNC_PATH+".attempt_freq_calc", return_value=(1000, [(1000, 0.9), (1100, 0.1)], 2000)), \
             patch(FUNC_PATH+".write_freq_corr") as wfc, \
             patch(FUNC_PATH+".file_check", side_effect=[TS_OUT, Path("r.out"), Path("p.out")]):
             attempt_freq_func(TS_OUT, Path("r.out"), Path("p.out"), corr_analysis=True, mode="gaussian")
        expected = 2
        assert wfc.call_count == expected, f"expected {expected}, got {wfc.call_count}"

    def test_write_freq_corr_skipped_when_false(self):
        """ Verify frequency and correlations function is not called when correlation analysis is disabled """
        with patch(FUNC_PATH+".attempt_freq_calc", return_value=(1000.0, [(1000.0, 0.9), (1100, 0.1)], 2000)) as afc, \
             patch(FUNC_PATH+".write_freq_corr") as wfc, \
             patch(FUNC_PATH+".file_check", side_effect=[TS_OUT, Path("r.out"), Path("p.out")]):
             attempt_freq_func(TS_OUT, Path("r.out"), Path("p.out"), corr_analysis=False, mode="gaussian")
        wfc.assert_not_called() # it should not be called
        expected = 2
        assert afc.call_count == expected, f"expected {expected}, got {expected}"

    def test_returns_max_correlated_frequency_when_requested(self):
        """ Verify the maximum-correlation frequency is selected when requested """
        expected_aver = 1895
        expected_max = (2000, 0.8)
        expected_corr = [(expected_max, 0.8), (expected_max[0] - 600, 0.2)]
        with patch(FUNC_PATH + ".attempt_freq_calc",
             return_value=(expected_aver, expected_corr, expected_max)):
             result = attempt_freq_func(TS_OUT, Path("r.out"), Path("p.out"), corr_analysis=False,
             mode="gaussian", pick_freq_max_corr=True)
        assert result.att_freq_react == expected_max[0], \
        f"expected {result.att_freq_react} == {expected_max[0]}; test #1"
        assert result.att_freq_prod == expected_max[0], \
        f"expected {result.att_freq_prod} == {expected_max[0]}; test #2"


# ===========================================================================
# --- number_of_levels_func - normal behaviour and input validation ---
# ===========================================================================
class TestNumberOfLevelsFunc:

    @patch(FUNC_PATH+".compute_number_of_vib_levels", side_effect=[5, 4])
    def test_returns_two_integers(self, cnvl):
        """ Verify the function returns the expected number of vibrational levels """
        ep = _make_energy_points()
        irc_data = _make_irc_data()
        r, p = number_of_levels_func(irc_data, ep, 1000.0, 900.0, 1.0, 1.0)
        expected = [5, 4]
        assert r == expected[0], f"expected {expected[0]}, got {r}; test #1"
        assert p == expected[1], f"expected {expected[1]}, got {p}; test #2"

    @patch(FUNC_PATH+".compute_number_of_vib_levels", side_effect=[5, 4])
    def test_called_with_correct_directions(self, cnvl):
        """ Verify vibrational level calculations are called with the correct directions """
        ep = _make_energy_points()
        irc_data = _make_irc_data()
        number_of_levels_func(irc_data, ep, 1000.0, 900.0, 1.0, 1.0)
        calls = [str(c) for c in cnvl.call_args_list]
        expected = ["reactant", "product"]
        for val in expected:
            assert any(val in c for c in calls), f"expected {val}, got {calls}"


# ===========================================================================
# --- potential_scal_factor_comp - normal behaviour and input validation ---
# ===========================================================================
class TestPotentialScalFactorComp:

    def test_returns_two_positive_floats(self):
        """ Verify the function returns two positive potential scaling factors """
        irc_data = _make_irc_data()
        ep = _make_energy_points()
        r, p = potential_scal_factor_comp(irc_data, ep)
        assert isinstance(r, float) and r > 0, f"expected val>0, got {r}, type={type(r)}; test #1"
        assert isinstance(p, float) and p > 0, f"expected val>0, got {p}, type={type(p)}; test #2"

    def test_flat_barrier_gives_unity_scaling(self):
        """ Verify a flat IRC barrier defaults both scaling factors to 1.0 """
        irc_data = MagicMock()
        # All energies identical -> zero barrier
        irc_data.electronic_energies = [(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
        irc_data.zpve_energies_forward = [(0.0, 0.0), (1.0, 0.0)]
        irc_data.zpve_energies_reverse = [(-1.0, 0.0), (0.0, 0.0)]
        ep = _make_energy_points(ts_el=0.0, ts_zpve=0.0, react_el=0.0, react_zpve=0.0,
            prod_el=0.0,  prod_zpve=0.0)
        with patch(FUNC_PATH+".logger") as mock_logger:
            r, p = potential_scal_factor_comp(irc_data, ep)
        assert math.isclose(r, 1.0), f"expected {1.0}, got {r}; test #1"
        assert math.isclose(p, 1.0), f"expected {1.0}, got {p}; test #2"

    def test_flat_barrier_warns(self):
        """ Verify a flat IRC barrier logs a warning """
        irc_data = MagicMock()
        # All energies identical -> zero barrier
        irc_data.electronic_energies = [(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
        irc_data.zpve_energies_forward = [(0.0, 0.0), (1.0, 0.0)]
        irc_data.zpve_energies_reverse = [(-1.0, 0.0), (0.0, 0.0)]
        ep = _make_energy_points(ts_el=0.0, ts_zpve=0.0, react_el=0.0, react_zpve=0.0,
            prod_el=0.0,  prod_zpve=0.0)
        with patch(FUNC_PATH+".logger") as mock_logger:
            potential_scal_factor_comp(irc_data, ep)
        assert mock_logger.warning.call_count >= 1, f"expected val>={1}, "
        f"got {mock_logger.warning.call_count}"

    def test_no_zpve_uses_zero(self):
        """ Verify missing ZPVE data defaults to zero without raising an error """
        irc_data = _make_irc_data(with_zpve=False)
        ep = _make_energy_points()
        r, p = potential_scal_factor_comp(irc_data, ep)
        assert r >= 0, f"expected val>=0, got {r}; test #1"
        assert p >= 0, f"expected val>=0, got {p}; test #2"


# ===========================================================================
# --- tunnex_input_writing - normal behaviour and input validation ---
# ===========================================================================
class TestTunnexInputWriting:

    @patch(FUNC_PATH+".writing_zpve_energy")
    @patch(FUNC_PATH+".writing_el_energy")
    @patch(FUNC_PATH+".write_input_head", side_effect=[Path("fwd.txt"), Path("bwd.txt")])
    @patch(FUNC_PATH+".file_check", return_value=TS_IN)
    def test_returns_two_paths(self, fc, wih, wel, wze):
        """ Verify the function returns input file paths for both forward and backward directions """
        irc_data = _make_irc_data()
        ep = _make_energy_points()
        af_data = _make_attempt_freq_data()
        num_levels = [5, 6]
        fwd, bwd = tunnex_input_writing(TS_IN, irc_data, ep, af_data, num_levels[0], num_levels[1])
        expected = [Path("fwd.txt"), Path("bwd.txt")]
        result = [fwd, bwd]
        for i, val in enumerate(result):
            assert val == expected[i], f"expected {expected[i]}, got {val}"

    @patch(FUNC_PATH+".writing_zpve_energy")
    @patch(FUNC_PATH+".writing_el_energy")
    @patch(FUNC_PATH+".write_input_head", side_effect=[Path("fwd.txt"), Path("bwd.txt")])
    @patch(FUNC_PATH+".file_check", return_value=TS_IN)
    def test_all_written_for_both_directions(self, fc, wih, wel, wze):
        """
        Verify headers, electronic energy data, and ZPVEs
        are written for both forward and backward directions
        """
        irc_data = _make_irc_data()
        ep = _make_energy_points()
        af_data = _make_attempt_freq_data()
        num_levels = [5, 6]
        tunnex_input_writing(TS_IN, irc_data, ep, af_data, num_levels[0], num_levels[1])
        expected = 2
        assert wih.call_count == expected, f"expected {expected}, got {wih.call_count}; test #1"
        assert wel.call_count == expected, f"expected {expected}, got {wel.call_count}; test #2"
        assert wze.call_count == expected, f"expected {expected}, got {wze.call_count}; test #3"
        assert fc.call_count == 1, f"expected {1}, got {fc.call_count}; test #4"


# ==============================================================================
# --- eckart_react_prod_optimization - normal behaviour and input validation ---
# ==============================================================================
class TestEckartReactProdOptimization:

    @patch(FUNC_PATH+".gauss_error_check")
    @patch(FUNC_PATH+".gauss_out_filename", side_effect=[Path("r.out"), Path("p.out")])
    @patch(FUNC_PATH+".run_software")
    @patch(FUNC_PATH+".create_gauss_input")
    def test_gaussian_returns_two_output_paths(self, cgi, rs, gof, gec):
        """ Verify Gaussian optimization returns output paths for both reactant and product """
        r, p = eckart_react_prod_optimization( Path("r.gjf"), Path("p.gjf"),
            CMD, mode="gaussian", run_software_key=False)
        expected = [Path("r.out"), Path("p.out")]
        result = [r, p]
        for i, val in enumerate(result):
            assert val == expected[i], f"expected {expected[i]}, got {val}; test #1"
        expected_list = [2, 0, 2, 2]
        result = [cgi.call_count, rs.call_count, gof.call_count, gec.call_count]
        for i, expected in enumerate(expected_list):
            assert result[i] == expected, f"expected {expected}, got {result[i]}; test #2"

    @patch(FUNC_PATH+".orca_error_check")
    @patch(FUNC_PATH+".orca_out_filename", side_effect=[Path("r.out"), Path("p.out")])
    @patch(FUNC_PATH+".run_software")
    @patch(FUNC_PATH+".create_orca_input")
    def test_orca_returns_two_output_paths(self, coi, rs, oof, oec):
        """ Verify ORCA optimization returns output paths for both reactant and product """
        r, p = eckart_react_prod_optimization( Path("r.gjf"), Path("p.gjf"),
            CMD, mode="orca", run_software_key=True)
        expected = [Path("r.out"), Path("p.out")]
        result = [r, p]
        for i, val in enumerate(result):
            assert val == expected[i], f"expected {expected[i]}, got {val}; test #1"
        expected_list = [2, 2, 2, 2]
        result = [coi.call_count, rs.call_count, oof.call_count, oec.call_count]
        for i, expected in enumerate(expected_list):
            assert result[i] == expected, f"expected {expected}, got {result[i]}; test #2"

    @patch(FUNC_PATH+".gauss_error_check")
    @patch(FUNC_PATH+".gauss_out_filename", side_effect=[Path("r.out"), Path("p.out")])
    @patch(FUNC_PATH+".run_software")
    @patch(FUNC_PATH+".create_gauss_input")
    def test_input_stem_has_react_and_prod(self, cgi, rs, gof, gec):
        """ Verify Gaussian input filenames contain the reactant and product identifiers """
        eckart_react_prod_optimization(Path("r.gjf"), Path("p.gjf"),
            CMD, mode="gaussian", run_software_key = True)
        stems = [str(c[0][3]) for c in cgi.call_args_list]
        expected = ["_react", "_prod"]
        for val in expected:
            assert any(val in s for s in stems), f"expected {val} in {stems}"

    def test_invalid_mode_raises(self):
        """ Verify an invalid program mode raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match="as program mode"):
            eckart_react_prod_optimization(Path("r.gjf"), Path("p.gjf"), CMD, mode="nwchem")


# ===========================================================================
# --- eckart_potential_results - normal behaviour and input validation ---
# ===========================================================================
class TestEckartPotentialResults:

    def _mock_reader(self, freq=-1500.0, n_modes=6):
        """ Returns the synthetic modes and masses """
        modes = [[freq + i * 100] + [0.1] * 6 for i in range(n_modes)]
        return (modes, (12.0, 1.0))

    @patch(FUNC_PATH+".eckart_potential_data")
    @patch(FUNC_PATH+".eckart_potential_parameters", return_value=(0.01, 0.08, 1.5))
    @patch(FUNC_PATH+".gauss_mode_coordinate_reader")
    def test_gaussian_returns_irc_data(self, gmcr, epp, epd):
        """ Verify 'gaussian' mode returns the IRC data produced by the Eckart potential calculations """
        gmcr.return_value = self._mock_reader()
        irc_mock = _make_irc_data()
        epd.return_value = irc_mock
        ep = _make_energy_points()
        result = eckart_potential_results(TS_OUT, ep, mode="gaussian", outlier_check=False)
        gmcr.assert_called_once() # it should be called once
        assert result is irc_mock, f"expected {result} in {irc_mock}"

    @patch(FUNC_PATH + "._heat_check_func")
    @patch(FUNC_PATH+".eckart_potential_data")
    @patch(FUNC_PATH+".eckart_potential_parameters", return_value=(0.01, 0.08, 1.5))
    @patch(FUNC_PATH+".orca_mode_coordinate_reader")
    def test_orca_returns_irc_data(self, omcr, epp, epd, hcf):
        """
        Verify 'orca' mode returns the IRC data produced by the Eckart potential calculations
        and calls _heat_check_func
        """
        omcr.return_value = self._mock_reader()
        irc_mock = _make_irc_data()
        epd.return_value = irc_mock
        ep = _make_energy_points()
        result = eckart_potential_results(TS_OUT, ep, mode="orca")
        omcr.assert_called_once() # it should be called once
        hcf.assert_called_once() # it should be called once
        assert result is irc_mock, f"expected {result} in {irc_mock}"

    @patch(FUNC_PATH+".eckart_potential_data")
    @patch(FUNC_PATH+".gauss_mode_coordinate_reader",
        return_value=([(-1500.0, 0.0), (100.0, 0.0), (500.0, 0.0)], None))
    def test_invalid_mode_raises(self, gmcr, epd):
        """ Verify an invalid program mode raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match="as program mode"):
            eckart_potential_results(TS_OUT, _make_energy_points(), mode="pyscf")
         
    @patch(FUNC_PATH+".eckart_potential_data")
    @patch(FUNC_PATH+".gauss_mode_coordinate_reader",
           return_value=([(-1500.0, 0.0), (100.0, 0.0), (500.0, 0.0)], None))
    def test_invalid_electronic_energy_raises(self, gmcr, epd):
        """ Verify an invalid Electronic energy array raises an error. Expecting a ValueError """       
        irc_data = MagicMock()
        irc_data.electronic_energies = None
        irc_data.zpve_energies_forward = None
        irc_data.zpve_energies_reverse = None
        epd.return_value = irc_data
        with pytest.raises(ValueError, match="Expected an array of"):
            eckart_potential_results(TS_OUT, _make_energy_points(), mode="gaussian")