# --- Test: test_attempt_freq_calc. Unit tests for attempt_freq_calc.py
# Run with: pytest test_attempt_freq_calc.py -v ---

# --- Modules ---
import pytest # type: ignore
from unittest.mock import MagicMock, patch

import math
import numpy as np
from pathlib import Path

from tunnex_2.constants_and_dataclasses.constants_and_settings import CM_M1_TO_HARTREE # type: ignore


# --- Module to test ---
from tunnex_2.irc_computations.ancillary_functions.attempt_freq_calc import ( # type: ignore
    _vib_modes_mass_scaling,
    attempt_freq_calc,
    write_freq_corr,
    _extract_energies_to_compute_number_of_vib_levels,
    compute_number_of_vib_levels,
    detect_outliers_local)


# --- Helpers and shared data ---
VIB_MASS_SCAL = "tunnex_2.irc_computations.ancillary_functions.attempt_freq_calc._vib_modes_mass_scaling"
# VIB_MASS_SCAL is needed for one test where no modes should be returned
LOGGER_IMP = "tunnex_2.irc_computations.ancillary_functions.attempt_freq_calc.logger"
FILE_CHECK = "tunnex_2.irc_computations.ancillary_functions.attempt_freq_calc.file_check"
GAUSS_COORD_READER = "tunnex_2.irc_computations.ancillary_functions.attempt_freq_calc.gauss_mode_coordinate_reader"
ORCA_COORD_READER = "tunnex_2.irc_computations.ancillary_functions.attempt_freq_calc.orca_mode_coordinate_reader"


def _make_modes(n_atoms: int = 2, n_modes: int = 3, freq_start: float = 100.0, freq_step: float = 100.0) -> list[list[float]]:
    """ Return synthetic normal modes: [freq, dx1, dy1, dz1, dx2, dy2, dz2, ...] """
    rng = np.random.default_rng(0) # fixing the seed, so the data are reproducible
    modes = []
    for i in range(n_modes):
        freq = freq_start + i * freq_step
        coords = rng.standard_normal(n_atoms * 3)
        norm = np.linalg.norm(coords)
        if np.isclose(norm, 0.0):
            raise ValueError(f"Cannot normalize a zero displacement vector. Got vector={coords}.")
        coords = coords / norm
        coords = coords.tolist()
        modes.append([freq] + coords)
    return modes

def _make_masses(n_atoms: int = 2) -> tuple[float, ...]:
    """ Return synthetic masses (12.0, 12.0, 12.0, ...) """
    return tuple([12.0] * n_atoms)

def _make_energy_points(ts_el=0.0, ts_zpve=0.05, react_el=-0.1, react_zpve=0.02, prod_el=-0.05, prod_zpve=0.02):
    """ Return the synthetic data for the energy points (irc, energy); 6 times """
    ep = MagicMock()
    ep.transition_state_el_energy = (0.0, ts_el)
    ep.transition_state_zpve = (0.0, ts_zpve)
    ep.reactant_el_energy = (-1.0, react_el)
    ep.reactant_zpve = (-1.0, react_zpve)
    ep.product_el_energy = (1.0, prod_el)
    ep.product_zpve = (1.0, prod_zpve)
    return ep

def _make_irc_data(scale_factor: int = 1, with_el_energies: bool = True, with_zpve: bool = True):
    """ Returns the dummy IRC curves """
    m = MagicMock()
    if with_el_energies:
        m.electronic_energies = [(x, energy * scale_factor) for x, energy in [
        (-1.0, -0.10),
        (-0.5, -0.08),
        (0.0, -0.05),
        (0.5, -0.07),
        (1.0, -0.12)]]
    else:
        m.electronic_energies = None
    if with_zpve:
        m.zpve_energies_forward = [(0.5, 0.01),
                                   (1.0, 0.02)]
        m.zpve_energies_reverse = [(-1.0, 0.03),
                                   (-0.5, 0.04)]
    else:
        m.zpve_energies_forward = None
        m.zpve_energies_reverse = None
    return m

def _make_irc_data_negative_barrier():
    """ Returns the dummy IRC curves, a negative barrier """
    m = MagicMock()
    m.electronic_energies = [(-1.0, -0.05),
                             (-0.5, -0.08),
                             (0.0, -0.10),
                             (0.5, -0.12),
                             (1.0, -0.15)]
    m.zpve_energies_forward = None
    m.zpve_energies_reverse = None
    return m


# ===========================================================================
# --- _vib_modes_mass_scaling - normal behaviour and input validation ---
# ===========================================================================
class TestVibModesMassScaling:

    def test_output_length_matches_input(self):
        """ Output length matches the input modes """
        modes = _make_modes(n_atoms=2, n_modes=3)
        masses = _make_masses(n_atoms=2)
        result = _vib_modes_mass_scaling(modes, masses)
        assert len(result) == len(modes), f"expected {len(modes)} modes, got {len(result)} modes"

    def test_frequency_column_unchanged(self):
        """ Frequencies remain unchanged after mass scaling """
        modes = _make_modes(n_atoms=2, n_modes=3)
        masses = _make_masses(n_atoms=2)
        result = _vib_modes_mass_scaling(modes, masses)
        for orig, scaled in zip(modes, result):
            assert orig[0] == scaled[0], f"frequency must not be modified: expected {orig[0]}, got {scaled[0]}"

    def test_coordinate_scaled_by_sqrt_mass(self):
        """ Each coordinate is scaled by the square root of its atom's mass """
        modes = [[500.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]  # freq, atom 0 (mass 12.0), atom 1 (mass 1.0)
        masses = (12.0, 1.0)
        result = _vib_modes_mass_scaling(modes, masses)
        scaled_coords = result[0][1:]
        expected = [1.0 * math.sqrt(12.0), 0.0 * math.sqrt(12.0), 0.0 * math.sqrt(12.0),
                    0.0 * math.sqrt(1.0), 1.0 * math.sqrt(1.0), 0.0 * math.sqrt(1.0)]
        result = np.allclose(scaled_coords, expected, atol=1e-12)
        assert result, f"expected scaled_modes={expected}, got {result}"

    def test_equal_masses_scales_uniformly(self):
        """ Equal masses result in uniform coordinate scaling """
        modes = [[200.0, 0.6, -0.8, 0.0, 0.8, 0.0, -0.6]]
        masses = (4.0, 4.0)
        result = _vib_modes_mass_scaling(modes, masses)
        scaled = result[0][1:]
        expected = [c * math.sqrt(4.0) for c in modes[0][1:]]
        result = np.allclose(scaled, expected, atol=1e-12)
        assert result, f"expected scaled_modes={expected}, got {result}"
    
    def test_different_masses_scales_differently(self):
        """ Different masses result in different coordinate scaling """
        modes = [[200.0, 0.6, -0.8, 0.0, 0.8, 0.0, -0.6]]
        masses = (4.0, 16.0)
        result = _vib_modes_mass_scaling(modes, masses)
        scaled = result[0][1:]
        expected = [c * math.sqrt(4.0) for c in modes[0][1:4]] + [c * math.sqrt(16.0) for c in modes[0][4:]]
        result = np.allclose(scaled, expected, atol=1e-12)
        assert result, f"expected scaled_modes={expected}, got {result}"

    def test_empty_modes_raises(self):
        """ Empty input list of modes raise an error. Expecting a ValueError """
        with pytest.raises(ValueError, match="least one mode"):
            _vib_modes_mass_scaling([], (12.0,))

    def test_no_freq_and_coordinates_raises(self):
        """ Empty list of a mode raise an error. Expecting a ValueError """
        with pytest.raises(ValueError, match="coordinates for"):
            _vib_modes_mass_scaling([[]], (12.0,))

    def test_freq_and_no_coordinates_raises(self):
        """ Empty coordinate list raise an error. Expecting a ValueError """
        with pytest.raises(ValueError, match="coordinates for"):
            _vib_modes_mass_scaling([[200]], (12.0,))

    def test_coordinate_count_mismatch_raises(self):
        """ The number of modes bigger than 3 * number of masses raises an error. Expecting a ValueError """
        modes = [[200.0, 0.6, -0.8, 0.0, 0.8, 0.0, -0.6]]
        masses = (4.0,)
        with pytest.raises(ValueError, match="coordinates for"):
            _vib_modes_mass_scaling(modes, masses)

    def test_coordinate_count_inconsistency_raises(self):
        """ The number of modes smaller than 3 * number of masses raises an error. Expecting a ValueError """
        modes = [[200.0, 0.6, -0.8, 0.0, 0.8, 0.0, -0.6]]
        masses = (1.0, 1.0, 1.0)
        with pytest.raises(ValueError, match="coordinates for"):
            _vib_modes_mass_scaling(modes, masses)


# ===========================================================================
# --- attempt_freq_calc - normal behaviour ---
# ===========================================================================
class TestAttemptFreqCalcNormal:

    def _run(self, ts_modes, min_modes, masses, prog_key="gaussian", species="reactant", hess_parsing_key=False):
        """ Run of the function with the mocked coordinate reader """
        reader = MagicMock(side_effect=[(ts_modes, masses), (min_modes, masses)])
        patch_target = (GAUSS_COORD_READER if prog_key == "gaussian" else ORCA_COORD_READER)
        with patch(patch_target, reader), patch(FILE_CHECK, return_value=Path("some.out")) as file_check:
            result = attempt_freq_calc(Path("ts.out"), Path("min.out"), species=species,
                prog_key=prog_key, hess_parsing=hess_parsing_key)
        return result, file_check

    def test_returns_float_list_and_tuple(self):
        """ Should return a frequency as a float, correlations as a list, and another frequency as a float """
        ts_modes = _make_modes(n_atoms=2, n_modes=1)
        min_modes = _make_modes(n_atoms=2, n_modes=3)
        masses = _make_masses(2)
        (freq_aver, corrs, freq_max), _ = self._run(ts_modes, min_modes, masses)
        assert isinstance(freq_aver, float), f"expected {float}, got {type(freq_aver)}; test #1"
        assert isinstance(corrs, list), f"expected {list}, got {type(corrs)}; test #2"
        assert isinstance(freq_max, tuple), f"expected {tuple}, got {type(freq_max)}; test #3"

    def test_filecheck_call_count(self):
        """ Should check the file via the internal functions as many times as expected """
        ts_modes = _make_modes(n_atoms=2, n_modes=1)
        min_modes = _make_modes(n_atoms=2, n_modes=3)
        masses = _make_masses(2)
        expected = 2
        # "gaussian", hess_parsing_key=False
        _, file_check = self._run(ts_modes, min_modes, masses, prog_key="gaussian", species="product", hess_parsing_key=False)
        assert file_check.call_count == expected, f"expected {expected}, got {file_check.call_count}; test #1"
        # "orca", hess_parsing_key=False
        _, file_check = self._run(ts_modes, min_modes, masses, prog_key="orca", species="reactant", hess_parsing_key=False)
        assert file_check.call_count == expected, f"expected {expected}, got {file_check.call_count}; test #2"
        # "gaussian", hess_parsing_key=True
        _, file_check = self._run(ts_modes, min_modes, masses, prog_key="gaussian", species="reactant", hess_parsing_key=True)
        assert file_check.call_count == expected, f"expected {expected}, got {file_check.call_count}; test #3"
        # "orca", hess_parsing_key=True
        _, file_check = self._run(ts_modes, min_modes, masses, prog_key="orca", species="product", hess_parsing_key=True)
        assert file_check.call_count == expected, f"expected {expected}, got {file_check.call_count}; test #4"

    def test_correlations_length_matches_modes(self):
        """ The number of correlations matches the number of modes """
        n_min = 5
        ts_modes = _make_modes(n_atoms=2, n_modes=1)
        min_modes = _make_modes(n_atoms=2, n_modes=n_min)
        masses = _make_masses(2)
        (_, corrs, _), _ = self._run(ts_modes, min_modes, masses)
        assert len(corrs) == n_min, f"expected {n_min}, got {len(corrs)}"

    def test_correlation_values_between_0_and_1(self):
        """ All correlation values are within the range [0, 1] """
        ts_modes = _make_modes(n_atoms=2, n_modes=1)
        min_modes = _make_modes(n_atoms=2, n_modes=4)
        masses = _make_masses(2)
        (_, corrs, _), _ = self._run(ts_modes, min_modes, masses)
        for _, corr in corrs:
            assert 0.0 <= corr <= 1.0 + 1e-12, f"correlation {corr} out of [0, 1]"

    def test_identical_ts_and_min_mode_gives_correlation_one(self, tol=1e-12):
        """ Identical TS and minimum modes give a correlation of 1.0, while orthogonal modes give 0.0 """
        mode_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        orthog_mode_vec = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        ts_modes = [[100.0] + mode_vec]
        min_modes = [[200.0] + mode_vec, [300.0] + orthog_mode_vec]
        masses = (1.0, 1.0)
        (_, corrs, _), _ = self._run(ts_modes, min_modes, masses)
        # Expecting 1.0 for the identical vectors
        assert math.isclose(corrs[0][1], 1.0, abs_tol=tol), f"expected {1.0}, got {corrs[0][1]}"
        # Expecting 0.0 for the orthogonal vectors
        assert math.isclose(corrs[1][1], 0.0, abs_tol=tol), f"expected {0.0}, got {corrs[0][1]}"

    def test_attempt_freq_is_positive(self):
        """ The calculated attempt frequency is positive, freq_aver != freq_max """
        ts_modes = _make_modes(n_atoms=3, n_modes=1)
        min_modes = _make_modes(n_atoms=3, n_modes=4)
        masses = _make_masses(3)
        (freq_aver, _, freq_corr_max), _ = self._run(ts_modes, min_modes, masses)
        assert freq_aver > 0.0, f"freq_aver > {0.0} is expected, got {freq_aver}; test #1"
        assert freq_corr_max[0] > 0.0, f"freq_max > {0.0} is expected, got {freq_corr_max[0]}; test #2"
        assert freq_aver != freq_corr_max[0], f"expected {freq_aver} != {freq_corr_max[0]}; test #3"

    def test_orca_prog_key_uses_orca_reader(self):
        """ The ORCA program key uses the ORCA mode reader """
        ts_modes = _make_modes(n_atoms=2, n_modes=1)
        min_modes = _make_modes(n_atoms=2, n_modes=3)
        masses = _make_masses(2)
        reader = MagicMock(side_effect=[(ts_modes, masses), (min_modes, masses)])
        expected = 2
        with patch(ORCA_COORD_READER, reader), patch(FILE_CHECK, side_effect=[Path("ts.out"), Path("min.out")]):
            attempt_freq_calc(Path("ts.out"), Path("min.out"), species="product", prog_key="orca")
        assert reader.call_count == expected, f"expected {expected}, got {reader.call_count}"

    def test_warns_when_correlation_sum_far_from_one(self):
        """ A correlation sum far from one triggers a logger warning """
        ts_modes = [[100.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        min_modes = [[200.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     [300.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
        masses = (1.0, 1.0) # use orthogonal modes so sum of correlations is 0.0
        reader = MagicMock(side_effect=[(ts_modes, masses), (min_modes, masses)])
        with patch(GAUSS_COORD_READER, reader), patch(FILE_CHECK, side_effect=[Path("ts.out"), Path("min.out")]):
            with patch(LOGGER_IMP) as mock_logger:
                attempt_freq_calc(Path("ts.out"), Path("min.out"), species="reactant", prog_key="gaussian")
        mock_logger.warning.assert_called()


# ===========================================================================
# --- attempt_freq_calc - input validation ---
# ===========================================================================
class TestAttemptFreqCalcValidation:

    def _mock_reader(self, n_atoms=2, n_modes=3, freq_start=100.0):
        """ Return synthetic normal modes and masses """
        modes = _make_modes(n_atoms, n_modes, freq_start)
        masses = _make_masses(n_atoms)
        return MagicMock(return_value=(modes, masses))

    def test_invalid_prog_key_raises(self):
        """ Unsupported program key should raise an error. Expecting a ValueError """
        with pytest.raises(ValueError, match="Unsupported program key"):
            attempt_freq_calc(Path("ts.out"), Path("min.out"), species="reactant", prog_key="molpro")

    def test_invalid_species_raises(self):
        """ Unsupported species type should raise an error. Expecting a ValueError """
        with pytest.raises(ValueError, match="Unsupported species type"):
            attempt_freq_calc(Path("ts.out"), Path("min.out"), species="unsupported_species", prog_key="gaussian")

    def test_zero_ts_mode_norm_raises(self):
        """A TS mode with all-zero coordinates should raise an error. Expecting a ValueError """
        modes_zero = [[100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        with patch(GAUSS_COORD_READER, return_value=(modes_zero, _make_masses(2))), \
            patch(FILE_CHECK, side_effect=[Path("ts.out"), Path("min.out")]):
            with pytest.raises(ValueError, match="normalization factor"):
                 attempt_freq_calc(Path("ts.out"), Path("min.out"), species="reactant", prog_key="gaussian")

    def test_zero_minimum_mode_norm_raises(self):
        """ A minimum mode with all-zero coordinates should raise an error. Expecting a ValueError """
        ts_modes = _make_modes(n_atoms=2, n_modes=1)
        min_modes = [[200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        masses = _make_masses(2)
        with patch(GAUSS_COORD_READER, side_effect=[(ts_modes, masses), (min_modes, masses)]), \
            patch(FILE_CHECK, side_effect=[Path("ts.out"), Path("min.out")]):
            with pytest.raises(ValueError, match="normalization factor"):
                 attempt_freq_calc(Path("ts.out"), Path("min.out"), species="reactant", prog_key="gaussian")

    def test_coordinate_count_mismatch_raises(self):
        """ A different number of coordinates for the TS and minimum modes should raise an error. Expecting a ValueError """
        ts_modes = [[100.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] # six coordinates
        min_modes = [[200.0, 1.0, 0.0, 0.0]]  # three coordinates
        with patch(GAUSS_COORD_READER, side_effect=[(ts_modes, _make_masses(2)), (min_modes, _make_masses(1))]), \
            patch(FILE_CHECK, side_effect=[Path("ts.out"), Path("min.out")]):
            with pytest.raises(ValueError, match="different numbers of coordinates"):
                attempt_freq_calc(Path("ts.out"), Path("min.out"), species="reactant", prog_key="gaussian")

    def test_empty_minimum_mode_raises(self):
        """ A different number of coordinates for the TS and minimum modes should raise an error. Expecting a ValueError """
        ts_modes = [[100.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] # six coordinates
        min_modes = [[200.0]]  # no coordinates
        with patch(GAUSS_COORD_READER, side_effect=[(ts_modes, _make_masses(2)), (min_modes, _make_masses(1))]), \
            patch(FILE_CHECK, side_effect=[Path("ts.out"), Path("min.out")]), patch(VIB_MASS_SCAL) as mock_scaling:
            mock_scaling.side_effect = [ts_modes, []]
            with pytest.raises(ValueError, match="No valid normal modes"):
                attempt_freq_calc(Path("ts.out"), Path("min.out"), species="reactant", prog_key="gaussian")


# ===========================================================================
# --- write_freq_corr - normal behaviour and input validation ---
# ===========================================================================
class TestWriteFreqCorr:

    def _write_and_read(self, tmp_path, corrs, max_freq_and_corr, att_freq=1234.56):
        """ Runs the function with the pre-defined parameters """
        out = tmp_path / "corr.txt"
        with patch(FILE_CHECK, return_value=out):
            write_freq_corr(out, att_freq, corrs, max_freq_and_corr)
            corr_file = out.with_stem(out.stem + "_corr")
        return corr_file.read_text(encoding="utf-8")

    def test_creates_file(self, tmp_path):
        """ Creates the output file and returns None """
        out = tmp_path / "corr.txt"
        with patch(FILE_CHECK, return_value=out):
            result = write_freq_corr(out, 1000.0, [(500.0, 0.9)], (500.0, 0.9))
            corr_file = out.with_stem(out.stem + "_corr")
        # Check of the correlation file
        assert corr_file.is_file(), f"expected {True}, got {out.is_file()}; test #1"
        assert corr_file.stem == "corr_corr", f"expected {"corr_corr"}, got {corr_file.stem}; test #2"
        assert corr_file.suffix == ".txt", f"expected {".txt"}, got {corr_file.suffix}; test #3"
        # Check of the function return value
        assert result is None, f"expected {None}, got {result}; test #4"

    def test_header_and_separator_present(self, tmp_path):
        """ Writes the expected column headers and a separator """
        text = self._write_and_read(tmp_path, [(500.0, 0.5)], (500.0, 0.5))
        expected_1 = ["Frequency", "Correlation"]
        # Check headers
        for val in expected_1:
            assert val in text, f"expected {val} in {text}; test #1"
        # Check a separator
        expected_2 = "-" * 10
        assert expected_2 in text, f"expected {expected_2} in {text}; test #2"

    def test_frequency_and_correlation_written(self, tmp_path):
        """ Writes frequency and correlation values with the expected formatting """
        expected = [(1234.5, 0.87654), (5678.9, 0.12345)]
        text = self._write_and_read(tmp_path, expected, (500.0, 0.5))
        for freq, corr in expected:
            assert str(freq) in text, f"expected {str(freq)} in {text}; test #1"
            assert str(corr) in text, f"expected {str(corr)} in {text}; test #2"

    def test_attempt_frequency_in_footer(self, tmp_path):
        """ Writes the attempt frequency in the footer """
        expected = 999.99
        text = self._write_and_read(tmp_path, [(500.0, 1.0)], (500.0, 1.0), att_freq=expected)
        assert str(expected) in text, f"expected {expected} in {text}"

    def test_multiple_rows_all_written(self, tmp_path):
        """ Writes all correlation rows """
        corrs = [(100.0, 0.1), (200.0, 0.5), (300.0, 0.4)]
        text = self._write_and_read(tmp_path, corrs, (500.0, 1.0))
        expected = [str(freq) for freq, _ in corrs]
        for val in expected:
            assert val in text, f"expected {val} in {text}"

    def test_empty_correlations_writes_only_header(self, tmp_path):
        """ Writes the header when no correlations are provided """
        text = self._write_and_read(tmp_path, [], (500.0, 1.0), att_freq=0.0)
        expected = "Frequency"
        assert expected in text, f"expected {expected} in {text}" 

    def test_string_path_accepted(self, tmp_path):
        """ Accepts a string path as the output file path """
        out = tmp_path / "corr.txt"
        with patch(FILE_CHECK, return_value=out):
            write_freq_corr(str(out), 1000.0, [(500.0, 0.9)], (500.0, 0.9))
            corr_file = out.with_stem(out.stem + "_corr")
        assert corr_file.is_file(), f"expected {True}, got {out.is_file()}"


# =================================================================================================
# --- _extract_energies_to_compute_number_of_vib_levels - normal behaviour and input validation ---
# =================================================================================================
class TestExtractEnergiesToComputeNumberOfVibLevels:

    def test_species_with_zpve(self, tol=1e-12):
        """ Should return the species energies including ZPVE (reactant) """
        ep = _make_energy_points()
        irc_data = _make_irc_data()
        ts_energy, point_energy = _extract_energies_to_compute_number_of_vib_levels(irc_data,
        ep, direction="reactant")
        expected = [0.00, -0.07]
        assert math.isclose(ts_energy, expected[0], abs_tol=tol), f"expected {expected[0]}, got {ts_energy}; test #1"
        assert math.isclose(point_energy, expected[1], abs_tol=tol), f"expected {expected[1]}, got {point_energy}; test #2"

    def test_reactant_without_zpve(self, tol=1e-12):
        """ Should return the species energies including ZPVE when ZPVE option is unavailable (reactant) """
        ep = _make_energy_points()
        irc_data = _make_irc_data(with_zpve=False)
        ts_energy, point_energy = _extract_energies_to_compute_number_of_vib_levels(irc_data,
        ep, direction="reactant")
        expected = [-0.05, -0.10]
        assert math.isclose(ts_energy, expected[0], abs_tol=tol), f"expected {expected[0]}, got {ts_energy}; test #1"
        assert math.isclose(point_energy, expected[1], abs_tol=tol), f"expected {expected[1]}, got {point_energy}; test #2"

    def test_product_with_zpve(self, tol=1e-12):
        """ Should return the species energies including ZPVE (product) """
        ep = _make_energy_points()
        irc_data = _make_irc_data()
        ts_energy, point_energy = _extract_energies_to_compute_number_of_vib_levels(irc_data,
        ep, direction="product")
        expected = [0.00, -0.10]
        assert math.isclose(ts_energy, expected[0], abs_tol=tol), f"expected {expected[0]}, got {ts_energy}; test #1"
        assert math.isclose(point_energy, expected[1], abs_tol=tol), f"expected {expected[1]}, got {point_energy}; test #2"

    def test_product_without_zpve(self, tol=1e-12):
        """ Should return the species energies including ZPVE when ZPVE option is unavailable (product) """
        ep = _make_energy_points()
        irc_data = _make_irc_data(with_zpve=False)
        ts_energy, point_energy = _extract_energies_to_compute_number_of_vib_levels(irc_data,
        ep, direction="product")
        expected = [-0.05, -0.12]
        assert math.isclose(ts_energy, expected[0], abs_tol=tol), f"expected {expected[0]}, got {ts_energy}; test #1"
        assert math.isclose(point_energy, expected[1], abs_tol=tol), f"expected {expected[1]}, got {point_energy}; test #2"

    def test_reactant_without_zpve_scaled(self, tol=1e-12):
        """ Should return the species energies including ZPVE when ZPVE option is unavailable (reactant, scaled) """
        ep = _make_energy_points()
        irc_data = _make_irc_data(scale_factor=0.5, with_zpve=False)
        ts_energy, point_energy = _extract_energies_to_compute_number_of_vib_levels(irc_data,
        ep, direction="reactant")
        expected = [-0.025, -0.05]
        assert math.isclose(ts_energy, expected[0], abs_tol=tol), f"expected {expected[0]}, got {ts_energy}; test #1"
        assert math.isclose(point_energy, expected[1], abs_tol=tol), f"expected {expected[1]}, got {point_energy}; test #2"

    def test_product_with_zpve_scaled(self, tol=1e-12):
        """ Should return the species energies including ZPVE (product, scaled) """
        ep = _make_energy_points()
        irc_data = _make_irc_data(scale_factor=2)
        ts_energy, point_energy = _extract_energies_to_compute_number_of_vib_levels(irc_data,
        ep, direction="product")
        expected = [-0.05, -0.22]
        assert math.isclose(ts_energy, expected[0], abs_tol=tol), f"expected {expected[0]}, got {ts_energy}; test #1"
        assert math.isclose(point_energy, expected[1], abs_tol=tol), f"expected {expected[1]}, got {point_energy}; test #2"

    def test_invalid_direction_raises(self):
        """ Should raise an error for an invalid direction. Expecting a ValueError """
        ep = _make_energy_points()
        irc_data = _make_irc_data()
        with pytest.raises(ValueError, match="as direction"):
            _extract_energies_to_compute_number_of_vib_levels(irc_data, ep, direction="unsupported")

    def test_missing_electronic_energies_raises(self):
        """Should raise an error when electronic energies are missing. Expecting a ValueError """
        ep = _make_energy_points()
        irc_data = _make_irc_data(with_el_energies=False)
        with pytest.raises(ValueError, match="array of electronic energies"):
            _extract_energies_to_compute_number_of_vib_levels(irc_data, ep, direction="reactant")


# ===========================================================================
# --- compute_number_of_vib_levels - normal behaviour ---
# ===========================================================================
class TestComputeNumberOfVibLevelsNormal:
    
    def test_returns_int(self):
        """ Returns the number of vibrational levels as an integer """
        ep = _make_energy_points(ts_el=0.0, ts_zpve=0.05, react_el=-0.1, react_zpve=0.02)
        irc_data = _make_irc_data()
        result = compute_number_of_vib_levels(1000.0, irc_data, ep, direction="reactant")
        assert isinstance(result, int), f"expected {int}, got {type(result)}"

    def test_non_negative_result(self):
        """ Returns a non-negative number of vibrational levels """
        ep = _make_energy_points(ts_zpve=0.0)
        # Check with_zpve=False
        irc_data = _make_irc_data(with_zpve=False)
        result = compute_number_of_vib_levels(1000.0, irc_data, ep, direction="reactant")
        assert result >= 0, f"expected value >= 0, got {result:.3g}; test #1"
        # Check scale_factor=0.0, with_zpve=False
        irc_data = _make_irc_data(scale_factor=0.0, with_zpve=False)
        result = compute_number_of_vib_levels(1000.0, irc_data, ep, direction="reactant")
        assert result == 0, f"expected value == 0, got {result:.3g}; test #2"
        # Check a negative barrier
        irc_data = _make_irc_data_negative_barrier()
        result = compute_number_of_vib_levels(1000.0, irc_data, ep, direction="reactant")
        assert result == 0, f"expected value == 0, got {result:.3g}; test #3"
        # Check a strictly positive number of levels
        irc_data = _make_irc_data(scale_factor=1.0, with_zpve=False)
        result = compute_number_of_vib_levels(1000.0, irc_data, ep, direction="reactant")
        assert result > 0, f"expected value > 0, got {result:.3g}; test #4"

    def test_higher_barrier_gives_more_levels(self):
        """ A higher energy barrier results in at least as many vibrational levels """
        ep = _make_energy_points(ts_el=0.0, ts_zpve=0.05, react_el=-0.05, react_zpve=0.02)
        # Check a scale factor
        irc_data_low = _make_irc_data(scale_factor=1)
        irc_data_high = _make_irc_data(scale_factor=2)
        n_low = compute_number_of_vib_levels(1000.0, irc_data_low, ep, direction="reactant")
        n_high = compute_number_of_vib_levels(1000.0, irc_data_high, ep, direction="reactant")
        assert n_high >= n_low, f"expected {n_high} >= {n_low}; test #1"
        # Check an effect of ZPVE
        irc_data_low = _make_irc_data(with_zpve=False)
        irc_data_high = _make_irc_data(with_zpve=True)
        n_low = compute_number_of_vib_levels(1000.0, irc_data_low, ep, direction="reactant")
        n_high = compute_number_of_vib_levels(1000.0, irc_data_high, ep, direction="reactant")
        assert n_high >= n_low, f"expected {n_high} >= {n_low}; test #2"

    def test_higher_frequency_gives_fewer_levels(self):
        """ A higher frequency results in fewer or equal vibrational levels """
        ep = _make_energy_points(ts_el=0.0, ts_zpve=0.05, react_el=-0.1, react_zpve=0.02)
        irc_data = _make_irc_data()
        n_low = compute_number_of_vib_levels(500.0, irc_data, ep, direction="reactant")
        n_high = compute_number_of_vib_levels(3000.0, irc_data, ep, direction="reactant")
        assert n_high <= n_low, f"expected {n_high} <= {n_low}"

    def test_product_direction_uses_product_energy(self):
        """ The product direction uses the product energy """
        ts_el, ts_zpve_val = -0.05, 0.05 # it should match the IRC data
        ts_total = ts_el + ts_zpve_val
        react_el, react_zpve = -0.10, 0.03 # it should match the IRC data
        react_total = react_el + react_zpve
        prod_el, prod_zpve = -0.12, 0.02 # it should match the IRC data
        prod_total = prod_el + prod_zpve
        ep = _make_energy_points(ts_zpve=ts_zpve_val)
        # Check with_zpve=False
        irc_data = _make_irc_data(with_zpve=False)
        frequency = 1000
        expected_react = max(0, math.ceil((ts_el - react_el) / (frequency * CM_M1_TO_HARTREE) + 0.5) - 1)
        expected_prod = max(0, math.ceil((ts_el - prod_el) / (frequency * CM_M1_TO_HARTREE) + 0.5) - 1)
        n_react = compute_number_of_vib_levels(frequency, irc_data, ep, direction="reactant")
        n_prod = compute_number_of_vib_levels(frequency, irc_data, ep, direction="product")
        assert n_react == expected_react, f"expected {expected_react}, got {n_react}; test #1"
        assert n_prod == expected_prod, f"expected {expected_prod}, got {n_prod}; test #2"
        # Check with_zpve=True
        irc_data = _make_irc_data(with_zpve=True)
        expected_react = max(0, math.ceil((ts_total - react_total) / (frequency * CM_M1_TO_HARTREE) + 0.5) - 1)
        expected_prod = max(0,math.ceil((ts_total - prod_total) / (frequency * CM_M1_TO_HARTREE) + 0.5) - 1)
        n_react = compute_number_of_vib_levels(frequency, irc_data, ep, direction="reactant")
        n_prod = compute_number_of_vib_levels(frequency, irc_data, ep, direction="product")
        assert n_react == expected_react, f"expected {expected_react}, got {n_react}; test #3"
        assert n_prod == expected_prod, f"expected {expected_prod}, got {n_prod}; test #4"

    def test_scaling_factor_scales_result(self):
        """ The potential scaling factor affects the number of vibrational levels """
        ep = _make_energy_points(ts_el=0.0, ts_zpve=0.05, react_el=-0.1, react_zpve=0.02)
        irc_data = _make_irc_data(with_zpve = False)
        n1 = compute_number_of_vib_levels(1000.0, irc_data, ep, potential_scaling_factor=1.0, direction="reactant")
        n2 = compute_number_of_vib_levels(1000.0, irc_data, ep, potential_scaling_factor=2.0, direction="reactant")
        assert n2 >= n1, f"expected {n2} >= {n1}"

    def test_the_formula_is_correct(self):
        """ The formula should be correct """
        ts_energies, point_energies = [-0.05, 0.0], [-0.10, 0.0]
        irc_data = _make_irc_data(with_zpve = False)
        psf, frequency = 2.0, 1000.0
        ep = _make_energy_points(ts_el=0.0, ts_zpve=0.05, react_el=-0.1, react_zpve=0.02)
        n_computed = compute_number_of_vib_levels(frequency, irc_data, ep,
            potential_scaling_factor=psf, direction="reactant")

        number_of_levels = math.ceil(psf * (sum(ts_energies) - sum(point_energies)) / (frequency * CM_M1_TO_HARTREE) + 0.5)
        n_expected = max(0, number_of_levels - 1)

        assert n_computed == n_expected, f"expected {n_computed} == {n_expected}"


# ===========================================================================
# --- compute_number_of_vib_levels - input validation ---
# ===========================================================================
class TestComputeNumberOfVibLevelsValidation:

    def test_invalid_direction_raises(self):
        """ Raises an error for an invalid direction. Expecting a ValueError """
        ep = _make_energy_points()
        irc_data = _make_irc_data()
        # Check an invalid direction
        with pytest.raises(ValueError, match="as direction") as exc_info_1:
            compute_number_of_vib_levels(1000.0, irc_data, ep, direction="unsupported")
        print("Caught (test #1):", exc_info_1.value)
        # Check electronic_energies=None (very non-typical case)
        irc_data_broken = _make_irc_data(with_el_energies=False)
        with pytest.raises(ValueError, match="array of electronic energies") as exc_info_2:
            compute_number_of_vib_levels(1000.0, irc_data_broken, ep, direction="reactant")
        print("Caught (test #2):", exc_info_2.value)

    def test_zero_or_negative_frequency_raises(self):
        """ Raises ValueError for a zero or negative frequency. Expecting a ValueError """
        ep = _make_energy_points()
        irc_data = _make_irc_data()
        # Check freq = 0
        with pytest.raises(ValueError, match="positive") as exc_info_1:
            compute_number_of_vib_levels(0.0, irc_data, ep, direction="reactant")
        print("Caught (test #1):", exc_info_1.value)
        # Check freq < 0
        with pytest.raises(ValueError, match="positive") as exc_info_2:
            compute_number_of_vib_levels(-500.0, irc_data, ep, direction="reactant")
        print("Caught (test #2):", exc_info_2.value)


# ===========================================================================
# --- detect_outliers_local - normal behaviour ---
# ===========================================================================
class TestDetectOutliersLocalNormal:

    def test_clean_data_no_warning(self):
        """ Does not issue a warning for smooth, outlier-free data """
        irc = np.linspace(-2.0, 2.0, 20)
        energy = np.sin(irc)
        with patch(LOGGER_IMP) as mock_logger: # smooth, no outliers
            result = detect_outliers_local(irc, energy, mode="el")
            mock_logger.warning.assert_not_called()
        assert result is None, f"expected {None}, got {result}"

    def test_obvious_spike_triggers_warning(self):
        """ Issues a warning when an obvious spike is detected """
        irc = np.linspace(-2.0, 2.0, 21)
        energy = np.sin(irc)
        energy[10] += 100.0
        with patch(LOGGER_IMP) as mock_logger: # huge spike at centre
            detect_outliers_local(irc, energy, mode="el")
        mock_logger.warning.assert_called()

    def test_single_point_logs_skipped_warning(self):
        """ Issues a warning when a single-point dataset cannot be analyzed reliably """
        with patch(LOGGER_IMP) as mock_logger: # a single point
            detect_outliers_local(np.array([0.0]), np.array([1.0]), mode="el")
        mock_logger.warning.assert_called()

    def test_unsorted_input_handled(self):
        """ Handles unsorted IRC input without issuing a warning for clean data """
        irc = np.array([3.0, 1.0, 2.0, 0.0, 4.0])
        energy = np.array([9.0, 1.0, 4.0, 0.0, 10.0])
        with patch(LOGGER_IMP) as mock_logger: # unsorted data
            detect_outliers_local(irc, energy, mode="el")
        mock_logger.warning.assert_not_called()

    def test_larger_radius_is_more_tolerant(self):
        """
        A larger radius uses a wider neighbourhood for local outlier detection.
        Verify that moderate deviations do not cause errors.
        """
        irc = np.linspace(-5.0, 5.0, 51)
        energy = np.cos(irc)
        energy[25] += 0.5
        with patch(LOGGER_IMP): # a moderate deviation
            detect_outliers_local(irc, energy, mode="el", radius=20)

    def test_zero_mad_flags_differing_point(self):
        """ When all neighbours are identical (MAD=0.0), any different value should be flagged as an outlier """
        irc = np.linspace(0.0, 10.0, 15)
        energy = np.ones(15)
        energy[7] = 999.0 # outlier among a flat background
        with patch(LOGGER_IMP) as mock_logger:
            detect_outliers_local(irc, energy, mode="el")
        expected = 1 # at least one warning is expected (for the outlier at index 7)
        result = mock_logger.warning.call_count
        assert result >= expected, f"expected at least {expected} warnings, got {result}" 

    def test_high_threshold_ignores_moderate_spike(self):
        """ Does not flag a moderate deviation when the threshold is very high """
        irc = np.linspace(-2.0, 2.0, 21)
        energy = np.sin(irc)
        energy[10] += 0.1 # a tiny spike
        with patch(LOGGER_IMP) as mock_logger:
            detect_outliers_local(irc, energy, mode="el", threshold=100.0)
        mock_logger.warning.assert_not_called()


# ===========================================================================
# --- detect_outliers_local - input validation ---
# ===========================================================================
class TestDetectOutliersLocalValidation:
    def test_empty_irc_raises(self):
        """ Raises an error when the IRC data is empty. Expecting a ValueError """
        with pytest.raises(ValueError, match="No"):
            detect_outliers_local(np.array([]), np.array([]), mode="el")

    def test_length_mismatch_raises(self):
        """ Raises an error when the IRC and energy arrays have different lengths. Expecting a ValueError """
        with pytest.raises(ValueError, match="same length"):
            detect_outliers_local(np.arange(5, dtype=float), np.arange(4, dtype=float), mode="zpve")

    def test_radius_less_than_1_raises(self):
        """ Raises an error when the radius is less than 1. Expecting a ValueError """
        with pytest.raises(ValueError, match="radius"):
            detect_outliers_local(np.arange(5, dtype=float), np.ones(5), mode="el", radius=0)

    def test_non_positive_threshold_raises(self):
        """ Raises an error when the threshold is not positive. Expecting a ValueError """
        # Check threshold = 0
        with pytest.raises(ValueError, match="threshold") as exc_info_1:
            detect_outliers_local(np.arange(5, dtype=float), np.ones(5), mode="zpve", threshold=0.0)
        print("Caught (test #1):", exc_info_1.value)
        # Check threshold < 0
        with pytest.raises(ValueError, match="threshold") as exc_info_2:
            detect_outliers_local(np.arange(5, dtype=float), np.ones(5), mode="zpve", threshold=-1.0)
        print("Caught (test #2):", exc_info_2.value)

    def test_duplicate_irc_raises(self):
        """ Raises an error when duplicate IRC values are present. Expecting a ValueError """
        irc = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Duplicate"):
            detect_outliers_local(irc, np.ones(5), mode="el")