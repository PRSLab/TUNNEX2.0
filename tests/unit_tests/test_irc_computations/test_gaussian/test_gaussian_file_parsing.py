# --- Test: test_gaussian_file_parsing. Unit tests for .\gaussian\gaussian_file_parsing.py
# Run with: pytest test_gaussian_file_parsing.py -v ---


# --- Modules ---
import pytest # type: ignore
from unittest.mock import MagicMock, mock_open, patch
from contextlib import ExitStack

import numpy as np
from pathlib import Path

from tunnex_2.constants_and_dataclasses.constants_and_settings import CM_M1_TO_HARTREE # type: ignore
from tunnex_2.constants_and_dataclasses.dataclasses import IRCData # type: ignore


# --- Module to test ---
from tunnex_2.irc_computations.gaussian.gaussian_file_parsing import ( # type: ignore
    _gauss_mass_extraction,
    gauss_irc_file_split,
    gauss_extract_geom_from_opt_file,
    gauss_extract_geom_from_irc_point,
    _gauss_freq_extraction_irc,
    reading_gauss_irc,
    _gauss_collect_freq_and_modes_tunnex,
    reading_gauss_struct_file,
    gauss_mode_coordinate_reader,
    _gauss_struct_irc_direction,
    _gauss_struct_extraction,
    reading_gauss_irc_tunnex)


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.irc_computations.gaussian.gaussian_file_parsing"
GAUSS_PATTERNS = "tunnex_2.irc_computations.gaussian.gaussian_file_parsing.gaussian_patterns"


# ===========================================================================
# --- _gauss_mass_extraction - normal behaviour ---
# ===========================================================================
class TestGaussMassExtractionNormal:

    def _setup(self, tmp_path, values):
        """Should set up a temporary Gaussian output file with mocked mass data """
        f = tmp_path / "mol.out"
        f.write_text("placeholder")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_MASS_LINES_AND_VALUES.findall.return_value = values
            result = _gauss_mass_extraction(f)
        return result

    def test_returns_numpy_array(self, tmp_path):
        """ Verify that extracted atomic masses are returned as a NumPy array """
        result = self._setup(tmp_path, [("Atom 1 has atomic number 1 and mass 1.008", "1.008"),
                                       ("Atom 2 has atomic number 6 and mass 12.0", "12.0")])
        assert isinstance(result, np.ndarray), f"expected {np.ndarray}, got {type(result)}"

    def test_correct_values_extracted(self, tmp_path):
        """ Verify that atomic masses are extracted with the correct values """
        result = self._setup(tmp_path,[ ("Atom 1 has atomic number 1 and mass 1.008", "1.008"),
                                        ("Atom 2 has atomic number 6 and mass 12.0", "12.0"),
                                        ("Atom 3 has atomic number 8 and mass 15.999", "15.999")])
        expected = [1.008, 12.0, 15.999]
        assert np.allclose(result, expected), f"expected {expected}, got {result}"

    def test_reads_from_cached_mass_file_if_exists(self, tmp_path):
        """ Verify that an existing cached mass file is used preferentially """
        mass_file = tmp_path / "gaussian_atom_masses.out"
        mass_file.write_text("cached")
        mol = str(tmp_path / "mol.out") # check if the function can accept str type
        Path(mol).write_text("original")
        with patch(FUNC_PATH +".open", mock_open(read_data="cached")) as mock_file, \
             patch(GAUSS_PATTERNS) as gp:
             gp.PATTERN_MASS_LINES_AND_VALUES.findall.return_value = [("Atom 2 has atomic number 6 and mass 12.0", "12.0")]
             result = _gauss_mass_extraction(mol)
        mock_file.assert_called_once_with(mass_file, encoding="utf-8") # it should be called once
        assert result[0] == 12.0, f"expected {12.0}, got {result[0]}"

    def test_writes_mass_cache_file_when_missing(self, tmp_path):
        """ Verify that a mass cache file is created when it does not exist """
        mol = tmp_path / "mol.out"
        mol.write_text("placeholder")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_MASS_LINES_AND_VALUES.findall.return_value = [("Atom 2 has atomic number 6 and mass 12.0", "12.0")]
            _gauss_mass_extraction(mol)
        result = (tmp_path / "gaussian_atom_masses.out").is_file()
        assert result, f"expected {True}, got {result}"


# ===========================================================================
# --- _gauss_mass_extraction - input validation ---
# ===========================================================================
class TestGaussMassExtractionValidation:

    def test_non_out_suffix_raises(self, tmp_path):
        """ Verifies that any file without an '.out' suffix file raises an error. Expecting a ValueError """
        f = tmp_path / "mol.log"
        f.write_text("")
        with pytest.raises(ValueError, match="an '.out' file"):
            _gauss_mass_extraction(f)

    def test_empty_mass_block_raises(self, tmp_path):
        """ Verifies that a missing mass block raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        f.write_text("No mass information here")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_MASS_LINES_AND_VALUES.findall.return_value = []
            with pytest.raises(ValueError, match="not found"):
                _gauss_mass_extraction(f)

    def test_empty_values_in_block_raises(self, tmp_path):
        """ Verifies that a mass block with no atomic masses raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        f.write_text("masses")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_MASS_LINES_AND_VALUES.findall.return_value = [("Atom 2 has atomic number 6 and mass ", "")]
            with pytest.raises(ValueError, match="no atomic masses"):
                _gauss_mass_extraction(f)

    def test_zero_mass_raises(self, tmp_path):
        """ Verifies that a zero atomic mass raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        f.write_text("masses")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_MASS_LINES_AND_VALUES.findall.return_value = [("Atom 2 has atomic number 6 and mass 0.0", "0.0")]
            with pytest.raises(ValueError, match="positive"):
                _gauss_mass_extraction(f)

    def test_negative_mass_raises(self, tmp_path):
        """ Verifies that a negative atomic mass raises ab error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        f.write_text("masses")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_MASS_LINES_AND_VALUES.findall.return_value = [("Atom 2 has atomic number 6 and mass -12.0", "-12.0")]
            with pytest.raises(ValueError, match="positive"):
                _gauss_mass_extraction(f)


# ===========================================================================
# --- gauss_irc_file_split - normal behaviour and input validation ---
# ===========================================================================
class TestGaussIRCSplit:

    def test_splits_file_into_forward_and_reverse(self, tmp_path):
        """ Tests that the file is correctly split into forward and reverse sections """
        f = tmp_path / "irc.out"
        marker = "=== MARKER ==="
        f.write_text("forward data=== MARKER ===reverse data")
        forward, reverse = gauss_irc_file_split(f, marker)
        assert forward == "forward data", f"expected {"forward data"}, got {forward}; test #1"
        assert reverse == "reverse data", f"expected {"reverse data"}, got {reverse}; test #2"

    def test_splits_only_at_first_marker(self, tmp_path):
        """ Tests that only the first marker is used to split the file """
        f = tmp_path / "irc.out"
        marker = "=== MARKER ==="
        f.write_text("forward=== MARKER ===reverse=== MARKER ===extra")
        forward, reverse = gauss_irc_file_split(f, marker)
        expected_f = "forward"
        assert forward == expected_f, f"expected {expected_f}, got {forward}; test #1"
        expected_r = "reverse=== MARKER ===extra"
        assert reverse == expected_r, f"expected {expected_r}, got {reverse}; test #2"

    def test_missing_marker_raises(self, tmp_path):
        """ Tests that a missing marker raises an error. Expecting a ValueError """
        f = tmp_path / "irc.out"
        marker = "=== MARKER ==="
        f.write_text("forward data without marker")
        with pytest.raises(ValueError, match="Cannot find"):
            gauss_irc_file_split(f, marker)


# ================================================================================
# --- gauss_extract_geom_from_opt_file - normal behaviour and input validation ---
# ================================================================================
class TestGaussExtractGeomFromOptFile:

    def test_returns_last_geometry_as_string(self, tmp_path):
        """ Test that the last optimized geometry is returned as a string """
        f = tmp_path / "opt.out"
        f.write_text("geometry blocks")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_OPTIMIZED_SPECIES_GEOMETRY_BLOCK.findall.return_value = ["first block", "last block"]
            gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.return_value = [("C", "1.0", "2.0", "3.0"),
                                                                    ("H", "4.0", "5.0", "6.0")]
            result = gauss_extract_geom_from_opt_file(f)
        expected = "C 1.0 2.0 3.0\nH 4.0 5.0 6.0"
        assert result == expected, f"expected {expected}, got {result}"

    def test_returns_geometry_as_numpy_array(self, tmp_path):
        """ Test that the optimized geometry is returned as a NumPy array """
        f = tmp_path / "opt.out"
        f.write_text("geometry block")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_OPTIMIZED_SPECIES_GEOMETRY_BLOCK.findall.return_value = ["geometry block"]
            gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.return_value = [("C", "1.0", "2.0", "3.0"),
                                                                    ("H", "4.0", "5.0", "6.0")]
            result = gauss_extract_geom_from_opt_file(f, as_array=True)
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert isinstance(result, np.ndarray), f"expected {np.ndarray}, got {type(result)}; test #1"
        assert np.array_equal(result, expected), f"expected {expected}, got {result}; test #2"

    def test_missing_geometry_block_raises(self, tmp_path):
        """ Test that a missing optimized geometry block raises an error. Expecting a ValueError """
        f = tmp_path / "opt.out"
        f.write_text("no geometry")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_OPTIMIZED_SPECIES_GEOMETRY_BLOCK.findall.return_value = []
            with pytest.raises(ValueError, match="was not found"):
                gauss_extract_geom_from_opt_file(f)

    def test_missing_geometry_coordinates_raises(self, tmp_path):
        """ Test that a geometry block without coordinates raises an error. Expecting a ValueError """
        f = tmp_path / "opt.out"
        f.write_text("geometry block")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_OPTIMIZED_SPECIES_GEOMETRY_BLOCK.findall.return_value = ["geometry block"]
            gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.return_value = []
            with pytest.raises(ValueError, match="no geometry coordinates"):
                gauss_extract_geom_from_opt_file(f)

    def test_invalid_coordinates_raise(self, tmp_path):
        """Test that invalid geometry coordinates raise an error. Expecting a ValueError """
        f = tmp_path / "opt.out"
        f.write_text("geometry block")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_OPTIMIZED_SPECIES_GEOMETRY_BLOCK.findall.return_value = ["geometry block"]
            gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.return_value = [("C", "not_a_number", "2.0", "3.0")]
            with pytest.raises(ValueError, match="Invalid coordinates"):
                gauss_extract_geom_from_opt_file(f, as_array=True)

    def test_raises_and_custom_species_is_used_in_error_message(self, tmp_path):
        """ Test that the specified species name is used in the error message (ValueError) """
        f = tmp_path / "opt.out"
        f.write_text("no geometry")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_OPTIMIZED_SPECIES_GEOMETRY_BLOCK.findall.return_value = []
            with pytest.raises(ValueError, match="reactant"):
                gauss_extract_geom_from_opt_file(f, species="reactant")


# =================================================================================
# --- gauss_extract_geom_from_irc_point - normal behaviour and input validation ---
# =================================================================================
class TestGaussExtractGeomFromIRCPoint:

    def test_returns_geometry_from_last_match(self, tmp_path):
        """ Test that the geometry after the last species marker is returned """
        f = tmp_path / "irc.out"
        text = "first geometry marker second geometry marker final geometry"
        match1 = MagicMock()
        match1.end.return_value = 10
        match2 = MagicMock()
        match2.end.return_value = 35
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer.return_value = [match1, match2]
            gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.return_value = [("C", "1.0", "2.0", "3.0"),
                                                                        ("H", "4.0", "5.0", "6.0")]
            result = gauss_extract_geom_from_irc_point(f, text)
        expected = "C 1.0 2.0 3.0\nH 4.0 5.0 6.0"
        assert result == expected, f"expected {expected}, got {result}"
        gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.assert_called_once()
        # it should be called once

    def test_geometry_lines_are_converted_to_string(self, tmp_path):
        """ Test that extracted geometry coordinates are returned as a formatted string """
        f = tmp_path / "irc.out"
        text = "geometry marker coordinates"
        match_mock = MagicMock()
        match_mock.end.return_value = 5
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer.return_value = [match_mock]
            gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.return_value = [("O", "1.1", "2.2", "3.3")]
            result = gauss_extract_geom_from_irc_point(f, text)
        expected = "O 1.1 2.2 3.3"
        assert result == expected, f"expected {expected}, got {result}"

    def test_geometry_pattern_search_starts_after_last_match(self, tmp_path):
        """ Test that geometry extraction starts after the last marker """
        f = tmp_path / "irc.out"
        text = "0123456789" * 5
        match_mock = MagicMock()
        match_mock.end.return_value = 30
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer.return_value = [match_mock]
            gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.return_value = [("C", "1.0", "2.0", "3.0")]
            gauss_extract_geom_from_irc_point(f, text)
        gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.assert_called_once()
        # it should be called once

    def test_multiple_geometry_matches_use_last_match(self, tmp_path):
        """ Test that only the last species marker is used """
        f = tmp_path / "irc.out"
        text = "abcdefghijklmnopqrstuvwxyz"
        first_match = MagicMock()
        first_match.end.return_value = 5
        second_match = MagicMock()
        second_match.end.return_value = 15
        third_match = MagicMock()
        third_match.end.return_value = 20
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer.return_value = [first_match, second_match, third_match]
            gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.return_value = [("H", "1.0", "2.0", "3.0")]
            gauss_extract_geom_from_irc_point(f, text)
        gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.assert_called_once()
        # it should be called once

    def test_multiple_geometry_lines_are_joined_with_newlines(self, tmp_path):
        """ Test that multiple geometry lines are joined with newline characters """
        f = tmp_path / "irc.out"
        text = "geometry"
        match_mock = MagicMock()
        match_mock.end.return_value = 0
        geometry_matches = [("C", "0.0", "1.0", "2.0"), ("H", "3.0", "4.0", "5.0"), ("O", "6.0", "7.0", "8.0")]
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer.return_value = [match_mock]
            gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.return_value = geometry_matches
            result = gauss_extract_geom_from_irc_point(f, text)
        expected = "C 0.0 1.0 2.0\nH 3.0 4.0 5.0\nO 6.0 7.0 8.0"
        assert result == expected, f"expected {expected}, got {result}"

    def test_geometry_coordinates_are_kept_as_strings(self, tmp_path):
        """ Test that extracted coordinate values are preserved as strings """
        f = tmp_path / "irc.out"
        text = "geometry"
        match_mock = MagicMock()
        match_mock.end.return_value = 0
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer.return_value = [match_mock]
            gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.return_value = [("Cl", "-1.234567", "2.000000", "3.141592")]
            result = gauss_extract_geom_from_irc_point(f, text)
        expected = "Cl -1.234567 2.000000 3.141592"
        assert result == expected, f"expected {expected}, got {result}"

    def test_missing_geometry_block_raises(self, tmp_path):
        """ Test that a missing species geometry block raises an error. Expecting a ValueError """
        f = tmp_path / "irc.out"
        text = "no geometry block"
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer.return_value = []
            with pytest.raises(ValueError, match="The geometry block of"):
                gauss_extract_geom_from_irc_point(f, text)

    def test_missing_geometry_lines_raises(self, tmp_path):
        """ Test that a geometry block without coordinates raises an error. Expecting a ValueError """
        f = tmp_path / "irc.out"
        text = "geometry marker"
        match_mock = MagicMock()
        match_mock.end.return_value = 10
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer.return_value = [match_mock]
            gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.return_value = []
            with pytest.raises(ValueError, match="The geometry of"):
                gauss_extract_geom_from_irc_point(f, text)

    @pytest.mark.parametrize("reactant_key, species", [(True, "reactant"), (False, "product")])
    def test_reactant_key_selects_species(self, tmp_path, reactant_key, species):
        """ Test that reactant_key selects the correct species for an error message (ValueError) """
        f = tmp_path / "irc.out"
        f.write_text("no geometry")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer.return_value = []
            with pytest.raises(ValueError, match=f"geometry block of a {species}"):
                gauss_extract_geom_from_irc_point(f, "text", reactant_key=reactant_key)


# ===========================================================================
# --- _gauss_freq_extraction_irc - normal behaviour and input validation ---
# ===========================================================================
class TestGaussFreqExtractionIRC:

    def test_returns_list_of_tuples(self):
        """ Verify that the function returns a list of (IRC, ZPVE) tuples """
        text = (
            "Frequencies -- 1000.0 2000.0\n"
            " NET REACTION COORDINATE UP TO THIS POINT =  1.5\n")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_FREQ_MODES_LINES.findall.return_value = ["1000.0", "2000.0"]
            gp.PATTERN_NET_REACTION_COORDINATE.search.return_value = MagicMock(group=lambda _: "1.5")
            result = _gauss_freq_extraction_irc(text, Path("f.out"), positive=True)
        assert isinstance(result, list), f"expected {list}, got {type(result)}; test #1"
        result_bool = all(isinstance(t, tuple) and len(t) == 2 for t in result)
        assert result_bool, f"expected {True}, got {result_bool}; test #2"

    def test_sign_applied_to_irc_coord(self):
        """ Verify that the IRC coordinate sign is reversed when positive is False """
        text = (
            "Frequencies -- 1000.0\n"
            " NET REACTION COORDINATE UP TO THIS POINT =  2.0\n")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_FREQ_MODES_LINES.findall.return_value = ["1000.0"]
            gp.PATTERN_NET_REACTION_COORDINATE.search.return_value = MagicMock(group=lambda _: "2.0")
            result = _gauss_freq_extraction_irc(text, Path("f.out"), positive=False)
        irc_coord = result[0][0]
        assert np.isclose(irc_coord, -2.0), f"expected {-2.0}, got {irc_coord}"

    def test_only_positive_freqs_summed(self):
        """ Verify that only positive frequencies contribute to the ZPVE calculation """
        text = (
            "Frequencies -- -500.0 1000.0 2000.0\n"
            " NET REACTION COORDINATE UP TO THIS POINT =  1.0\n")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_FREQ_MODES_LINES.findall.return_value = ["-500.0", "1000.0", "2000.0"]
            gp.PATTERN_NET_REACTION_COORDINATE.search.return_value = MagicMock(group=lambda _: "1.0")
            result = _gauss_freq_extraction_irc(text, Path("f.out"), positive=False)
        expected_zpve = 0.5 * 3000.0 * CM_M1_TO_HARTREE
        assert np.isclose(result[0][1], expected_zpve), f"expected {expected_zpve}, got {result[0][1]}"

    def test_empty_text_raises(self):
        """ Verify that an empty input raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match="No frequencies"):
            _gauss_freq_extraction_irc("no relevant content", Path("f.out"))

    def test_no_frequencies_before_irc_coordinate_raises(self):
        """ Verify that an IRC coordinate without preceding frequencies raises an error. Expecting a ValueError """
        text = ("NET REACTION COORDINATE UP TO THIS POINT = 1.0\n")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_NET_REACTION_COORDINATE.search.return_value = MagicMock(group=lambda _: "1.0")
            with pytest.raises(ValueError, match="No frequencies"):
                _gauss_freq_extraction_irc(text, Path("f.out"))

    def test_zero_frequency_sum_raises(self):
        """ Verify that a zero positive-frequency sum raises an error. Expecting a ValueError """
        text = ("Frequencies -- -500.0 0.0\n"
            "NET REACTION COORDINATE UP TO THIS POINT = 1.0\n")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_FREQ_MODES_LINES.findall.return_value = ["-500.0", "0.0"]
            gp.PATTERN_NET_REACTION_COORDINATE.search.return_value = MagicMock(group=lambda _: "1.0")
            with pytest.raises(ValueError, match="No frequencies"):
                _gauss_freq_extraction_irc(text, Path("f.out"))
    
    def test_missing_irc_coordinate_raises(self):
        """ Verify that a missing IRC coordinate raises an error. Expecting a ValueError """
        text = ("Frequencies -- 1000.0\n"
            "NET REACTION COORDINATE UP TO THIS POINT = invalid\n")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_FREQ_MODES_LINES.findall.return_value = ["1000.0"]
            match_mock = MagicMock()
            match_mock.group.return_value = None
            gp.PATTERN_NET_REACTION_COORDINATE.search.return_value = match_mock
            with pytest.raises(ValueError, match="IRC coordinate"):
                _gauss_freq_extraction_irc(text, Path("f.out"))


# ===========================================================================
# --- _gauss_freq_extraction_irc - normal behaviour and input validation ---
# ===========================================================================
class TestReadingGaussIRC:

    def _base_patches(self, file_path):
        """ Should apply base patches for reading_gauss_irc """
        stack = ExitStack()
        stack.enter_context(
        patch(FUNC_PATH + ".file_check", return_value=Path(file_path)))
        stack.enter_context(patch(FUNC_PATH + ".gauss_irc_file_split",
            return_value=("forward_text", "reverse_text")))
        return stack

    def test_proj_freq_false_returns_none_zpves(self, tmp_path):
        """ Verify that IRC data is returned with ZPVEs omitted when projected frequencies are disabled """
        f = tmp_path / "irc.out"
        f.write_text("")
        match_mock = MagicMock()
        match_mock.group.return_value = "-153.0"
        match_mock.end.return_value = 0
        with self._base_patches(f), patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_TS_ENERGY_IRC.search.return_value = match_mock
            gp.PATTERN_EL_ENERGY_IRC.findall.return_value = [("0.01", "1.0")]
            result = reading_gauss_irc(f, proj_freq=False)
        expected = [1.0, -152.99]
        for i, val in enumerate(result.electronic_energies[0]):
            assert np.isclose(val, expected[i]), f"expected {expected[i]}, "
            f"got {val}, point {i}; test #1"
        assert result.zpve_energies_forward is None, f"expected {None}, "
        f"got {result.zpve_energies_forward}; test #2"
        assert result.zpve_energies_reverse is None, f"expected {None}, "
        f"got {result.zpve_energies_reverse}; test #3"

    def test_proj_freq_true_calls_freq_extraction(self, tmp_path):
        """
        Verify that frequency extraction is called twice
        and ZPVE data is returned when projected frequencies are enabled
        """
        f = tmp_path / "irc.out"
        f.write_text("")
        match_mock = MagicMock()
        match_mock.group.return_value = "-153.0"
        match_mock.end.return_value = 0
        with self._base_patches(f), patch(GAUSS_PATTERNS) as gp, \
             patch(FUNC_PATH + "._gauss_freq_extraction_irc", return_value=[(1.0, 0.03)]) as freq_ext:
            gp.PATTERN_TS_ENERGY_IRC.search.return_value = match_mock
            gp.PATTERN_EL_ENERGY_IRC.findall.return_value = [("0.01", "1.0")]
            result = reading_gauss_irc(f, proj_freq=True)
        assert freq_ext.call_count == 2, f"expected {2}, got {freq_ext.call_count}; test #1"
        # The function is called twice: forward and reverse ZPVEs
        assert result.zpve_energies_forward is not None, f"expected not {None}, "
        f"got {result.zpve_energies_forward}; test #2"
        assert result.zpve_energies_forward is not None, f"expected not {None}, "
        f"got {result.zpve_energies_reverse}; test #3"

    def test_returns_irc_data_object(self, tmp_path): 
        """ Verify that the function returns an IRCData object """
        f = tmp_path / "irc.out"
        f.write_text("")
        match_mock = MagicMock()
        match_mock.group.return_value = "-153.0"
        match_mock.end.return_value = 0
        with self._base_patches(f), \
            patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_TS_ENERGY_IRC.search.return_value = match_mock
            gp.PATTERN_EL_ENERGY_IRC.findall.return_value = [("0.01", "1.0")]
            result = reading_gauss_irc(f, proj_freq=False)
        assert isinstance(result, IRCData), f"expected {IRCData}, got {result}" 

    def test_raises_when_ts_energy_block_not_found(self, tmp_path):
        """ Verify that missing transition state block raises an error. Expecting a ValueError """
        f = tmp_path / "irc.out"
        f.write_text("")
        with self._base_patches(f), patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_TS_ENERGY_IRC.search.return_value = None
            with pytest.raises(ValueError, match="transition state energy"):
                reading_gauss_irc(f)

    def test_raises_when_ts_energy_value_is_missing(self, tmp_path):
        """ Verify that missing transition state energy value raises an error. Expecting a ValueError """
        f = tmp_path / "irc.out"
        f.write_text("")
        with self._base_patches(f), patch(GAUSS_PATTERNS) as gp:
            match_mock = MagicMock()
            match_mock.group.return_value = None
            gp.PATTERN_TS_ENERGY_IRC.search.return_value = match_mock
            with pytest.raises(ValueError, match="transition state energy"):
                reading_gauss_irc(f)
            match_mock.group.assert_called_once_with(1) # it should be called once


# ====================================================================================
# --- _gauss_collect_freq_and_modes_tunnex - normal behaviour and input validation ---
# ====================================================================================
class TestGaussCollectFreqAndModesTunnex:

    def test_returns_expected_values(self, tmp_path):
        """ Verify that the function returns ZPVE, frequencies, modes, and atom masses """
        f = tmp_path / "mol.out"
        f.write_text("placeholder")
        atom_masses = np.array([1.008, 12.0])
        opt_structure = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]])
        hessian = np.eye(6)
        zpve = 0.123
        vibs = np.array([1000.0, 2000.0])
        modes = np.array([[0.1, 0.2], [0.3, 0.4]])
        with patch(FUNC_PATH + "._gauss_mass_extraction", return_value=atom_masses) as mass_mock, \
             patch(FUNC_PATH + ".gauss_extract_geom_from_opt_file", return_value=opt_structure) as geom_mock, \
             patch(FUNC_PATH + ".gauss_hessian_reader", return_value=hessian) as hessian_mock, \
             patch(FUNC_PATH + ".freq_and_modes_extraction_tunnex", return_value=(
             zpve,vibs, modes)) as extraction_mock:
             result = _gauss_collect_freq_and_modes_tunnex(f)
        assert result[0] == zpve, f"expected {zpve}, got {result[0]}; test #1"
        assert np.array_equal(result[1], vibs), f"expected {vibs}, got {result[1]}; test #2"
        assert np.array_equal(result[2], modes), f"expected {modes}, got {result[2]}; test #3"
        assert np.array_equal(result[3], atom_masses), f"expected {atom_masses}, got {result[3]}; test #4"
        mass_mock.assert_called_once_with(f) # it should be called once
        geom_mock.assert_called_once_with(f, "species", as_array=True) # it should be called once
        hessian_mock.assert_called_once_with(f, "opt") # it should be called once
        extraction_mock.assert_called_once_with(opt_structure, atom_masses, hessian) # it should be called once


# ===========================================================================
# --- reading_gauss_struct_file - normal behaviour and input validation ---
# ===========================================================================
class TestReadingGaussStructFile:

    def test_mode_positions(self, tmp_path):
        """ Test that the mode position is set correctly for each calculation mode """
        for mode, expected_pos in [("ts", 0.0), ("react", float("-inf")), ("prod", float("inf"))]:
            f = tmp_path / f"{mode}.out"
            el_mock = MagicMock()
            el_mock.group.return_value = "-153.0"
            content = "SCF Done: E = -153.0\nFrequencies -- 1000.0\n- Thermochemistry -\n"
            f.write_text(content)
            with patch(FUNC_PATH + ".file_check", return_value=f), \
            patch(GAUSS_PATTERNS) as gp:
                call_count = [0]
                def side_effect(line):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return el_mock
                    return None
                gp.PATTERN_EL_ENERGY_SINGLE_POINT.search.side_effect = side_effect
                gp.PATTERN_FREQ_MODES_LINES.findall.return_value = ["1000.0"]
                el_energy, _ = reading_gauss_struct_file(f, mode=mode)
            if np.isinf(expected_pos):
                result = np.isinf(el_energy[0]) and np.copysign(1, el_energy[0]) == np.copysign(1, expected_pos)
                assert result, f"expected {expected_pos}, got {el_energy[0]} for {mode} : {expected_pos}"
            else:
                result = np.isclose(el_energy[0], expected_pos)
                assert result, f"expected {expected_pos}, got {el_energy[0]} for {mode} : {expected_pos}"

    def test_zpve_computed_as_half_sum_of_freqs(self, tmp_path):
        """ Test that ZPVE is computed as half the sum of vibrational frequencies """
        f = tmp_path / "mol.out"
        f.write_text("SCF\nFrequencies -- 1000.0 2000.0\n- Thermochemistry -\n")
        el_mock = MagicMock()
        el_mock.group.return_value = "-153.0"
        with patch(FUNC_PATH + ".file_check", return_value=f), \
             patch(GAUSS_PATTERNS) as gp:
            call_n = [0]
            def el_side(line):
                call_n[0] += 1
                return el_mock if call_n[0] == 1 else None
            gp.PATTERN_EL_ENERGY_SINGLE_POINT.search.side_effect = el_side
            gp.PATTERN_FREQ_MODES_LINES.findall.side_effect = lambda l: [
            "1000.0", "2000.0"] if "Frequencies" in l else []
            el_e, zpve = reading_gauss_struct_file(f, mode="ts")
        expected = 0.5 * 3000.0 * CM_M1_TO_HARTREE
        assert np.isclose(zpve[1], expected), f"expected {expected}, got {zpve[1]}; test #1"
        assert len(el_e) == 2, f"expected {2}, got {len(el_e)}; test #2"
        assert len(zpve) == 2, f"expected {2}, got {len(zpve)}; test #3"

    def test_hess_parsing_true_uses_hessian_zpve(self, tmp_path):
        """ Test that Hessian-based parsing uses the ZPVE obtained from the Hessian """
        f = tmp_path / "mol.out"
        f.write_text("SCF\nFrequencies -- 1000.0\n- Thermochemistry -\n")
        el_mock = MagicMock()
        el_mock.group.return_value = "-153.0"
        with patch(FUNC_PATH + ".file_check", return_value=f), \
             patch(GAUSS_PATTERNS) as gp, \
             patch(FUNC_PATH + "._gauss_collect_freq_and_modes_tunnex",
                   return_value=(0.099, None, None, None)) as hess_mock:
            call_n = [0]
            def el_side(line):
                call_n[0] += 1
                return el_mock if call_n[0] == 1 else None
            gp.PATTERN_EL_ENERGY_SINGLE_POINT.search.side_effect = el_side
            gp.PATTERN_FREQ_MODES_LINES.findall.side_effect = lambda l: ["1000.0"] if "Frequencies" in l else []
            el_e, zpve = reading_gauss_struct_file(f, mode="ts", hess_parsing=True)
        assert np.isclose(zpve[1], 0.099), f"expected {0.099}, got {zpve[1]}; test #1"
        assert len(el_e) == 2, f"expected {2}, got {len(el_e)}; test #2"
        assert len(zpve) == 2, f"expected {2}, got {len(zpve)}; test #3"
        hess_mock.assert_called_once() # it should be called once
    
    def test_invalid_mode_raises(self, tmp_path):
        """ Test that an invalid calculation mode raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        f.write_text("")
        with patch(FUNC_PATH + ".file_check", return_value=f):
            with pytest.raises(ValueError, match="incorrect"):
                reading_gauss_struct_file(f, mode="minimum")

    def test_missing_el_energy_raises(self, tmp_path):
        """ Test that missing electronic energy raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        f.write_text("Frequencies -- 1000.0\n- Thermochemistry -\n")
        with patch(FUNC_PATH + ".file_check", return_value=f), \
        patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_EL_ENERGY_SINGLE_POINT.search.return_value = None
            gp.PATTERN_FREQ_MODES_LINES.findall.return_value = ["1000.0"]
            with pytest.raises(ValueError, match="electronic energy"):
                reading_gauss_struct_file(f, mode="ts")

    def test_missing_zpve_raises(self, tmp_path):
        """ Test that missing ZPVE raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        f.write_text("SCF Done: E = -153.0\n")
        match_mock = MagicMock()
        match_mock.group.return_value = "-153.0"
        with patch(FUNC_PATH + ".file_check", return_value=f), \
        patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_EL_ENERGY_SINGLE_POINT.search.side_effect = [match_mock] + [None] * 100
            gp.PATTERN_FREQ_MODES_LINES.findall.return_value = []
            with pytest.raises(ValueError, match="ZPVE"):
                reading_gauss_struct_file(f, mode="ts")

    def test_zero_freq_sum_raises(self, tmp_path):
        """ Test that zero frequency sum raises an error. Expecting a ValueError """
        lines = [ "SCF Done: E = -153.0\n", "- Thermochemistry -\n"]
        f = tmp_path / "mol.out"
        f.write_text("".join(lines))
        el_mock = MagicMock()
        el_mock.group.return_value = "-153.0"
        with patch(FUNC_PATH + ".file_check", return_value=f), \
            patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_EL_ENERGY_SINGLE_POINT.search.side_effect = [el_mock, None, None]
            gp.PATTERN_FREQ_MODES_LINES.findall.return_value = []
            with pytest.raises(ValueError, match="No frequencies"):
                reading_gauss_struct_file(f, mode="ts")


# ============================================================================
# --- gauss_mode_coordinate_reader - normal behaviour and input validation ---
# ============================================================================
class TestGaussModeCoordinateReader:

    def nums_side_effect(self, line):
        """ Should return a mock frequency value and three normal mode coordinates """
        if "Frequencies" in line: return ["1000.0"]
        if "0.1" in line: return ["0.1", "0.2", "0.3"]
        return []

    def nums_side_effect_missing_coord(self, line):
        """ Should return a mock frequency value and two normal mode coordinates (one is missing) """
        if "Frequencies" in line: return ["1000.0"]
        if "0.1" in line: return ["0.1", "0.2"]
        return []

    def test_file_check_called_with_path(self, tmp_path):
        """ Test that file_check is called once with a Path object """
        f = tmp_path / "mol.out"
        f.write_text("")
        with patch(FUNC_PATH + ".file_check", return_value=f) as mock_fc, \
             patch(GAUSS_PATTERNS) as gp, \
             patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([1.0])) as gme:
            block_mock = MagicMock()
            block_mock.group.return_value = ""
            gp.PATTERN_FREQ_MODES_BLOCK.search.return_value = block_mock
            # Path input
            try:
                gauss_mode_coordinate_reader(f)
            except ValueError:
                pass
            mock_fc.assert_called_once_with(f) # it should be called once
            gme.assert_called_once_with(f) # it should be called once

    def test_file_check_called_with_str(self, tmp_path):
        """ Test that file_check is called once with a string path """
        f = tmp_path / "mol.out"
        f.write_text("")
        with patch(FUNC_PATH + ".file_check", return_value=f) as mock_fc, \
             patch(GAUSS_PATTERNS) as gp, \
             patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([1.0])) as gme:
            block_mock = MagicMock()
            block_mock.group.return_value = ""
            gp.PATTERN_FREQ_MODES_BLOCK.search.return_value = block_mock
            try:
                gauss_mode_coordinate_reader(str(f))
            except ValueError:
                pass
            mock_fc.assert_called_once_with(str(f)) # it should be called once
            gme.assert_called_once_with(f) # it should be called once

    def test_returns_tuple_of_list_and_tuple(self, tmp_path):
        """ Test that the function returns normal mode data as a list and atomic masses as a tuple """
        f = tmp_path / "mol.out"
        f.write_text("block")
        block_text = (
            "                     \n"
            "Frequencies -- 1000.0\n"
            "Atom  AN  X  Y  Z\n"
            "1  6  0.1  0.2  0.3\n"
            "\n")
        block_mock = MagicMock()
        block_mock.group.return_value = block_text
        with patch(GAUSS_PATTERNS) as gp, \
             patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([12.0])):
            gp.PATTERN_FREQ_MODES_BLOCK.search.return_value = block_mock
            gp.PATTERN_FREQ_MODES_LINES.findall.side_effect = self.nums_side_effect
            result = gauss_mode_coordinate_reader(f, hess_parsing=False)
        freq_and_modes, masses = result
        assert isinstance(freq_and_modes, list), f"expected {list}, got {type(freq_and_modes)}; test #1"
        expected = [[1000.0, 0.1, 0.2, 0.3]]
        assert freq_and_modes == expected, f"expected {expected}, got {freq_and_modes}; test #2"
        assert isinstance(masses, tuple), f"expected {tuple}, got {type(masses)}; test #3"
        assert masses == (12.0,), f"expected {(12.0,)}, got {masses}; test #4"

    def test_hess_parsing_calls_hessian_functions_once(self, tmp_path):
        """ Test that Hessian parsing calls the Hessian parser and format transformer once """
        f = tmp_path / "mol.out"
        f.write_text("hessian data")
        expected_data = (None,
            (1000.0, 2000.0),
            [[[0.1, 0.2, 0.3]]],
            np.array([1.0, 12.0]))
        expected_result = [[1000.0, 0.1, 0.2, 0.3]]
        with patch(FUNC_PATH + "._gauss_collect_freq_and_modes_tunnex", return_value=expected_data) as collect_mock, \
            patch(FUNC_PATH + ".vibrations_format_transform", return_value=expected_result) as transform_mock:
            result = gauss_mode_coordinate_reader(f, hess_parsing=True)
        collect_mock.assert_called_once_with(f, "species")
        transform_mock.assert_called_once_with(f, expected_data[1], expected_data[2], expected_data[3], 1e-2)
        assert result == (expected_result, tuple(expected_data[3]))

    def test_missing_normal_vectors_block_raises(self, tmp_path):
        """Test that missing normal mode vectors block raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        f.write_text("no normal coords here")
        with patch(FUNC_PATH + ".file_check", return_value=f), patch(GAUSS_PATTERNS) as gp, \
             patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([1.0, 12.0])):
            gp.PATTERN_FREQ_MODES_BLOCK.search.return_value = None
            with pytest.raises(ValueError, match="normal mode vectors block"):
                gauss_mode_coordinate_reader(f, hess_parsing=False)

    def test_missing_frequencies_raises(self, tmp_path):
        """Test that missing frequencies raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        f.write_text("no frquencies here")
        with patch(FUNC_PATH + ".file_check", return_value=f), patch(GAUSS_PATTERNS) as gp, \
             patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([1.0, 12.0])):
            block_mock = MagicMock()
            block_mock.group.return_value = "Frequencies --"
            gp.PATTERN_FREQ_MODES_BLOCK.search.return_value = block_mock
            gp.PATTERN_FREQ_MODES_LINES.findall.return_value = []
            with pytest.raises(ValueError, match="Frequencies were not found"):
                gauss_mode_coordinate_reader(f, hess_parsing=False)

    def test_missing_normal_vectors_table_raises(self, tmp_path):
        """Test that a missing normal mode vector table raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        f.write_text("block")
        block_mock = MagicMock()
        block_mock.group.return_value = "Frequencies -- 1000.0\nno atom table\n"
        with patch(FUNC_PATH + ".file_check", return_value=f), patch(GAUSS_PATTERNS) as gp, \
             patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([1.0])):
            gp.PATTERN_FREQ_MODES_BLOCK.search.return_value = block_mock
            gp.PATTERN_FREQ_MODES_LINES.findall.side_effect = lambda line: ["1000.0"] if "Frequencies" in line else []
            with pytest.raises(ValueError, match="Normal mode vector table"):
                gauss_mode_coordinate_reader(f, hess_parsing=False)

    def test_coordinate_count_mismatch_raises(self, tmp_path):
        """Test that a mismatch in the number of normal mode coordinates raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        f.write_text("block")
        # for 1 frequency 3 coords per atom are expected; supplying 2 coords means a mismatch
        block_text = "Frequencies -- 1000.0\nAtom  AN  X  Y  Z\n1  6  0.1  0.2\n\n"
        block_mock = MagicMock()
        block_mock.group.return_value = block_text
        with patch(GAUSS_PATTERNS) as gp, \
             patch(FUNC_PATH + "._gauss_mass_extraction",
                   return_value=np.array([12.0])):
            gp.PATTERN_FREQ_MODES_BLOCK.search.return_value = block_mock
            gp.PATTERN_FREQ_MODES_LINES.findall.side_effect = self.nums_side_effect_missing_coord
            with pytest.raises(ValueError, match="Expected"):
                gauss_mode_coordinate_reader(f, hess_parsing=False)

    def test_missing_normal_vectors_raises(self, tmp_path):
        """ Test that missing normal mode vectors raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        f.write_text("block")
        block_mock = MagicMock()
        block_mock.group.return_value = "Frequencies -- 1000.0\nAtom  AN  X  Y  Z\n"
        with patch(FUNC_PATH + ".file_check", return_value=f), patch(GAUSS_PATTERNS) as gp, \
             patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([1.0])):
            gp.PATTERN_FREQ_MODES_BLOCK.search.return_value = block_mock
            gp.PATTERN_FREQ_MODES_LINES.findall.side_effect = lambda line: ["1000.0"] if "Frequencies" in line else []
            with pytest.raises(ValueError, match="Normal mode vectors"):
                gauss_mode_coordinate_reader(f, hess_parsing=False)
    
    def test_missing_frequencies_and_modes_raises(self, tmp_path):
        """ Test that missing frequencies and normal modes raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        f.write_text("no frquencies and modes here")
        match_mock = MagicMock()
        match_mock.group.return_value = ""
        with patch(FUNC_PATH + ".file_check", return_value=f), \
            patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([1.0])), \
            patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_FREQ_MODES_BLOCK.search.return_value = match_mock
            with pytest.raises(ValueError, match="Frequencies and modes"):
                gauss_mode_coordinate_reader(f)


# ============================================================================
# --- _gauss_struct_irc_direction - normal behaviour and input validation ---
# ============================================================================
class TestGaussStructIRCDirection:

    def test_forward_sign_positive(self, tmp_path):
        """ Test that the forward IRC direction preserves a positive coordinate """
        f = tmp_path / "irc.out"
        f.write_text("")
        block_mock = MagicMock()
        block_mock.group.return_value = "block"
        coord_mock = MagicMock()
        coord_mock.group.return_value = "1.5"
        geom_matches = [("C", "0.0", "0.0", "0.0")]
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer.return_value = iter([block_mock])
            gp.PATTERN_NET_REACTION_COORDINATE.search.return_value = coord_mock
            gp.PATTERN_IRC_CURRENT_STRUCTURE_LINE.findall.return_value = geom_matches
            irc_coords, _ = _gauss_struct_irc_direction(f, "text", forward_key=True)
        assert np.isclose(irc_coords[0], 1.5), f"expected {1.5}, got {irc_coords[0]}"

    def test_reverse_sign_negative(self, tmp_path):
        """ Test that the reverse IRC direction negates the coordinate """
        f = tmp_path / "irc.out"
        f.write_text("")
        block_mock = MagicMock()
        block_mock.group.return_value = "block"
        coord_mock = MagicMock()
        coord_mock.group.return_value = "1.5"
        geom_matches = [("C", "0.0", "0.0", "0.0")]
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer.return_value = iter([block_mock])
            gp.PATTERN_NET_REACTION_COORDINATE.search.return_value = coord_mock
            gp.PATTERN_IRC_CURRENT_STRUCTURE_LINE.findall.return_value = geom_matches
            irc_coords, _ = _gauss_struct_irc_direction(f, "text", forward_key=False)
        assert np.isclose(irc_coords[0], -1.5), f"expected {-1.5}, got {irc_coords[0]}"

    def test_geometry_stored_as_numpy_array(self, tmp_path):
        """ Test that the geometry is stored as a NumPy array with the expected shape """
        f = tmp_path / "irc.out"
        f.write_text("")
        block_mock = MagicMock()
        block_mock.group.return_value = "block"
        coord_mock = MagicMock()
        coord_mock.group.return_value = "1.0"
        geom_matches = [("C", "1.0", "2.0", "3.0"), ("H", "4.0", "5.0", "6.0")]
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer.return_value = iter([block_mock])
            gp.PATTERN_NET_REACTION_COORDINATE.search.return_value = coord_mock
            gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.return_value = geom_matches
            _, geoms = _gauss_struct_irc_direction(f, "text")
        result = isinstance(geoms[0], np.ndarray)
        assert result, f"expected {np.ndarray}, got {type(geoms[0])}; test #1"
        assert geoms[0].shape == (2, 3), f"expected {(2, 3)}, got {geoms[0].shape}; test #2"

    def test_multiple_structure_blocks_are_parsed(self, tmp_path):
        """ Test that multiple IRC structure blocks are parsed correctly """
        f = tmp_path / "irc.out"
        f.write_text("")
        block1 = MagicMock()
        block1.group.return_value = "block1"
        block2 = MagicMock()
        block2.group.return_value = "block2"
        coord1 = MagicMock()
        coord1.group.return_value = "1.0"
        coord2 = MagicMock()
        coord2.group.return_value = "2.0"
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer.return_value = iter([block1, block2])
            gp.PATTERN_NET_REACTION_COORDINATE.search.side_effect = [coord1, coord2]
            gp.PATTERN_IRC_CURRENT_STRUCTURE_LINE.findall.side_effect = [[("C", "0.0", "0.0", "0.0")],
                                                                         [("C", "1.0", "1.0", "1.0")]]
            irc_coords, geometries = _gauss_struct_irc_direction(f, "text")
        assert irc_coords == [1.0, 2.0], f"expected {[1.0, 2.0]}, got {irc_coords}; test #1"
        assert len(geometries) == 2, f"expected {2}, got {len(geometries)}; test #2"

    def test_no_structure_blocks_raises(self, tmp_path):
        """ Test that missing structure blocks raise an error. Expecting a ValueError """
        f = tmp_path / "irc.out"
        f.write_text("")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer.return_value = iter([])
            with pytest.raises(ValueError, match="geometry blocks"):
                _gauss_struct_irc_direction(f, "text")

    def test_missing_irc_coordinate_raises(self, tmp_path):
        """ Test that a missing IRC coordinate raises an error. Expecting a ValueError """
        f = tmp_path / "irc.out"
        f.write_text("")
        block_mock = MagicMock()
        block_mock.group.return_value = "block"
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer.return_value = iter([block_mock])
            gp.PATTERN_NET_REACTION_COORDINATE.search.return_value = None
            with pytest.raises(ValueError, match="IRC coordinate"):
                _gauss_struct_irc_direction(f, "text")

    def test_missing_geometry_raises(self, tmp_path):
        """ Test that missing geometry raises an error. Expecting a ValueError """
        f = tmp_path / "irc.out"
        f.write_text("")
        block_mock = MagicMock()
        block_mock.group.return_value = "block"
        coord_mock = MagicMock()
        coord_mock.group.return_value = "1.5"
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer.return_value = iter([block_mock])
            gp.PATTERN_NET_REACTION_COORDINATE.search.return_value = coord_mock
            gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.return_value = []
            with pytest.raises(ValueError, match="geometry of the IRC point"):
                _gauss_struct_irc_direction(f, "text")


# ===========================================================================
# --- _gauss_struct_extraction - normal behaviour and input validation ---
# ===========================================================================
class TestGaussStructExtraction:

    def test_extracts_ts_forward_and_reverse_structures(self, tmp_path):
        """ Tests extraction of the transition state, forward, and reverse IRC structures """
        f = tmp_path / "irc.out"
        f.write_text("")
        ts_block = MagicMock()
        ts_block.group.return_value = "ts block"
        ts_geom_matches = [("C", "0.0", "0.1", "0.2"), ("H", "1.0", "1.1", "1.2")]
        forward_coords = [0.5, 1.0]
        forward_geometries = [np.array([[2.0, 2.1, 2.2]]), np.array([[3.0, 3.1, 3.2]])]
        reverse_coords = [-0.5, -1.0]
        reverse_geometries = [np.array([[4.0, 4.1, 4.2]]), np.array([[5.0, 5.1, 5.2]])]
        with patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("forward", "reverse")), \
             patch(GAUSS_PATTERNS) as gp, \
             patch(FUNC_PATH + "._gauss_struct_irc_direction",
                side_effect=[(forward_coords, forward_geometries),
                             (reverse_coords, reverse_geometries)]):
            gp.PATTERN_IRC_INPUT_ORIENTATION_BLOCK.search.return_value = ts_block
            gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.return_value = ts_geom_matches
            result = _gauss_struct_extraction(f)
        assert result[0][0] == 0.0, f"expected {0.0}, got {result[0][0]}; test #1"
        expected_array = np.array([[0.0, 0.1, 0.2],[1.0, 1.1, 1.2]])
        assert np.array_equal(result[0][1], expected_array), f"expected {expected_array}, "
        f"got {result[0][1]}; test #2"
        expected_irc_coord = [0.0, 0.5, 1.0, -0.5, -1.0]
        result_irc_coord = [coord for coord, _ in result]
        assert result_irc_coord == expected_irc_coord, f"expected {expected_irc_coord}, "
        f"got {result_irc_coord}; test #3"

    def test_calls_irc_direction_for_forward_and_reverse(self, tmp_path):
        """ Tests that IRC structures are extracted for both forward and reverse directions """
        f = tmp_path / "irc.out"
        f.write_text("")
        ts_block = MagicMock()
        ts_block.group.return_value = "ts block"
        with patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("forward text", "reverse text")), \
            patch(GAUSS_PATTERNS) as gp, \
            patch(FUNC_PATH + "._gauss_struct_irc_direction", return_value=([], [])) as direction_mock:
            gp.PATTERN_INPUT_ORIENTATION_BLOCK.search.return_value = ts_block
            gp.PATTERN_INPUT_ORIENTATION_LINES.findall.return_value = [("C", "0.0", "0.0", "0.0")]
            _gauss_struct_extraction(f)
        assert direction_mock.call_count == 2, f"expected {2}, got {direction_mock.call_count}"
        direction_mock.assert_any_call(f, "forward text", forward_key=True) # it should be called
        direction_mock.assert_any_call(f, "reverse text", forward_key=False) # it should be called

    def test_ts_geometry_is_converted_to_numpy_array(self, tmp_path):
        """ Tests that the transition state geometry is converted to a NumPy array """
        f = tmp_path / "irc.out"
        f.write_text("")
        ts_block = MagicMock()
        ts_block.group.return_value = "ts block"
        geom_matches = [("C", "1.0", "2.0", "3.0"), ("H", "4.0", "5.0", "6.0")]
        with patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("forward", "reverse")), \
            patch(GAUSS_PATTERNS) as gp, \
            patch(FUNC_PATH + "._gauss_struct_irc_direction", return_value=([], [])):
            gp.PATTERN_IRC_INPUT_ORIENTATION_BLOCK.search.return_value = ts_block
            gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.return_value = geom_matches
            result = _gauss_struct_extraction(f)
        ts_geom = result[0][1]
        assert isinstance(ts_geom, np.ndarray), f"expected {np.ndarray}, got {type(ts_geom)}; test #1"
        assert ts_geom.shape == (2, 3), f"expected {(2, 3)}, got {ts_geom.shape}; test #2"
        expected_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert np.array_equal(ts_geom, expected_array), f"expected {expected_array}, "
        f"got {ts_geom}; test #3"
    
    def test_missing_ts_geometry_block_raises(self, tmp_path):
        """ Tests that a missing transition state geometry block raises an error. Expecting a ValueError """
        f = tmp_path / "irc.out"
        f.write_text("")
        with patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("forward", "reverse")), \
            patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_INPUT_ORIENTATION_BLOCK.search.return_value = None
            with pytest.raises(ValueError, match="geometry block of the IRC point"):
                _gauss_struct_extraction(f)

    def test_missing_ts_geometry_raises(self, tmp_path):
        """ Tests that a missing transition state geometry raises an error. Expecting a ValueError """
        f = tmp_path / "irc.out"
        f.write_text("")
        ts_block = MagicMock()
        ts_block.group.return_value = "ts block"
        with patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("forward", "reverse")), \
            patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_IRC_INPUT_ORIENTATION_BLOCK.search.return_value = ts_block
            gp.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall.return_value = []
            with pytest.raises(ValueError, match="geometry of the IRC point"):
                _gauss_struct_extraction(f)


# ===========================================================================
# --- reading_gauss_irc_tunnex - normal behaviour and input validation ---
# ===========================================================================
class TestReadingGaussIRCTunnex:

    def _make_ts_match(self):
        """ Should creates a mock transition state energy match """
        m = MagicMock()
        m.group.return_value = "-153.0"
        m.end.return_value = 0
        return m

    def test_subfunctions_calls_made(self, tmp_path):
        """ Tests that all required helper functions are called once """
        irc = tmp_path / "irc.out"
        ts  = tmp_path / "ts.out"
        irc.write_text("")
        ts.write_text("")
        with patch(FUNC_PATH + ".file_check", side_effect=[irc, ts]) as mock_fc, \
             patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([1.0, 12.0])) as gme, \
             patch(FUNC_PATH + "._gauss_struct_extraction", return_value=[(0.0, np.zeros((2, 3)))]) as gse, \
             patch(FUNC_PATH + ".irc_steps_comp", return_value=np.array([0.1])) as ise, \
             patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("fwd", "rev")) as ifs, \
             patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_TS_ENERGY_IRC.search.return_value = self._make_ts_match()
            gp.PATTERN_EL_ENERGY_IRC.findall.return_value = [("-0.01", "0.0"), ("-0.02", "0.1")]
            reading_gauss_irc_tunnex(irc, ts, proj_freq=False)
        assert mock_fc.call_count == 2, f"expected {2}, got {mock_fc.call_count}"
        mock_fc.assert_any_call(irc) # it should be called at least once
        mock_fc.assert_any_call(ts) # it should be called at least once
        gme.assert_called_once() # it should be called once
        gse.assert_called_once() # it should be called once
        ise.assert_called_once() # it should be called once
        ifs.assert_called_once() # it should be called once
    
    def test_ts_at_irc_zero_after_shift(self, tmp_path):
        """ Tests that the transition state is shifted to IRC = 0.0 """
        irc = tmp_path / "irc.out"
        ts  = tmp_path / "ts.out"
        irc.write_text("")
        ts.write_text("")
        with patch(FUNC_PATH + ".file_check", side_effect=[irc, ts]), \
             patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([1.0])), \
             patch(FUNC_PATH + "._gauss_struct_extraction", return_value=[(0.0, np.zeros((1, 3))),
                                                                          (0.5, np.ones((1, 3))),
                                                                          (-0.5, np.ones((1, 3)) * 2)]), \
             patch(FUNC_PATH + ".irc_steps_comp", return_value=np.array([0.5, 0.5])), \
             patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("fwd", "rev")), \
             patch(GAUSS_PATTERNS) as gp:
             m = self._make_ts_match()
             m.group.return_value = "-100.0"
             gp.PATTERN_TS_ENERGY_IRC.search.return_value = m
             gp.PATTERN_EL_ENERGY_IRC.findall.return_value = [ ("0.0",   "0.0"), ("-0.05", "0.5"), ("-0.03", "-0.5")]
             result = reading_gauss_irc_tunnex(irc, ts, proj_freq=False)
        max_irc = max(result.electronic_energies, key=lambda x: x[1])[0]
        assert np.isclose(max_irc, 0.0, atol=1e-12), f"expected irc_ts == {0.0}, got {max_irc}; test #1"
        assert isinstance(result, IRCData), f"expected {IRCData}, got {type(result)}; test #2"

    def test_proj_freq_false_returns_none_zpves(self, tmp_path):
        """ Tests that ZPVE values are None when projected frequencies are disabled """
        irc = tmp_path / "irc.out"
        ts  = tmp_path / "ts.out"
        irc.write_text("")
        ts.write_text("")
        with patch(FUNC_PATH + ".file_check", side_effect=[irc, ts]), \
             patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([1.0])), \
             patch(FUNC_PATH + "._gauss_struct_extraction", return_value=[(0.0, np.zeros((1, 3)))]), \
             patch(FUNC_PATH + ".irc_steps_comp", return_value=np.array([])), \
             patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("fwd", "rev")), \
             patch(GAUSS_PATTERNS) as gp:
             m = self._make_ts_match()
             gp.PATTERN_TS_ENERGY_IRC.search.return_value = m
             gp.PATTERN_EL_ENERGY_IRC.findall.return_value = [("0.0", "0.0")]
             result = reading_gauss_irc_tunnex(irc, ts, proj_freq=False)
        assert result.zpve_energies_forward is None, f"expected {None}, got {result.zpve_energies_forward}; test #1"
        assert result.zpve_energies_reverse is None, f"expected {None}, got {result.zpve_energies_reverse}; test #2"

    def test_proj_freq_true_returns_zpves(self, tmp_path):
        """ Tests that projected ZPVEs are returned for both IRC directions """
        f = tmp_path / "irc.out"
        ts = tmp_path / "ts.out"
        f.write_text("")
        ts.write_text("")
        with patch(FUNC_PATH + ".file_check", side_effect=[f, ts]), \
            patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([1.0])), \
            patch(FUNC_PATH + "._gauss_struct_extraction", return_value=[
                    (-0.5, np.zeros((1, 3))),
                    (0.0, np.zeros((1, 3))),
                    (0.5, np.zeros((1, 3)))]), \
            patch(FUNC_PATH + ".irc_steps_comp", return_value=np.array([0.5, 0.5])), \
            patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("fwd", "rev")), \
            patch(FUNC_PATH + ".gauss_collect_proj_zpves", return_value=[0.01, 0.02, 0.03]), \
            patch(GAUSS_PATTERNS) as gp:
            match_mock = self._make_ts_match()
            gp.PATTERN_TS_ENERGY_IRC.search.return_value = match_mock
            gp.PATTERN_EL_ENERGY_IRC.findall.return_value = [
                ("-0.02", "-0.5"),
                ("0.0", "0.0"),
                ("-0.01", "0.5")]
            result = reading_gauss_irc_tunnex(f, ts, proj_freq=True)
        assert result.zpve_energies_forward is not None, f"expected not {None}, "
        f"got {result.zpve_energies_forward}; test #1"
        assert result.zpve_energies_reverse is not None, f"expected not {None}, "
        f"got {result.zpve_energies_reverse}; test #1"
        assert len(result.zpve_energies_forward) == 1, f"expected {1}, got {result.zpve_energies_forward}; test #3"
        assert len(result.zpve_energies_reverse) == 1, f"expected {1}, got {result.zpve_energies_reverse}; test #4"

    def test_proj_freq_calls_zpve_function_once(self, tmp_path):
        """ Tests that the ZPVE collection function is called once """
        f = tmp_path / "irc.out"
        ts = tmp_path / "ts.out"
        f.write_text("")
        ts.write_text("")
        structures = [(0.0, np.zeros((1, 3)))]
        masses = np.array([1.0])
        zpves = [0.01]
        with patch(FUNC_PATH + ".file_check", side_effect=[f, ts]), \
            patch(FUNC_PATH + "._gauss_mass_extraction", return_value=masses), \
            patch(FUNC_PATH + "._gauss_struct_extraction", return_value=structures), \
            patch(FUNC_PATH + ".irc_steps_comp", return_value=np.array([])), \
            patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("fwd", "rev")), \
            patch(FUNC_PATH + ".gauss_collect_proj_zpves", return_value=zpves) as zpve_mock, \
            patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_TS_ENERGY_IRC.search.return_value = self._make_ts_match()
            gp.PATTERN_EL_ENERGY_IRC.findall.return_value = [("0.0", "0.0")]
            reading_gauss_irc_tunnex(f, ts, proj_freq=True)
        zpve_mock.assert_called_once_with(f, structures, masses) # it should be called once

    def test_zpve_forward_contains_only_positive_irc(self, tmp_path):
        """ Tests that forward ZPVE values contain only positive IRC coordinates """
        f = tmp_path / "irc.out"
        ts = tmp_path / "ts.out"
        f.write_text("")
        ts.write_text("")
        structures = [(-0.5, np.zeros((1, 3))), (0.0, np.zeros((1, 3))), (0.5, np.zeros((1, 3)))]
        with patch(FUNC_PATH + ".file_check", side_effect=[f, ts]), \
            patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([1.0])), \
            patch(FUNC_PATH + "._gauss_struct_extraction", return_value=structures), \
            patch(FUNC_PATH + ".irc_steps_comp", return_value=np.array([0.5, 0.5])), \
            patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("fwd", "rev")), \
            patch(FUNC_PATH + ".gauss_collect_proj_zpves", return_value=[0.01, 0.02, 0.03]), \
            patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_TS_ENERGY_IRC.search.return_value = self._make_ts_match()
            gp.PATTERN_EL_ENERGY_IRC.findall.return_value = [
                ("-0.02", "-0.5"),
                ("0.0", "0.0"),
                ("-0.01", "0.5")]
            result = reading_gauss_irc_tunnex(f, ts, proj_freq=True)
        result_key = all(irc > 0 for irc, _ in result.zpve_energies_forward)
        assert result_key, f"expected {True}, got {result_key}; test #1"
        assert result.zpve_energies_forward == [(0.5, 0.03)], f"expected {[(0.5, 0.03)]}, "
        f"got {result.zpve_energies_forward}; test #2"

    def test_zpve_reverse_contains_only_negative_irc(self, tmp_path):
        """ Tests that reverse ZPVE values contain only negative IRC coordinates """
        f = tmp_path / "irc.out"
        ts = tmp_path / "ts.out"
        f.write_text("")
        ts.write_text("")
        structures = [
            (-0.5, np.zeros((1, 3))),
            (0.0, np.zeros((1, 3))),
            (0.5, np.zeros((1, 3)))]
        with patch(FUNC_PATH + ".file_check", side_effect=[f, ts]), \
            patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([1.0])), \
            patch(FUNC_PATH + "._gauss_struct_extraction", return_value=structures), \
            patch(FUNC_PATH + ".irc_steps_comp", return_value=np.array([0.5, 0.5])), \
            patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("fwd", "rev")), \
            patch(FUNC_PATH + ".gauss_collect_proj_zpves", return_value=[0.01, 0.02, 0.03]), \
            patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_TS_ENERGY_IRC.search.return_value = self._make_ts_match()
            gp.PATTERN_EL_ENERGY_IRC.findall.return_value = [
                ("-0.02", "-0.5"),
                ("0.0", "0.0"),
                ("-0.01", "0.5")]
            result = reading_gauss_irc_tunnex(f, ts, proj_freq=True)
        result_key = all(irc < 0 for irc, _ in result.zpve_energies_reverse)
        assert result_key, f"expected {True}, got {result_key}; test #1"
        assert result.zpve_energies_reverse == [(-0.5, 0.01)], f"expected {[(-0.5, 0.01)]}, "
        f"got {result.zpve_energies_reverse}; test #2"

    def test_zpve_ts_point_is_excluded_from_both_directions(self, tmp_path):
        """ Tests that the transition state ZPVE is excluded from both directions """
        f = tmp_path / "irc.out"
        ts = tmp_path / "ts.out"
        f.write_text("")
        ts.write_text("")
        structures = [(-0.5, np.zeros((1, 3))), (0.0, np.zeros((1, 3))), (0.5, np.zeros((1, 3)))]
        with patch(FUNC_PATH + ".file_check", side_effect=[f, ts]), \
            patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([1.0])), \
            patch(FUNC_PATH + "._gauss_struct_extraction", return_value=structures), \
            patch(FUNC_PATH + ".irc_steps_comp", return_value=np.array([0.5, 0.5])), \
            patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("fwd", "rev")), \
            patch(FUNC_PATH + ".gauss_collect_proj_zpves", return_value=[0.01, 0.02, 0.03]), \
            patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_TS_ENERGY_IRC.search.return_value = self._make_ts_match()
            gp.PATTERN_EL_ENERGY_IRC.findall.return_value = [
                ("-0.02", "-0.5"),
                ("0.0", "0.0"),
                ("-0.01", "0.5")]
            result = reading_gauss_irc_tunnex(f, ts, proj_freq=True)
        result_key_f = all(irc != 0.0 for irc, _ in result.zpve_energies_forward)
        assert result_key_f, f"expected {True}, got {result_key_f}; test #1"
        result_key_r = all(irc != 0.0 for irc, _ in result.zpve_energies_reverse)
        assert result_key_r, f"expected {True}, got {result_key_r}; test #2"

    def test_zpve_length_mismatch_raises(self, tmp_path):
        """ Tests that a ZPVE and IRC coordinate count mismatch raises an error. Expecting a ValueError """
        irc = tmp_path / "irc.out"
        ts  = tmp_path / "ts.out"
        irc.write_text("")
        ts.write_text("")
        with patch(FUNC_PATH + ".file_check", side_effect=[irc, ts]), \
             patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([1.0])), \
             patch(FUNC_PATH + "._gauss_struct_extraction", return_value=[(0.0, np.zeros((1, 3))),
                                                                          (0.5, np.ones((1, 3)))]), \
             patch(FUNC_PATH + ".irc_steps_comp", return_value=np.array([0.5])), \
             patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("fwd", "rev")), \
             patch(GAUSS_PATTERNS) as gp, \
             patch(FUNC_PATH + ".gauss_collect_proj_zpves", return_value=[0.03]):
             # 1 ZPVE for 2 IRC points is a mismatch
             m = self._make_ts_match()
             gp.PATTERN_TS_ENERGY_IRC.search.return_value = m
             gp.PATTERN_EL_ENERGY_IRC.findall.return_value = [("0.0", "0.0"), ("-0.05", "0.5")]
             with pytest.raises(ValueError, match="must match"):
                reading_gauss_irc_tunnex(irc, ts, proj_freq=True)

    def test_missing_ts_energy_raises(self, tmp_path):
        """ Tests that missing transition state energy raises an error. Expecting a ValueError """
        f = tmp_path / "irc.out"
        ts = tmp_path / "ts.out"
        f.write_text("")
        ts.write_text("")
        with patch(FUNC_PATH + ".file_check", side_effect=[f, ts]), \
            patch(FUNC_PATH + "._gauss_mass_extraction", return_value=np.array([1.0])), \
            patch(FUNC_PATH + "._gauss_struct_extraction", return_value=[(0.0, np.zeros((1, 3)))]), \
            patch(FUNC_PATH + ".irc_steps_comp", return_value=np.array([])), \
            patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("fwd", "rev")), \
            patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_TS_ENERGY_IRC.search.return_value = None
            with pytest.raises(ValueError, match="transition state energy"):
                reading_gauss_irc_tunnex(f, ts, proj_freq=False)