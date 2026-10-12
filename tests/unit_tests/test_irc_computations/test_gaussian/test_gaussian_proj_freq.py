# --- Test: test_gaussian_proj_freq. Unit tests for .\gaussian\gaussian_proj_freq.py
# Run with: pytest test_gaussian_proj_freq.py -v ---

# --- Modules ---
import pytest  # type: ignore
from unittest.mock import patch

import numpy as np


# --- Module to test ---
from tunnex_2.irc_computations.gaussian.gaussian_proj_freq import ( # type: ignore
    _gauss_hessian_values_extract,
    gauss_hessian_reader,
    gauss_collect_proj_zpves)


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.irc_computations.gaussian.gaussian_proj_freq"
GAUSS_PATTERNS = "tunnex_2.irc_computations.gaussian.gaussian_proj_freq.gaussian_patterns"


def _make_file(tmp_path, name, content):
    """ Should write a file with the content (Hessian) """
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return f

def _make_hessian_text(n: int, chunk_size: int = 5) -> str:
    """
    Should build a minimal lower-triangular Hessiantext block
    for an NxN matrix (Gaussian format)
    """
    lines = []
    chunk = chunk_size  # Gaussian prints columns in groups of 5 (default value)
    for col_start in range(0, n, chunk):
        col_end = min(col_start + chunk, n) # column header
        lines.append("  " + "  ".join(str(c + 1) for c in range(col_start, col_end)))
        for row in range(col_start, n):  # lower-triangular
            n_cols = min(row + 1, col_end) - col_start
            if n_cols <= 0:
                continue
            values = " ".join(f"{float(row * n + col):.6E}".replace("E", "D")
                for col in range(col_start, col_start + n_cols))
            lines.append(f"  {row + 1}  {values}")
    return "\n".join(lines)

def _make_structures(n):
    """ Should make dummy structural data """
    result = [(float(i), np.eye(3)) for i in range(n)]
    return result


# ===========================================================================
# --- _gauss_hessian_values_extract - normal behaviour ---
# ===========================================================================
class TestGaussHessianValuesExtractNormal:

    def test_returns_numpy_array(self, tmp_path):
        """ Tests that the extracted Hessian is returned as a NumPy array """
        f = tmp_path / "mol.out"
        text = _make_hessian_text(2)
        result = _gauss_hessian_values_extract(f, text)
        assert isinstance(result, np.ndarray), f"expected {np.ndarray}, got {type(result)}"

    def test_matrix_is_square(self, tmp_path):
        """ Tests that the extracted Hessian matrix is square """
        f = tmp_path / "mol.out"
        for n in (2, 3, 6):
            text = _make_hessian_text(n)
            result = _gauss_hessian_values_extract(f, text)
            assert result.shape == (n, n), f"Expected {(n, n)}, got {result.shape}"

    def test_matrix_is_symmetric(self, tmp_path):
        """ Tests that the extracted Hessian matrix is symmetric """
        f = tmp_path / "mol.out"
        text = _make_hessian_text(4)
        result = _gauss_hessian_values_extract(f, text)
        assert np.allclose(result, result.T), f"Expected {result} == {result.T}"

    def test_no_nans_in_result(self, tmp_path):
        """ Tests that the extracted Hessian contains no NaN values """
        f = tmp_path / "mol.out"
        text = _make_hessian_text(3)
        result = _gauss_hessian_values_extract(f, text)
        assert not np.isnan(result).any(), f"Expected not {np.nan} in {result}"

    def test_fortran_d_notation_parsed(self, tmp_path):
        """ Tests that Fortran D notation is correctly parsed as floating-point values """
        f = tmp_path / "mol.out"
        text = "  1\n  1  4.42310D-01\n"
        result = _gauss_hessian_values_extract(f, text)
        assert result.shape == (1, 1), f"Expected {(1, 1)}, got {result.shape}; test #1"
        assert np.isclose(result[0, 0], 0.442310, rtol=1e-5), f"Expected {0.442310}, "
        f"got {result[0, 0]}; test #2"

    def test_diagonal_values_correct(self, tmp_path):
        """ Tests that the Hessian diagonal and symmetric off-diagonal values are extracted correctly """
        f = tmp_path / "mol.out"
        text = "  \n  1  2\n  1  1.0D+00\n  2  0.5D+00  2.0D+00\n"
        result = _gauss_hessian_values_extract(f, text)
        assert np.isclose(result[0, 0], 1.0), f"Expected {1.0}, got {result[0, 0]}; test #1"
        assert np.isclose(result[1, 1], 2.0), f"Expected {2.0}, got {result[1, 1]}; test #2"
        assert np.isclose(result[0, 1], 0.5), f"Expected {0.5}, got {result[0, 1]}; test #3"
        assert np.isclose(result[1, 0], 0.5), f"Expected {0.5}, got {result[1, 0]}; test #4"

    def test_more_than_one_block_handling(self, tmp_path):
        """ Tests that Hessian values split across multiple blocks are handled correctly """
        f = tmp_path / "mol.out"
        text = _make_hessian_text(7)
        result = _gauss_hessian_values_extract(f, text)
        assert result.shape == (7, 7), f"Expected {(7, 7)}, got {result.shape}; test #1"
        assert np.allclose(result, result.T), f"Expected {result} == {result.T}; test #2"
        assert not np.isnan(result).any(), f"Expected not {np.nan} in {result}; test #3"

    def test_large_block_chunk_handling(self, tmp_path):
        """ Tests that Hessian values in a block larger than the default chunk size are handled correctly """
        f = tmp_path / "mol.out"
        text = _make_hessian_text(7, chunk_size=7)
        result = _gauss_hessian_values_extract(f, text)
        assert result.shape == (7, 7), f"Expected {(7, 7)}, got {result.shape}; test #1"
        assert np.allclose(result, result.T), f"Expected {result} == {result.T}; test #2"
        assert not np.isnan(result).any(), f"Expected not {np.nan} in {result}; test #3"


# ===========================================================================
# --- _gauss_hessian_values_extract - input validation ---
# ===========================================================================
class TestGaussHessianValuesExtractValidation:

    def test_duplicate_column_block_raises(self, tmp_path):
        """ Tests that duplicate Hessian column indices raise an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        # Provide columns (1,1) and rows (1)
        text = "  1  1\n  1  1.0D+00  2.0D+00\n"
        with pytest.raises(ValueError, match="Duplicate Hessian column"):
            _gauss_hessian_values_extract(f, text)
    
    def test_invalid_fortran_value_raises(self, tmp_path):
        """ Tests that an invalid Fortran Hessian value raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        # Provide NOT_A_NUMBER
        text = "  1\n  1  NOT_A_NUMBER\n"
        with pytest.raises(ValueError, match="Invalid Hessian value"):
            _gauss_hessian_values_extract(f, text)
    
    def test_empty_text_raises(self, tmp_path):
        """ Tests that empty input text raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        # Provide ""
        text = ""
        with pytest.raises(ValueError, match="Hessian lines were not found"):
            _gauss_hessian_values_extract(f, text)

    def test_no_data_rows_raises(self, tmp_path):
        """ Tests that a column header without data rows raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        # Provide column header only, no data rows
        text = "  1  2  3\n"
        with pytest.raises(ValueError, match="Hessian lines were not found"):
            _gauss_hessian_values_extract(f, text)

    def test_zero_row_value_raises(self, tmp_path):
        """ Tests that a zero Hessian row index raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        # Provide columns (1,2) and rows (0,1)
        text = "  1  2\n  0  1.0D+00\n  1  2.0D+00  3.0D+00\n"
        with pytest.raises(ValueError, match="Expected"):
            _gauss_hessian_values_extract(f, text)

    def test_negative_row_value_raises(self, tmp_path):
        """ Tests that a negative Hessian row index raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        # Provide columns (1,2) and rows (-1,1)
        text = "  1  2\n -1  1.0D+00\n  1  2.0D+00  3.0D+00\n"
        with pytest.raises(ValueError, match="Expected"):
            _gauss_hessian_values_extract(f, text)

    def test_zero_column_value_raises(self, tmp_path):
        """ Tests that a zero Hessian column index raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        # Provide columns (0,1) and rows (1,2)
        text = "  0  1\n  1  1.0D+00\n  2  2.0D+00  3.0D+00\n"
        with pytest.raises(ValueError, match="Invalid Hessian column index"):
            _gauss_hessian_values_extract(f, text)

    def test_negative_column_value_raises(self, tmp_path):
        """ Tests that a negative Hessian column index raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        # Provide columns (-1,1) and rows (1,2)
        text = "  -1  1\n  1  1.0D+00\n  2  2.0D+00  3.0D+00\n  2\n 3  4.0D+00\n"
        with pytest.raises(ValueError, match="Missing rows"):
            _gauss_hessian_values_extract(f, text)

    def test_incomplete_hessian_missing_row_raises(self, tmp_path):
        """ Tests that a Hessian with a missing row raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        # Provide columns (1,2) and rows (1)
        text = "  1  2\n  1  1.0D+00\n"
        with pytest.raises(ValueError, match="Missing rows"):
            _gauss_hessian_values_extract(f, text)

    def test_incomplete_hessian_missing_column_raises(self, tmp_path):
        """ Tests that a Hessian with a missing column raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        # Provide columns (1,3) and rows (1,2,3)
        text = "  1  3\n  1  1.0D+00\n  2  2.0D+00  3.0D+00\n  3  4.0D+00  5.0D+00\n"
        with pytest.raises(ValueError, match="Missing columns"):
            _gauss_hessian_values_extract(f, text)

    def test_incomplete_hessian_columns_raises(self, tmp_path):
        """ Tests that a Hessian with incomplete column data raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        # Provide columns (1,2,3) and rows (1,2,3),
        # but there are two values in the last row instead of three
        text = "  1 2 3\n  1  1.0D+00\n  2  2.0D+00  3.0D+00\n  3  4.0D+00  5.0D+00\n"
        with pytest.raises(ValueError, match="Missing columns"):
            _gauss_hessian_values_extract(f, text)

    def test_incomplete_hessian_values_raises(self, tmp_path):
        """ Tests that a Hessian with missing matrix values raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        # Provide columns (1,2,3) and rows (1,2,3),
        # but there is one value in the middle row instead of two
        text = "  1 2 3\n  1  1.0D+00\n  2  2.0D+00\n  3  3.0D+00  4.0D+00 5.0D+00\n"
        with pytest.raises(ValueError, match="Some matrix elements"):
            _gauss_hessian_values_extract(f, text)
    
    def test_hessian_with_extra_values_raises(self, tmp_path):
        """ Tests that a Hessian row with extra values raises an error. Expecting a ValueError """
        f = tmp_path / "mol.out"
        # Provide columns (1,2,3) and rows (1,2,3),
        # but there are four vlues in the last row instead of three
        text = "  1 2 3\n  1  1.0D+00\n  2  2.0D+00  3.0D+00\n  3  4.0D+00  5.0D+00  6.0D+00  7.0D+00\n"
        with pytest.raises(ValueError, match="Expected"):
            _gauss_hessian_values_extract(f, text)


# ===========================================================================
# --- gauss_hessian_reader - normal behaviour ---
# ===========================================================================
class TestGaussHessianReaderNormal:

    def test_irc_mode_returns_list(self, tmp_path):
        """ Tests that IRC mode returns the extracted Hessians as a list """
        f = _make_file(tmp_path, "irc.out", "")
        block = _make_hessian_text(2)
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_HESSIAN_IN_IRC_BLOCK.findall.return_value = [block]
            result = gauss_hessian_reader(f, mode="irc")
        assert isinstance(result, list), f"Expected {list}, got {type(result)}; test #1"
        assert len(result) == 1, f"Expected {1}, got {len(result)}; test #2"

    def test_irc_mode_multiple_hessians(self, tmp_path):
        """ Tests that IRC mode correctly handles multiple Hessians """
        f = _make_file(tmp_path, "irc.out", "")
        block = _make_hessian_text(2)
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_HESSIAN_IN_IRC_BLOCK.findall.return_value = [block, block, block]
            result = gauss_hessian_reader(f, mode="irc")
        assert len(result) == 3, f"Expected {3}, got {len(result)}"

    def test_irc_mode_each_element_is_ndarray(self, tmp_path):
        """ Tests that each extracted IRC Hessian is returned as a NumPy array """
        f = _make_file(tmp_path, "irc.out", "")
        block = _make_hessian_text(2)
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_HESSIAN_IN_IRC_BLOCK.findall.return_value = [block]
            result = gauss_hessian_reader(f, mode="irc")
        assert isinstance(result[0], np.ndarray), f"Expected {np.ndarray}, got {type(result[0])}"

    def test_opt_mode_returns_ndarray(self, tmp_path):
        """ Tests that optimization mode returns the extracted Hessian as a NumPy array """
        f = _make_file(tmp_path, "mol.out", "")
        block = _make_hessian_text(2)
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_HESSIAN_IN_OPT_BLOCK.findall.return_value = [block]
            result = gauss_hessian_reader(f, mode="opt")
        assert isinstance(result, np.ndarray), f"Expected {np.ndarray}, got {type(result)}"

    def test_opt_mode_uses_last_hessian(self, tmp_path):
        """ Tests that optimization mode uses the last Hessian found in the output """
        f = _make_file(tmp_path, "mol.out", "")
        block_first = _make_hessian_text(2)
        block_last = "  1  2\n  1  9.99999D+00\n  2  8.88888D+00  0.5D+00\n"
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_HESSIAN_IN_OPT_BLOCK.findall.return_value = [block_first, block_last]
            result = gauss_hessian_reader(f, mode="opt")
        assert np.isclose(result[0, 0], 9.99999, rtol=1e-5), f"Expected {9.99999}, got {result[0, 0]}"

    def test_default_mode_is_irc(self, tmp_path):
        """ Tests that IRC mode is used by default when no mode is specified """
        f = _make_file(tmp_path, "irc.out", "")
        block = _make_hessian_text(2)
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_HESSIAN_IN_IRC_BLOCK.findall.return_value = [block]
            result = gauss_hessian_reader(f)
        assert isinstance(result, list), f"Expected {list}, got {type(result)}"


# ===========================================================================
# --- gauss_hessian_reader - input validation ---
# ===========================================================================
class TestGaussHessianReaderValidation:

    def test_invalid_mode_raises(self, tmp_path):
        """ Tests that an invalid Hessian reader mode raises an error. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.out", "")
        with pytest.raises(ValueError, match="Expected"):
            gauss_hessian_reader(f, mode="freq")

    def test_irc_mode_no_hessian_raises(self, tmp_path):
        """ Tests that 'irc' mode raises an error when no Hessian is found. Expecting a ValueError """
        f = _make_file(tmp_path, "irc.out", "No Hessian here.")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_HESSIAN_IN_IRC_BLOCK.findall.return_value = []
            with pytest.raises(ValueError, match="Hessian was not found"):
                gauss_hessian_reader(f, mode="irc")

    def test_opt_mode_no_hessian_raises(self, tmp_path):
        """ Tests that 'opt' mode raises an error when no Hessian is found. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.out", "No Hessian here.")
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_HESSIAN_IN_OPT_BLOCK.findall.return_value = []
            with pytest.raises(ValueError, match="Hessian was not found"):
                gauss_hessian_reader(f, mode="opt")


# ===========================================================================
# --- gauss_collect_proj_zpves - normal behaviour ---
# ===========================================================================
class TestGaussCollectProjZpvesNormal:

    def test_returns_tuple(self, tmp_path):
        """ Tests that the projected ZPVEs are returned as a tuple """
        f = _make_file(tmp_path, "irc.out", "")
        n = 3
        structures = _make_structures(n)
        hessians = [np.eye(9)] * n
        zpves = tuple(0.03 + 0.001 * i for i in range(n))
        with patch(FUNC_PATH+".gauss_hessian_reader", return_value=hessians), \
             patch(FUNC_PATH+".proj_freq_extraction_tunnex", return_value=(zpves, None, None)):
             result = gauss_collect_proj_zpves(f, structures, np.ones(3))
        assert isinstance(result, tuple), f"Expected {tuple}, got {type(result)}"

    def test_length_matches_number_of_structures(self, tmp_path):
        """ Tests that the number of projected ZPVEs matches the number of structures """
        f = _make_file(tmp_path, "irc.out", "")
        n = 4
        structures = _make_structures(n)
        hessians = [np.eye(9)] * n
        zpves = tuple(0.03 * i for i in range(n))
        with patch(FUNC_PATH+".gauss_hessian_reader", return_value=hessians), \
             patch(FUNC_PATH+".proj_freq_extraction_tunnex", return_value=(zpves, None, None)):
             result = gauss_collect_proj_zpves(f, structures, np.ones(3))
        assert len(result) == n, f"Expected {n}, got {len(result)}"

    def test_structures_sorted_by_irc_before_passing_to_proj_freq(self, tmp_path):
        """
        Tests that structures are sorted by IRC coordinate
        before being passed to projected frequency extraction
        """
        f = _make_file(tmp_path, "irc.out", "")
        structures = [(2.0, np.eye(3)), (0.0, np.eye(3)), (1.0, np.eye(3))]
        hessians = [np.eye(9)] * 3
        zpves = (0.01, 0.02, 0.03)
        captured = {}
        def capture_coords(file, coords, masses, hess):
            captured["coords"] = coords
            return zpves, None, None
        with patch(FUNC_PATH+".gauss_hessian_reader", return_value=hessians), \
             patch(FUNC_PATH+".proj_freq_extraction_tunnex", side_effect=capture_coords):
             gauss_collect_proj_zpves(f, structures, np.ones(3))
        assert captured["coords"] is not None, f"Expected not {None}, got {captured["coords"]}; test #1"
        assert len(captured["coords"]) == 3, f"Expected {3}, got {len(captured["coords"])}; test #2"

    def test_hessians_sorted_consistently_with_structures(self, tmp_path):
        """ Tests that Hessians are sorted consistently with their corresponding structures """
        f = _make_file(tmp_path, "irc.out", "")
        structures = [(1.0, np.eye(3)), (0.0, np.eye(3))]
        hessians = [np.eye(9) * 1.0, np.eye(9) * 2.0]
        captured = {}
        def capture(file, coords, masses, hess):
            captured["hess"] = hess
            return (0.01, 0.02), None, None
        with patch(FUNC_PATH+".gauss_hessian_reader", return_value=hessians), \
             patch(FUNC_PATH+".proj_freq_extraction_tunnex", side_effect=capture):
             gauss_collect_proj_zpves(f, structures, np.ones(3))
        assert len(captured["hess"]) == 2, f"Expected {2}, got {len(captured["hess"])}"

    def test_hessians_follow_structures_after_sorting(self, tmp_path):
        """ Tests that Hessians remain associated with their structures after sorting by IRC coordinate """
        f = _make_file(tmp_path, "irc.out", "")
        structures = [(2.0, np.eye(3)), (0.0, np.eye(3)), (1.0, np.eye(3))]
        hessians = [np.eye(9) * 20.0, np.eye(9) * 0.0, np.eye(9) * 10.0]
        captured = {}
        def capture(file, coords, masses, hess):
            captured["coords"] = coords
            captured["hess"] = hess
            return (0.01, 0.02, 0.03), None, None
        with patch(FUNC_PATH+".gauss_hessian_reader", return_value=hessians), \
             patch(FUNC_PATH+".proj_freq_extraction_tunnex", side_effect=capture):
             gauss_collect_proj_zpves(f, structures, np.ones(3))
        result = [float(np.trace(h)) for h in captured["hess"]]
        assert result == [0.0, 90.0, 180.0], \
            f"Expected Hessians in IRC order, got {result}"

    def test_zpve_values_returned_from_proj_freq(self, tmp_path):
        """ Tests that the ZPVE values returned by projected frequency extraction are returned unchanged """
        f = _make_file(tmp_path, "irc.out", "")
        structures = [(0.0, np.eye(3))]
        hessians = [np.eye(9)]
        expected = (0.054321,)
        with patch(FUNC_PATH+".gauss_hessian_reader", return_value=hessians), \
             patch(FUNC_PATH+".proj_freq_extraction_tunnex", return_value=(expected, None, None)):
            result = gauss_collect_proj_zpves(f, structures, np.ones(3))
        assert result == expected, f"Expected {expected}, got {result}"


# ===========================================================================
# --- gauss_collect_proj_zpves - input validation ---
# ===========================================================================
class TestGaussCollectProjZpvesValidation:

    def test_hessian_count_mismatch_raises(self, tmp_path):
        """
        Tests that a mismatch between the number of Hessians
        and structures raises an error. Expecting a ValueError
        """
        f = _make_file(tmp_path, "irc.out", "")
        structures = _make_structures(3)
        with patch(FUNC_PATH+".gauss_hessian_reader",return_value=[np.eye(9)] * 2):
            with pytest.raises(ValueError, match="must match"):
                gauss_collect_proj_zpves(f, structures, np.ones(3))

    def test_irc_and_structures_mismatch_one_element_raises(self, tmp_path):
        """ Tests that an IRC structure entry with one element raises an error. Expecting a ValueError """
        f = _make_file(tmp_path, "irc.out", "")
        structures = _make_structures(3) + [(4,)]
        with patch(FUNC_PATH+".gauss_hessian_reader", return_value=[np.eye(9)] * 4):
            with pytest.raises(ValueError, match="Each IRC structure"):
                gauss_collect_proj_zpves(f, structures, np.ones(3))

    def test_irc_and_structures_mismatch_no_elements_raises(self, tmp_path):
        """ Tests that an empty IRC structure entry raises an error. Expecting a ValueError """
        f = _make_file(tmp_path, "irc.out", "")
        structures = _make_structures(3) + [()]
        with patch(FUNC_PATH+".gauss_hessian_reader", return_value=[np.eye(9)] * 4):
            with pytest.raises(ValueError, match="Each IRC structure"):
                gauss_collect_proj_zpves(f, structures, np.ones(3))

    def test_irc_and_structures_mismatch_etra_elements_raises(self, tmp_path):
        """
        Tests that an IRC structure entry with more than two elements
        raises an error. Expecting a ValueError
        """
        f = _make_file(tmp_path, "irc.out", "")
        structures = _make_structures(3) + [(4, np.eye(3), 5)]
        with patch(FUNC_PATH+".gauss_hessian_reader", return_value=[np.eye(9)] * 4):
            with pytest.raises(ValueError, match="Each IRC structure"):
                gauss_collect_proj_zpves(f, structures, np.ones(3))