# --- Test: test_orca_proj_freq. Unit tests for .\orca\orca_proj_freq.py
# Run with: pytest test_orca_proj_freq.py -v ---

# --- Modules ---
import pytest  # type: ignore
from unittest.mock import MagicMock, patch

import numpy as np
from pathlib import Path


# --- Module to test ---
from tunnex_2.irc_computations.orca.orca_proj_freq import ( # type: ignore
    orca_hessian_reader,
    orca_collect_proj_zpves)


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.irc_computations.orca.orca_proj_freq"
ORCA_PATTERNS = "tunnex_2.irc_computations.orca.orca_proj_freq.orca_patterns"


def _make_file(tmp_path, name, content=""):
    """ Should write a file with the content (Hessian) """
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return f

def _make_hessian_text(n: int, symm_matrix: bool = True, chunk_size: int = 5) -> str:
    """ Should build a full Hessian text block for an NxN matrix (ORCA format) """
    chunk = chunk_size
    shift = 0 if symm_matrix else 1
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = i + j
            matrix[j][i] = i + j - shift
    lines = []
    for col_start in range(0, n, chunk):
        col_end = min(col_start + chunk, n)
        lines.append("  " + "  ".join(str(c) for c in range(col_start, col_end)))
        for row in range(n):
            values = "  ".join(f"{float(matrix[row][col]):.10E}" for col in range(col_start, col_end))
            lines.append(f"  {row}  {values}")
        result = "\n".join(lines)
    return result

def _make_structures(n):
    """ Should make dummy structural data and masses """
    result = [(float(i), np.eye(3)) for i in range(n)]
    atom_masses = np.full(n, 12.0)
    return result, atom_masses


# ===========================================================================
# --- orca_hessian_reader - normal behaviour ---
# ===========================================================================
class TestOrcaHessianReaderNormal:

    def _run(self, tmp_path, n, chunk_size=5):
        """ Should return the dummy Hessian data """
        f = _make_file(tmp_path, "mol.hess", "")
        text = _make_hessian_text(n, chunk_size)
        match_mock = MagicMock()
        match_mock.groups.return_value = (str(n), text)
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_HESSIAN_BLOCK.search.return_value = match_mock
            return orca_hessian_reader(f)

    def test_returns_ndarray(self, tmp_path):
        """ Tests returning a NumPy array """
        result = self._run(tmp_path, 3)
        assert isinstance(result, np.ndarray), f"Expected {np.ndarray}, got {type(result)}"

    def test_correct_shape(self, tmp_path):
        """ Tests returning a Hessian with the correct shape """
        for n in (2, 3, 6, 9):
            result = self._run(tmp_path, n)
            assert result.shape == (n, n), f"Expected {(n, n)}, got {result.shape}"

    def test_no_nans(self, tmp_path):
        """ Tests returning a Hessian without NaN values """
        hessian = self._run(tmp_path, 4)
        result = not np.isnan(hessian).any()
        assert result, f"Expected {True}, got {result}"

    def test_values_correct(self, tmp_path):
        """ Tests returning the correct Hessian values """
        n = 3
        result = self._run(tmp_path, n)
        for row in range(n):
            for col in range(n):
                expected = row + col
                assert np.isclose(result[row, col], expected), \
                    f"[{row},{col}]: expected {expected}, got {result[row, col]}"

    def test_large_matrix_chunk_handling(self, tmp_path):
        """ Tests handling Hessian matrices larger than the chunk size """
        hessian = self._run(tmp_path, 7)
        result = not np.isnan(hessian).any()
        assert hessian.shape == (7, 7), f"Expected {(7, 7)}, got {hessian.shape}; test #1"
        assert result, f"Expected {True}, got {result}; test #2"

    def test_bigger_chunk_size_handling(self, tmp_path):
        """ Tests handling chunk sizes larger than the default """
        hessian = self._run(tmp_path, 10, chunk_size=7)
        result = not np.isnan(hessian).any()
        assert hessian.shape == (10, 10), f"Expected {(10, 10)}, got {hessian.shape}; test #1"
        assert result, f"Expected {True}, got {result}; test #2"

    def test_accepts_path_object(self, tmp_path):
        """ Tests accepting a Path object as input """
        f = _make_file(tmp_path, "mol.hess", "")
        text = _make_hessian_text(2)
        match_mock = MagicMock()
        match_mock.groups.return_value = ("2", text)
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_HESSIAN_BLOCK.search.return_value = match_mock
            result = orca_hessian_reader(Path(f))
        assert isinstance(result, np.ndarray), f"Expected {np.ndarray}, got {type(result)}"

    def test_accepts_string_path(self, tmp_path):
        """ Tests accepting a string path as input """
        f = _make_file(tmp_path, "mol.hess", "")
        text = _make_hessian_text(2)
        match_mock = MagicMock()
        match_mock.groups.return_value = ("2", text)
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_HESSIAN_BLOCK.search.return_value = match_mock
            result = orca_hessian_reader(str(f))
        assert isinstance(result, np.ndarray), f"Expected {np.ndarray}, got {type(result)}"


# ===========================================================================
# --- orca_hessian_reader - input validation ---
# ===========================================================================
class TestOrcaHessianReaderValidation:

    def test_no_hessian_block_raises(self, tmp_path):
        """ Tests that the function raises an error when no Hessian is found. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.hess", "no hessian here")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_HESSIAN_BLOCK.search.return_value = None
            with pytest.raises(ValueError, match="Hessian was not found"):
                orca_hessian_reader(f)

    def test_empty_hessian_text_raises(self, tmp_path):
        """
        Tests that the function raises an error when
        no Hessian lines are found. Expecting a ValueError
        """
        f = _make_file(tmp_path, "mol.hess", "block")
        match_mock = MagicMock()
        match_mock.groups.return_value = ("3", "")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_HESSIAN_BLOCK.search.return_value = match_mock
            with pytest.raises(ValueError, match="Lines of hessian were not found"):
                orca_hessian_reader(f)

    def test_invalid_column_index_raises(self, tmp_path):
        """ Tests raising an error for an invalid Hessian column index. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.hess", "block")
        # Provide column #5 for n=2
        text = "  1 5\n  0  1.0 2.0\n  1  2.0 4.0\n"
        match_mock = MagicMock()
        match_mock.groups.return_value = ("2", text)
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_HESSIAN_BLOCK.search.return_value = match_mock
            with pytest.raises(ValueError, match="Invalid Hessian column"):
                orca_hessian_reader(f)

    def test_invalid_row_index_raises(self, tmp_path):
        """ Tests raising an error for an invalid Hessian row index. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.hess", "block")
        # Provide row #5 for n=2
        text = "  \n  0 1\n  0  1.0 2.0\n  5  2.0 4.0\n"
        match_mock = MagicMock()
        match_mock.groups.return_value = ("2", text)
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_HESSIAN_BLOCK.search.return_value = match_mock
            with pytest.raises(ValueError, match="Invalid Hessian row"):
                orca_hessian_reader(f)

    def test_lack_value_count_raises(self, tmp_path):
        """ Tests raising an error for an incorrect number of Hessian values. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.hess", "block")
        # Provide one value for two columns
        text = "  0 1\n  0  1.0 2.0\n  1  2.0\n"
        match_mock = MagicMock()
        match_mock.groups.return_value = ("2", text)
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_HESSIAN_BLOCK.search.return_value = match_mock
            with pytest.raises(ValueError, match="Unexpected number of Hessian values"):
                orca_hessian_reader(f)

    def test_extra_value_count_raises(self, tmp_path):
        """ Tests raising an error for an excessive number of Hessian values. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.hess", "block")
        # Provide three values for two columns
        text = "  0 1\n  0  1.0 2.0\n  1  2.0 4.0 6.0\n"
        match_mock = MagicMock()
        match_mock.groups.return_value = ("2", text)
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_HESSIAN_BLOCK.search.return_value = match_mock
            with pytest.raises(ValueError, match="Unexpected number of Hessian values"):
                orca_hessian_reader(f)

    def test_missing_rows_raises(self, tmp_path):
        """ Tests raising an error when Hessian rows are missing. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.hess", "block")
        # Provide two rows for n=3
        text = "  0  1  2\n  0  1.0  0.5  0.3\n  1  0.5  2.0  0.1\n"
        match_mock = MagicMock()
        match_mock.groups.return_value = ("3", text)
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_HESSIAN_BLOCK.search.return_value = match_mock
            with pytest.raises(ValueError, match="Missing rows"):
                orca_hessian_reader(f)

    def test_missing_columns_raises(self, tmp_path):
        """ Tests raising an error when Hessian columns are missing. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.hess", "block")
        # Provide two colums for n=3
        text = "  0  1 \n  0  1.0  0.5\n  1  0.5  2.0\n 2  0.4  1.5\n"
        match_mock = MagicMock()
        match_mock.groups.return_value = ("3", text)
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_HESSIAN_BLOCK.search.return_value = match_mock
            with pytest.raises(ValueError, match="Some matrix elements"):
                orca_hessian_reader(f)

    def test_nan_in_matrix_raises(self, tmp_path):
        """ Tests raising an error for a non-numeric Hessian value. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.hess", "block")
        # Provide NOT_A_NUMBER in one of the lines
        text = "  0 1\n  0  1.0 2.0\n  1  2.0 NOT_A_NUMBER\n"
        match_mock = MagicMock()
        match_mock.groups.return_value = ("2", text)
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_HESSIAN_BLOCK.search.return_value = match_mock
            with pytest.raises(ValueError, match="Invalid numeric Hessian value"):
                orca_hessian_reader(f)

    def test_non_symmetric_matrix_raises(self, tmp_path):
        """ Tests raising an error for a non-symmetric Hessian matrix. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.hess", "block")
        # Provide non-symmetric Hessian
        text = _make_hessian_text(5, symm_matrix=False)
        match_mock = MagicMock()
        match_mock.groups.return_value = ("5", text)
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_HESSIAN_BLOCK.search.return_value = match_mock
            with pytest.raises(ValueError, match="Hessian matrix is not symmetric"):
                orca_hessian_reader(f)


# ===========================================================================
# --- orca_collect_proj_zpves - normal behaviour ---
# ===========================================================================
class TestOrcaCollectProjZpvesNormal:

    def _run(self, tmp_path, n_points=3, run_key=False):
        """ Should make a dummy run of the function """
        irc = _make_file(tmp_path, "irc.xyz", "")
        ts_in = _make_file(tmp_path, "ts.inp",  "")
        energies = np.linspace(-153.0, -152.9, n_points)
        coords = np.zeros((n_points, 2, 3))
        texts = [f"geom_{i}" for i in range(n_points)]
        masses = np.array([12.0, 1.0])
        zpves = tuple(0.03 + 0.001 * i for i in range(n_points))
        with patch(FUNC_PATH+".create_orca_hess_comp",
             side_effect=[tmp_path / f"hess_{i}.inp" for i in range(n_points)]) as coh, \
             patch(FUNC_PATH+".run_software") as rs, \
             patch(FUNC_PATH+".orca_out_filename",
             side_effect=[tmp_path / f"hess_{i}.hess" for i in range(n_points)]) as oof, \
             patch(FUNC_PATH+".orca_error_check") as oec, \
             patch(FUNC_PATH+".orca_hessian_reader", return_value=np.eye(6)) as ohr, \
             patch(FUNC_PATH+".proj_freq_extraction_tunnex", return_value=(zpves, None, None)):
             result = orca_collect_proj_zpves(ts_in, irc, "cmd.txt",
             energies, coords, texts, masses, run_software_key=run_key)
        return result, zpves, coh, rs, oof, oec, ohr

    def test_returns_tuple(self, tmp_path):
        """ Tests returning a tuple """
        result, _, _, _, _, _, _ = self._run(tmp_path)
        assert isinstance(result, tuple), f"Expected {tuple}, got {type(result)}"

    def test_length_matches_n_points(self, tmp_path):
        """ Tests returning one result for each IRC point """
        n = 4
        result, _, _, _, _, _, _ = self._run(tmp_path, n_points=n)
        assert len(result) == n, f"Expected {n}, got {len(result)}"

    def test_zpve_values_from_proj_freq(self, tmp_path):
        """ Tests returning the expected projected frequency values """
        result, expected, _, _, _, _, _ = self._run(tmp_path, n_points=3)
        assert result == expected, f"Expected {expected}, got {len(result)}"

    def test_run_software_not_called_when_key_false(self, tmp_path):
        """ Tests not running the software when the run key is false """
        _, _, _, rs, _, _, _ = self._run(tmp_path, n_points=3, run_key=False)
        rs.assert_not_called() # it should not be called

    def test_run_software_called_for_each_point_when_key_true(self, tmp_path):
        """ Tests running the software for each IRC point when the run key is true """
        n = 3
        _, _, coh, rs, _, _, _ = self._run(tmp_path, n_points=n, run_key=True)
        assert rs.call_count == n, f"Expected {n}, got {rs.call_count}; test #1"
        assert coh.call_count == n, f"Expected {n}, got {coh.call_count}; test #2"

    def test_orca_out_filename_called_for_each_hess(self, tmp_path):
        """ Tests calling the ORCA output filename function for each Hessian """
        n = 3
        _, _, _, _, oof, _, ohr = self._run(tmp_path, n_points=n, run_key=False)
        assert oof.call_count == n, f"Expected {n}, got {oof.call_count}"

    def test_orca_hessian_reader_called_for_each_hess(self, tmp_path):
        """ Tests calling the ORCA Hessian reader for each Hessian """
        n = 3
        _, _, _, _, oof, _, ohr = self._run(tmp_path, n_points=n, run_key=False)
        assert ohr.call_count == n, f"Expected {n}, got {ohr.call_count}"

    def test_create_hess_comp_called_with_correct_step_numbers(self, tmp_path):
        """ Tests creating Hessian inputs with the correct step numbers """
        irc   = _make_file(tmp_path, "irc.xyz", "")
        ts_in = _make_file(tmp_path, "ts.inp",  "")
        n = 3
        captured_steps = []
        def capture_hess(ts_inp, geom, energy, step):
            captured_steps.append(step)
            return tmp_path / f"hess_{step}.inp"
        with patch(FUNC_PATH+".create_orca_hess_comp", side_effect=capture_hess), \
             patch(FUNC_PATH+".run_software"), \
             patch(FUNC_PATH+".orca_out_filename",
             side_effect=[tmp_path / f"h{i}.hess" for i in range(n)]), \
             patch(FUNC_PATH+".orca_error_check"), \
             patch(FUNC_PATH+".orca_hessian_reader", return_value=np.eye(6)), \
             patch(FUNC_PATH+".proj_freq_extraction_tunnex",
             return_value=((0.03,) * n, None, None)) as pfet:
             orca_collect_proj_zpves(ts_in, irc, "cmd.txt", np.linspace(-153.0, -152.9, n),
             np.zeros((n, 2, 3)), [f"g{i}" for i in range(n)], np.array([12.0, 1.0]))
        assert captured_steps == list(range(n)), f"Expected {list(range(n))}, "
        f"got {captured_steps}; test #1"
        assert pfet.call_count == 1, f"Expected {1}, got {pfet.call_count}; test #2"

    def test_orca_error_check_called_for_each_hess(self, tmp_path):
        """ Tests checking each Hessian output for errors """
        irc   = _make_file(tmp_path, "irc.xyz", "")
        ts_in = _make_file(tmp_path, "ts.inp",  "")
        n = 4
        with patch(FUNC_PATH+".create_orca_hess_comp",
             side_effect=[tmp_path / f"h{i}.inp" for i in range(n)]), \
             patch(FUNC_PATH+".run_software"), \
             patch(FUNC_PATH+".orca_out_filename",
             side_effect=[tmp_path / f"h{i}.hess" for i in range(n)]), \
             patch(FUNC_PATH+".orca_error_check") as mock_check, \
             patch(FUNC_PATH+".orca_hessian_reader", return_value=np.eye(6)), \
             patch(FUNC_PATH+".proj_freq_extraction_tunnex",
             return_value=((0.03,) * n, None, None)):
             orca_collect_proj_zpves(ts_in, irc, "cmd.txt", np.linspace(-153.0, -152.9, n),
             np.zeros((n, 2, 3)), [f"g{i}" for i in range(n)], np.array([12.0, 1.0]))
        assert mock_check.call_count == n, f"Expected {n}, got {mock_check.call_count}"

    def test_hessians_passed_to_proj_freq_extraction(self, tmp_path):
        """ Tests passing all Hessian matrices to the projected frequency extraction """
        irc   = _make_file(tmp_path, "irc.xyz", "")
        ts_in = _make_file(tmp_path, "ts.inp",  "")
        n = 3
        eye = np.eye(6)
        captured = {}
        def capture_proj(file, coords, masses, hess_list):
            captured["hess_list"] = hess_list
            return ((0.03,) * n, None, None)
        with patch(FUNC_PATH+".create_orca_hess_comp",
             side_effect=[tmp_path / f"h{i}.inp" for i in range(n)]), \
             patch(FUNC_PATH+".run_software"), \
             patch(FUNC_PATH+".orca_out_filename",
             side_effect=[tmp_path / f"h{i}.hess" for i in range(n)]), \
             patch(FUNC_PATH+".orca_error_check"), \
             patch(FUNC_PATH+".orca_hessian_reader", return_value=eye), \
             patch(FUNC_PATH+".proj_freq_extraction_tunnex", side_effect=capture_proj):
             orca_collect_proj_zpves(ts_in, irc, "cmd.txt", np.linspace(-153.0, -152.9, n),
             np.zeros((n, 2, 3)), [f"g{i}" for i in range(n)], np.array([12.0, 1.0]))
        result = all(np.allclose(h, eye) for h in captured["hess_list"])
        assert len(captured["hess_list"]) == n, f"Expected {n}, "
        f"got {len(captured["hess_list"])}; test #1"
        assert result, f"Expected {True}, got {result}; test #2"


# ===========================================================================
# --- orca_collect_proj_zpves - input validation ---
# ===========================================================================
class TestOrcaCollectProjZpvesValidation:

    def test_energies_coords_mismatch_raises(self, tmp_path):
        """ Tests raising an error when energies and coordinate arrays mismatch. Expecting a ValueError """
        irc = _make_file(tmp_path, "irc.xyz", "")
        ts_in = _make_file(tmp_path, "ts.inp", "")
        energies = np.array([-153.0, -152.9])
        coords, masses = _make_structures(3)
        texts = ["g", "g"]
        with pytest.raises(ValueError, match=r"and geometries \(array\)"):
            orca_collect_proj_zpves(ts_in, irc, "cmd.txt", energies, coords, texts, masses)
    
    def test_energies_text_mismatch_raises(self, tmp_path):
        """ Tests raising an error when energies and geometry texts mismatch. Expecting a ValueError """
        irc = _make_file(tmp_path, "irc.xyz", "")
        ts_in = _make_file(tmp_path, "ts.inp", "")
        energies = np.array([-153.0, -152.9, -153.1])
        coords, masses = _make_structures(3)
        texts = ["geom1", "geom2"]
        with pytest.raises(ValueError, match=r"and geometries \(text\)"):
            orca_collect_proj_zpves(ts_in, irc, "cmd.txt", energies, coords, texts, masses)