# --- Test: test_gaussian_create_input_file. Unit tests for .\gaussian\gaussian_create_input_file.py
# Run with: pytest test_gaussian_create_input_file.py -v ---


# --- Modules ---
import pytest # type: ignore
from unittest.mock import MagicMock, patch


# --- Module to test ---
from tunnex_2.irc_computations.gaussian.gaussian_create_input_file import ( # type: ignore
    create_gauss_input_file)


# --- Helpers and shared data ---
GAUSS_PATTERNS = "tunnex_2.irc_computations.gaussian.gaussian_create_input_file.gaussian_patterns"


GAUSS_INPUT_SIMPLE = """\
%mem=6GB
%nprocs=4
#p B3LYP/6-311++G freq

Comment

0 1
H  0.000  0.000  0.000
H  1.089  0.000  0.000

"""

GAUSS_INPUT_NO_METHOD = """\
%mem=6GB

Comment

0 1
H  0.000  0.000  0.000
H  1.089  0.000  0.000

"""

GAUSS_INPUT_TWO_METHODS = """\
#p B3LYP/6-311++G freq
#p MP2/6-31G* opt

Comment

0 1
H  0.000  0.000  0.000
H  1.089  0.000  0.000

"""


def _write(tmp_path, name, content):
    """ Should write the input data """
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return f


# ===========================================================================
# --- create_gauss_input_file - normal behaviour and input validation ---
# ===========================================================================
class TestCreateGaussInputFile:

    def test_creates_output_file(self, tmp_path):
        """ Tests creation of the output file """
        src = _write(tmp_path, "mol.gjf", GAUSS_INPUT_SIMPLE)
        dst = tmp_path / "mol_ts.gjf"
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_COMPUTATIONAL_METHOD_GAUSS.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("#p", "B3LYP/6-31G*")) if "#p" in l else None)
            gp.PATTERN_COMMENT = "Comment"
            gp.PATTERN_GEOMETRY_LINE_INPUT_GAUSS.match.return_value = None
            create_gauss_input_file(src, dst, None, "opt")
        assert dst.is_file(), f"expected {True}, got {dst.is_file()}"

    def test_method_tail_appended_to_method_line(self, tmp_path):
        """ Tests appending the method tail to the computational method line """
        src = _write(tmp_path, "mol.gjf", GAUSS_INPUT_SIMPLE)
        dst = tmp_path / "mol_ts.gjf"
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_COMPUTATIONAL_METHOD_GAUSS.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("#p", "B3LYP/6-31G*")) if "#p" in l else None)
            gp.PATTERN_COMMENT = "NONEXISTENT"
            gp.PATTERN_GEOMETRY_LINE_INPUT_GAUSS.match.return_value = None
            create_gauss_input_file(src, dst, None, "freq=noraman")
        content = dst.read_text(encoding="utf-8")
        assert "freq=noraman" in content, f"expected {"freq=noraman"} in {content}; test #1"
        assert "B3LYP/6-31G*" in content, f"expected {"B3LYP/6-31G*"} in {content}; test #2"

    def test_species_geometry_written_when_provided(self, tmp_path):
        """ Tests writing the provided species geometry to the output file """
        src = _write(tmp_path, "mol.gjf", GAUSS_INPUT_SIMPLE)
        dst = tmp_path / "mol_react.gjf"
        geom = "C  1.0  0.0  0.0\nH  2.0  0.0  0.0"
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_COMPUTATIONAL_METHOD_GAUSS.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("#p", "B3LYP/6-31G*")) if "#p" in l else None)
            gp.PATTERN_COMMENT = "NONEXISTENT"
            gp.PATTERN_GEOMETRY_LINE_INPUT_GAUSS.match.side_effect = lambda l: (MagicMock() if "0 1" in l else None)
            create_gauss_input_file(src, dst, geom, "opt")
        content = dst.read_text(encoding="utf-8")
        assert "C  1.0  0.0  0.0" in content, f"expected {"C  1.0  0.0  0.0"} in {content}"

    def test_original_geometry_preserved_when_species_geometry_is_none(self, tmp_path):
        """ Tests that the original geometry is preserved when no replacement geometry is provided """
        expected = ["H  0.000  0.000  0.000", "H  1.089  0.000  0.000", "#p B3LYP/6-31G* freq=noraman"]
        src = _write(tmp_path, "mol.gjf", GAUSS_INPUT_SIMPLE)
        dst = tmp_path / "mol_ts.gjf"
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_COMPUTATIONAL_METHOD_GAUSS.match.side_effect = lambda l: (
                MagicMock(groups=lambda: ("#p", "B3LYP/6-31G*")) if "#p" in l else None)
            gp.PATTERN_COMMENT = "NONEXISTENT"
            gp.PATTERN_GEOMETRY_LINE_INPUT_GAUSS.match.return_value = None
            create_gauss_input_file(src, dst, None, "freq=noraman")
        output = dst.read_text(encoding="utf-8")
        for val in expected:
            assert val in output, f"expected {val} in {output!r}"

    def test_blank_line_added_after_comment(self, tmp_path):
        """ Tests that a blank line is added after a comment line """
        src = _write(tmp_path, "mol.gjf", GAUSS_INPUT_SIMPLE)
        dst = tmp_path / "mol_ts.gjf"
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_COMPUTATIONAL_METHOD_GAUSS.match.side_effect = lambda l: (
                MagicMock(groups=lambda: ("#p", "B3LYP/6-31G*")) if "#p" in l else None)
            gp.PATTERN_COMMENT = "Comment"
            gp.PATTERN_GEOMETRY_LINE_INPUT_GAUSS.match.return_value = None
            create_gauss_input_file(src, dst, None, "opt")
        output = dst.read_text(encoding="utf-8")
        expected = "Comment\n\n"
        assert expected in output, f"expected {expected} in {output!r}"

    def test_returns_none(self, tmp_path):
        """ Tests that the function returns None on success """
        src = _write(tmp_path, "mol.gjf", GAUSS_INPUT_SIMPLE)
        dst = tmp_path / "mol_ts.gjf"
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_COMPUTATIONAL_METHOD_GAUSS.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("#p", "B3LYP/6-31G*")) if "#p" in l else None)
            gp.PATTERN_COMMENT = "NONEXISTENT"
            gp.PATTERN_GEOMETRY_LINE_INPUT_GAUSS.match.return_value = None
            result = create_gauss_input_file(src, dst, None, "opt")
        assert result is None, f"expected {None}, got {result}"
    
    def test_no_method_line_raises(self, tmp_path):
        """ Tests that a missing computational method line raises an error. Expecting a ValueError """
        src = _write(tmp_path, "mol.gjf", GAUSS_INPUT_NO_METHOD)
        dst = tmp_path / "mol_ts.gjf"
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_COMPUTATIONAL_METHOD_GAUSS.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("#p", "B3LYP/6-31G*")) if "#p" in l else None)
            gp.PATTERN_COMMENT = "NONE"
            gp.PATTERN_GEOMETRY_LINE_INPUT_GAUSS.match.return_value = None
            with pytest.raises(ValueError, match="method was not found"):
                create_gauss_input_file(src, dst, None, "opt")
    
    def test_multiple_method_lines_raises(self, tmp_path):
        """ Tests that multiple computational method lines raise an error. Expecting a ValueError """
        src = _write(tmp_path, "mol.gjf", GAUSS_INPUT_TWO_METHODS)
        dst = tmp_path / "mol_ts.gjf"
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_COMPUTATIONAL_METHOD_GAUSS.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("#p", "B3LYP/6-31G*")) if "#p" in l else None)
            gp.PATTERN_COMMENT = "NONE"
            gp.PATTERN_GEOMETRY_LINE_INPUT_GAUSS.match.return_value = None
            with pytest.raises(ValueError, match="Multiple computational method lines"):
                create_gauss_input_file(src, dst, None, "opt")

    def test_missing_geometry_raises(self, tmp_path):
        """ Tests that a missing geometry raises an error. Expecting a ValueError """
        content = "#p B3LYP/6-31G* opt\nTitle\n"
        src = _write(tmp_path, "mol.gjf", content)
        dst = tmp_path / "mol_ts.gjf"
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_COMPUTATIONAL_METHOD_GAUSS.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("#p", "B3LYP/6-31G*")) if "#p" in l else None)
            gp.PATTERN_COMMENT = "NONE"
            gp.PATTERN_GEOMETRY_LINE_INPUT_GAUSS.match.return_value = None
            with pytest.raises(ValueError, match="geometry"):
                create_gauss_input_file(src, dst, "C 0 0 0", "opt")
    
    def test_output_file_deleted_on_error(self, tmp_path):
        """ Tests that the output file is deleted when an error occurs (ValueError) """
        src = _write(tmp_path, "mol.gjf", GAUSS_INPUT_TWO_METHODS)
        dst = tmp_path / "mol_ts.gjf"
        with patch(GAUSS_PATTERNS) as gp:
            gp.PATTERN_COMPUTATIONAL_METHOD_GAUSS.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("#p", "B3LYP/6-31G*")) if "#p" in l else None)
            gp.PATTERN_COMMENT = "NONE"
            gp.PATTERN_GEOMETRY_LINE_INPUT_GAUSS.match.return_value = None
            with pytest.raises(ValueError):
                create_gauss_input_file(src, dst, None, "opt")
        assert not dst.exists(), f"expected {True}, got {not dst.exists()}"