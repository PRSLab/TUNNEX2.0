# --- Test: test_orca_create_input_file. Unit tests for .\orca\orca_create_input_file.py
# Run with: pytest test_orca_create_input_file.py -v ---


# --- Modules ---
import pytest # type: ignore
from unittest.mock import MagicMock, patch

from pathlib import Path


# --- Module to test ---
from tunnex_2.irc_computations.orca.orca_create_input_file import ( # type: ignore
    create_orca_input_file,
    create_orca_hess_comp)


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.irc_computations.orca.orca_create_input_file"
ORCA_PATTERNS = "tunnex_2.irc_computations.orca.orca_create_input_file.orca_patterns"


ORCA_INPUT_SIMPLE = """\
! B3LYP 6-311++G** OptTS Freq

* xyz 0 1
H  0.000  0.000  0.000
H  1.089  0.000  0.000
*
"""

ORCA_INPUT_WITH_GEOM_BLOCK = """\
! B3LYP def2-SVP OPT

%geom
Calc_Hess true
end

* xyz 0 1
C  0.000  0.000  0.000
O  1.000  0.000  0.000
*
"""

ORCA_INPUT_NO_METHOD = """\
* xyz 0 1
H -1.000  0.000  0.000
C  0.000  0.000  0.000
N  1.000  0.000  0.000
*
"""

ORCA_INPUT_TWO_METHODS = """\
! B3LYP def2-SVP opt
! MP2 def2-SVP freq

* xyz 0 1
H  0.000  0.000  0.000
F  0.000  1.000  0.000
*
"""

ORCA_INPUT_UNCLOSED_GEOM = """\
! B3LYP def2-SVP opt

%geom
NumHess true

* xyz 0 1
H  0.000  0.000  0.000
Cl 0.000  0.000  1.000
*
"""


def _write(tmp_path, name, content=""):
    """ Should write the input data """
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return f


# ===========================================================================
# --- create_orca_input_file - normal behaviour and input validation ---
# ===========================================================================
class TestCreateOrcaInputFile:

    def test_creates_output_file(self, tmp_path):
        """ Tests creation of the output file """
        src = _write(tmp_path, "mol.inp", ORCA_INPUT_SIMPLE)
        dst = tmp_path / "mol_ts.inp"
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_COMPUTATIONAL_METHOD_ORCA.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("B3LYP", "def2-SVP")) if "B3LYP" in l else None)
            op.PATTERN_GEOMETRY_LINE_ORCA.match.return_value = None
            create_orca_input_file(src, dst, None, "freq")
        assert dst.is_file(), f"expected {True}, got {dst.is_file()}"

    def test_method_tail_appended_to_method_line(self, tmp_path):
        """ Tests appending the method tail to the computational method line """
        src = _write(tmp_path, "mol.inp", ORCA_INPUT_SIMPLE)
        dst = tmp_path / "mol_ts.inp"
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_COMPUTATIONAL_METHOD_ORCA.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("B3LYP", "def2-SVP")) if "B3LYP" in l else None)
            op.PATTERN_GEOMETRY_LINE_ORCA.match.return_value = None
            create_orca_input_file(src, dst, None, "freq=noraman")
        content = dst.read_text(encoding="utf-8")
        expected = ["freq=noraman", "B3LYP", "def2-SVP"]
        for val in expected:
            assert val in content, f"expected {val} in {content}"

    def test_species_geometry_written_when_provided(self, tmp_path):
        """ Tests writing the provided species geometry to the output file """
        src = _write(tmp_path, "mol.inp", ORCA_INPUT_SIMPLE)
        dst = tmp_path / "mol_react.inp"
        geom = "C  1.0  0.0  0.0\nH  2.0  0.0  0.0"
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_COMPUTATIONAL_METHOD_ORCA.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("B3LYP", "def2-SVP")) if "B3LYP" in l else None)
            op.PATTERN_GEOMETRY_LINE_ORCA.match.side_effect = lambda l: (
            MagicMock() if "C " in l or "H " in l else None)
            create_orca_input_file(src, dst, geom, "opt")
        content = dst.read_text(encoding="utf-8")
        assert "C  1.0  0.0  0.0" in content, f"expected {"C  1.0  0.0  0.0"} in {content}"

    def test_geometry_appended_with_asterisk(self, tmp_path):
        """ Tests appending geometry with an asterisk """
        src = _write(tmp_path, "mol.inp", ORCA_INPUT_SIMPLE)
        dst = tmp_path / "mol_react.inp"
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_COMPUTATIONAL_METHOD_ORCA.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("B3LYP", "def2-SVP")) if "B3LYP" in l else None)
            op.PATTERN_GEOMETRY_LINE_ORCA.match.side_effect = lambda l: (
            MagicMock() if "H " in l else None)
            create_orca_input_file(src, dst, "C 0 0 0", "opt")
        content = dst.read_text(encoding="utf-8")
        assert "*" in content, f"expected {"*"} in {content}"

    def test_geom_block_lines_skipped(self, tmp_path):
        """ Tests skipping geometry block lines """
        src = _write(tmp_path, "mol.inp", ORCA_INPUT_WITH_GEOM_BLOCK)
        dst = tmp_path / "mol_ts.inp"
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_COMPUTATIONAL_METHOD_ORCA.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("B3LYP", "def2-SVP")) if "B3LYP" in l else None)
            op.PATTERN_GEOMETRY_LINE_ORCA.match.return_value = None
            create_orca_input_file(src, dst, None, "opt")
        content = dst.read_text(encoding="utf-8")
        assert "MaxIter" not in content, f"expected {"MaxIter"} not in {content}; test #1"
        assert "%geom" not in content, f"expected {"%geom"} not in {content}; test #2"

    def test_returns_none(self, tmp_path):
        """ Tests returning None """
        src = _write(tmp_path, "mol.inp", ORCA_INPUT_SIMPLE)
        dst = tmp_path / "mol_ts.inp"
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_COMPUTATIONAL_METHOD_ORCA.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("B3LYP", "def2-SVP")) if "B3LYP" in l else None)
            op.PATTERN_GEOMETRY_LINE_ORCA.match.return_value = None
            result = create_orca_input_file(src, dst, None, "opt")
        assert result is None, f"expected {None}, got {result}"
    
    def test_missing_method_raises(self, tmp_path):
        """ Tests that a missing computational method line raises an error. Expecting a ValueError """
        src = _write(tmp_path, "mol.inp", ORCA_INPUT_NO_METHOD)
        dst = tmp_path / "mol_ts.inp"
        with patch(ORCA_PATTERNS) as gp:
            gp.PATTERN_COMPUTATIONAL_METHOD_ORCA.match.return_value = None
            gp.PATTERN_GEOMETRY_LINE_ORCA.match.return_value = None
            with pytest.raises(ValueError, match="computational method was not found"):
                create_orca_input_file(src, dst, None, "opt")

    def test_multiple_method_lines_raises(self, tmp_path):
        """ Tests that multiple computational method lines raise an error. Expecting a ValueError """
        src = _write(tmp_path, "mol.inp", ORCA_INPUT_TWO_METHODS)
        dst = tmp_path / "mol_ts.inp"
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_COMPUTATIONAL_METHOD_ORCA.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("B3LYP", "def2-SVP")) if "!" in l else None)
            op.PATTERN_GEOMETRY_LINE_ORCA.match.return_value = None
            with pytest.raises(ValueError, match="Multiple computational method"):
                create_orca_input_file(src, dst, None, "opt")

    def test_missing_geometry_raises(self, tmp_path):
        """ Tests raising an error when geometry is missing. Expecting a ValueError """
        content = "! B3LYP def2-SVP opt\n\nsome text\n"
        src = _write(tmp_path, "mol.inp", content)
        dst = tmp_path / "mol_ts.inp"
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_COMPUTATIONAL_METHOD_ORCA.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("B3LYP", "def2-SVP")) if "B3LYP" in l else None)
            op.PATTERN_GEOMETRY_LINE_ORCA.match.return_value = None
            with pytest.raises(ValueError, match="geometry of the species was not found"):
                create_orca_input_file(src, dst, "C 0 0 0", "opt")

    def test_unclosed_geom_block_raises(self, tmp_path):
        """ Tests raising an error for an unclosed geometry block. Expecting a ValueError """
        src = _write(tmp_path, "mol.inp", ORCA_INPUT_UNCLOSED_GEOM)
        dst = tmp_path / "mol_ts.inp"
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_COMPUTATIONAL_METHOD_ORCA.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("B3LYP", "def2-SVP")) if "B3LYP" in l else None)
            op.PATTERN_GEOMETRY_LINE_ORCA.match.return_value = None
            with pytest.raises(ValueError, match="block was not closed with"):
                create_orca_input_file(src, dst, None, "opt")

    def test_output_deleted_on_error(self, tmp_path):
        """ Tests that the output file is deleted when an error occurs (ValueError) """
        src = _write(tmp_path, "mol.inp", ORCA_INPUT_TWO_METHODS)
        dst = tmp_path / "mol_ts.inp"
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_COMPUTATIONAL_METHOD_ORCA.match.side_effect = lambda l: (
            MagicMock(groups=lambda: ("B3LYP", "def2-SVP")) if "!" in l else None)
            op.PATTERN_GEOMETRY_LINE_ORCA.match.return_value = None
            with pytest.raises(ValueError, match="Multiple computational method"):
                create_orca_input_file(src, dst, None, "opt")
        assert not dst.exists(), f"expected {True}, got {not dst.exists()}"


# ===========================================================================
# --- create_orca_hess_comp - normal behaviour and input validation ---
# ===========================================================================
class TestCreateOrcaHessComp:

    def test_output_calls_file_check_once(self, tmp_path):
        """ Tests calling file_check exactly once """
        f = _write(tmp_path, "mol.inp", "")
        with patch(FUNC_PATH+".file_check", return_value=f) as fc, \
                patch(FUNC_PATH+".create_orca_input_file"), \
                patch(ORCA_PATTERNS) as op:
                op.LINE_HESS_COMP = "Freq {}  # el_energy placeholder"
                create_orca_hess_comp(f, "C 0 0 0", -153.0, 5)
        fc.assert_called_once() # it should be called once

    def test_create_input_called_once(self, tmp_path):
        """ Tests calling the input creation function exactly once """
        f = _write(tmp_path, "mol.inp", "")
        with patch(FUNC_PATH+".file_check", return_value=f), \
            patch(FUNC_PATH+".create_orca_input_file") as coi, \
            patch(ORCA_PATTERNS) as op:
            op.LINE_HESS_COMP = "Freq {}"
            create_orca_hess_comp(f, "C 0 0 0", -153.0, 1)
        coi.assert_called_once() # it should be called once

    def test_output_stem_has_step_number(self, tmp_path):
        """ Tests including the step number in the output filename """
        f = _write(tmp_path, "mol.inp", "")
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+".create_orca_input_file"), \
             patch(ORCA_PATTERNS) as op:
             op.LINE_HESS_COMP = "Freq {}  # el_energy placeholder"
             result = create_orca_hess_comp(f, "C 0 0 0", -153.0, 5)
        assert "_hess_irc_5" in result.stem, f"expected {"_hess_irc_5"} in {result.stem}"

    def test_returns_path(self, tmp_path):
        """ Tests returning the output path """
        f = _write(tmp_path, "mol.inp", "")
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+".create_orca_input_file"), \
             patch(ORCA_PATTERNS) as op:
             op.LINE_HESS_COMP = "Freq {}"
             result = create_orca_hess_comp(f, "C 0 0 0", -153.0, 1)
        assert isinstance(result, Path), f"expected {Path}, got {type(result)}"

    def test_el_energy_formatted_into_method_tail(self, tmp_path):
        """ Tests formatting the electronic energy into the method tail """
        f = _write(tmp_path, "mol.inp", "")
        captured = {}
        def capture(src_f, dst_f, geom, tail):
            captured["tail"] = tail
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+".create_orca_input_file", side_effect=capture), \
             patch(ORCA_PATTERNS) as op:
             op.LINE_HESS_COMP = "Freq {}"
             create_orca_hess_comp(f, "C 0 0 0", -153.456, 3)
        assert "-153.456" in captured["tail"], f"expected {"-153.456"} in {captured["tail"]}"

    def test_geometry_passed_to_create_input(self, tmp_path):
        """ Tests passing the geometry to the input creation function """
        f = _write(tmp_path, "mol.inp", "")
        captured = {}
        def capture(src_f, dst_f, geom, tail):
            captured["geom"] = geom
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+".create_orca_input_file", side_effect=capture), \
             patch(ORCA_PATTERNS) as op:
             op.LINE_HESS_COMP = "Freq {}"
             create_orca_hess_comp(f, "C 1.0 2.0 3.0", -153.0, 1)
        assert "C 1.0 2.0 3.0" in captured["geom"], f"expected {"C 1.0 2.0 3.0"} in {captured["geom"]}"

    def test_different_step_numbers_give_different_filenames(self, tmp_path):
        """ Tests generating different filenames for different step numbers """
        f = _write(tmp_path, "mol.inp", "")
        results = []
        for step in (1, 5, 10):
            with patch(FUNC_PATH+".file_check", return_value=f), \
                 patch(FUNC_PATH+".create_orca_input_file"), \
                 patch(ORCA_PATTERNS) as op:
                 op.LINE_HESS_COMP = "Freq {}"
                 results.append(create_orca_hess_comp(f, "C 0 0 0", -153.0, step))
        result = len(set(r.stem for r in results))
        assert result == 3, f"expected {3}, got {result}"

    def test_input_file_called_with_expected_paths(self, tmp_path):
        """ Tests passing the expected input and output paths """
        f = _write(tmp_path, "mol.inp", "")
        with patch(FUNC_PATH+".file_check", return_value=f), \
            patch(FUNC_PATH+".create_orca_input_file") as create_input, \
            patch(ORCA_PATTERNS) as op:
            op.LINE_HESS_COMP = "Freq {}"
            result = create_orca_hess_comp(f, "C 0 0 0", -153.0, 5)
        create_input.assert_called_once_with(f, result, "C 0 0 0", "Freq -153.0")
        # it should be called once

    def test_method_tail_uses_given_energy(self, tmp_path):
        """ Tests using the given electronic energy in the method tail """
        f = _write(tmp_path, "mol.inp", "")
        with patch(FUNC_PATH+".file_check", return_value=f), \
            patch(FUNC_PATH+".create_orca_input_file") as create_input, \
            patch(ORCA_PATTERNS) as op:
            op.LINE_HESS_COMP = "Freq {}"
            create_orca_hess_comp(f, "C 0 0 0", -200.25, 1)
        result = create_input.call_args.args[3]
        assert result == "Freq -200.25", f"expected {"Freq -200.25"}, got {result}"