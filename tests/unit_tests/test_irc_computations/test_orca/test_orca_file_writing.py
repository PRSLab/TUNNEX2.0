# --- Test: test_orca_file_writing. Unit tests for .\orca\orca_file_writing.py
# Run with: pytest test_orca_file_writing.py -v ---


# --- Modules ---
import pytest # type: ignore
from unittest.mock import patch, call

from pathlib import Path


# --- Module to test ---
from tunnex_2.irc_computations.orca.orca_file_writing import ( # type: ignore
    create_orca_input,
    create_orca_irc_input,
    create_orca_react_prod_input_from_irc)


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.irc_computations.orca.orca_file_writing"
ORCA_PATTERNS = "tunnex_2.irc_computations.orca.orca_file_writing.orca_patterns"


def _write(tmp_path, name, content=""):
    """ Should write the input data """
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return f


# ===========================================================================
# --- create_orca_input - normal behaviour and input validation ---
# ===========================================================================
class TestCreateOrcaInput:

    def test_output_stem_has_suffix(self, tmp_path):
        """ Tests including the suffix in the output filename """
        f = _write(tmp_path, "mol.inp", "")
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+".create_orca_input_file"):
             result = create_orca_input(f, None, "opt", "_ts")
        assert "_ts" in result.stem, f"expected {"_ts"} in {result.stem}"

    def test_returns_path(self, tmp_path):
        """ Tests returning the output path """
        f = _write(tmp_path, "mol.inp", "")
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+".create_orca_input_file"):
             result = create_orca_input(f, None, "opt", "_ts")
        assert isinstance(result, Path), f"expected {Path}, got {type(result)}"

    def test_callscreate_orca_input_file(self, tmp_path):
        """ Tests calling the ORCA input file creation function """
        f = _write(tmp_path, "mol.inp", "")
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+".create_orca_input_file") as mock_create:
             create_orca_input(f, "geom", "freq", "_prod")
        mock_create.assert_called_once() # it should be called once

    @pytest.mark.parametrize("suffix", ["_ts", "_react", "_prod"])
    def test_all_valid_suffixes_accepted(self, tmp_path, suffix):
        """ Tests accepting all valid output suffixes """
        f = _write(tmp_path, "mol.inp", "")
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+".create_orca_input_file"):
             result = create_orca_input(f, None, "opt", suffix)
        assert suffix in result.stem, f"expected {suffix} in {result.stem}"

    def test_invalid_suffix_raises(self, tmp_path):
        """ Tests raising an error for an invalid output suffix. Expecting a ValueError """
        f = _write(tmp_path, "mol.inp", "")
        with patch(FUNC_PATH+".file_check", return_value=f):
            with pytest.raises(ValueError, match="Expected suffix"):
                create_orca_input(f, None, "opt", "_irc")


# ===========================================================================
# --- create_orca_irc_input - normal behaviour and input validation ---
# ===========================================================================
class TestCreateOrcaIRCInput:

    def test_output_stem_ends_with_irc(self, tmp_path):
        """ Tests creating an output file with the IRC suffix """
        src = _write(tmp_path, "mol.inp", "")
        ts  = _write(tmp_path, "mol_ts.out", "")
        with patch(FUNC_PATH+".file_check", side_effect=[src, ts]) as fc, \
             patch(FUNC_PATH+".orca_extract_geom_from_opt_file", return_value="C 0 0 0") as oeg, \
             patch(FUNC_PATH+".create_orca_input_file") as coi, \
             patch(ORCA_PATTERNS):
             result = create_orca_irc_input(src, ts)
        value = result.stem.endswith("_irc")
        assert value, f"expected {True}, got {value}"

    def test_file_check_is_called_twice(self, tmp_path):
        """ Tests calling file_check twice """
        src = _write(tmp_path, "mol.inp", "")
        ts  = _write(tmp_path, "mol_ts.out", "")
        with patch(FUNC_PATH+".file_check", side_effect=[src, ts]) as fc, \
             patch(FUNC_PATH+".orca_extract_geom_from_opt_file", return_value="C 0 0 0"), \
             patch(FUNC_PATH+".create_orca_input_file"), \
             patch(ORCA_PATTERNS):
             create_orca_irc_input(src, ts)
        assert fc.call_count == 2, f"expected {2}, got {fc.call_count}"

    def test_orca_extract_geom_is_called_once(self, tmp_path):
        """ Tests calling the geometry extraction function once """
        src = _write(tmp_path, "mol.inp", "")
        ts  = _write(tmp_path, "mol_ts.out", "")
        with patch(FUNC_PATH+".file_check", side_effect=[src, ts]), \
             patch(FUNC_PATH+".orca_extract_geom_from_opt_file", return_value="C 0 0 0") as oeg, \
             patch(FUNC_PATH+".create_orca_input_file"), \
             patch(ORCA_PATTERNS):
             create_orca_irc_input(src, ts)
        assert oeg.call_count == 1, f"expected {1}, got {oeg.call_count}"

    def test_create_orca_input_file_is_called_once(self, tmp_path):
        """ Tests calling the ORCA input file creation function once """
        src = _write(tmp_path, "mol.inp", "")
        ts  = _write(tmp_path, "mol_ts.out", "")
        with patch(FUNC_PATH+".file_check", side_effect=[src, ts]), \
             patch(FUNC_PATH+".orca_extract_geom_from_opt_file", return_value="C 0 0 0"), \
             patch(FUNC_PATH+".create_orca_input_file") as coi, \
             patch(ORCA_PATTERNS):
             create_orca_irc_input(src, ts)
        assert coi.call_count == 1, f"expected {1}, got {coi.call_count}"

    def test_block_tail_irc_used_as_tail(self, tmp_path):
        """ Tests using the IRC block tail """
        src = _write(tmp_path, "mol.inp", "")
        ts  = _write(tmp_path, "mol_ts.out", "")
        captured = {}
        def capture(src_f, dst_f, geom, tail):
            captured["tail"] = tail
        with patch(FUNC_PATH+".file_check", side_effect=[src, ts]), \
             patch(FUNC_PATH+".orca_extract_geom_from_opt_file", return_value="C 0 0 0"), \
             patch(FUNC_PATH+".create_orca_input_file", side_effect=capture), \
             patch(ORCA_PATTERNS) as op:
             op.BLOCK_TAIL_IRC = "IRC_PARAMS"
             create_orca_irc_input(src, ts)
        assert captured["tail"] == "IRC_PARAMS", f"expected {"IRC_PARAMS"}, got {captured["tail"]}"

    def test_returns_path(self, tmp_path):
        """ Tests returning the output path """
        src = _write(tmp_path, "mol.inp", "")
        ts  = _write(tmp_path, "mol_ts.out", "")
        with patch(FUNC_PATH+".file_check", side_effect=[src, ts]), \
             patch(FUNC_PATH+".orca_extract_geom_from_opt_file", return_value="C 0 0 0"), \
             patch(FUNC_PATH+".create_orca_input_file"), \
             patch(ORCA_PATTERNS):
             result = create_orca_irc_input(src, ts)
        assert isinstance(result, Path), f"expected {Path}, got {type(result)}"


# =====================================================================================
# --- create_orca_react_prod_input_from_irc - normal behaviour and input validation ---
# =====================================================================================
class TestCreateOrcaReactProdInputFromIRC:

    def test_returns_two_paths(self, tmp_path):
        """ Tests returning two output paths """
        irc = _write(tmp_path, "irc.xyz", "")
        ts  = _write(tmp_path, "mol.inp", "")
        react_xyz = _write(tmp_path, "mol_irc_IRC_B.xyz", "")
        prod_xyz = _write(tmp_path, "mol_irc_IRC_F.xyz", "")
        with patch(FUNC_PATH+".file_check", side_effect=[ts, irc, react_xyz, prod_xyz]), \
             patch(FUNC_PATH+".orca_extract_geom_from_irc_point", return_value="C 0 0 0"), \
             patch(FUNC_PATH+".create_orca_input_file"), \
             patch(ORCA_PATTERNS):
             react, prod = create_orca_react_prod_input_from_irc(irc, ts)
        assert isinstance(react, Path), f"expected {Path}, got {type(react)}; test #1"
        assert isinstance(prod, Path), f"expected {Path}, got {type(prod)}; test #2"

    def test_output_stems_have_react_and_prod(self, tmp_path):
        """ Tests including react and prod suffixes in output filenames """
        irc = _write(tmp_path, "irc.xyz", "")
        ts = _write(tmp_path, "mol.inp", "")
        react_xyz = _write(tmp_path, "mol_irc_IRC_B.xyz", "")
        prod_xyz = _write(tmp_path, "mol_irc_IRC_F.xyz", "")
        with patch(FUNC_PATH+".file_check", side_effect=[ts, irc, react_xyz, prod_xyz]), \
             patch(FUNC_PATH+".orca_extract_geom_from_irc_point", return_value="C 0 0 0"), \
             patch(FUNC_PATH+".create_orca_input_file"), \
             patch(ORCA_PATTERNS):
             react, prod = create_orca_react_prod_input_from_irc(irc, ts)
        assert "_react" in react.stem, f"expected {"_react"} in {react.stem}; test #1"
        assert "_prod" in prod.stem, f"expected {"_prod"} in {prod.stem}; test #2"

    def test_file_check_called_twice(self, tmp_path):
        """ Tests calling file_check twice """
        irc = _write(tmp_path, "irc.xyz", "")
        ts  = _write(tmp_path, "mol.inp", "")
        react_xyz = _write(tmp_path, "mol_irc_IRC_B.xyz", "")
        prod_xyz  = _write(tmp_path, "mol_irc_IRC_F.xyz", "")
        with patch(FUNC_PATH+".file_check", side_effect=[ts, irc, react_xyz, prod_xyz]) as fc, \
             patch(FUNC_PATH+".orca_extract_geom_from_irc_point", return_value="C 0 0 0"), \
             patch(FUNC_PATH+".create_orca_input_file"), \
             patch(ORCA_PATTERNS):
             create_orca_react_prod_input_from_irc(irc, ts)
        assert fc.call_count == 2, f"expected {2}, got {fc.call_count}"

    def test_orca_extract_geom_called_twice(self, tmp_path):
        """ Tests calling the ORCA geometry extraction function twice """
        irc = _write(tmp_path, "irc.xyz", "")
        ts  = _write(tmp_path, "mol.inp", "")
        react_xyz = _write(tmp_path, "mol_irc_IRC_B.xyz", "")
        prod_xyz  = _write(tmp_path, "mol_irc_IRC_F.xyz", "")
        with patch(FUNC_PATH+".file_check", side_effect=[ts, irc, react_xyz, prod_xyz]), \
             patch(FUNC_PATH+".orca_extract_geom_from_irc_point", return_value="C 0 0 0") as oeg, \
             patch(FUNC_PATH+".create_orca_input_file"), \
             patch(ORCA_PATTERNS):
             create_orca_react_prod_input_from_irc(irc, ts)
        assert oeg.call_count == 2, f"expected {2}, got {oeg.call_count}"

    def test_create_orca_input_called_twice(self, tmp_path):
        """ Tests calling the ORCA input file creation function twice """
        irc = _write(tmp_path, "irc.xyz", "")
        ts  = _write(tmp_path, "mol.inp", "")
        react_xyz = _write(tmp_path, "mol_irc_IRC_B.xyz", "")
        prod_xyz  = _write(tmp_path, "mol_irc_IRC_F.xyz", "")
        with patch(FUNC_PATH+".file_check", side_effect=[ts, irc, react_xyz, prod_xyz]), \
             patch(FUNC_PATH+".orca_extract_geom_from_irc_point", return_value="C 0 0 0"), \
             patch(FUNC_PATH+".create_orca_input_file") as coi, \
             patch(ORCA_PATTERNS):
             create_orca_react_prod_input_from_irc(irc, ts)
        assert coi.call_count == 2, f"expected {2}, got {coi.call_count}"

    def test_create_orca_input_called_with_correct_arguments(self, tmp_path):
        """ Tests calling the ORCA input file creation function with correct arguments """
        irc = _write(tmp_path, "irc.xyz", "")
        ts = _write(tmp_path, "mol.inp", "")
        react_xyz = _write(tmp_path, "mol_irc_IRC_B.xyz", "")
        prod_xyz = _write(tmp_path, "mol_irc_IRC_F.xyz", "")
        with patch(FUNC_PATH+".file_check", side_effect=[ts, irc, react_xyz, prod_xyz]), \
            patch(FUNC_PATH+".orca_extract_geom_from_irc_point", return_value="C 0 0 0"), \
            patch(FUNC_PATH+".create_orca_input_file") as coi, \
            patch(ORCA_PATTERNS) as op:
            op.LINE_TAIL_REACT_PROD = "PARAMS"
            create_orca_react_prod_input_from_irc(irc, ts)
        expected = [call(ts, tmp_path / "mol_react.inp", "C 0 0 0", "PARAMS"),
                    call(ts, tmp_path / "mol_prod.inp", "C 0 0 0", "PARAMS")]
        assert coi.call_args_list == expected, f"expected {expected}, got {coi.call_args_list}"

    def test_react_uses_IRC_B_file_prod_uses_IRC_F_file(self, tmp_path):
        """ Tests using the IRC_B file for react and the IRC_F file for prod """
        irc = _write(tmp_path, "irc.xyz", "")
        ts = _write(tmp_path, "mol.inp", "")
        react_xyz = _write(tmp_path, "mol_irc_IRC_B.xyz", "")
        prod_xyz = _write(tmp_path, "mol_irc_IRC_F.xyz", "")
        with patch(FUNC_PATH+".file_check", side_effect=[ts, irc, react_xyz, prod_xyz]), \
             patch(FUNC_PATH+".orca_extract_geom_from_irc_point", return_value="C 0 0 0"), \
             patch(FUNC_PATH+".create_orca_input_file"), \
             patch(ORCA_PATTERNS):
             create_orca_react_prod_input_from_irc(irc, ts)
        result_react = any("IRC_B" in str(p) for p in [react_xyz])
        result_prod = any("IRC_F" in str(p) for p in [prod_xyz])
        assert result_react, f"expected {True}, got {result_react}; test #1"
        assert result_prod, f"expected {True}, got {result_prod}; test #2"