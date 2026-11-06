# --- Test: test_gaussian_file_writing. Unit tests for .\gaussian\gaussian_file_writing.py
# Run with: pytest test_gaussian_file_writing.py -v ---


# --- Modules ---
import pytest # type: ignore
from unittest.mock import patch

from pathlib import Path


# --- Module to test ---
from tunnex_2.irc_computations.gaussian.gaussian_file_writing import ( # type: ignore
    create_gauss_input,
    create_gauss_irc_input,
    create_gauss_react_prod_input_from_irc)


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.irc_computations.gaussian.gaussian_file_writing"
GAUSS_PATTERNS = "tunnex_2.irc_computations.gaussian.gaussian_file_writing.gaussian_patterns"


def _write(tmp_path, name, content):
    """ Should write the input data """
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return f


# ===========================================================================
# --- create_gauss_input - normal behaviour and input validation ---
# ===========================================================================
class TestCreateGaussInput:

    def test_output_path_has_correct_stem(self, tmp_path):
        """ Tests that the output path has the expected filename stem """
        f = _write(tmp_path, "mol.gjf", "")
        with patch(FUNC_PATH+".file_check", return_value=f) as fc, \
             patch(FUNC_PATH+".create_gauss_input_file") as cgi:
            result = create_gauss_input(f, None, "opt", "_ts")
        fc.assert_called_once() # it should be called once
        cgi.assert_called_once() # it should be called once
        assert result.stem == "mol_ts", f"expected {"mol_ts"}, got {result.stem}; test #1"
        assert isinstance(result, Path), f"expected {Path}, got {type(result)}; test #2"

    @pytest.mark.parametrize("suffix", ["_ts", "_react", "_prod"])
    def test_all_valid_suffixes_accepted(self, tmp_path, suffix):
        """ Tests that all valid suffixes are accepted """
        f = _write(tmp_path, "mol.gjf", "")
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+".create_gauss_input_file"):
             result = create_gauss_input(f, None, "opt", suffix)
        assert suffix in result.stem, f"expected {suffix} in {result.stem}"

    def test_invalid_suffix_raises(self, tmp_path):
        """ Tests that an invalid suffix raises an error. Expecting a ValueError """
        f = _write(tmp_path, "mol.gjf", "")
        with patch(FUNC_PATH+".file_check", return_value=f) as fc:
            with pytest.raises(ValueError, match="Expected suffix"):
                create_gauss_input(f, None, "opt", "_irc")
        fc.assert_called_once() # it should be called once


# ===========================================================================
# --- create_gauss_irc_input - normal behaviour and input validation ---
# ===========================================================================
class TestCreateGaussIRCInput:

    def test_output_stem_ends_with_irc(self, tmp_path):
        """ Tests that the output file stem ends with '_irc' """
        src = _write(tmp_path, "mol.gjf", "")
        ts  = _write(tmp_path, "mol_ts.out", "")
        with patch(FUNC_PATH+".file_check", side_effect=[src, ts]) as fc, \
             patch(FUNC_PATH+".gauss_extract_geom_from_opt_file", return_value="C 0 0 0") as geof, \
             patch(FUNC_PATH+".create_gauss_input_file") as cgi:
             result = create_gauss_irc_input(src, ts)
        assert result.stem.endswith("_irc"), f"expected {"_irc"} in {result.stem}; test #1"
        assert fc.call_count == 2, f"expected {2}, got {fc.call_count}; test #2"
        geof.assert_called_once() # it should be called once
        cgi.assert_called_once() # it should be called once

    def test_calcall_in_method_tail(self, tmp_path):
        """ Tests that 'calcall' is included in the method tail when calc_all is True """
        src = _write(tmp_path, "mol.gjf", "")
        ts  = _write(tmp_path, "mol_ts.out", "")
        captured = {}
        def capture(src_f, dst_f, geom, tail):
            captured["tail"] = tail
        with patch(FUNC_PATH+".file_check", side_effect=[src, ts]), \
             patch(FUNC_PATH+".gauss_extract_geom_from_opt_file", return_value="C 0 0 0"), \
             patch(FUNC_PATH+".create_gauss_input_file", side_effect=capture):
             create_gauss_irc_input(src, ts, calc_all=True)
        assert "calcall" in captured["tail"], f"expected {"calcall"} in {captured["tail"]}"

    def test_calcfc_when_calc_all_false(self, tmp_path):
        """ Tests that 'calcfc' is included in the method tail when calc_all is False """
        src = _write(tmp_path, "mol.gjf", "")
        ts  = _write(tmp_path, "mol_ts.out", "")
        captured = {}
        def capture(src_f, dst_f, geom, tail):
            captured["tail"] = tail
        with patch(FUNC_PATH+".file_check", side_effect=[src, ts]), \
             patch(FUNC_PATH+".gauss_extract_geom_from_opt_file", return_value="C 0 0 0"), \
             patch(FUNC_PATH+".create_gauss_input_file", side_effect=capture):
             create_gauss_irc_input(src, ts, calc_all=False)
        assert "calcfc" in captured["tail"], f"expected {"calcfc"} in {captured["tail"]}"

    def test_iop_added_when_proj_freq_true(self, tmp_path):
        """ Tests that the projected frequency option is added to the method tail when proj_freq is True """
        src = _write(tmp_path, "mol.gjf", "")
        ts  = _write(tmp_path, "mol_ts.out", "")
        captured = {}
        def capture(src_f, dst_f, geom, tail):
            captured["tail"] = tail
        with patch(FUNC_PATH+".file_check", side_effect=[src, ts]), \
             patch(FUNC_PATH+".gauss_extract_geom_from_opt_file", return_value="C 0 0 0"), \
             patch(FUNC_PATH+".create_gauss_input_file", side_effect=capture):
             create_gauss_irc_input(src, ts, proj_freq=True)
        assert "iop(1/73=2)" in captured["tail"], f"expected {"iop(1/73=2)"} in {captured["tail"]}"

    def test_iop_absent_when_proj_freq_false(self, tmp_path):
        """ Tests that the projected frequency option is absent from the method tail when proj_freq is False """
        src = _write(tmp_path, "mol.gjf", "")
        ts  = _write(tmp_path, "mol_ts.out", "")
        captured = {}
        def capture(src_f, dst_f, geom, tail):
            captured["tail"] = tail
        with patch(FUNC_PATH+".file_check", side_effect=[src, ts]), \
             patch(FUNC_PATH+".gauss_extract_geom_from_opt_file", return_value="C 0 0 0"), \
             patch(FUNC_PATH+".create_gauss_input_file", side_effect=capture):
             create_gauss_irc_input(src, ts, proj_freq=False)
        assert "iop" not in captured["tail"], f"expected {"iop"} not in {captured["tail"]}"

    def test_custom_parameters_in_tail(self, tmp_path):
        """ Tests that custom parameters are included in the method tail """
        src = _write(tmp_path, "mol.gjf", "")
        ts  = _write(tmp_path, "mol_ts.out", "")
        captured = {}
        def capture(src_f, dst_f, geom, tail):
            captured["tail"] = tail
        with patch(FUNC_PATH+".file_check", side_effect=[src, ts]), \
             patch(FUNC_PATH+".gauss_extract_geom_from_opt_file", return_value="C 0 0 0"), \
             patch(FUNC_PATH+".create_gauss_input_file", side_effect=capture):
             create_gauss_irc_input(src, ts, max_points=100, max_cycle=60, step_size=-5)
        expected = ["maxpoints=100", "maxcycle=60", "stepsize=-5"]
        for val in expected:
            assert val in captured["tail"], f"expected {val} in {captured["tail"]}"


# ======================================================================================
# --- create_gauss_react_prod_input_from_irc - normal behaviour and input validation ---
# ======================================================================================

class TestCreateGaussReactProdInputFromIRC:

    def test_reactant_result_is_path(self, tmp_path):
        """ Tests that the reactant input file path is returned as a Path object """
        irc = _write(tmp_path, "irc.out", "FWD<M>REV")
        ts = _write(tmp_path, "mol.gjf", "")
        with patch(FUNC_PATH + ".file_check", side_effect=[ts, irc]), \
             patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("forward_text", "reverse_text")), \
             patch(FUNC_PATH + ".gauss_extract_geom_from_irc_point", return_value="C 0 0 0"), \
             patch(FUNC_PATH + ".create_gauss_input_file"), \
             patch(GAUSS_PATTERNS):
             react, _ = create_gauss_react_prod_input_from_irc(irc, ts)
        assert isinstance(react, Path), f"expected {Path}, got {type(react)}"
    
    def test_product_result_is_path(self, tmp_path):
        """ Tests that the product input file path is returned as a Path object """
        irc = _write(tmp_path, "irc.out", "FWD<M>REV")
        ts = _write(tmp_path, "mol.gjf", "")
        with patch(FUNC_PATH + ".file_check", side_effect=[ts, irc]), \
             patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("forward_text", "reverse_text")), \
             patch(FUNC_PATH + ".gauss_extract_geom_from_irc_point", return_value="C 0 0 0"), \
             patch(FUNC_PATH + ".create_gauss_input_file"), \
             patch(GAUSS_PATTERNS):
             _, prod = create_gauss_react_prod_input_from_irc(irc, ts)
        assert isinstance(prod, Path), f"expected {Path}, got {type(prod)}"

    def test_file_check_called_twice(self, tmp_path):
        """ Tests that file_check is called twice """
        irc = _write(tmp_path, "irc.out", "FWD<M>REV")
        ts = _write(tmp_path, "mol.gjf", "")
        with patch(FUNC_PATH + ".file_check", side_effect=[ts, irc]) as fc, \
             patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("forward_text", "reverse_text")), \
             patch(FUNC_PATH + ".gauss_extract_geom_from_irc_point", return_value="C 0 0 0"), \
             patch(FUNC_PATH + ".create_gauss_input_file"), \
             patch(GAUSS_PATTERNS):
             create_gauss_react_prod_input_from_irc(irc, ts)
        assert fc.call_count == 2,f"expected {2}, got {fc.call_count}"

    def test_geometry_extraction_called_twice(self, tmp_path):
        """ Tests that the IRC geometry is extracted twice """
        irc = _write(tmp_path, "irc.out", "FWD<M>REV")
        ts = _write(tmp_path, "mol.gjf", "")
        with patch(FUNC_PATH + ".file_check", side_effect=[ts, irc]), \
             patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("forward_text", "reverse_text")), \
             patch(FUNC_PATH + ".gauss_extract_geom_from_irc_point", return_value="C 0 0 0") as gegi, \
             patch(FUNC_PATH + ".create_gauss_input_file"), \
             patch(GAUSS_PATTERNS):
             create_gauss_react_prod_input_from_irc(irc, ts)
        assert gegi.call_count == 2, f"expected {2}, got {gegi.call_count}"

    def test_create_gauss_input_file_called_twice(self, tmp_path):
        """ Tests that create_gauss_input_file is called twice """
        irc = _write(tmp_path, "irc.out", "FWD<M>REV")
        ts = _write(tmp_path, "mol.gjf", "")
        with patch(FUNC_PATH + ".file_check", side_effect=[ts, irc]), \
             patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("forward_text", "reverse_text")), \
             patch(FUNC_PATH + ".gauss_extract_geom_from_irc_point", return_value="C 0 0 0"), \
             patch(FUNC_PATH + ".create_gauss_input_file") as cgi, \
             patch(GAUSS_PATTERNS):
             create_gauss_react_prod_input_from_irc(irc, ts)
        assert cgi.call_count == 2, f"expected {2}, got {cgi.call_count}"

    def test_gauss_irc_file_split_called_once(self, tmp_path):
        """ Tests that gauss_irc_file_split is called once """
        irc = _write(tmp_path, "irc.out", "FWD<M>REV")
        ts = _write(tmp_path, "mol.gjf", "")
        with patch(FUNC_PATH + ".file_check", side_effect=[ts, irc]), \
             patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("forward_text", "reverse_text")) as gif, \
             patch(FUNC_PATH + ".gauss_extract_geom_from_irc_point", return_value="C 0 0 0"), \
             patch(FUNC_PATH + ".create_gauss_input_file"), \
             patch(GAUSS_PATTERNS):
             create_gauss_react_prod_input_from_irc(irc, ts)
        gif.assert_called_once()  # it should be called once

    def test_geometry_extraction_uses_correct_reactant_key(self, tmp_path):
        """
        Tests that reactant geometry is extracted with reactant_key=True
        and product geometry with reactant_key=False
        """
        irc = _write(tmp_path, "irc.out", "")
        ts = _write(tmp_path, "mol.gjf", "")
        with patch(FUNC_PATH + ".file_check", side_effect=[ts, irc]), \
             patch(FUNC_PATH + ".gauss_irc_file_split", return_value=("FORWARD_TEXT", "REVERSE_TEXT")), \
             patch(FUNC_PATH + ".gauss_extract_geom_from_irc_point", return_value="C 0 0 0") as gegi, \
             patch(FUNC_PATH + ".create_gauss_input_file"), \
             patch(GAUSS_PATTERNS):
            create_gauss_react_prod_input_from_irc(irc, ts)
        result = [gegi.call_args_list[0].kwargs["reactant_key"], gegi.call_args_list[1].kwargs["reactant_key"]]
        expected = [True, False]
        for i, val in enumerate(result):
            assert val is expected[i], f"expected {expected[i]}, got {val}"

    def test_output_stems_have_react_and_prod(self, tmp_path):
        """ Tests that the output file stems contain '_react' and '_prod' """
        irc = _write(tmp_path, "irc.out", "")
        ts  = _write(tmp_path, "mol.gjf", "")
        with patch(FUNC_PATH+".file_check", side_effect=[ts, irc]), \
             patch(FUNC_PATH+".gauss_irc_file_split", return_value=("fwd", "rev")), \
             patch(FUNC_PATH+".gauss_extract_geom_from_irc_point", return_value="C 0 0 0"), \
             patch(FUNC_PATH+".create_gauss_input_file"), \
             patch(GAUSS_PATTERNS):
             react, prod = create_gauss_react_prod_input_from_irc(irc, ts)
        assert "_react" in react.stem, f"expected {"_react"} in {react.stem}; test #1"
        assert "_prod" in prod.stem, f"expected {"_prod"} in {prod.stem}; test #2"

    def test_react_uses_reverse_text_prod_uses_forward(self, tmp_path):
        """ Tests that the reactant uses the reverse IRC text and the product uses the forward IRC text """
        irc = _write(tmp_path, "irc.out", "")
        ts  = _write(tmp_path, "mol.gjf", "")
        geom_calls = []
        with patch(FUNC_PATH+".file_check", side_effect=[ts, irc]), \
             patch(FUNC_PATH+".gauss_irc_file_split", return_value=("FORWARD_TEXT", "REVERSE_TEXT")), \
             patch(FUNC_PATH + ".gauss_extract_geom_from_irc_point", side_effect=lambda f, text, **kwargs:
             geom_calls.append(text) or "C 0 0 0"), \
             patch(FUNC_PATH+".create_gauss_input_file"), \
             patch(GAUSS_PATTERNS):
             create_gauss_react_prod_input_from_irc(irc, ts)
        expected = ["REVERSE_TEXT", "FORWARD_TEXT"]
        for val in expected:
            assert val in geom_calls, f"expected {val} in {geom_calls}"