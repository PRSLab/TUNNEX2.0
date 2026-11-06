# --- Test: test_reader. Unit tests for reader.py ---
# Run with: pytest test_reader.py -v ---


# --- Modules ---
import pytest # type: ignore
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np


# --- Module to test ---
from tunnex_2.qmt_computations.data_reader_writer.reader import ( # type: ignore
    _parse_line,
    _validate_positive,
    _read_irc_matrix,
    read_input)


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.qmt_computations.data_reader_writer.reader"

HEADER_LINE_1 = "freq=101.3975; T=11"
HEADER_LINE_2 = "E0=-155.776; corr-ZPVE0=0.083; potential_scaling_factor=2.0; N=3"
HEADER_LINE_3 = "T_arrhenius_min=5; T_arrhenius_max=1000; T_step=5; tunn_prob_aver=1"

E_BLOCK = ["IRC; E", "-1.0; -10.0", "0.0; -9.5", "1.0; -10.0", "END; E"]
ZPVE_BLOCK = ["IRC; ZPVE", "-1.0; 0.01", "0.0; 0.02", "1.0; 0.01", "END; ZPVE"]

def _make_input_text(header1=HEADER_LINE_1, header2=HEADER_LINE_2, header3=HEADER_LINE_3,
        e_block=None, zpve_block=None):
    """ Should assemble a full, valid input-file text from header lines and IRC blocks """
    e_block = e_block if e_block is not None else E_BLOCK
    zpve_block = zpve_block if zpve_block is not None else ZPVE_BLOCK
    lines = [header1, header2, header3] + e_block + zpve_block
    return "\n".join(lines)

def _write_input_file(tmp_path, text, name="input.txt"):
    """ Should write the given text to a temporary input file and return its path """
    f = tmp_path / name
    f.write_text(text)
    return f


def _read_input_with_mocked_dataclasses(path):
    """ Should call read_input with TunnexInputSettings and IRCDataQMT """
    with ExitStack() as stack:
        settings_mock = stack.enter_context(
            patch(FUNC_PATH + ".TunnexInputSettings",
                  side_effect=lambda **kw: ("TunnexInputSettings", kw)))
        irc_mock = stack.enter_context(
            patch(FUNC_PATH + ".IRCDataQMT",
                  side_effect=lambda **kw: ("IRCDataQMT", kw)))
        result = read_input(path)
    return settings_mock, irc_mock, result


# ===========================================================================
# --- _parse_line - normal behaviour and input validation ---
# ===========================================================================
class TestParseLine:

    def test_parses_single_key_value_pair(self):
        """ Verify that a single key=value pair is parsed into a dict with a float value """
        result = _parse_line("freq=1500.0", 1)
        expected = {"freq": 1500.0}
        assert result == {"freq": 1500.0}, f"expected {expected}, got {result}"

    def test_parses_multiple_key_value_pairs(self):
        """ Verify that multiple semicolon-separated key=value pairs are all parsed """
        result = _parse_line("freq=1500.0; T=298.15", 1)
        expected = {"freq": 1500.0, "T": 298.15}
        assert result == expected, f"expected {expected}, got {result}"

    def test_strips_whitespace_around_keys_and_values(self):
        """ Verify that whitespace around keys and values is stripped before parsing """
        result = _parse_line(" freq = 1500.0 ; T = 298.15 ", 1)
        expected = {"freq": 1500.0, "T": 298.15}
        assert result == expected, f"expected {expected}, got {result}"

    def test_trailing_semicolon_is_tolerated(self):
        """ Verify that a trailing semicolon is tolerated, since the resulting empty item is skipped """
        result = _parse_line("freq=1500.0;", 1)
        assert result == {"freq": 1500.0}, f"expected {{'freq': 1500.0}}, got {result}"

    def test_double_semicolon_is_tolerated(self):
        """ Verify that a double semicolon (producing an empty item in between) is tolerated """
        result = _parse_line("freq=1500.0;;T=298.15", 1)
        expected = {"freq": 1500.0, "T": 298.15}
        assert result == expected, f"expected {expected}, got {result}"

    def test_whitespace_only_segment_still_raises(self):
        """ Verify that a segment containing only whitespace raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match="Invalid key=value format"):
            _parse_line("freq=1500.0; ;T=298.15", 1)
    
    def test_raises_on_missing_equals_sign(self):
        """ Verify that a line without an '=' sign raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match = "Invalid key=value format"):
            _parse_line("freq1500.0", 3)

    def test_raises_on_duplicate_keys(self):
        """ Verify that a duplicate key raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match = "duplicate key"):
            _parse_line("freq=1500.0; freq=2000.0; T=50", 3)

    def test_raises_on_non_numeric_value(self):
        """ Verify that a non-numeric value raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match = "Invalid key=value format"):
            _parse_line("freq=abc", 2)

    def test_raises_on_non_a_number(self):
        """ Verify that not-a-number value raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match = "Invalid key=value format"):
            _parse_line("freq=NAN", 2)
    
    def test_error_message_includes_line_number(self):
        """
        Verify that the raised error message includes
        the given line number. Expecting a ValueError """
        with pytest.raises(ValueError, match=r"\[Line 7\]"):
            _parse_line("bad", 7)

    @pytest.mark.parametrize("bad_value", ["inf", "-inf", "Infinity"])
    def test_rejects_infinite_values(self, bad_value):
        """
        Verify that +inf, -inf, and spelled-out Infinity values
        raise an error. Expecting a ValueError
        """
        with pytest.raises(ValueError, match="Invalid key=value format"):
            _parse_line(f"freq={bad_value}", 1)

    def test_raises_on_multiple_equals_signs_in_one_pair(self):
        """
        Verify that a pair with more than one '=' sign
        raises an error. Expecting a ValueError
        """
        with pytest.raises(ValueError, match="Invalid key=value format"):
            _parse_line("freq=1500=0", 1) # unpacking failure

    def test_mode_like_string_in_a_numeric_field_fails_at_parse_stage(self):
        """
        Verify that a non-numeric, mode-like string value
        raises an error. Expecting a ValueError
        """
        with pytest.raises(ValueError, match="Invalid key=value format"):
            _parse_line("freq=finite_sum", 1)


# ===========================================================================
# --- _validate_positive - normal behaviour and input validation ---
# ===========================================================================
class TestValidatePositive:

    def test_accepts_positive_value(self):
        """ Verify that a strictly positive value passes validation without error """
        result = _validate_positive("freq", 1500.0)
        assert result is None, f"expected {None}, got {result}"

    def test_accepts_zero_when_allow_zero_is_true(self):
        """ Verify that zero is accepted when allow_zero is True """
        result = _validate_positive("max_level", 0, allow_zero=True)
        assert result is None, f"expected {None}, got {result}"

    def test_accepts_positive_value_when_allow_zero_is_true(self):
        """ Verify that a positive value still passes validation when allow_zero is True """
        result = _validate_positive("max_level", 3, allow_zero=True)
        assert result is None, f"expected {None}, got {result}"

    def test_raises_on_zero_by_default(self):
        """ Verify that zero raises an when allow_zero is False. Expecting a ValueError """
        with pytest.raises(ValueError):
            _validate_positive("freq", 0)

    def test_raises_on_negative_value_by_default(self):
        """ Verify that a negative value raises an when allow_zero is False. Expecting a ValueError """
        with pytest.raises(ValueError):
            _validate_positive("freq", -1.0)

    def test_raises_on_negative_value_even_when_allow_zero_is_true(self):
        """ Verify that a negative value raises an even when allow_zero is True. Expecting a ValueError """
        with pytest.raises(ValueError):
            _validate_positive("max_level", -1, allow_zero=True)

    def test_error_message_includes_name_and_value(self):
        """
        Verify that the raised error message includes
        the parameter name and offending value. Expecting a ValueError
        """
        with pytest.raises(ValueError, match=r"freq.*-5"):
            _validate_positive("freq", -5)


# ===========================================================================
# --- _read_irc_matrix - normal behaviour and input validation ---
# ===========================================================================
class TestReadIrcMatrix:

    def test_extracts_x_and_y_values_between_tags(self):
        """ Verify that x and y values are correctly extracted from within the start and end tags """
        lines = E_BLOCK
        x_vals, y_vals = _read_irc_matrix(lines, "IRC; E", "END; E")
        assert np.allclose(x_vals, [-1.0, 0.0, 1.0]), f"expected {[-1.0, 0.0, 1.0]}, "
        f"got {x_vals}; test #1"
        assert np.allclose(y_vals, [-10.0, -9.5, -10.0]), f"expected {[-10.0, -9.5, -10.0]}, "
        f"got {y_vals}; test #2"

    def test_returns_numpy_arrays(self):
        """ Verify that both returned values are NumPy arrays """
        x_vals, y_vals = _read_irc_matrix(ZPVE_BLOCK, "IRC; ZPVE", "END; ZPVE")
        assert isinstance(x_vals, np.ndarray), f"expected {np.ndarray}, got {type(x_vals)}; test #1"
        assert isinstance(y_vals, np.ndarray), f"expected {np.ndarray}, got {type(y_vals)}; test #2"

    def test_skips_blank_lines_inside_block(self):
        """ Verify that blank lines inside the block are skipped without affecting the extraction """
        lines = ["IRC; ZPVE", "-1.0; -10.0", "", "0.0; -9.5", "END; ZPVE"]
        x_vals, _ = _read_irc_matrix(lines, "IRC; ZPVE", "END; ZPVE")
        assert np.allclose(x_vals, [-1.0, 0.0]), f"expected {[-1.0, 0.0]}, got {x_vals}"

    def test_ignores_lines_outside_block(self):
        """ Verify that lines outside the start and end tags are not included in the extraction """
        lines = ["junk; line"] + E_BLOCK + ["also; junk"]
        x_vals, _ = _read_irc_matrix(lines, "IRC; E", "END; E")
        assert len(x_vals) == 3, f"expected {3}, got {len(x_vals)}"

    def test_accepts_exactly_two_points(self):
        """ Verify that a block with exactly two data points is accepted (the minimum allowed) """
        lines = ["IRC; E", "0.0; -10.0", "1.0; -9.0", "END; E"]
        x_vals, _ = _read_irc_matrix(lines, "IRC; E", "END; E")
        assert len(x_vals) == 2, f"expected {2}, got {len(x_vals)}"
    
    def test_raises_on_missing_block(self):
        """
        Verify that an error is raised when the start tag
        is not found at all. Expecting a ValueError
        """
        with pytest.raises(ValueError, match="empty or missing"):
            _read_irc_matrix(["some", "other", "lines"], "IRC; E", "END; E")

    def test_raises_on_empty_block(self):
        """
        Verify that an error is raised when the block
        between tags contains no data rows. Expecting a ValueError
        """
        lines = ["IRC; E", "END; E"]
        with pytest.raises(ValueError, match="empty or missing"):
            _read_irc_matrix(lines, "IRC; E", "END; E")

    def test_raises_on_key_swap(self):
        """
        Verify that an error is raised when the start and end markers
        are swapped in the text. Expecting a ValueError
        """
        lines = ["END; E", "1.0; -10.0", "0.5; -9.5", "IRC; E"]
        with pytest.raises(ValueError, match="empty or missing"):
            _read_irc_matrix(lines, "IRC; E", "END; E")

    def test_raises_on_other_block(self):
        """
        Verify that an error is raised when the start and end markers
        are applied for the other block in the text. Expecting a ValueError
        """
        lines = ["IRC; E", "1.0; -10.0", "0.5; -9.5", "END; E"]
        with pytest.raises(ValueError, match="empty or missing"):
            _read_irc_matrix(lines, "IRC; ZPVE", "END; ZPVE")
            # markers here are different from ones in lines

    def test_raises_on_malformed_row(self):
        """
        Verify that a row without a semicolon-separated pair
        raises an error. Expecting a ValueError
        """
        lines = ["IRC; E", "not_a_valid_row", "END; E"]
        with pytest.raises(ValueError, match="Invalid IRC row"):
            _read_irc_matrix(lines, "IRC; E", "END; E")

    def test_raises_on_non_increasing_x_values(self):
        """
        Verify that non-strictly-increasing IRC x values
        raise an error. Expecting a ValueError
        """
        lines = ["IRC; E", "1.0; -10.0", "0.5; -9.5", "END; E"]
        with pytest.raises(ValueError, match="must be strictly increasing"):
            _read_irc_matrix(lines, "IRC; E", "END; E")

    def test_raises_on_duplicate_x_values(self):
        """
        Verify that a repeated (non-strictly-increasing) IRC x value
        raises an error. Expecting a ValueError
        """
        lines = ["IRC; E", "0.0; -10.0", "0.0; -9.5", "END; E"]
        with pytest.raises(ValueError, match="must be strictly increasing"):
            _read_irc_matrix(lines, "IRC; E", "END; E")

    def test_raises_on_non_a_number_values(self):
        """ Verify that not-a-number value raises an error. Expecting a ValueError """
        lines = ["IRC; E", "0.0; -10.0", "NaN; -9.5", "END; E"]
        with pytest.raises(ValueError, match="Invalid IRC row:"):
            _read_irc_matrix(lines, "IRC; E", "END; E")

    def test_rejects_infinite_x_value(self):
        """ Verify that an infinite x-value in an IRC row an error. Expecting a ValueError """
        lines = ["IRC; E", "-inf; -10.0", "1.0; -9.5", "END; E"]
        with pytest.raises(ValueError, match="Invalid IRC row"):
            _read_irc_matrix(lines, "IRC; E", "END; E")

    def test_rejects_infinite_y_value(self):
        """ Verify that an infinite y-value in an IRC row an error. Expecting a ValueError """
        lines = ["IRC; E", "0.0; inf", "1.0; -9.5", "END; E"]
        with pytest.raises(ValueError, match="Invalid IRC row"):
            _read_irc_matrix(lines, "IRC; E", "END; E")

    def test_raises_on_single_point_block(self):
        """ Verify that a block with only one data point raises a ValueError """
        lines = ["IRC; E", "0.0; -10.0", "END; E"]
        with pytest.raises(ValueError, match="Expected at least 2 points"):
            _read_irc_matrix(lines, "IRC; E", "END; E")

    def test_error_message_names_the_correct_block(self):
        """ Verify that the too-few-points error message references the given start_tag """
        lines = ["IRC; ZPVE", "0.0; 0.01", "END; ZPVE"]
        with pytest.raises(ValueError, match=r"IRC; ZPVE"):
            _read_irc_matrix(lines, "IRC; ZPVE", "END; ZPVE")


# ===========================================================================
# --- read_input - normal behaviour and input validation ---
# ===========================================================================
class TestReadInputFile:

    def test_accepts_str_path_as_well_as_path_object(self, tmp_path):
        """ Verify that read_input accepts a plain string path, not just a Path object """
        f = _write_input_file(tmp_path, _make_input_text())
        result = _read_input_with_mocked_dataclasses(str(f))
        assert result is not None, f"expected not {None}, got {result}"
    
    def test_returns_a_two_element_tuple(self, tmp_path):
        """ Verify that read_input returns exactly a (params, IRC) tuple """
        f = _write_input_file(tmp_path, _make_input_text())
        _, _, result = _read_input_with_mocked_dataclasses(f)
        assert len(result) == 2, f"expected {2}, got {len(result)}"

    def test_constructs_settings_with_correct_scalar_parameters(self, tmp_path):
        """ Verify that TunnexInputSettings is constructed with the correctly parsed scalar parameters """
        f = _write_input_file(tmp_path, _make_input_text())
        settings_mock, _, _ = _read_input_with_mocked_dataclasses(f)
        kwargs = settings_mock.call_args.kwargs
        assert kwargs["freq"] == 101.3975, f"expected {101.3975}, got {kwargs['freq']}; test #1"
        assert kwargs["T"] == 11, f"expected {11}, got {kwargs['T']}; test #2"
        assert kwargs["E0"] == -155.776, f"expected {-155.776}, got {kwargs['E0']}; test #3"
        assert kwargs["ZPVE0"] == 0.083, f"expected {0.083}, got {kwargs['ZPVE0']}; test #4"
        assert kwargs["potential_scaling_factor"] == 2.0, f"expected {2.0}, "
        f"got {kwargs['potential_scaling_factor']}; test #5"

    def test_constructs_settings_with_upper_level_as_int(self, tmp_path):
        """ Verify that "number_of_levels" is passed to TunnexInputSettings as an int """
        f = _write_input_file(tmp_path, _make_input_text())
        settings_mock, _, _ = _read_input_with_mocked_dataclasses(f)
        kwargs = settings_mock.call_args.kwargs
        assert kwargs["number_of_levels"] == 3, f"expected {3}, "
        f"got {kwargs["number_of_levels"]}; test #1"
        assert isinstance(kwargs["number_of_levels"], int), f"expected {int}, "
        f"got {type(kwargs["number_of_levels"])}; test #2"

    def test_constructs_settings_with_correct_arrhenius_parameters(self, tmp_path):
        """
        Verify that TunnexInputSettings is constructed
        with the correctly parsed Arrhenius temperature range
        """
        f = _write_input_file(tmp_path, _make_input_text())
        settings_mock, _, _ = _read_input_with_mocked_dataclasses(f)
        kwargs = settings_mock.call_args.kwargs
        assert kwargs["T_arrhenius_min"] == 5.0, f"expected {5.0}, "
        f"got {kwargs['T_arrhenius_min']}; test #1"
        assert kwargs["T_arrhenius_max"] == 1000.0, f"expected {1000.0}, "
        f"got {kwargs['T_arrhenius_max']}; test #2"
        assert kwargs["T_step"] == 5.0, f"expected {5.0}, "
        f"got {kwargs['T_step']}; test #3"

    def test_constructs_irc_data_with_correct_arrays(self, tmp_path):
        """ Verify that IRCDataQMT is constructed with the correctly parsed E and ZPVE arrays """
        f = _write_input_file(tmp_path, _make_input_text())
        _, irc_mock, _ = _read_input_with_mocked_dataclasses(f)
        kwargs = irc_mock.call_args.kwargs
        assert np.allclose(kwargs["E_x"], [-1.0, 0.0, 1.0]), f"expected {[-1.0, 0.0, 1.0]}, "
        f"got {kwargs['E_x']}; test #1"
        assert np.allclose(kwargs["E_y"], [-10.0, -9.5, -10.0]), f"expected {[-10.0, -9.5, -10.0]}, "
        f"got {kwargs['E_y']}; test #2"
        assert np.allclose(kwargs["ZPVE_x"], [-1.0, 0.0, 1.0]), f"expected {[-1.0, 0.0, 1.0]}, "
        f"got {kwargs['ZPVE_x']}; test #3"
        assert np.allclose(kwargs["ZPVE_y"], [0.01, 0.02, 0.01]), f"expected {[0.01, 0.02, 0.01]}, "
        f"got {kwargs['ZPVE_y']}; test #4"

    def test_returned_tuple_contains_the_constructed_objects(self, tmp_path):
        """
        Verify that the returned tuple contains the objects
        built from TunnexInputSettings and IRCDataQMT
        """
        f = _write_input_file(tmp_path, _make_input_text())
        _, _, (params, irc) = _read_input_with_mocked_dataclasses(f)
        expected = ["TunnexInputSettings", "IRCDataQMT"]
        assert params[0] == expected[0], f"expected {expected[0]}, got {params[0]}; test #1"
        assert irc[0] == expected[1], f"expected {expected[1]}, got {irc[0]}; test #2"
       
    def test_accepts_e_and_zpve_blocks_with_different_number_of_points(self, tmp_path):
        """
        Verify that E and ZPVE blocks may have a different number of rows,
        as long as endpoints match
        """
        e_block = ["IRC; E", "-1.0; -10.0", "-0.5; -9.7", "0.0; "
                   "-9.5", "0.5; -9.8", "1.0; -10.0", "END; E"]
        zpve_block = ["IRC; ZPVE", "-1.0; 0.01", "1.0; 0.01", "END; ZPVE"]
        text = _make_input_text(e_block=e_block, zpve_block=zpve_block)
        f = _write_input_file(tmp_path, text)
        result = _read_input_with_mocked_dataclasses(f)
        assert result is not None, f"expected not {None}, got {result}"
    
    @pytest.mark.parametrize("code,expected_mode", [(1, "finite_sum"), (2, "infinite_sum"), (3, "integral")])
    def test_valid_codes_map_to_expected_mode_names(self, tmp_path, code, expected_mode):
        """ Verify that each allowed numeric code maps to its corresponding mode name """
        text = _make_input_text(header3=f"T_arrhenius_min=200.0; T_arrhenius_max=400.0; "
                                         f"T_step=10.0; tunn_prob_aver={code}")
        f = _write_input_file(tmp_path, text)
        settings_mock, _, _ = _read_input_with_mocked_dataclasses(f)
        kwargs = settings_mock.call_args.kwargs
        assert kwargs["prob_aver_mode_qmt"] == expected_mode, f"expected {expected_mode}, "
        f"got {kwargs['prob_aver_mode_qmt']}"
    
    def test_raises_file_not_found_error_for_missing_file(self, tmp_path):
        """
        Verify that an error is raised when the input file
        does not exist. Expecting a FileNotFoundError
        """
        missing = tmp_path / "does_not_exist.txt"
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            read_input(missing)

    def test_raises_on_too_few_header_lines(self, tmp_path):
        """
        Verify that an is raised when the file has fewer
        than 3 header lines. Expecting a ValueError
        """
        f = _write_input_file(tmp_path, "freq=1500.0; T=298.15\nE0=-1.0")
        with pytest.raises(ValueError, match="Input file must contain at least"):
            read_input(f)

    def test_raises_on_missing_key_in_line_one(self, tmp_path):
        """
        Verify that a missing key in the first header line
        raises an error. Expecting a ValueError
        """
        text = _make_input_text(header1="freq=1500.0; bogus=1.0")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="Missing required keys"):
            read_input(f)

    def test_raises_on_unexpected_key_in_line_two(self, tmp_path):
        """
        Verify that an unexpected key in the second header line
        raises an error. Expecting a ValueError
        """
        text = _make_input_text(header2="E0=-1.0; corr-ZPVE0=0.01; "
                                "potential_scaling_factor=1.0; N=2; extra=9")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="Unrecognized keys"):
            read_input(f)

    def test_raises_on_unexpected_key_in_line_three(self, tmp_path):
        """
        Verify that an unexpected key in the third header line
        raises an error. Expecting a ValueError
        """
        text = _make_input_text(header3="T_arrhenius_min=200.0; T_arrhenius_max=400.0; T_step=10.0; "
                                "tunn_prob_aver=2; extra=1")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="Unrecognized keys"):
            read_input(f)

    def test_raises_on_non_positive_freq(self, tmp_path):
        """ Verify that a non-positive freq value raises an error. Expecting a ValueError """
        text = _make_input_text(header1="freq=0.0; T=298.15")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="must be > 0"):
            read_input(f)

    def test_raises_on_non_positive_temperature(self, tmp_path):
        """ Verify that a non-positive temperature value raises an error. Expecting a ValueError """
        text = _make_input_text(header1="freq=1500.0; T=-1.0")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="must be > 0"):
            read_input(f)

    def test_raises_on_non_positive_scaling_factor(self, tmp_path):
        """
        Verify that a non-positive potential_scaling_factor
        raises an error. Expecting a ValueError
        """
        text = _make_input_text(header2="E0=-1.0; corr-ZPVE0=0.01; potential_scaling_factor=0.0; N=2")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="must be > 0"):
            read_input(f)

    def test_raises_on_negative_upper_level_to_compute(self, tmp_path):
        """
        Verify that a negative N (the_upper_level_to_compute)
        raises an error. Expecting a ValueError
        """
        text = _make_input_text(header2="E0=-1.0; corr-ZPVE0=0.01; potential_scaling_factor=1.0; N=-1")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="must be >= 0"):
            read_input(f)

    def test_raises_on_non_integer_upper_level_to_compute(self, tmp_path):
        """
        Verify that non-integer (the_upper_level_to_compute)
        raises an error. Expecting a ValueError
        """
        text = _make_input_text(header2="E0=-1.0; corr-ZPVE0=0.01; potential_scaling_factor=1.0; N=0.5")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="must be an integer"):
            read_input(f)
    
    def test_raises_on_non_positive_T_arrhenius_min(self, tmp_path):
        """
        Verify that a non-positive T_arrhenius_min
        raises an error. Expecting a ValueError
        """
        text = _make_input_text(header3="T_arrhenius_min=0.0; T_arrhenius_max=400.0; "
                                "T_step=10.0; tunn_prob_aver=3")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="must be > 0"):
            read_input(f)

    def test_raises_on_non_positive_T_arrhenius_max(self, tmp_path):
        """ Verify that a non-positive T_arrhenius_max raises an error. Expecting a ValueError """
        text = _make_input_text(header3="T_arrhenius_min=200.0; T_arrhenius_max=0.0; "
                                "T_step=10.0; tunn_prob_aver=2")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="must be > 0"):
            read_input(f)

    def test_raises_on_non_positive_T_step(self, tmp_path):
        """ Verify that a non-positive T_step raises an error. Expecting a ValueError """
        text = _make_input_text(header3="T_arrhenius_min=200.0; T_arrhenius_max=400.0; "
                                "T_step=0.0; tunn_prob_aver=1")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="must be > 0"):
            read_input(f)

    def test_raises_when_T_arrhenius_min_greater_than_max(self, tmp_path):
        """
        Verify that an error is raised when T_arrhenius_min
        exceeds T_arrhenius_max. Expecting a ValueError
        """
        text = _make_input_text(header3="T_arrhenius_min=500.0; T_arrhenius_max=400.0; "
                                "T_step=10.0; tunn_prob_aver=1")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="cannot be >="):
            read_input(f)

    def test_raises_when_initial_irc_values_differ(self, tmp_path):
        """
        Verify that an error is raised when the E and ZPVE blocks
        start at different IRC values. Expecting a ValueError
        """
        e_block = ["IRC; E", "-1.0; -10.0", "0.0; -9.5", "1.0; -10.0", "END; E"]
        zpve_block = ["IRC; ZPVE", "-2.0; 0.01", "0.0; 0.02", "1.0; 0.01", "END; ZPVE"]
        text = _make_input_text(e_block=e_block, zpve_block=zpve_block)
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="Initial IRC values"):
            read_input(f)

    def test_raises_when_final_irc_values_differ(self, tmp_path):
        """
        Verify that an error is raised when the E and ZPVE blocks
        end at different IRC values. Expecting a ValueError
        """
        e_block = ["IRC; E", "-1.0; -10.0", "0.0; -9.5", "1.0; -10.0", "END; E"]
        zpve_block = ["IRC; ZPVE", "-1.0; 0.01", "0.0; 0.02", "2.0; 0.01", "END; ZPVE"]
        text = _make_input_text(e_block=e_block, zpve_block=zpve_block)
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="Final IRC values"):
            read_input(f)

    def test_negative_non_integer_N_raises_non_negative_error(self, tmp_path):
        """ Verify that for a negative, non-integer N raises an error. Expecting a ValueError """
        text = _make_input_text(header2="E0=-1.0; corr-ZPVE0=0.01; potential_scaling_factor=1.0; N=-0.5")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="must be >= 0"):
            read_input(f)

    def test_raises_on_unknown_numeric_code(self, tmp_path):
        """ Verify that a numeric code outside {1, 2, 3} raises an error. Expecting a ValueError """
        text = _make_input_text(header3="T_arrhenius_min=200.0; T_arrhenius_max=400.0; "
                                         "T_step=10.0; tunn_prob_aver=4")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="temperature averaging mode"):
            read_input(f)

    def test_raises_on_non_integer_code(self, tmp_path):
        """
        Verify that a non-integer numeric value for the mode code
        raises an error. Expecting a ValueError
        """
        text = _make_input_text(header3="T_arrhenius_min=200.0; T_arrhenius_max=400.0; "
                                         "T_step=10.0; tunn_prob_aver=1.5")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError, match="temperature averaging mode"):
            read_input(f)