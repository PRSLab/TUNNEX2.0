# --- Test: test_gaussian_patterns. Unit tests for .\gaussian\gaussian_patterns.py
# Run with: pytest test_gaussian_patterns.py -v ---


# --- Modules ---
import math


# --- Module to test ---
from tunnex_2.irc_computations.gaussian import gaussian_patterns # type: ignore


# ============================================================================
# --- String patterns - settings ---
# ============================================================================
class TestStringPatterns:
    
    def test_string_pattenrs(self):
        """ Tests that Gaussian string pattern constants have the expected values """
        pat = gaussian_patterns.METHOD_TAIL_TS
        exp_pat = r"opt=(ts,calcfc,noeigen) freq=NoRaman"
        assert pat == exp_pat, f"expected {exp_pat}, got {pat}; test #1"
        pat = gaussian_patterns.METHOD_TAIL_REACT_PROD
        exp_pat = r"Opt freq=NoRaman"
        assert pat == exp_pat, f"expected {exp_pat}, got {pat}; test #2"
        pat = gaussian_patterns.PATTERN_COMMENT
        exp_pat = r"Comment"
        assert pat == exp_pat, f"expected {exp_pat}, got {pat}; test #3"
        pat = gaussian_patterns.IRC_SPLIT_MARKER_FORWARD
        exp_pat = r"Calculation of FORWARD path complete."
        assert pat == exp_pat, f"expected {exp_pat}, got {pat}; test #4"
        pat = gaussian_patterns.IRC_SPLIT_MARKER_REVERSE
        exp_pat = r"Calculation of REVERSE path complete."
        assert pat == exp_pat, f"expected {exp_pat}, got {pat}; test #5"


# ============================================================================
# --- PATTERN_COMPUTATIONAL_METHOD_GAUSS - match and no match ---
# ============================================================================
class TestPatternComputationalMethodGauss:

    P = gaussian_patterns.PATTERN_COMPUTATIONAL_METHOD_GAUSS

    def test_matches_standard_method_line(self):
        """ Tests that a standard Gaussian computational method line is matched """
        m = self.P.match("#p B3LYP/6-31G* opt")
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_prefix_and_method(self):
        """ Tests that the computational method line prefix and method are captured correctly """
        m = self.P.match("#N B3LYP/6-311++G** freq=NoRaman")
        expected = ["#N", "B3LYP/6-311++G**"]
        for i, val in enumerate(expected):
            assert m.group(i+1) == val, f"expected {val}, got {m.group(i+1)}"

    def test_no_match_hash_without_letter(self):
        """ Tests that a hash symbol without a following letter is not matched """
        m = self.P.match("# MP2/6-31G* opt")
        assert m is None, f"expected {None}, got {m}"

    def test_no_match_for_blank_line(self):
        """ Tests that a blank line is not matched """
        m = self.P.match("")
        assert m is None, f"expected {None}, got {m}"

    def test_no_match_for_geometry_line(self):
        """ Tests that a Gaussian geometry line is not matched """
        m1 = self.P.match("C  0.000  0.000  0.000")
        m2 = self.P.match("Comment")
        assert m1 is None, f"expected {None}, got {m1}; test #1"
        assert m2 is None, f"expected {None}, got {m2}; test #2"


# ============================================================================
# --- PATTERN_GEOMETRY_LINE_INPUT_GAUSS - match and no match ---
# ============================================================================
class TestPatternGeometryLineInputGauss:

    P = gaussian_patterns.PATTERN_GEOMETRY_LINE_INPUT_GAUSS

    def test_matches_symbol_and_three_coords(self):
        """ Tests that an atomic symbol followed by three coordinates is matched """
        m = self.P.search("C  0.000000  1.234567 -0.123456")
        assert m is not None, f"expected not {None}, got {m}"

    def test_matches_two_letter_element(self):
        """ Tests that a two-letter atomic symbol is matched """
        m = self.P.search("Cl  0.000000  1.234567  0.000000")
        assert m is not None, f"expected not {None}, got {m}"

    def test_matches_numeric_atom_type(self):
        """ Tests that a numeric atom type is matched """
        m = self.P.search("6  0.000000  1.234567  0.000000")
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_element_and_coords(self):
        """ Tests that the atomic symbol and three coordinates are captured """
        line = "O  1.100000  2.200000  3.300000"
        m = self.P.search(line)
        expected = line.split()
        for i, val in enumerate(expected):
            assert m.group(i+1) == val, f"expected {val}, got {m.group(i+1)}; test #{i+1}"

    def test_matches_scientific_notation(self):
        """ Tests that coordinates in scientific notation are matched """
        m = self.P.search("H  1.0E-5  2.3E+1  -4.5e-3")
        assert m is not None, f"expected not {None}, got {m}"

    def test_no_match_for_method_line(self):
        """ Tests that a computational method line is not matched """
        m = self.P.search("# B3LYP/6-31G* opt")
        assert m is None, f"expected {None}, got {m}"

    def test_no_match_for_blank_line(self):
        """ Tests that a blank line is not matched """
        m = self.P.search("")
        assert m is None, f"expected {None}, got {m}"


# ============================================================================
# --- PATTERN_OPTIMIZED_SPECIES_GEOMETRY_BLOCK - match and no match ---
# ============================================================================
class TestPatternOptimizedSpeciesGeometryBlock:

    P = gaussian_patterns.PATTERN_OPTIMIZED_SPECIES_GEOMETRY_BLOCK
    
    SINGLE_BLOCK_1 = (
        "Standard orientation:\n"
        " ---------------------------------------------------------------------\n"
        " Center     Atomic      Atomic             Coordinates (Angstroms)\n"
        " Number     Number      Type               X          Y          Z\n"
        " ---------------------------------------------------------------------\n"
        "      1          1           0        0.000000    0.635864    0.000000\n"
        "      2          1           0        0.924490    1.200305    0.000000\n"
        " ---------------------------------------------------------------------\n"
        " Rotational constants (GHZ):")

    SINGLE_BLOCK_2 = (
        "Standard orientation:\n"
        "----------------------------------------------------------------------\n"
        " Center     Atomic      Atomic             Coordinates (Angstroms)    \n"
        " Number     Number       Type             X           Y           Z   \n"
        "----------------------------------------------------------------------\n"
        "      1          6           0        0.000000    0.635864    0.000000\n"
        "      2          1           0        0.924490    1.200305    0.050000\n"
        "----------------------------------------------------------------------\n"
        " Rotational constants (GHZ):     57.1296489   10.1447927     9.0979548")

    EXPECTED_CAPTURE = (
    "1          6           0        0.000000    0.635864    0.000000\n"
    "      2          1           0        0.924490    1.200305    0.050000")
    
    
    def test_finds_geometry_block(self):
        """ Tests that a standard orientation geometry block is found """
        m = self.P.search(self.SINGLE_BLOCK_1)
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_coordinate_lines(self):
        """ Tests that the coordinate lines are captured from the geometry block """
        m = self.P.search(self.SINGLE_BLOCK_1)
        assert "0.635864" in m.group(1), f"expected {"0.635864"} in {m.group(1)}"

    def test_no_match_without_rotational_constants(self):
        """ Tests that a geometry block without rotational constants is not matched """
        text = "Standard orientation:\n ---\n Center\n ---\n data\n ---\n"
        m = self.P.search(text)
        assert m is None, f"expected not {None}, got {m}"

    def test_captures_only_geometry_lines(self):
        """ Tests that the captured group contains only the atom number and coordinate lines """
        m = self.P.search(self.SINGLE_BLOCK_2)
        assert m.group(1) == self.EXPECTED_CAPTURE, (f"expected {self.EXPECTED_CAPTURE}, got {m.group(1)}")

    def test_capture_excludes_header_row(self):
        """ Tests that the header text is not captured """
        m = self.P.search(self.SINGLE_BLOCK_2)
        assert "Center" not in m.group(1), f"expected no {"Center"} in capture, got {m.group(1)}; test #1"
        assert "Coordinates" not in m.group(1), f"expected no header text in capture, got {m.group(1)}; test #2"

    def test_dotall_matches_across_newlines(self):
        """ Tests that the matched block can span multiple lines """
        m = self.P.search(self.SINGLE_BLOCK_2)
        assert m is not None, f"expected not {None}, got {m}; test #1"
        assert "\n" in m.group(1), f"expected {"\n"} in {m.group(1)}; test #2"

    def test_no_match_without_standard_orientation_header(self):
        """ Tests that a block missing the 'Standard orientation' header is not matched """
        text = self.SINGLE_BLOCK_2.replace("Standard orientation:", "Input orientation:")
        m = self.P.search(text)
        assert m is None, f"expected {None}, got {m}"

    def test_no_match_without_rotational_constants(self):
        """ Tests that a block missing the trailing 'Rotational constants' marker is not matched """
        text = self.SINGLE_BLOCK_2.replace(
        "Rotational constants (GHZ)", "")
        m = self.P.search(text)
        assert m is None, f"expected {None}, got {m}"

    def test_finditer_returns_one_match_per_block(self):
        """ Tests that two consecutive geometry blocks each produce their own match """
        second_block = self.SINGLE_BLOCK_2.replace("0.635864", "0.700000")
        text = self.SINGLE_BLOCK_2 + "\nSome intermediate text\n\n" + second_block
        matches = list(self.P.finditer(text))
        assert len(matches) == 2, f"expected {2} matches, got {len(matches)}"

    def test_non_greedy_does_not_leak_into_next_block(self):
        """ Tests that the first block's match does not capture coordinates from the second block """
        second_block = self.SINGLE_BLOCK_2.replace("0.635864", "0.700000")
        text = self.SINGLE_BLOCK_2 + "\nSome intermediate text\n\n" + second_block
        matches = list(self.P.finditer(text))
        first_capture = matches[0].group(1)
        assert "0.700000" not in first_capture, f"expected {"0.700000"} not in {first_capture}"

    def test_second_match_captures_its_own_geometry(self):
        """ Tests that the second block's match captures its own (distinct) coordinates """
        second_block = self.SINGLE_BLOCK_2.replace("0.635864", "0.700000")
        text = self.SINGLE_BLOCK_2 + "\nSome intermediate text\n\n" + second_block
        matches = list(self.P.finditer(text))
        second_capture = matches[1].group(1)
        assert "0.700000" in second_capture, f"expected {"0.700000"} in {second_capture}; test #1"
        assert "0.635864" not in second_capture, f"expected {"0.635864"} not in {second_capture}; test #2"

    def test_no_match_on_empty_string(self):
        """ Tests that an empty string does not match """
        m = self.P.search("")
        assert m is None, f"expected {None}, got {m}"


# ============================================================================
# --- PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS - match and no match ---
# ============================================================================
class TestPatternGeometryLineOutputGauss:

    P = gaussian_patterns.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS

    def test_matches_standard_output_line(self):
        """ Tests that a standard Gaussian geometry output line is matched """
        line = "      1          6           0        0.000000    0.635864    0.000000"
        m = self.P.search(line)
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_four_groups(self):
        """ Tests that the atom index, atomic number, charge, and three coordinates are captured """
        line = "      2          1           0        0.924490    1.200305   -0.050000"
        m = self.P.search(line)
        expected = ["1", "0.924490", "1.200305", "-0.050000"]
        for i, val in enumerate(expected):
            assert m.group(i+1) == val, f"expected {val}, got {m.group(i+1)}; test #{i+1}"

    def test_matches_negative_coordinates(self):
        """ Tests that geometry lines with negative coordinates are matched """
        line = "      3          8           0       -1.234567   -0.000001    2.345678"
        m = self.P.search(line)
        assert m is not None, f"expected not {None}, got {m}"

    def test_no_match_for_header_line(self):
        """ Tests that a geometry table header line is not matched """
        m = self.P.search(" Center     Atomic")
        assert m is None, f"expected {None}, got {m}"

    def test_no_match_for_dashes(self):
        """ Tests that a separator line containing dashes is not matched """
        m = self.P.search(" -----------------------------------")
        assert m is None, f"expected {None}, got {m}"

    def test_matches_line_without_atomic_type(self):
            """ Tests that a 5-column geometry line is matched """
            line = "      1          6           0.000000    0.635864    0.000000"
            m = self.P.search(line)
            assert m is not None, f"expected not {None}, got {m}; test #1"
            assert m.group(1) == "6", f"expected {"6"}, got {m.group(1)}; test #2"
            assert m.group(2) == "0.000000", f"expected {"0.000000"}, got {m.group(2)}; test #3"
    
    def test_finditer_does_not_merge_across_broken_line(self):
        """ Tests that a truncated line does not bleed into the next valid line """
        block = (# the first line truncated, missing Z coordinate
            "      1          6           0        0.000000    0.635864\n"
            "      2          1           0        0.924490    1.200305   -0.050000\n")
        matches = list(self.P.finditer(block))
        assert len(matches) == 1, f"expected {1} match, got {len(matches)}; test #1"
        assert matches[0].group(1) == "1", f"expected {"1"}, got {matches[0].group(1)}; test #2"

    def test_finditer_matches_multiple_valid_lines(self):
        """ Tests that a real multi-line block yields one match per geometry line """
        block = (
            "      1          6           0        0.000000    0.635864    0.000000\n"
            "      2          1           0        0.924490    1.200305   -0.050000\n")
        matches = list(self.P.finditer(block))
        assert len(matches) == 2, f"expected {2} matches, got {len(matches)}"

    def test_no_match_with_trailing_text(self):
        """ Tests that a line with trailing non-whitespace content is not matched """
        line = "      1          6           0        0.000000    0.635864    0.000000  extra"
        m = self.P.search(line)
        assert m is None, f"expected {None}, got {m}"


# ============================================================================
# --- PATTERN_EL_ENERGY_SINGLE_POINT - match and no match ---
# ============================================================================
class TestPatternElEnergySinglePoint:

    P = gaussian_patterns.PATTERN_EL_ENERGY_SINGLE_POINT

    def test_matches_scf_done_line(self):
        """ Tests that a standard electronic energy line is matched """
        line = " SCF Done:  E(RB3LYP) =  -153.799148571     A.U. after   11 cycles"
        m = self.P.search(line)
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_energy_value(self):
        """ Tests that the electronic energy value is captured correctly """
        line = " SCF Done:  E(RB3LYP) =  -153.799148571     A.U."
        m = self.P.search(line)
        result = float(m.group(1))
        assert math.isclose(result, -153.799148571), f"expected {-153.799148571}, got {result}"

    def test_matches_mp2_energy(self):
        """ Tests that an MP2 electronic energy line is matched """
        line = " SCF Done:  E(UMP2) =  -75.123456789"
        m = self.P.search(line)
        assert m is not None, f"expected not {None}, got {m}"

    def test_matches_positive_energy(self):
        """ Tests that a positive electronic energy value is matched """
        line = " SCF Done:  E(RHF) =  0.123456789"
        m = self.P.search(line)
        assert m is not None, f"expected not {None}, got {m}"

    def test_no_match_for_irc_energy_line(self):
        """ Tests that an IRC energy line is not matched """
        m = self.P.search("   16   -0.049479    1.68034")
        assert m is None, f"expected {None}, got {m}"

    def test_no_match_for_ts_energy_line(self):
        """ Tests that a transition state relative energy line is not matched """
        line = "Energies reported relative to the TS energy of -153.0"
        m = self.P.search(line)
        assert m is None, f"expected {None}, got {m}"

# ============================================================================
# --- PATTERN_MASS_LINES_AND_VALUES - match and no match ---
# ============================================================================
class TestPatternMassLinesAndValues:

    P = gaussian_patterns.PATTERN_MASS_LINES_AND_VALUES

    MASS_LINE_C = "Atom  1 has atomic number  6 and mass  12.00000"
    MASS_LINE_H = "Atom  2 has atomic number  1 and mass   1.00783"

    def test_mass_lines_and_values_finds_line(self):
        """ Tests that an atomic mass line is matched """
        m = self.P.search(self.MASS_LINE_C)
        assert m is not None, f"expected not {None}, got {m}"

    def test_mass_lines_and_values_captures_mass(self):
        """ Tests that the atomic mass value is captured correctly """
        m = self.P.findall(self.MASS_LINE_C)
        assert m is not None, f"expected not {None}, got {m}; test #1"
        result = [float(mass) for _, mass in m]
        assert math.isclose(result[0], 12.0), f"expected {12.0}, got {result[0]}; test #2"

    def test_mass_values_captures_hydrogen(self):
        """ Tests that a hydrogen atomic mass is captured correctly """
        m = self.P.findall(self.MASS_LINE_H)
        result = [float(mass) for _, mass in m]
        assert math.isclose(result[0], 1.00783), f"expected {1.00783}, got {result[0]}"

    def test_finds_multiple_mass_lines(self):
        """ Tests that multiple atomic mass lines are matched """
        text = self.MASS_LINE_C + "\n" + self.MASS_LINE_H
        matches = self.P.findall(text)
        assert len(matches) == 2, f"expected {2}, got {len(matches)}"

    def test_no_match_for_scf_line(self):
        """ Tests that an electronic energy line is not matched """
        m = self.P.search("SCF Done: E(RB3LYP) = -153.0")
        assert m is None, f"expected {None}, got {m}"

# ============================================================================
# --- PATTERN_IRC_POINT_REACT_PROD_GEOMETRY - match and no match ---
# ============================================================================
class TestPatternIrcPointReactProdGeometry:

    P = gaussian_patterns.PATTERN_IRC_POINT_REACT_PROD_GEOMETRY

    def test_matches_end_of_projected_freq_marker(self):
        """ Tests that the corresponding marker is matched """
        line = "**** End of Projected Frequency Analysis ****"
        m = self.P.search(line)
        assert m is not None, f"expected not {None}, got {m}"

    def test_no_match_for_similar_but_wrong_text(self):
        """ Tests that a similar but incomplete marker is not matched """
        m = self.P.search("End of Projected Frequency Analysis")
        assert m is None, f"expected {None}, got {m}"

    def test_no_match_for_blank(self):
        """ Tests that a blank line is not matched """
        m = self.P.search("")
        assert m is None, f"expected {None}, got {m}"


# ============================================================================
# --- PATTERN_IRC_INPUT_ORIENTATION_BLOCK - match and no match ---
# ============================================================================
class TestPatternIrcInputOrientationBlock:

    P = gaussian_patterns.PATTERN_IRC_INPUT_ORIENTATION_BLOCK

    BLOCK = (
        "Input orientation:\n"
        "                         Standard orientation:                     \n"
        " ------------------------------------------------------------------\n"
        "                    Distance matrix\n")

    def test_matches_input_orientation_block(self):
        """ Tests that an input orientation block ending with the 'Distance matrix' is matched """
        m = self.P.search(self.BLOCK)
        assert m is not None, f"expected not {None}, got {m}"

    def test_no_match_without_distance_matrix(self):
        """ Tests that an input orientation block without the 'Distance matrix' is not matched """
        text = "Input orientation:\n --\n Center\n --\n"
        m = self.P.search(text)
        assert m is None, f"expected {None}, got {m}"


# ============================================================================
# --- PATTERN_IRC_CURRENT_STRUCTURE_BLOCK - match and no match ---
# ============================================================================
class TestPatternIRCCurrentStructureBlock:

    P = gaussian_patterns.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK

    BLOCK = (
        "CURRENT STRUCTURE\n"
        " Cartesian Coordinates (Ang):\n"
        "  1  6  0.000  0.000  0.000\n"
        " NET REACTION COORDINATE UP TO THIS POINT = 1.5")

    def test_matches_structure_block(self):
        """ Tests that a current structure block ending with the 'NET REACTION COORDINATE' is matched """
        m = self.P.search(self.BLOCK)
        assert m is not None, f"expected not {None}, got {m}"

    def test_no_match_without_net_reaction_coordinate(self):
        """ Tests that a current structure block without the 'NET REACTION COORDINATE' is not matched """
        text = "CURRENT STRUCTURE\n Cartesian Coordinates (Ang):\n data\n"
        m = self.P.search(text)
        assert m is None, f"expected {None}, got {m}"


# ============================================================================
# --- PATTERN_TS_ENERGY_IRC - match and no match ---
# ============================================================================
class TestPatternTsEnergyIrc:

    P = gaussian_patterns.PATTERN_TS_ENERGY_IRC

    LINE = "Energies reported relative to the TS energy of    -153.749670"

    def test_matches_standard_line(self):
        """ Tests that a standard energy line is matched """
        m = self.P.search(self.LINE)
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_energy(self):
        """ Tests that the TS energy value is captured correctly """
        m = self.P.search(self.LINE)
        result = float(m.group(1))
        assert math.isclose(result, -153.749670), f"expected {-153.749670}, got {result}"

    def test_matches_positive_energy(self):
        """ Tests that a positive TS energy value is matched and captured correctly """
        line = "Energies reported relative to the TS energy of +12.345678"
        m = self.P.search(line)
        result = float(m.group(1))
        assert m is not None, f"expected not {None}, got {m}; test #1"
        assert math.isclose(result, 12.345678), f"expected {12.345678}, got {result}; test #2"

    def test_no_match_for_random_line(self):
        """ Tests that an unrelated energy line is not matched """
        m = self.P.search("SCF Done: E(RB3LYP) = -153.0")
        assert m is None, f"expected {None}, got {m}"


# ============================================================================
# --- PATTERN_EL_ENERGY_IRC - match and no match ---
# ============================================================================
class TestPatternElEnergyIrc:

    P = gaussian_patterns.PATTERN_EL_ENERGY_IRC

    def test_matches_irc_table_line(self):
        """ Tests that a standard IRC table line is matched """
        line = "   16   -0.049479    1.68034"
        m = self.P.search(line)
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_relative_energy_and_irc(self):
        """ Tests that the relative energy and IRC coordinate are captured correctly """
        line = "   16   -0.049479    1.68034"
        m = self.P.search(line)
        expected = [-0.049479, 1.68034]
        for i, val in enumerate(expected):
            result = float(m.group(i+1))
            assert math.isclose(result, val), f"expected {val}, got {result}"

    def test_matches_negative_irc(self):
        """ Tests that negative relative energy and IRC coordinate values are matched """
        line = "    5   -0.012345   -0.56789"
        m = self.P.search(line)
        expected = [-0.012345, -0.56789]
        for i, val in enumerate(expected):
            result = float(m.group(i+1))
            assert math.isclose(result, val), f"expected {val}, got {result}"

    def test_no_match_for_header(self):
        """ Tests that an IRC table header line is not matched """
        m = self.P.search("Energy    RxCoord")
        assert m is None, f"expected {None}, got {m}"

# ============================================================================
# --- PATTERN_NET_REACTION_COORDINATE - match and no match ---
# ============================================================================
class TestPatternNetReactionCoordinate:

    P = gaussian_patterns.PATTERN_NET_REACTION_COORDINATE

    def test_matches_full_line(self):
        """ Tests that a standard reaction coordinate line is matched and its value is captured correctly """
        line = " NET REACTION COORDINATE UP TO THIS POINT =  1.68034"
        m = self.P.search(line)
        result = float(m.group(1))
        assert m is not None, f"expected not {None}, got {m}; test #1"
        assert math.isclose(result, 1.68034), f"expected {1.68034}, got {result}; test #2"

    def test_no_match_integer_value(self):
        """ Tests that an integer reaction coordinate value is not matched """
        line = " NET REACTION COORDINATE UP TO THIS POINT = 0"
        m = self.P.search(line)
        assert m is None, f"expected {None}, got {m}"

    def test_no_match_for_unrelated_line(self):
        """ Tests that an unrelated SCF energy line is not matched """
        m = self.P.search("SCF Done: E = -153.0")
        assert m is None, f"expected {None}, got {m}"

    def test_matches_negative_value(self):
        """ Tests that a negative reaction coordinate value is matched and captured correctly """
        line = " NET REACTION COORDINATE UP TO THIS POINT = -0.42130"
        m = self.P.search(line)
        assert m is not None, f"expected not {None}, got {m}; test #1"
        assert math.isclose(float(m.group(1)), -0.42130), f"expected {-0.42130}, "
        f"got {float(m.group(1))}; test #2"
    
    def test_no_match_for_plus_sign(self):
        """ Tests that a reaction coordinate with an explicit plus sign is not matched """
        line = " NET REACTION COORDINATE UP TO THIS POINT = +1.68034"
        m = self.P.search(line)
        assert m is None, f"expected {None}, got {m}"

    def test_matches_within_larger_irc_block(self):
        """ Tests that a reaction coordinate is matched correctly within a larger IRC block """
        block = (
            "CURRENT STRUCTURE\n"
            "Cartesian Coordinates (Ang):\n"
            "      1  6   0.000000   0.635864   0.000000\n"
            " NET REACTION COORDINATE UP TO THIS POINT = -0.15820\n")
        m = self.P.search(block)
        assert m is not None, f"expected not {None}, got {m}; test #1"
        assert math.isclose(float(m.group(1)), -0.15820), f"expected {-0.15820}, "
        f"got {float(m.group(1))}; test #2"

    def test_tolerates_no_spaces_around_equals(self):
        """ Tests that a reaction coordinate is matched when no spaces surround the equals sign """
        line = "NET REACTION COORDINATE UP TO THIS POINT=1.68034"
        m = self.P.search(line)
        assert m is not None, f"expected not {None}, got {m}; test #1"
        assert math.isclose(float(m.group(1)), 1.68034), f"expected {1.68034}, "
        f"got {float(m.group(1))}; test #2"


# ============================================================================
# --- PATTERN_HESSIAN_IN_IRC_BLOCK - match and no match ---
# --- PATTERN_HESSIAN_IN_OPT_BLOCK - match and no match ---
# ============================================================================
class TestPatternHessianBlock:

    PIRC = gaussian_patterns.PATTERN_HESSIAN_IN_IRC_BLOCK
    POPT = gaussian_patterns.PATTERN_HESSIAN_IN_OPT_BLOCK

    HESSIAN_BLOCK_IRC = (
        "Force constants in Cartesian coordinates: \n"
        "                1             2\n"
        "      1  0.442310D+00\n"
        "      2  0.237978D-01  0.376006D+00\n"
        " Leave Link")

    HESSIAN_BLOCK_OPT = (
        "Force constants in Cartesian coordinates: \n"
        "                1             2\n"
        "      1  0.442310D+00\n"
        " FormGI")

    def test_irc_hessian_pattern_matches(self):
        """ Tests that the IRC Hessian block is matched """
        m = self.PIRC.search(self.HESSIAN_BLOCK_IRC)
        assert m is not None, f"expected not {None}, got {m}"

    def test_irc_hessian_captures_content(self):
        """ Tests that the IRC Hessian matrix content is captured """
        m = self.PIRC.search(self.HESSIAN_BLOCK_IRC)
        assert "0.442310D+00" in m.group(1), f"expected {"0.442310D+00"} in {m.group(1)}"

    def test_opt_hessian_pattern_matches(self):
        """ Tests that the optimization Hessian block is matched """
        m = self.POPT.search(self.HESSIAN_BLOCK_OPT)
        assert m is not None, f"expected not {None}, got {m}"

    def test_opt_hessian_captures_content(self):
        """ Tests that the optimization Hessian matrix content is captured """
        m = self.POPT.search(self.HESSIAN_BLOCK_OPT)
        assert "0.442310D+00" in m.group(1), f"expected {"0.442310D+00"} in {m.group(1)}"

    def test_irc_pattern_does_not_match_opt_terminator(self):
        """ Tests that the IRC Hessian pattern does not match an optimization Hessian block """
        m = self.PIRC.search(self.HESSIAN_BLOCK_OPT)
        assert m is None, f"expected {None}, got {m}"

    def test_opt_pattern_does_not_match_irc_terminator(self):
        """ Tests that the optimization Hessian pattern does not match an IRC Hessian block """
        m = self.POPT.search(self.HESSIAN_BLOCK_IRC)
        assert m is None, f"expected {None}, got {m}"

    def test_no_match_without_header(self):
        """ Tests that an incomplete Hessian block without the required header is not matched """
        m = self.PIRC.search("      1  0.442310D+00\n Leave Link")
        assert m is None, f"expected {None}, got {m}"


# ============================================================================
# --- PATTERN_FREQ_MODES_BLOCK - match and no match ---
# ============================================================================
class TestPatternFreqModesBlock:

    P = gaussian_patterns.PATTERN_FREQ_MODES_BLOCK

    SINGLE_BLOCK_2 = (
        " Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering\n"
        " activities (A**4/AMU), depolarization ratios for plane and unpolarized\n"
        " incident light, reduced masses (AMU), force constants (mDyne/A),\n"
        " and normal coordinates:\n"
        "                    1                      2                      3\n"
        "                    A'                     A'                     A'\n"
        " Frequencies --  -2019.5774               135.7838               474.0512\n"
        " Red. masses --      1.1069                 1.3253                 2.8267\n"
        " Frc consts  --      2.6600                 0.0144                 0.3743\n"
        " IR Inten    --    607.4247                 1.3244                 6.3898\n"
        "  Atom  AN      X      Y      Z        X      Y      Z        X      Y      Z\n"
        " 1   6    -0.08   0.00   0.00     0.00   0.00   0.13     0.02   0.22   0.00\n"
        " 2   8     0.00  -0.05   0.00     0.00   0.00  -0.09     0.20  -0.14   0.00\n"
        "                                                                   \n"
        " -------------------\n"
        " - Thermochemistry -\n")

    def test_matches_basic_block(self):
        """ Tests that a complete frequency modes block is matched """
        m = self.P.search(self.SINGLE_BLOCK_2)
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_content_between_markers(self):
        """
        Tests that the block content between the normal-coordinate
        and thermochemistry markers is captured
        """
        m = self.P.search(self.SINGLE_BLOCK_2)
        capture = m.group(1)
        expected = ["Frequencies --", "-2019.5774", "Atom  AN", "-0.05"]
        for i, val in enumerate(expected):
            assert val in capture, f"expected {val} in {capture}; test #{i+1}"

    def test_no_match_without_thermochemistry_marker(self):
        """ Tests that a block without the thermochemistry marker is not matched """
        text = self.SINGLE_BLOCK_2.replace(" - Thermochemistry -\n", "")
        m = self.P.search(text)
        assert m is None, f"expected {None}, got {m}"

    def test_no_match_without_dashes_before_thermochemistry(self):
        """
        Tests that a block without the separator before
        the thermochemistry marker is not matched
        """
        text = self.SINGLE_BLOCK_2.replace("-------------------\n - Thermochemistry -\n", "")
        m = self.P.search(text)
        assert m is None, f"expected {None}, got {m}"

    def test_dotall_content_spans_multiple_lines(self):
        """ Tests that the captured block content can span multiple lines """
        m = self.P.search(self.SINGLE_BLOCK_2)
        assert m is not None, f"expected not {None}, got {m}; test #1"
        assert "\n" in m.group(1), f"expected {"\n"} in {m.group(1)}; test #2"

    def test_multiple_normal_coordinates_headers_merge_into_one_capture(self):
        """ Tests that multiple normal-coordinate sections are captured as one block """
        second_mode_block = (
            "                     4                      5                      6\n"
            "                    A                      A                      A\n"
            " Frequencies --    620.1122    701.3344    855.6677\n"
            "    1   6     0.10   0.20  -0.10    -0.20   0.30   0.10     0.10   0.20  -0.20\n")
        text = (self.SINGLE_BLOCK_2.replace(
                "-------------------\n"
                " - Thermochemistry -\n",
                "")
            + second_mode_block
            + "-------------------\n"
            " - Thermochemistry -\n")
        m = self.P.search(text)
        capture = m.group(1)
        assert m is not None, f"expected not {None}, got {m}; test #1"
        assert "-2019.5774" in capture, f"expected {"-2019.5774"} in {capture}; test #2"
        assert "620.1122" in capture, f"expected {"620.1122"} in {capture}; test #3"


# ============================================================================
# --- PATTERN_FREQ_MODES_LINES - match and no match ---
# ============================================================================
class TestPatternFreqModsLines:

    P = gaussian_patterns.PATTERN_FREQ_MODES_LINES
        
    def test_finds_positive_numbers(self):
        """ Tests that positive decimal numbers are matched and captured """
        line = "Frequencies --  36.5420  306.8203  1234.5678"
        expected = ["36.5420", "306.8203", "1234.5678"]
        result = self.P.findall(line)
        assert result == expected, f"expected {expected}, got {result}" 

    def test_finds_negative_numbers(self):
        """ Tests that negative decimal numbers are matched and captured """
        result = self.P.findall("Frequencies -- -2091.8757  36.5420")
        expected = ["-2091.8757", "36.5420"]
        for val in expected:
            assert val in result, f"expected {val} in {result}"

    def test_finds_mixed_sign_numbers(self):
        """ Tests that decimal numbers with mixed signs are matched and captured """
        result = self.P.findall("-1.0  2.5  -3.14")
        expected = ["-1.0", "2.5", "-3.14"]
        assert result == expected, f"expected {expected}, got {result}" 

    def test_no_match_for_integers_without_decimal(self):
        """ Tests that integers without a decimal point are not matched """
        result = self.P.findall("Step 16 Path 1")
        expected = []
        assert result == expected, f"expected {expected}, got {result}"

    def test_finds_numbers_in_coordinate_line(self):
        """ Tests that decimal coordinate values are matched and captured """
        line = "      1          6           0        0.000000    0.635864    0.000002"
        expected = ["0.000000", "0.635864", "0.000002"]
        result = self.P.findall(line)
        for val in expected:
            assert val in result, f"expected {val} in {result}"

    def test_plus_sign_is_not_captured(self):
        """ Tests that a leading plus sign is not included in the captured positive number """
        result = self.P.findall("+1.5 -2.5")
        expected = ["1.5", "-2.5"]
        assert result == expected, f"expected {expected}, got {result}"
    
    def test_no_match_for_number_without_leading_digit(self):
        
        result = self.P.findall(".5000 0.6000")
        expected = ["0.6000"]
        assert result == expected, f"expected {expected}, got {result}"

    def test_finds_adjacent_numbers_without_separator(self):
        """ Tests that decimal numbers without a leading digit are not matched """
        result = self.P.findall("306.8203-2091.8757")
        expected = ["306.8203", "-2091.8757"]
        assert result == expected, f"expected {expected}, got {result}"

    def test_finds_numbers_in_mode_vector_line(self):
        """ Tests that decimal values in a vibrational mode vector line are matched """
        line = " 1   6     0.00   0.00  -0.08    -0.06   0.17   0.00     0.00   0.00  -0.12"
        expected = ["0.00", "0.00", "-0.08", "-0.06", "0.17", "0.00", "0.00", "0.00", "-0.12"]
        result = self.P.findall(line)
        assert result == expected, f"expected {expected}, got {result}"

    def test_no_match_on_empty_string(self):
        """ Tests that no numbers are matched in an empty string """
        result = self.P.findall("")
        assert result == [], f"expected [], got {result}"

    def test_finds_numbers_across_multiple_lines(self):
        """ Tests that decimal numbers are matched across multiple lines """
        block = "Frequencies --  -2091.8757    36.5420\n Red. masses --   1.0500     2.3400\n"
        expected = ["-2091.8757", "36.5420", "1.0500", "2.3400"]
        result = self.P.findall(block)
        assert result == expected, f"expected {expected}, got {result}"