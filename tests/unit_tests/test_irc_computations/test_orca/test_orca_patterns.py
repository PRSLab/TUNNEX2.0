# --- Test: test_orca_file_parsing. Unit tests for .\orca\orca_file_parsing.py
# Run with: pytest test_gaussian_file_parsing.py -v ---


# --- Modules ---
import math
import re


# --- Module to test ---
from tunnex_2.irc_computations.orca import orca_patterns # type: ignore


# ============================================================================
# --- String patterns - settings ---
# ============================================================================
class TestStringPatterns:
    
    def test_string_pattenrs(self):
        """ Tests that ORCA string pattern constants have the expected values """
        pat = orca_patterns.BLOCK_TAIL_TS
        exp_pat = r"""OptTS Freq"""
        assert exp_pat in pat, f"expected {exp_pat} in {pat}; test #1"
        pat = orca_patterns.BLOCK_TAIL_IRC
        exp_pat = r"IRC"
        assert exp_pat in pat, f"expected {exp_pat} in {pat}; test #2"
        pat = orca_patterns.LINE_TAIL_REACT_PROD
        exp_pat = r"Opt Freq"
        assert exp_pat in pat, f"expected {exp_pat} in {pat}; test #3"
        pat = orca_patterns.LINE_HESS_COMP
        exp_pat = r"Freq # E {}"
        assert exp_pat in pat, f"expected {exp_pat} in {pat}; test #4"


# ============================================================================
# --- PATTERN_COMPUTATIONAL_METHOD_ORCA - match and no match ---
# ============================================================================
class TestPatternComputationalMethodORCA:

    P = orca_patterns.PATTERN_COMPUTATIONAL_METHOD_ORCA

    def test_matches_standard_orca_method_line(self):
        """ Tests that a standard ORCA method line is matched """
        m = self.P.match("! B3LYP def2-SVP opt")
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_method_and_basis(self):
        """ Tests that the method and basis set are captured correctly """
        m = self.P.match("! BP86 def2-TZVP freq")
        assert m.group(1) == "BP86", f"expected {"BP86"}, got {m.group(1)}: test #1"
        assert m.group(2) == "def2-TZVP", f"expected {"def2-TZVP"}, got {m.group(2)}: test #2"

    def test_matches_with_leading_spaces(self):
        """ Tests that an ORCA method line with leading spaces is matched """
        m = self.P.match("  ! B3LYP def2-SVP")
        assert m is not None, f"expected not {None}, got {m}"

    def test_no_match_for_comment_line(self):
        """ Tests that a comment line is not matched """
        m = self.P.match("# This is a comment")
        assert m is None, f"expected {None}, got {m}"

    def test_no_match_for_geometry_line(self):
        """ Tests that a geometry line is not matched """
        m = self.P.match("C  0.000  0.000  0.000")
        assert m is None, f"expected {None}, got {m}"

    def test_no_match_for_blank_line(self):
        """ Tests that a blank line is not matched """
        m1 = self.P.match("")
        m2 = self.P.match("%geom")
        assert m1 is None, f"expected {None}, got {m1}; test #1"
        assert m2 is None, f"expected {None}, got {m2}; test #2"


# ============================================================================
# --- PATTERN_OPTIMIZED_SPECIES_GEOM_BLOCK - match and no match ---
# ============================================================================
class TestPatternOptimizedSpeciesGeomBlock:

    P = orca_patterns.PATTERN_OPTIMIZED_SPECIES_GEOM_BLOCK

    BLOCK = (
        "CARTESIAN COORDINATES (ANGSTROEM)\n"
        "---------------------------------\n"
        "  C      0.000000    0.635864    0.000000\n"
        "  H      0.924490    1.200305    0.000000\n")

    def test_finds_geometry_block(self):
        """ Tests that a Cartesian geometry block is found """
        m = self.P.search(self.BLOCK)
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_coordinate_lines(self):
        """ Tests that coordinate lines are captured correctly """
        m = self.P.search(self.BLOCK)
        assert "0.635864" in m.group(1), f"expected {"0.635864"} in {m.group(1)}"

    def test_no_match_without_header(self):
        """ Tests that a geometry block without the required header is not matched """
        text = "C  0.000  0.635864  0.000\nH  0.924  1.200  0.000\n"
        m = self.P.search(text)
        assert m is None, f"expected {None}, got {m}"

    def test_no_match_without_dashes(self):
        """ Tests that a geometry block without the separator line is not matched """
        text = "CARTESIAN COORDINATES (ANGSTROEM)\nC  0.0  0.0  0.0\n"
        m = self.P.search(text)
        assert m is None, f"expected {None}, got {m}"


# ============================================================================
# --- PATTERN_GEOMETRY_LINE_ORCA - match and no match ---
# ============================================================================
class TestPatternGeometryLineORCA:

    P = orca_patterns.PATTERN_GEOMETRY_LINE_ORCA

    def test_matches_standard_line(self):
        """ Tests that a standard geometry line is matched """
        m = self.P.search("  C   0.000000   0.635864   0.000000")
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_four_groups(self):
        """ Tests that the atom type and three coordinates are captured correctly """
        m = self.P.search("  O  -1.234567   0.000001   2.345678")
        expected = ["O", "-1.234567", "0.000001", "2.345678"]
        for i, val in enumerate(expected):
            assert m.group(i+1) == val, f"expected {val}, got {m.group(i+1)}: test #{i+1}"

    def test_matches_two_letter_element(self):
        """ Tests that a two-letter element symbol is matched """
        m = self.P.search("  Cl   0.000   1.000   2.000")
        assert m is not None, f"expected not {None}, got {m}"

    def test_matches_integer_atom_type(self):
        """ Tests that an integer atom type is matched """
        m = self.P.search("  6   0.000   1.000   2.000")
        assert m is not None, f"expected not {None}, got {m}"

    def test_no_match_for_header_line(self):
        """ Tests that a geometry header line is not matched """
        m = self.P.search("  NO  LB  ZA  FRAG  MASS  X  Y  Z")
        assert m is None, f"expected {None}, got {m}"

    def test_no_match_for_dash_separator(self):
        """ Tests that a dash separator line is not matched """
        m = self.P.search("---------------------------")
        assert m is None, f"expected {None}, got {m}"


# ============================================================================
# --- PATTERN_EL_ENERGY_ZPVE_SINGLE_POINT - match and no match ---
# ============================================================================
class TestPatternElEnergyZPVESinglePoint:

    P = orca_patterns.PATTERN_EL_ENERGY_ZPVE_SINGLE_POINT

    BLOCK = (
        "Electronic energy                ... -153.799148571 Eh\n"
        "Zero point energy                ...     0.04905392 Eh   30.78 kcal/mol\n")

    def test_matches_block(self):
        """ Tests that an electronic energy and ZPVE block is matched """
        m = self.P.search(self.BLOCK)
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_el_energy(self):
        """ Tests that the electronic energy is captured correctly """
        m = self.P.search(self.BLOCK)
        result = float(m.group(1))
        assert math.isclose(result, -153.799148571), f"expected {-153.799148571}, got {m}"

    def test_captures_zpve(self):
        """ Tests that the ZPVE is captured correctly """
        m = self.P.search(self.BLOCK)
        result = float(m.group(2))
        assert math.isclose(result, 0.04905392), f"expected {0.04905392}, got {m}"

    def test_no_match_without_zpve_line(self):
        """ Tests that a block without a ZPVE line is not matched """
        text = "Electronic energy                ... -153.799148571 Eh\n"
        m = self.P.search(text)
        assert m is None, f"expected {None}, got {m}"

    def test_no_match_for_scf_done_line(self):
        """ Tests that an 'SCF Done' line is not matched """
        m = self.P.search("SCF Done: E(RB3LYP) = -153.0")
        assert m is None, f"expected {None}, got {m}"


# ============================================================================
# --- PATTERN_MASS_BLOCK - match and no match ---
# ============================================================================
class TestPatternMassBlock:

    P = orca_patterns.PATTERN_MASS_BLOCK

    MASS_BLOCK = (
        "CARTESIAN COORDINATES (A.U.)\n"
        "----------------------------\n"
        "  NO LB   ZA   FRAG   MASS      X           Y           Z   \n"
        "   0  C  6.0000  0  12.0000   0.000000  -1.201567   0.000000\n"
        "   1  H  1.0000  0   1.0078   1.747400   2.268218   0.000000\n")

    def test_matches_mass_block(self):
        """ Tests that an atomic mass block is matched """
        m = self.P.search(self.MASS_BLOCK)
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_atom_data(self):
        """ Tests that atomic data including masses are captured correctly """
        m = self.P.search(self.MASS_BLOCK)
        assert "12.0000" in m.group(1), f"expected {"12.0000"} in {m.group(1)}"

    def test_no_match_without_header(self):
        """ Tests that atomic data without the required header is not matched """
        text = "   1  C  6.0  0  12.0  0.0  0.0  0.0\n"
        m = self.P.search(text)
        assert m is None, f"expected {None}, got {m}"


# ============================================================================
# --- PATTERN_IRC_COORDINATES_BLOCK - match and no match ---
# ============================================================================
class TestPatternIRCCoordinatesBlock:

    P = orca_patterns.PATTERN_IRC_COORDINATES_BLOCK

    BLOCK = ("2\n"
        "Coordinates from ORCA-job all_IRC_Full E -153.799148\n"
        "C  0.000000 -0.635864  0.000000\n"
        "H  0.924490  1.200305  0.000000\n")

    def test_matches_energy_and_coords(self):
        """ Tests that an energy value and coordinate block are matched """
        m = self.P.search(self.BLOCK)
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_energy(self):
        """ Tests that the energy value is captured correctly """
        m = self.P.search(self.BLOCK)
        result = float(m.group(1))
        assert math.isclose(result, -153.799148), f"expected {-153.799148}, got {result}"

    def test_captures_coordinate_block(self):
        """ Tests that the coordinate block is captured correctly """
        m = self.P.search(self.BLOCK)
        assert "-0.635864" in m.group(2), f"expected {"-0.635864"} in {m.group(2)}"

    def test_no_match_without_correct_prefix(self):
        """ Tests that an energy value without the E prefix is not matched """
        text = "-153.799148\nC  0.0  0.0  0.0\n"
        m = self.P.search(text)
        assert m is None, f"expected {None}, got {m}"

    def test_matches_positive_energy(self):
        """ Tests that a positive energy value is matched correctly """
        text = "E +0.123456\nC  0.0  0.0  0.0\n"
        m = self.P.search(text)
        result = float(m.group(1))
        assert m is not None, f"expected not {None}, got {m}; test #1"
        assert math.isclose(result, 0.123456), f"expected {0.123456}, got {result}; test #2"


# ============================================================================
# --- PATTERN_HESSIAN_BLOCK - match and no match ---
# ============================================================================
class TestPatternHessianBlock:

    P = orca_patterns.PATTERN_HESSIAN_BLOCK

    BLOCK = (
        "$hessian\n"
        "3\n"
        "          0          1          2\n"
        "0       0.1        0.2        0.3\n"
        "1       0.4        0.5        0.6\n"
        "2       0.7        0.8        0.9\n"
        "\n"
        "$vibrational_frequencies\n")

    def test_matches_hessian_block(self):
        """ Tests that a Hessian block is matched """
        m = self.P.search(self.BLOCK)
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_dimension(self):
        """ Tests that the Hessian dimension is captured correctly """
        m = self.P.search(self.BLOCK)
        result = int(m.group(1))
        assert result == 3, f"expected {3}, got {result}"

    def test_captures_data_block(self):
        """ Tests that Hessian data is captured correctly """
        m = self.P.search(self.BLOCK)
        assert "0.1" in m.group(2), f"expected {"0.1"} in {m.group(2)}"

    def test_no_match_without_vibrational_frequencies_marker(self):
        """ Tests that a Hessian block without the vibrational frequencies marker is not matched """
        text = "$hessian\n6\n0  0.1  0.2\n"
        m = self.P.search(text)
        assert m is None, f"expected {None}, got {m}"


# ============================================================================
# --- PATTERN_VIBR_BLOCK - match and no match ---
# ============================================================================
class TestPatternVibrBlock:

    P = orca_patterns.PATTERN_VIBR_BLOCK

    BLOCK = (
        "-----------\n"
        "VIBRATIONAL FREQUENCIES\n"
        "-----------\n"
        "\n"
        "Scaling factor for frequencies =  1.000000000  (already applied!)\n"
        "   0:     0.00 cm**-1\n"
        "   1:  -100.00 cm**-1  ***imaginary mode***\n"
        "   2:   200.00 cm**-1\n"
        "   3:   300.00 cm**-1\n"
        "   4:   400.00 cm**-1\n"
        "   5:   500.00 cm**-1\n"
        "-----------\n"
        "NORMAL MODES\n"
        "-----------\n")

    def test_matches_block(self):
        """ Tests that a vibrational frequencies block is matched """
        m = self.P.search(self.BLOCK)
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_frequency_section(self):
        """ Tests that the vibrational frequency section is captured correctly """
        m = self.P.search(self.BLOCK)
        assert "-100.00" in m.group(1), f"expected {"-100.00"} in {m.group(1)}"

    def test_no_match_without_normal_modes_header(self):
        """ Tests that a frequency block without the normal modes header is not matched """
        text = "-----------\nVIBRATIONAL FREQUENCIES\n-----------\n0: 0.0 cm**-1\n"
        m = self.P.search(text)
        assert m is None, f"expected {None}, got {m}"


# ============================================================================
# --- PATTERN_VIBR_FREQUENCIES - match and no match ---
# ============================================================================
class TestPatternVibrFrequencies:

    P = orca_patterns.PATTERN_VIBR_FREQUENCIES

    def test_matches_frequency_line(self):
        """ Tests that a vibrational frequency line is matched """
        m = self.P.search("   0:     0.00 cm**-1")
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_frequency_value(self):
        """ Tests that the vibrational frequency value is captured correctly """
        m = self.P.search("   6:  1234.56 cm**-1")
        result = float(m.group(1))
        assert math.isclose(result, 1234.56), f"expected {1234.56}, got {result}"

    def test_matches_negative_frequency(self):
        """ Tests that a negative vibrational frequency is matched correctly """
        m = self.P.search("   0:  -2091.88 cm**-1")
        result = float(m.group(1))
        assert m is not None, f"expected not {None}, got {m}; test #1"
        assert math.isclose(result, -2091.88), f"expected {-2091.88}, got {result}; test #2"

    def test_finds_all_in_block(self):
        """ Tests that all vibrational frequencies in a block are found """
        text = "   0:  0.00 cm**-1\n   1:  1000.0 cm**-1\n   2:  2000.0 cm**-1\n"
        m = self.P.findall(text)
        result = float(m[2])
        assert len(m) == 3, f"expected {3}, got {len(m)}; test #1"
        assert math.isclose(result, 2000.0), f"expected {2000.0}, got {result}; test #2"

    def test_no_match_for_energy_line(self):
        """ Tests that an energy line and a standalone number are not matched """
        m1 = self.P.search("Electronic energy ... -153.0 Eh")
        m2 = self.P.search("1234.56")
        assert m1 is None, f"expected {None}, got {m1}; test #1"
        assert m2 is None, f"expected {None}, got {m2}; test #2"


# ============================================================================
# --- PATTERN_NORMAL_MODES_BLOCK - match and no match ---
# ============================================================================
class TestPatternNormalModesBlock:

    P = orca_patterns.PATTERN_NORMAL_MODES_BLOCK

    BLOCK = (
        "------------\n"
        "NORMAL MODES\n"
        "------------\n"
        "These modes are the Cartesian displacements weighted by the diagonal matrix\n"
        "M(i,i)=1/sqrt(m[i]) where m[i] is the mass of the displaced atom\n"
        "Thus, these vectors are normalized but *not* orthogonal\n"
        "\n"
        "     0    1    2\n"
        "0  0.1  0.2  0.3\n"
        "1  0.4  0.5  0.6\n"
        "2  0.7  0.8  0.9\n"
        "\n"
        "\n"
        "-----------\n"
        "IR SPECTRUM\n")

    def test_matches_normal_modes_block(self):
        """ Tests that a normal modes block is matched and its data is captured """
        m = self.P.search(self.BLOCK)
        assert m is not None, f"expected not {None}, got {m}; test #1"
        assert "0.1" in m.group(1), f"expected {"0.1"} in {m.group(1)}; test #2"

    def test_no_match_without_ir_spectrum(self):
        """ Tests that a normal modes block without the IR spectrum marker is not matched """
        text = (
            "------------\nNORMAL MODES\n------------\n"
            "Thus, these vectors are normalized but *not* orthogonal\n"
            "data\n-----------\n")
        m = self.P.search(text)
        assert m is None, f"expected {None}, got {m}"

    def test_no_match_without_normalized_vectors_line(self):
        """ Tests that a normal modes block without the normalized vectors line is not matched """
        text = (
            "------------\nNORMAL MODES\n------------\n"
            "data\n-----------\nIR SPECTRUM\n")
        m = self.P.search(text)
        assert m is None, f"expected {None}, got {m}"


# ============================================================================
# --- PATTERN_HESSIAN_FLOAT - match and no match ---
# ============================================================================
class TestPatternHessianFloat:

    P = orca_patterns.PATTERN_HESSIAN_FLOAT

    def test_pattern_hessian_float_matches_numbers_no_match_text(self):
        """ Tests that the Hessian float pattern matches valid numeric values and rejects invalid text """
        p = re.compile(orca_patterns.PATTERN_HESSIAN_FLOAT)
        assert isinstance(orca_patterns.PATTERN_HESSIAN_FLOAT, str), f"expected {str}, "
        f"got {type(orca_patterns.PATTERN_HESSIAN_FLOAT)}; test #1"
        assert p.search("1.234") is not None, f"expected not {None}, got {p.search("1.234")}; test #2"
        assert p.search("-1.234") is not None, f"expected not {None}, got {p.search("-1.234")}; test #3"
        assert p.search("1.23E-05") is not None, f"expected not {None}, got {p.search("1.23E-05")}; test #4"
        assert p.search("abc") is None, f"expected {None}, got {p.search("abc")}; test #5"


# ============================================================================
# --- PATTERN_HESSIAN_LINE - match and no match ---
# ============================================================================
class TestPatternHessianLine:

    P = orca_patterns.PATTERN_HESSIAN_LINE

    def test_matches_five_column_hessian_line(self):
        """ Tests that a Hessian line with five values is matched """
        line = "   0   0.1  0.2  -0.3  4.5E-06  -1.23E+02"
        m = self.P.search(line)
        assert m is not None, f"expected not {None}, got {m}"

    def test_captures_row_index(self):
        """ Tests that the Hessian row index is captured correctly """
        line = "   3   0.1  0.2  -0.3  4.5  -1.23"
        m = self.P.search(line)
        assert m.group(1) == "3", f"expected {"3"} in {m.group(1)}"

    def test_no_match_for_column_header(self):
        """ Tests that a Hessian column header is not matched """
        line = "          0          1          2"
        m = self.P.search(line)
        assert m is None, f"expected {None}, got {m}"

    def test_matches_scientific_notation_values(self):
        """ Tests that Hessian values in scientific notation are matched """
        line = "  0  1.23456789E-05  -2.34567890E+02  3.45E+00  -4.56E-01  5.67E+01"
        m = self.P.search(line)
        assert m is not None, f"expected not {None}, got {m}"