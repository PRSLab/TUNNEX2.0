# --- Test: test_constants_and_settings. Unit tests for constants_and_settings.py
# Run with: pytest test_constants_and_settings.py -v ---


# --- Comment ---
# Test physical correctness and internal consistency of constants,
# not just their raw values (which would just duplicate the source file).
# --- Comment ---


# --- Modules ---
import pytest # type: ignore

import math
from pathlib import Path


# --- Module to test ---
from tunnex_2.constants_and_dataclasses.constants_and_settings import ( # type: ignore
    COMMAND_FILE_PATH, RUN_SOFTWARE_KEY_MARKER, SLURM_MODE_KEY, ENVR,
    POTENTIAL_SCAL_FACTOR_DEFAULT, CORR_ANALYSIS_BOOL, PICK_MAX_FREQ_FLAG,
    TUNN_PROB_AVERAGING_ALLOWED_MODES,
    CM_M1_TO_HARTREE, KJ_MOL_M1_TO_HARTREE, CM_M1_TO_HZ, HARTREE_TO_J,
    BOHR_RADIUS_TO_M, ANGSTROM_TO_BOHR_RADIUS,
    AMU_TO_KG, LIGHT_SPEED,
    CONVERSION_FACTOR_HESSIAN_VALUES, CONVERSION_FACTOR_F_STAR_ECKART,
    REVELO, K_B, GRID_SIZE, INTEGRAL_RTOL, HOUR, DAY, YEAR)


# ===========================================================================
# --- constants - sanity check: all constants must be positive ---
# ===========================================================================
class TestAllConstantsPositive:

    @pytest.mark.parametrize("name, value", [
        ("POTENTIAL_SCAL_FACTOR_DEFAULT", POTENTIAL_SCAL_FACTOR_DEFAULT),
        ("CM_M1_TO_HARTREE", CM_M1_TO_HARTREE),
        ("KJ_MOL_M1_TO_HARTREE", KJ_MOL_M1_TO_HARTREE),
        ("CM_M1_TO_HZ", CM_M1_TO_HZ),
        ("HARTREE_TO_J", HARTREE_TO_J),
        ("BOHR_RADIUS_TO_M", BOHR_RADIUS_TO_M),
        ("ANGSTROM_TO_BOHR_RADIUS", ANGSTROM_TO_BOHR_RADIUS),
        ("AMU_TO_KG", AMU_TO_KG),
        ("LIGHT_SPEED", LIGHT_SPEED),
        ("CONVERSION_FACTOR_HESSIAN_VALUES", CONVERSION_FACTOR_HESSIAN_VALUES),
        ("CONVERSION_FACTOR_F_STAR_ECKART", CONVERSION_FACTOR_F_STAR_ECKART),
        ("REVELO", REVELO),
        ("K_B", K_B),
        ("HOUR", HOUR),
        ("DAY", DAY),
        ("YEAR", YEAR)])
    def test_positive(self, name, value):
        assert value > 0, f"{name} must be positive, got {value}"


# ===========================================================================
# --- Settings and flags check ---
# ===========================================================================
class TestSettings:

    def test_command_file_path(self):
        f""" Checking the command file path (Path) """
        assert isinstance(COMMAND_FILE_PATH, Path), f"expected {Path}, "
        f"got {type(COMMAND_FILE_PATH)}; test #1"
        expected = ".txt"
        assert COMMAND_FILE_PATH.suffix == expected, f"expected {expected}, "
        f"got {COMMAND_FILE_PATH.suffix}; test #2"

    def test_run_software_parameters(self):
        f""" Checking the run software key, SLURM mode key, and environment for the computations """
        assert isinstance(RUN_SOFTWARE_KEY_MARKER, bool), f"expected {bool}, "
        f"got {type(RUN_SOFTWARE_KEY_MARKER)}; test #1"
        assert isinstance(SLURM_MODE_KEY, bool), f"expected {bool}, "
        f"got {type(SLURM_MODE_KEY)}; test #2"
        assert ENVR is None or isinstance(ENVR, dict), f"expected {None} or {dict}, "
        f"got {type(ENVR)}; test #3"

    def test_corr_analysis_pick_max_freq_are_bool(self):
        f""" Checking the correlation analysis marker """
        assert isinstance(CORR_ANALYSIS_BOOL, bool), f"expected {bool}, got {type(CORR_ANALYSIS_BOOL)}; test #1"
        assert isinstance(PICK_MAX_FREQ_FLAG, bool), f"expected {bool}, got {type(PICK_MAX_FREQ_FLAG)}; test #2"
        assert isinstance(TUNN_PROB_AVERAGING_ALLOWED_MODES, dict), f"expected {bool}, "
        f"got {type(TUNN_PROB_AVERAGING_ALLOWED_MODES)}; test #3"

    def test_potential_scal_factor_default_is_one(self):
        f""" Checking the default potential scaling factor """
        result = POTENTIAL_SCAL_FACTOR_DEFAULT == 1.0
        assert result, f"expected {1.0}, got {POTENTIAL_SCAL_FACTOR_DEFAULT}"

    def test_grid_size_integral_rtol_are_integers(self):
        f""" Checking the grid size and relative tolerance of integration (QMT module) """
        assert isinstance(GRID_SIZE, int), f"expected {int}, got {type(GRID_SIZE)}; test #1"
        assert GRID_SIZE > 2, f"expected > {2}, got {GRID_SIZE}; test #2"
        assert isinstance(INTEGRAL_RTOL, float), f"expected {float}, got {type(INTEGRAL_RTOL)}; test #3"
        assert INTEGRAL_RTOL > 0.0, f"expected > {0.0}, got {INTEGRAL_RTOL}; test #4"


# =========================================================================================
# --- Energy unit conversions - cross-check against literature (NCTU) ---
# --- Reference: https://wild.life.nctu.edu.tw/class/common/energy-unit-conv-table.html ---
# =========================================================================================
class TestEnergyConversions:

    def test_cm_m1_to_hartree_known_value(self):
        expected = 4.55633e-6
        f""" 1 cm-1 ≈ {expected} Hartree (NCTU) """
        result = math.isclose(CM_M1_TO_HARTREE, expected, rel_tol=1e-3)
        assert result, f"expected {expected}, got {CM_M1_TO_HARTREE}"

    def test_kj_mol_m1_to_hartree_known_value(self):
        expected = 3.8088e-4
        f""" 1 kJ mol-1 ≈ {expected} Hartree (NCTU) """
        result = math.isclose(KJ_MOL_M1_TO_HARTREE, expected, rel_tol=1e-3)
        assert result, f"expected {expected}, got {KJ_MOL_M1_TO_HARTREE}"

    def test_cm_m1_hz_known_value(self):
        expected = 2.99793e+10
        f""" 1 cm-1 ≈ {expected} Hz (NCTU) """
        result = math.isclose(CM_M1_TO_HZ, expected, rel_tol=1e-3)
        assert result, f"expected {expected}, got {CM_M1_TO_HZ}"

    def test_hartree_to_j_known_value(self):
        expected = 4.360e-18
        f""" 1 Hartree ≈ {expected} J (NCTU) """
        result = math.isclose(HARTREE_TO_J, expected, rel_tol=1e-3)
        assert result, f"expected {expected}, got {HARTREE_TO_J}"

    def test_energy_ratio_kj_mol_m1_to_cm_m1(self):
        expected = 83.593
        f""" The ratio kJ mol-1 to cm-1 should be equal ≈ {expected} (NCTU) """
        ratio = KJ_MOL_M1_TO_HARTREE / CM_M1_TO_HARTREE
        result = math.isclose(ratio, expected, rel_tol=1e-3)
        assert result, f"expected {expected}, got {ratio}"

    def test_energy_ratio_j_to_cm_m1(self):
        expected = 5.03445e+22
        f""" The ratio J to cm-1 should be equal ≈ {expected} (NCTU) """
        ratio = 1 / (HARTREE_TO_J * CM_M1_TO_HARTREE)
        result = math.isclose(ratio, expected, rel_tol=1e-3)
        assert result, f"expected {expected}, got {ratio}"

    def test_energy_ratio_kj_mol_m1_to_j(self):
        expected = 6.02e+20
        f""" The ratio kJ mol-1 to J should be equal ≈ {expected} (NCTU) """
        ratio = 1 / (HARTREE_TO_J * KJ_MOL_M1_TO_HARTREE)
        result = math.isclose(ratio, expected, rel_tol=1e-3)
        assert result, f"expected {expected}, got {ratio}"

    def test_energy_ratio_hartree_to_hz(self):
        expected = 6.57966e+15
        f""" The ratio Hartree to Hz should be equal ≈ {expected} (NCTU) """
        ratio = CM_M1_TO_HZ / CM_M1_TO_HARTREE
        result = math.isclose(ratio, expected, rel_tol=1e-3)
        assert result, f"expected {expected}, got {ratio}"

    def test_energy_ratio_kj_mol_m1_to_hz(self):
        expected = 2.50607e+12
        f""" The ratio kJ mol-1 to Hz should be equal ≈ {expected} (NCTU) """
        ratio = (KJ_MOL_M1_TO_HARTREE * CM_M1_TO_HZ) / (CM_M1_TO_HARTREE)
        result = math.isclose(ratio, expected, rel_tol=1e-3)
        assert result, f"expected {expected}, got {ratio}"

    def test_energy_ratio_j_to_hz(self):
        expected = 1.50930e+33
        f""" The ratio J to Hz should be equal ≈ {expected} (NCTU) """
        ratio = (CM_M1_TO_HZ) / (HARTREE_TO_J * CM_M1_TO_HARTREE)
        result = math.isclose(ratio, expected, rel_tol=1e-3)
        assert result, f"expected {expected}, got {ratio}"


# =================================================================================
# --- Length unit conversions - cross-check against literature (NIST) ---
# --- Reference: https://physics.nist.gov/cgi-bin/cuu/Results?search_for=length ---
# =================================================================================
class TestLengthConversions:

    def test_angstrom_to_bohr_known_value(self):
        expected = 1.88972613
        f""" 1 Angstrom ≈ {expected} bohr (NIST) """
        result = math.isclose(ANGSTROM_TO_BOHR_RADIUS, expected, rel_tol=1e-4)
        assert result, f"expected {expected}, got {result}"

    def test_bohr_radius_to_m_known_value(self):
        expected = 5.29177211e-11
        f""" 1 bohr ≈ {expected} m (NIST) """
        result = math.isclose(BOHR_RADIUS_TO_M, expected, rel_tol=1e-4)
        assert result, f"expected {expected}, got {result}"

    def test_angstrom_in_bohr_consistent_with_bohr_in_m(self):
        expected = 1e-10
        f""" 1 Angstrom ≈ {expected} m (NIST) """
        ratio = ANGSTROM_TO_BOHR_RADIUS * BOHR_RADIUS_TO_M
        result = math.isclose(ratio, expected, rel_tol=1e-4)
        assert result, f"expected {expected}, got {ratio}"


# ===============================================================================
# --- Mass unit conversions - cross-check against literature (NIST) ---
# --- Reference: https://physics.nist.gov/cgi-bin/cuu/Results?search_for=mass ---
# ===============================================================================
class TestMassConversions:

    def test_amu_to_kg_known_value(self):
        expected = 1.66053906892e-27
        f""" 1 amu ≈ {expected} kg (NIST) """
        result = math.isclose(AMU_TO_KG, expected, rel_tol=1e-4)
        assert result, f"expected {expected}, got {result}"

    def test_revelo_known_value(self):
        expected = 1822.88848628
        f""" 1 amu ≈ {expected} electron masses (NIST) """
        result = math.isclose(REVELO, expected, rel_tol=1e-4)
        assert result, f"expected {expected}, got {result}"

    def test_electron_mass_known_value(self):
        expected = 9.1093837139e-31
        f""" 1 electron mass ≈ {expected} electron masses (NIST) """
        ratio = AMU_TO_KG / REVELO
        result = math.isclose(ratio, expected, rel_tol=1e-4)
        assert result, f"expected {expected}, got {ratio}"


# ===========================================================================================
# --- Speed of light - cross-check against literature (NIST) ---
# --- Reference 1: https://physics.nist.gov/cgi-bin/cuu/Value?c|search_for=adopted_in! ---
# --- Reference 2: https://wild.life.nctu.edu.tw/class/common/energy-unit-conv-table.html ---
# ===========================================================================================
class TestSpeedOfLight:

    def test_light_speed_exact(self):
        expected = 299_792_458
        f""" c = {expected} m s-1 (NIST) """
        result = LIGHT_SPEED == expected
        assert result, f"expected {expected}, got {LIGHT_SPEED}"

    def test_cm_m1_to_hz_consistent_with_light_speed(self):
        """ 1 cm-1 to Hz should be equal LIGHT_SPEED * 100 (m s-1 -> cm s-1) """
        result = math.isclose(CM_M1_TO_HZ, LIGHT_SPEED * 100, rel_tol=1e-4)
        assert result, f"expected {CM_M1_TO_HZ}, got {LIGHT_SPEED * 100}"


# ===============================================================================
# --- Conversion factors - cross-check against my formulas ---
# ===============================================================================
class TestConversionFactors:

    def test_conversion_factor_hessian_values(self):
        """ Check of the conversion factor for Hessian values according to my formula """
        expected = 1 / (2 * math.pi * LIGHT_SPEED * 100) * (
        math.sqrt(HARTREE_TO_J / (AMU_TO_KG * BOHR_RADIUS_TO_M ** 2)))
        result = math.isclose(CONVERSION_FACTOR_HESSIAN_VALUES, expected, rel_tol=1e-4)
        assert result, f"expected {expected}, got {CONVERSION_FACTOR_HESSIAN_VALUES}"

    def test_conversion_factor_f_star_eckart(self):
        """ Check of the conversion factor for F_star (Eckart potential) according to my formula """
        expected = CM_M1_TO_HZ ** 2 * AMU_TO_KG * BOHR_RADIUS_TO_M ** 2 / HARTREE_TO_J
        result = math.isclose(CONVERSION_FACTOR_F_STAR_ECKART, expected, rel_tol=1e-4)
        assert result, f"expected {expected}, got {CONVERSION_FACTOR_F_STAR_ECKART}"


# ====================================================================================
# --- Boltzmann constant - cross-check against literature (NIST) ---
# --- Reference: https://physics.nist.gov/cgi-bin/cuu/Value?k|search_for=boltzmann ---
# ====================================================================================
class TestBoltzmannConstant:

    def test_k_b_hartree_known_value(self):
        expected = 3.16681156e-6
        f""" k_B ≈ {expected} Hartree K-1 """
        result = math.isclose(K_B, expected, rel_tol=1e-4)
        assert result, f"expected {expected}, got {K_B}"

    def test_k_b_j_known_value(self):
        expected = 1.380649e-23
        f""" k_B ≈ {expected} J K-1 """
        result = math.isclose(K_B * HARTREE_TO_J, expected, rel_tol=1e-4)
        assert result, f"expected {expected}, got {K_B * HARTREE_TO_J}"


# ===============================================================================
# --- Time units - cross-check against literature (NIST) ---
# --- Reference: https://physics.nist.gov/cgi-bin/cuu/Results?search_for=time ---
# ===============================================================================
class TestTimeConversions:

    def test_seconds_in_minute(self):
        expected = 60
        f""" 1 minute = {expected} seconds (NIST) """
        result = math.isclose(1 / (HOUR * 60), expected, rel_tol=1e-12)
        assert result, f"expected {expected}, got {1 / (HOUR * 60)}"
    
    def test_seconds_in_hour(self):
        expected = 3600
        f""" 1 hour = {expected} seconds (NIST) """
        result = math.isclose(1 / HOUR, expected, rel_tol=1e-12)
        assert result, f"expected {expected}, got {1 / HOUR}"

    def test_seconds_in_day(self):
        expected = 86400
        f""" 1 day = {expected} seconds (NIST) """
        result = math.isclose(1 / DAY, expected, rel_tol=1e-12)
        assert result, f"expected {expected}, got {1 / DAY}"

    def test_seconds_in_year(self):
        expected = 31536000
        f""" 1 year = {expected} seconds (NIST) """
        result = math.isclose(1 / YEAR, expected, rel_tol=1e-12)
        assert result, f"expected {expected}, got {1 / YEAR}"

    def test_hours_in_day(self):
        expected = 24
        f""" 1 day = {expected} hours (NIST) """
        result = math.isclose(HOUR / DAY, expected, rel_tol=1e-12)
        assert result, f"expected {expected}, got {HOUR / DAY}"
    
    def test_hours_in_year(self):
        expected = 8760
        f""" 1 year = {expected} days (NIST) """
        result = math.isclose(HOUR / YEAR, expected, rel_tol=1e-12)
        assert result, f"expected {expected}, got {HOUR / YEAR}"

    def test_days_in_year(self):
        expected = 365
        f""" 1 year = {expected} days (NIST) """
        result = math.isclose(DAY / YEAR, expected, rel_tol=1e-12)
        assert result, f"expected {expected}, got {DAY / YEAR}"