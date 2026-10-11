# --- Test: test_irc_helpers. Unit tests for irc_helpers.py
# Run with: pytest test_irc_helpers.py -v ---

# --- Modules ---
import pytest # type: ignore
from unittest.mock import patch

import numpy as np
from pathlib import Path

from tunnex_2.constants_and_dataclasses.constants_and_settings import ( # type: ignore
ANGSTROM_TO_BOHR_RADIUS,
CM_M1_TO_HARTREE,
CONVERSION_FACTOR_HESSIAN_VALUES)


# --- Module to test ---
from tunnex_2.irc_computations.ancillary_functions.irc_helpers import ( # type: ignore
    _com_correction,
    _rot_correction,
    _struct_dist,
    _find_tangent,
    _hessian_mass_weighting,
    _build_transl_rot_space,
    _proj_modes_out,
    irc_steps_comp,
    proj_freq_extraction_tunnex,
    freq_and_modes_extraction_tunnex,
    vibrations_format_transform)


# --- Helpers and shared data ---
DUMMY_FILE = Path("test.log")


def _rot_matrix_for_given_angle_z(theta):
    """ It returns the transposed rotation matrix about the z-axis for a given angle """
    Rz = np.array([[ np.cos(theta), np.sin(theta), 0],
                   [-np.sin(theta), np.cos(theta), 0],
                   [ 0,             0,             1]])
    return Rz

def _rot_matrix_random():
    """ It returns the transposed random rotation matrix """
    Q, _ = np.linalg.qr(np.random.default_rng(42).standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


@pytest.fixture
def homonuclear_diatomic():
    """ Two equal-mass atoms along x-axis (H-H). COM (center of masses) should be at the midpoint """
    geometry = np.array([[1.0, 0.0, 0.0],
                         [3.0, 0.0, 0.0]])
    masses = np.array([1.0, 1.0])
    return geometry, masses

@pytest.fixture
def acetylene_like():
    """ Acetylene-like molecule with the COM at (0, 0, 0) """
    geometry = np.array([[-2.0, 0.0, 0.0],
                         [-1.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [2.0, 0.0, 0.0]])
    masses = np.array([1.0, 12.0, 12.0, 1.0])
    return geometry, masses

@pytest.fixture
def water_like():
    """ Three-atom geometry with unequal masses (H-O-H analogue) """
    geometry = np.array([[0.0,  0.0,  0.0],
                         [0.96, 0.0,  0.0],
                         [-0.24, 0.93, 0.0]])
    masses = np.array([16.0, 1.0, 1.0])
    return geometry, masses

@pytest.fixture
def ammonia_like():
    """ Four-atom non-planar geometry with unequal masses (NH3 analogue) """
    geometry = np.array([[0.0, 0.0, 0.07],
                         [-0.47, 0.81, -0.32],
                         [-0.47, -0.81, -0.32],
                         [0.94, 0.0, -0.32]])
    masses = np.array([14.0, 1.0, 1.0, 1.0])
    return geometry, masses

@pytest.fixture
def two_point_trajectory():
    """ Two structures to test the tangent vectors computation """
    geometry_1 = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    geometry_2 = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    structures = [geometry_1, geometry_2]
    masses = np.array([1.0, 1.0])
    return structures, masses


@pytest.fixture
def four_point_trajectory():
    """ Four structures to test the tangent vectors computation """
    geometry_1 = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    geometry_2 = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    geometry_3 = np.array([[4.0, 0.0, 0.0], [-4.0, 0.0, 0.0]])
    geometry_4 = np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    structures = [geometry_1, geometry_2, geometry_3, geometry_4]
    masses = np.array([1.0, 1.0])
    return structures, masses

@pytest.fixture
def hessian_projection():
    """ Hessian and vectors to test the matrix projection """
    hessian = np.diag([1.0, 4.0, 9.0, 16.0])
    transl_rot_matrix = np.array([[1.0], [0.0], [0.0],[0.0]])
    irc_tangent = np.array([0.0, 1.0, 0.0, 0.0])
    return hessian, transl_rot_matrix, irc_tangent


# ===========================================================================
# --- _com_correction - normal behaviour ---
# ===========================================================================
class TestComCorrectionNormal:

    def test_com_at_origin_after_correction(self, homonuclear_diatomic):
        """ Moving the diatomic molecule COM to (0, 0, 0) """
        geometry, masses = homonuclear_diatomic
        result = _com_correction(geometry, masses)
        com = np.average(result, weights=masses, axis=0)
        assert np.allclose(com, 0.0, atol=1e-12), f"COM should be zero after correction, got {com}"

    def test_already_centred_unchanged(self, acetylene_like):
        """ If COM is already at origin, the result should be equal the input """
        geometry, masses = acetylene_like
        result = _com_correction(geometry, masses)
        assert np.allclose(result, geometry, atol=1e-12), f"expected {result} == {geometry}"

    def test_single_atom_returns_origin(self):
        """ For a single atom the code should return (0, 0, 0) """
        geometry = np.array([[3.0, -2.0, 5.0]])
        masses = np.array([12.0])
        result = _com_correction(geometry, masses)
        expected = np.zeros((1, 3))
        assert np.allclose(result, expected, atol=1e-12), f"expected {result} == {expected}"

    def test_water_like_com_at_origin(self, water_like):
        """ Moving the water-like molecule COM to (0, 0, 0) """
        geometry, masses = water_like
        result = _com_correction(geometry, masses)
        com = np.average(result, weights=masses, axis=0)
        assert np.allclose(com, 0.0, atol=1e-12), f"COM should be zero after correction, got {com}"

    def test_shape_preserved(self, water_like):
        """ The array shape of a water-like molecule should be the same after the correction """
        geometry, masses = water_like
        result = _com_correction(geometry, masses)
        assert result.shape == geometry.shape, f"expected {result.shape} == {geometry.shape}"

    def test_relative_distances_unchanged(self, water_like):
        """ COM shift must not change the distances between atoms """
        geometry, masses = water_like
        result = _com_correction(geometry, masses)
        dist_before = np.linalg.norm(geometry[0] - geometry[1])
        dist_after = np.linalg.norm(result[0] - result[1])
        assert np.isclose(dist_before, dist_after, atol=1e-12), f"expected {dist_before} == {dist_after}"

    def test_returns_new_array(self, homonuclear_diatomic):
        """ COM function must not modify the input array in-place """
        geometry, masses = homonuclear_diatomic
        original = geometry.copy()
        _com_correction(geometry, masses)
        assert np.allclose(geometry, original), f"expected {geometry} == {original}"


# ===========================================================================
# --- _com_correction - input validation ---
# ===========================================================================
class TestComCorrectionValidation:

    def test_1d_geometry_raises(self):
        """ Got (3,) geometry shape instead of (N,3). Expecting a ValueError """
        with pytest.raises(ValueError, match="shape"):
            _com_correction(np.array([1.0, 2.0, 3.0]), np.array([1.0]))

    def test_geometry_wrong_columns_raises(self):
        """ Got (3,2) geometry shape instead of (3,3). Expecting a ValueError """
        with pytest.raises(ValueError, match="shape"):
            _com_correction(np.ones((3, 2)), np.ones(3))

    def test_2d_masses_raises(self):
        """ Got (3,1) masses instead of (3,). Expecting a ValueError """
        with pytest.raises(ValueError, match="shape"):
            _com_correction(np.ones((3, 3)), np.ones((3, 1)))

    def test_atom_count_mismatch_raises(self):
        """ Got 3 atoms in a molecule and 4 masses. Expecting a ValueError """
        with pytest.raises(ValueError, match="does not match"):
            _com_correction(np.ones((3, 3)), np.ones(4))

# ===========================================================================
# --- _rot_correction - normal behaviour ---
# ===========================================================================
class TestRotCorrectionNormal:

    def test_identical_geometries_unchanged(self, water_like):
        """ Identical geometries. The function should not change anything """
        geometry, masses = water_like
        result = _rot_correction(geometry, geometry, masses, DUMMY_FILE)
        assert np.allclose(result, geometry, atol=1e-10), f"expected {result} == {geometry}"

    def test_rotation_around_z(self, water_like):
        """ A 90-degree rotation around z should be corrected back to the reference geometry """
        reference, masses = water_like
        theta = np.pi / 2
        Rz = _rot_matrix_for_given_angle_z(theta)
        rotated = reference @ Rz.T
        result = _rot_correction(reference, rotated, masses, DUMMY_FILE)
        assert np.allclose(result, reference, atol=1e-10), f"expected {result} == {reference}"

    def test_rotation_around_arbitrary_axis(self, water_like):
        """
        Random rotation should be fully corrected.
        A random proper rotation is builded via QR decomposition
        """
        reference, masses = water_like
        Q = _rot_matrix_random()
        rotated = reference @ Q.T
        result = _rot_correction(reference, rotated, masses, DUMMY_FILE)
        assert np.allclose(result, reference, atol=1e-10), f"expected {result} == {reference}"

    def test_output_shape_preserved(self, water_like):
        """ The array shape of a water-like molecule should be the same after the correction """
        geometry, masses = water_like
        result = _rot_correction(geometry, geometry, masses, DUMMY_FILE)
        assert result.shape == geometry.shape, f"expected {result.shape} == {geometry.shape}"

    def test_inter_atomic_distances_preserved(self, water_like):
        """ Rotation must not change the distances between atoms """
        reference, masses = water_like
        theta = np.pi / 3
        Rz = _rot_matrix_for_given_angle_z(theta)
        rotated = reference @ Rz.T
        result = _rot_correction(reference, rotated, masses, DUMMY_FILE)
        for i in range(len(reference)):
            for j in range(i + 1, len(reference)):
                d_ref = np.linalg.norm(reference[i] - reference[j])
                d_res = np.linalg.norm(result[i]    - result[j])
                assert np.isclose(d_ref, d_res, atol=1e-10), (
                    f"Distance {i}-{j} changed: {d_ref} vs {d_res}")

    def test_no_reflection_in_result(self, ammonia_like):
        """
        The rotation matrix implied by the result must have det = +1 (a proper rotation), never -1 (a reflection).
        The test should be run on the non-planar molecules, because det could be 0.0 otherwise
        """
        # --- Setting th input ---
        reference, masses = ammonia_like
        theta = np.pi / 5
        Rz = _rot_matrix_for_given_angle_z(theta)
        rotated = reference @ Rz.T
        result = _rot_correction(reference, rotated, masses, DUMMY_FILE)
        # --- Recovering the effective rotation matrix from before/after ---
        R_eff, _, _, _ = np.linalg.lstsq(rotated, result, rcond=None)
        # --- Checking if the matrix is orthogonal (as the rotation matrix should be) ---
        assert np.allclose(R_eff.T @ R_eff, np.eye(3), atol=1e-6), (
        "Effective transformation must be orthogonal, "
        f"got R.T @ R =\n{R_eff.T @ R_eff}")
        # --- Checking if the matrix det is +1.0 ---
        assert np.isclose(np.linalg.det(R_eff), 1.0, atol=1e-6), (
            f"Result must be a proper rotation (det = +1.0), got det={np.linalg.det(R_eff)}")

    def test_degenerate_planar_molecule_logs_warning(self, water_like):
            """ A planar molecule (rank-2 geometry) should trigger a logger warning, not raise an exception """
            geometry, masses = water_like
            import_logger = "tunnex_2.irc_computations.ancillary_functions.irc_helpers.logger"
            with patch(import_logger) as mock_logger:
                result = _rot_correction(geometry, geometry, masses, DUMMY_FILE)
                mock_logger.warning.assert_called_once()
            assert result.shape == geometry.shape, f"expected {result.shape} == {geometry.shape}"

    def test_degenerate_linear_molecule_logs_warning(self, acetylene_like):
        """
        A linear molecule (rank-1 geometry) and an atom (rank-0 geometry)
        should trigger a logger warning, not raise an exception
        """
        geometry, masses = acetylene_like
        atom, mass = np.array([[0.0, 0.0, 0.0]]), np.array([1.0])         
        import_logger = "tunnex_2.irc_computations.ancillary_functions.irc_helpers.logger"
        with patch(import_logger) as mock_logger:
            result = _rot_correction(geometry, geometry, masses, DUMMY_FILE)
            mock_logger.warning.assert_called_once()
        with patch(import_logger) as mock_logger:    
            result_atom = _rot_correction(atom, atom, mass, DUMMY_FILE)
            mock_logger.warning.assert_called_once()
        assert result.shape == geometry.shape, f"expected {result.shape} == {geometry.shape}; test #1"
        assert result_atom.shape == atom.shape, f"expected {result_atom.shape} == {atom.shape}; test #2"


# ===========================================================================
# _rot_correction - input validation
# ===========================================================================
class TestRotCorrectionValidation:

    def test_shape_mismatch_raises(self):
        """ Got geometries with different shapes. Expecting a ValueError """
        g1 = np.ones((3, 3))
        g2 = np.ones((4, 3))
        with pytest.raises(ValueError, match="do not match"):
            _rot_correction(g1, g2, np.ones(3), DUMMY_FILE)

    def test_wrong_columns_raises(self):
        """ Got (N,2) geometry shape instead of (N,3). Expecting a ValueError """
        g = np.ones((3, 2))
        with pytest.raises(ValueError, match="shape"):
            _rot_correction(g, g, np.ones(3), DUMMY_FILE)

    def test_reference_geometry_wrong_ndim_raises(self):
        """ Got 1D reference geometry instead of 2D. Expecting a ValueError """
        with pytest.raises(ValueError, match="shape"):
            _rot_correction(np.ones(3), np.ones((3, 3)), np.ones(3), DUMMY_FILE)

    def test_2d_masses_raises(self):
        """ Got (N,1) masses shape instead of (N,). Expecting a ValueError """
        g = np.ones((3, 3))
        with pytest.raises(ValueError, match="shape"):
            _rot_correction(g, g, np.ones((3, 1)), DUMMY_FILE)

    def test_atom_count_mismatch_raises(self):
        """ Got 4 masses for 3 atoms. Expecting a ValueError """
        g = np.ones((3, 3))
        with pytest.raises(ValueError, match="does not match"):
            _rot_correction(g, g, np.ones(4), DUMMY_FILE)


# ===========================================================================
# --- _struct_dist - normal behaviour ---
# ===========================================================================
class TestStructDistNormal:

    def test_identical_geometries_zero_distance(self, water_like):
        """ Identical geometries should have zero structural distance """
        geometry, masses = water_like
        result = _struct_dist(geometry, geometry, masses)
        expected = 0.0
        assert np.isclose(result, expected), f"expected {result} == {expected}"

    def test_single_atom_unit_displacement(self):
        """ A 1 amu atom displaced by 1 Angstrom should give the conversion factor (ANGSTROM_TO_BOHR_RADIUS) """
        reference = np.array([[0.0, 0.0, 0.0]])
        geometry = np.array([[1.0, 0.0, 0.0]])
        masses = np.array([1.0])
        result = _struct_dist(reference, geometry, masses)
        expected = ANGSTROM_TO_BOHR_RADIUS
        assert np.isclose(result, expected), f"expected {result} == {expected}"

    def test_mass_weighting(self):
        """ Structural distance should scale with the square root of mass """
        reference = np.array([[0.0, 0.0, 0.0]])
        geometry = np.array([[1.0, 0.0, 0.0]])
        masses = np.array([4.0])
        result = _struct_dist(reference, geometry, masses)
        expected = 2.0 * ANGSTROM_TO_BOHR_RADIUS
        assert np.isclose(result, expected), f"expected {result} == {expected}"

    def test_three_dimensional_displacement(self):
        """ Distance should account for displacement along all three coordinates """
        reference = np.array([[0.0, 0.0, 0.0]])
        geometry = np.array([[1.0, 2.0, 2.0]])
        masses = np.array([1.0])
        result = _struct_dist(reference, geometry, masses)
        expected = 3.0 * ANGSTROM_TO_BOHR_RADIUS
        assert np.isclose(result, expected), f"expected {result} == {expected}"


# ===========================================================================
# --- _find_tangent - normal behaviour ---
# ===========================================================================
class TestFindTangentNormal:

    def test_two_points_give_normalized_tangent(self, two_point_trajectory):
        """ Two structures should give a normalized tangent along the x-axis """
        structures, masses = two_point_trajectory
        result = _find_tangent(structures, masses, DUMMY_FILE)
        expected = [1/np.sqrt(2), 0.0, 0.0, -1/np.sqrt(2), 0.0, 0.0]
        expected_len = 2
        assert len(result) == expected_len, f"expected {len(result)} == {expected_len}; test #1"
        for i in range(2):
            norm = np.linalg.norm(result[i])
            assert np.isclose(norm, 1.0), f"expected {norm} == {1.0}, {i} point; test #2"
            assert np.allclose(result[i], expected), f"expected {result} == {expected}, {i} point; test #3"

    def test_four_points_give_normalized_tangent(self, four_point_trajectory):
        """ Multiple structures should give a normalized tangent along the x-axis """
        structures, masses = four_point_trajectory
        result = _find_tangent(structures, masses, DUMMY_FILE)
        expected = [1/np.sqrt(2), 0.0, 0.0, -1/np.sqrt(2), 0.0, 0.0]
        expected_len = 4
        assert len(result) == expected_len, f"expected {len(result)} == {expected_len}; test #1"
        for i in range(4):
            norm = np.linalg.norm(result[i])
            assert np.isclose(norm, 1.0), f"expected {norm} == {1.0}, {i} point; test #2"
            assert np.allclose(result[i], expected), f"expected {result} == {expected}, {i} point; test #3"

    def test_tangent_has_correct_shape(self, four_point_trajectory):
        """ Each tangent should contain 3 Cartesian components per atom """
        structures, masses = four_point_trajectory
        result = _find_tangent(structures, masses, DUMMY_FILE)
        assert len(result) == len(structures), f"expected {len(result)} == {len(structures)}; test #1"
        for tangent in result:
            expected = (3 * len(masses),)
            assert tangent.shape == expected, f"expected {tangent.shape} == {expected}; test #2"

    def test_mass_weighting_gives_expected_tangent(self):
        """ Mass weighting should produce the expected normalized tangent """
        geometry_1 = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        geometry_2 = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        structures = [geometry_1, geometry_2]
        masses = np.array([1.0, 4.0])
        result = _find_tangent(structures, masses, DUMMY_FILE)
        expected = np.array([np.sqrt(0.8), 0.0, 0.0, -np.sqrt(0.2), 0.0, 0.0])
        for i in range(2):
            assert np.allclose(result[i], expected), f"expected {result[i]} == {expected}; {i} point"

    def test_three_points_equal_spacing_use_central_difference(self):
        """ Three equally spaced structures should use the central difference tangent (middle point) """
        geometry_1 = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
        geometry_2 = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        geometry_3 = np.array([[1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]])
        structures = [geometry_1, geometry_2, geometry_3]
        masses = np.array([1.0, 4.0])
        result = _find_tangent(structures, masses, DUMMY_FILE)
        expected = np.array([np.sqrt(0.8), 0.0, 0.0, -np.sqrt(0.2), 0.0, 0.0])
        assert np.allclose(result[1], expected), f"expected {result[1]} == {expected}"

    def test_three_points_nonuniform_spacing(self):
        """ Three unequally spaced structures should use the non-uniform three-point derivative (middle point) """
        geometry_1 = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
        geometry_2 = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        geometry_3 = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        structures = [geometry_1, geometry_2, geometry_3]
        masses = np.array([1.0, 4.0])
        result = _find_tangent(structures, masses, DUMMY_FILE)
        expected = np.array([np.sqrt(0.8), 0.0, 0.0, -np.sqrt(0.2), 0.0, 0.0])
        assert np.allclose(result[1], expected), f"expected {result[1]} == {expected}"


# ===========================================================================
# --- _find_tangent - input validation ---
# ===========================================================================
class TestFindTangentValidation:

    def test_pure_translation_zero_irc(self):
        """ The same structures but shifted should give zero IRC step. Expecting a ValueError """
        geometry_1 = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        geometry_2 = np.array([[1.0, 5.0, 0.0], [-1.0, 5.0, 0.0]])
        structures = [geometry_1, geometry_2]
        masses = np.array([1.0, 4.0])
        with pytest.raises(ValueError, match="Zero IRC step"):
            _find_tangent(structures, masses, DUMMY_FILE)

    def test_pure_rotaion_zero_irc(self, water_like):
        """ The same structures but rotated should give zero IRC step. Expecting a ValueError """
        reference, masses = water_like
        Q = _rot_matrix_random()
        rotated = reference @ Q.T
        structures = [reference, rotated]
        with pytest.raises(ValueError, match="Zero IRC step"):
            _find_tangent(structures, masses, DUMMY_FILE)

    def test_zero_tangent(self):
        """A symmetric trajectory with zero central derivative. Expecting a ValueError """
        geometry_1 = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
        geometry_2 = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        structures = [geometry_1, geometry_2, geometry_1]
        masses = np.array([1.0, 4.0])
        with pytest.raises(ValueError, match="Zero tangent"):
            _find_tangent(structures, masses, DUMMY_FILE)

    def test_single_structure_raises(self, water_like):
        """ At least two structures should be submitted. Expecting a ValueError """
        geometry, masses = water_like
        structures = [geometry]
        with pytest.raises(ValueError, match="two structures"):
            _find_tangent(structures, masses, DUMMY_FILE)


# ===========================================================================
# --- _hessian_mass_weighting - normal behaviour ---
# ===========================================================================
class TestHessianMWNormal:

    def test_unit_masses_leave_hessian_unchanged(self):
        """ Unit masses should leave the Hessian unchanged """
        hessian = np.ones((3, 3))
        masses = np.array([1.0])
        result = _hessian_mass_weighting(hessian, masses)
        assert np.allclose(result, hessian), f"expected {result} == {hessian}"

    def test_mass_four_scales_hessian_by_four(self):
        """ A mass of 4 should scale all Hessian elements by 1/4 """
        hessian = np.ones((3, 3))
        masses = np.array([4.0])
        result = _hessian_mass_weighting(hessian, masses)
        expected = np.ones((3, 3)) / 4.0
        assert np.allclose(result, expected), f"expected {result} == {expected}"

    def test_different_masses_scale_hessian_correctly(self):
        """ Different atomic masses should scale Hessian elements correctly and preserve Hessian symmetry """
        hessian = np.ones((6, 6))
        masses = np.array([1.0, 4.0])
        result = _hessian_mass_weighting(hessian, masses)
        expected = np.array([
            [1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5, 0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5, 0.25, 0.25, 0.25]])
        assert np.allclose(result, expected), f"expected {result} == {expected}; test #1"
        assert np.allclose(result, result.T), f"expected {result} == {result.T}; test #2"


# ===========================================================================
# --- _hessian_mass_weighting - input validation ---
# ===========================================================================
class TestHessianMWValidation:
        
    def test_hessian_shape_must_match_number_of_atoms(self):
        """ Hessian shape should match three coordinates per atom """
        hessian = np.eye(3)
        masses = np.array([1.0, 4.0])
        with pytest.raises(ValueError, match="incompatible"):
            _hessian_mass_weighting(hessian, masses)


# ===========================================================================
# --- _build_transl_rot_space - normal behaviour ---
# ===========================================================================
class TestBuildTRSpaceNormal:

    def test_single_atom_gives_only_translation_vectors(self):
        """ A single atom should have three mass-weighted translation vectors """
        coordinates = np.array([[1.0, 2.0, 3.0]])
        masses = np.array([4.0])
        result = _build_transl_rot_space(coordinates, masses)
        expected = np.array([
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0]])
        assert result.shape == (3, 6), "shape"
        assert np.allclose(result, expected), f"expected {result} == {expected}"

    def test_translation_vectors_are_mass_weighted(self):
        """ Translation vectors should contain sqrt(mass) for each atom """
        coordinates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        masses = np.array([1.0, 4.0])
        result = _build_transl_rot_space(coordinates, masses)
        expected = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]])
        expected_share = (6, 6)
        assert result.shape == expected_share, f"expected {result.shape} == {expected_share}; test #1"
        assert np.allclose(result[:, :3], expected), f"expected {result} == {expected}; test #2"

    def test_rotation_vectors_for_linear_x_geometry(self):
        """ Rotation vectors should correspond to cross products with the rotation axes """
        coordinates = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        masses = np.array([1.0, 1.0])
        result = _build_transl_rot_space(coordinates, masses)
        expected_rotation = np.array([
            [0.0,  0.0,  0.0],
            [0.0,  0.0, -1.0],
            [0.0,  1.0,  0.0],
            [0.0,  0.0,  0.0],
            [0.0,  0.0,  1.0], 
            [0.0, -1.0,  0.0]])
        assert np.allclose(result[:, 3:], expected_rotation), f"expected {result[:, 3:]} == {expected_rotation}"

    def test_rotation_vectors_are_mass_weighted(self):
        """ Rotation vectors should be scaled by the square root of atomic masses """
        coordinates = np.array([[-1.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
        masses = np.array([1.0, 4.0])
        result = _build_transl_rot_space(coordinates, masses)
        expected_rotation = np.array([
            [0.0,  0.0,  0.0],
            [0.0,  0.0, -1.0],
            [0.0,  1.0,  0.0],
            [0.0,  0.0,  0.0],
            [0.0,  0.0,  0.5],
            [0.0, -0.5,  0.0]])
        assert np.allclose(result[:, 3:], expected_rotation), f"expected {result[:, 3:]} == {expected_rotation}"

    def test_translation_of_geometry_does_not_change_tr_space(self):
        """ A rigid translation of the geometry should not change the TR space """
        coordinates = np.array([[-1.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
        masses = np.array([1.0, 4.0])
        translated_coordinates = coordinates + np.array([10.0, -5.0, 7.0])
        result = _build_transl_rot_space(coordinates, masses)
        translated_result = _build_transl_rot_space(translated_coordinates, masses)
        assert np.allclose(translated_result, result), f"expected {translated_result} == {result}"

    def test_rotation_vectors_for_nonlinear_geometry(self, water_like):
        """ A non-linear geometry should have three non-zero rotation vectors """
        coordinates, masses = water_like
        result = _build_transl_rot_space(coordinates, masses)
        rotation = result[:, 3:]
        assert result.shape[1] == 6, f"this function always should give {6} vectors, even for single atoms "
        f"and linear molecules, got {result.shape[1]}; test #1"
        assert result.shape == (9, 6), f"expected {result.shape} == {(9, 6)}; test #2"
        assert rotation.shape == (9, 3), f"expected {result.shape} == {(9, 3)}; test #3"
        result = np.all(np.linalg.norm(rotation, axis=0) > 0.0)
        assert result, f"expected {True}, got {result}; test #4"


# ===========================================================================
# --- _proj_modes_out - normal behaviour ---
# ===========================================================================
class TestProjModesOutNormal:

    def test_projection_removes_projected_direction(self, hessian_projection):
        """ The projected direction should not contribute to the vibrational modes """
        hessian, transl_rot_matrix, _ = hessian_projection
        _, wavenums, _ = _proj_modes_out(hessian, transl_rot_matrix)
        expected_eigenvalue = 16.0
        expected_wavenum = (CONVERSION_FACTOR_HESSIAN_VALUES * np.sqrt(expected_eigenvalue))
        result = np.allclose(wavenums[-1], [expected_wavenum])
        assert result, f"expected {True}, got {result}"

    def test_projection_keeps_all_vibrational_modes(self, hessian_projection):
        """ Projection should preserve all modes outside the projected subspace """
        hessian, transl_rot_matrix, _ = hessian_projection
        _, wavenums, modes = _proj_modes_out(hessian, transl_rot_matrix)
        expected_eigenvalues = np.array([4.0, 9.0, 16.0])
        expected_wavenums = (CONVERSION_FACTOR_HESSIAN_VALUES * np.sqrt(expected_eigenvalues))
        assert wavenums.shape == (3,), f"expected {wavenums.shape} == {(3,)}; test #1"
        assert modes.shape == (3, 4), f"expected {modes.shape} == {(3, 4)}; test #2"
        result = np.allclose(wavenums, expected_wavenums)
        assert result, f"expected {True}, got {result}; test #3"

    def test_irc_tangent_projects_out_one_additional_mode(self, hessian_projection):
        """ IRC tangent should remove one additional vibrational degree of freedom """
        hessian, transl_rot_matrix, irc_tangent = hessian_projection
        _, wavenums_without_irc, modes_without_irc = _proj_modes_out(hessian, transl_rot_matrix)
        _, wavenums_with_irc, modes_with_irc = _proj_modes_out(hessian, transl_rot_matrix, irc_tangent)
        assert wavenums_without_irc.shape == (3,), f"expected {wavenums_without_irc.shape} == {(3,)}; test #1"
        assert modes_without_irc.shape == (3, 4), f"expected {modes_without_irc.shape} == {(3, 4)}; test #2"
        assert wavenums_with_irc.shape == (2,), f"expected {wavenums_with_irc.shape} == {(2,)}; test #3"
        assert modes_with_irc.shape == (2, 4), f"expected {modes_with_irc.shape} == {(2, 4)}; test #4"
        expected_eigenvalues = np.array([9.0, 16.0])
        expected_wavenums = (CONVERSION_FACTOR_HESSIAN_VALUES * np.sqrt(expected_eigenvalues))
        result = np.allclose(wavenums_with_irc, expected_wavenums)
        assert result, f"expected {True}, got {result}; test #5"

    def test_mode_vectors_are_orthonormal(self, hessian_projection):
        """ Returned mode vectors should form an orthonormal basis """
        hessian, transl_rot_matrix, _ = hessian_projection
        _, _, modes = _proj_modes_out(hessian, transl_rot_matrix)
        overlap = modes @ modes.T
        expected = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        for mode, expected_mode in zip(modes, expected):
            assert np.isclose(abs(np.dot(mode, expected_mode)), 1.0), \
            f"Mode {mode} does not match expected direction {expected_mode}"
        # An orthonormal basis is built with the vector sign precision
        result_1 = np.allclose(modes, expected)
        assert result_1, f"expected {True}, got {result_1}; test #1"
        result_2 = np.allclose(overlap, np.eye(len(modes)))
        assert result_2, f"expected {True}, got {result_2}; test #2"

    def test_negative_eigenvalue_gives_negative_wavenumber(self):
        """ A negative Hessian eigenvalue should produce a negative wavenumber """
        hessian = np.diag([-4.0, 9.0])
        transl_rot_matrix = np.array([[0.0], [1.0]])
        _, wavenums, _ = _proj_modes_out(hessian, transl_rot_matrix)
        expected = -np.array([CONVERSION_FACTOR_HESSIAN_VALUES * np.sqrt(4.0)])
        result = np.allclose(wavenums, expected)
        assert result, f"expected {True}, got {result}"

    def test_zpve_uses_only_positive_wavenumbers(self):
        """ ZPVE should include only positive vibrational wavenumbers """
        hessian = np.diag([-4.0, 9.0, 16.0])
        transl_rot_matrix = np.array([[0.0], [1.0], [0.0]])
        zpve, wavenums, _ = _proj_modes_out(hessian, transl_rot_matrix)
        expected_wavenums = np.array([-CONVERSION_FACTOR_HESSIAN_VALUES * np.sqrt(4.0),
                                      CONVERSION_FACTOR_HESSIAN_VALUES * np.sqrt(16.0)])
        expected_zpve = (0.5 * CONVERSION_FACTOR_HESSIAN_VALUES * np.sqrt(16.0) * CM_M1_TO_HARTREE)
        result_1 = np.allclose(wavenums, expected_wavenums)
        assert result_1, f"expected {True}, got {result_1}; test #1"
        result_2 = np.isclose(zpve, expected_zpve)
        assert result_2, f"expected {True}, got {result_2}; test #2"


# ===========================================================================
# --- _proj_modes_out - input validation ---
# ===========================================================================
class TestProjModesOutValidation:
        
    def test_empty_hessian_raises(self):
        """ A Hessian should be a non-empty square matrix. Expecting a ValueError """
        hessian_empty = np.empty((0, 0))
        transl_rot_matrix = np.eye(3, 1)
        with pytest.raises(ValueError, match="empty"):
            _proj_modes_out(hessian_empty, transl_rot_matrix)

    def test_1d_hessian_raises(self):
        """ A Hessian should be a square matrix, not a 1D, 3D matrix etc. Expecting a ValueError """
        hessian_vector = np.ones(6)
        transl_rot_matrix = np.eye(3, 1)
        with pytest.raises(ValueError, match="square matrix"):
            _proj_modes_out(hessian_vector, transl_rot_matrix)

    def test_non_square_hessian_raises(self):
        """ A Hessian should be a square matrix, not (NxM). Expecting a ValueError """
        hessian = np.ones((3, 4))
        transl_rot_matrix = np.eye(3, 1)
        with pytest.raises(ValueError, match="square matrix"):
            _proj_modes_out(hessian, transl_rot_matrix)

    def test_non_2d_transl_rot_matrix_raises(self):
        """ A non-2D translation and rotation matrix is not allowed. Expecting a ValueError """
        hessian = np.eye(6)
        transl_rot_matrix = np.ones(6)
        with pytest.raises(ValueError, match="Translation and rotation matrix"):
            _proj_modes_out(hessian, transl_rot_matrix)

    def test_transl_rot_matrix_wrong_number_of_rows_raises(self):
        """A translation and rotation matrix with the wrong number of rows is not allowed. Expecting a ValueError """
        hessian = np.eye(6)
        transl_rot_matrix = np.ones((5, 6))
        with pytest.raises(ValueError, match="Translation and rotation matrix"):
            _proj_modes_out(hessian, transl_rot_matrix)

    def test_non_1d_irc_tangent_raises(self):
        """ A non-1D IRC tangent is not allowed. Expecting a ValueError """
        hessian = np.eye(6)
        transl_rot_matrix = np.ones((6, 6))
        irc_tangent = np.ones((6, 1))
        with pytest.raises(ValueError, match="IRC tangent"):
            _proj_modes_out(hessian, transl_rot_matrix, irc_tangent)

    def test_irc_tangent_wrong_size_raises(self):
        """An IRC tangent with the wrong number of coordinates is not allowed. Expecting a ValueError """
        hessian = np.eye(6)
        transl_rot_matrix = np.ones((6, 6))
        irc_tangent = np.ones(5)
        with pytest.raises(ValueError, match="IRC tangent"):
            _proj_modes_out(hessian, transl_rot_matrix, irc_tangent)


# ===========================================================================
# --- irc_steps_comp - normal behaviour ---
# ===========================================================================
class TestIRCStepCompNormal:

    def test_two_structures_give_one_irc_step(self):
        """ Two structures should produce one IRC step """
        structures = np.array([[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
        masses = np.array([1.0, 1.0])
        result = irc_steps_comp(DUMMY_FILE, structures, masses)
        assert len(result) == 1, "number of steps"
        result = np.isclose(result[0], np.sqrt(0.5) * ANGSTROM_TO_BOHR_RADIUS)
        assert result, f"expected {True}, got {result}"

    def test_n_structures_give_n_minus_one_steps(self, four_point_trajectory):
        """ N structures should produce N-1 IRC steps """
        structures, masses = four_point_trajectory
        result = irc_steps_comp(DUMMY_FILE, np.array(structures), masses)
        expected = len(structures) - 1
        assert len(result) == expected, f"expected {len(result)} == {expected}"

    def test_pure_translation_gives_zero_step(self):
        """ Pure translation should not contribute to the IRC step """
        structures = np.array([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], [[1.0, 5.0, 0.0], [-1.0, 5.0, 0.0]]])
        masses = np.array([1.0, 1.0])
        steps = irc_steps_comp(DUMMY_FILE, structures, masses)
        result = np.isclose(steps[0], 0.0)
        assert result, f"expected {True}, got {result}"

    def test_pure_rotation_gives_zero_step(self, water_like):
        """ Pure rotation should not contribute to the IRC step """
        reference, masses = water_like
        Q = _rot_matrix_random()
        rotated = reference @ Q.T
        structures = np.array([reference, rotated])
        steps = irc_steps_comp(DUMMY_FILE, structures, masses)
        result = np.isclose(steps[0], 0.0)
        assert result, f"expected {True}, got {result}"

    def test_mass_weighting_changes_irc_step(self):
        """ Atomic masses should affect the IRC step """
        structures = np.array([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]])
        result_equal = irc_steps_comp(DUMMY_FILE, structures, np.array([1.0, 1.0]))
        result_weighted = irc_steps_comp(DUMMY_FILE, structures, np.array([1.0, 4.0]),)
        result = not np.isclose(result_equal[0], result_weighted[0])
        assert result, f"expected {False}, got {result}"


# ===========================================================================
# --- irc_steps_comp - input validation ---
# ===========================================================================
class TestIRCStepCompValidation:

    def test_less_than_two_structures_raises(self, water_like):
        """ No structures or a single structure are not allowed. Expecting a ValueError """
        geometry, masses = water_like
        structures_0 = np.array([])
        structures_1 = np.array([geometry])
        # Zero structures
        with pytest.raises(ValueError, match="two structures")  as exc_info_1:
            irc_steps_comp(DUMMY_FILE, structures_0, masses), "0 structures"
        print("Caught (test #1):", exc_info_1.value)
        # One structure
        with pytest.raises(ValueError, match="two structures")  as exc_info_2:
            irc_steps_comp(DUMMY_FILE, structures_1, masses), "1 structure"
        print("Caught (test #2):", exc_info_2.value)


# ===========================================================================
# --- proj_freq_extraction_tunnex - normal behaviour ---
# ===========================================================================
class TestProjFreqExtTUNNormal:

    def test_returns_one_result_per_structure(self, four_point_trajectory):
        """ The function should return one ZPVE, frequency set, and mode set per structure """
        structures, masses = four_point_trajectory
        hessians = [np.eye(6)] * 4
        zpves, vibs, modes = proj_freq_extraction_tunnex(DUMMY_FILE, np.array(structures), masses, hessians)
        assert len(zpves) == len(structures), f"expected {len(zpves)} == {len(structures)}; test #1"
        assert len(vibs) == len(structures), f"expected {len(vibs)} == {len(structures)}; test #2"
        assert len(modes) == len(structures), f"expected {len(modes)} == {len(structures)}; test #3"

    def test_result_shapes(self, four_point_trajectory):
        """Returned ZPVEs, frequencies, and modes should have the expected shapes """
        structures, masses = four_point_trajectory
        hessians = [np.eye(6)] * 4
        zpves, vibs, modes = proj_freq_extraction_tunnex(DUMMY_FILE, np.array(structures), masses, hessians)
        for zpve, frequencies, mode_vectors in zip(zpves, vibs, modes):
            assert isinstance(zpve, float), f"expected {float}, got {zpve}; test #1"
            assert frequencies.ndim == 1, f"expected {frequencies.ndim} == {1}; test #2"
            assert mode_vectors.ndim == 2, f"expected {mode_vectors.ndim} == {2}; test #3"
            result = mode_vectors.shape[1]
            expected = 3 * len(masses)
            assert result == expected, f"expected {result} == {expected}; test #4"

    def test_different_hessians_produce_different_frequencies(self, water_like):
        """ Different Hessians should produce different projected frequencies """
        geometry, masses = water_like
        structures = np.array([geometry, geometry * 1.1, geometry * 1.2, geometry * 1.3])
        n = 3 * len(masses)
        hessians = np.array([np.eye(n), 2.0 * np.eye(n), 4.0 * np.eye(n), 6.0 * np.eye(n)])
        _, vibs, _ = proj_freq_extraction_tunnex(DUMMY_FILE, structures, masses, hessians)
        for i in range(3):
            assert not np.allclose(vibs[i], vibs[i+1]), \
                f"frequencies for {i} and {i+1} Hessians should not match"

    def test_negative_hessian_produces_negative_frequencies(self, water_like):
        """ A negative Hessian should produce negative projected frequencies """
        geometry, masses = water_like
        structures = np.array([geometry, geometry * 1.1, geometry * 1.2, geometry * 1.3])
        n = 3 * len(masses)
        hessian = -1.0 * np.eye(n)
        hessians = np.array([hessian] * len(structures))
        _, vibs, _ = proj_freq_extraction_tunnex(DUMMY_FILE, structures, masses, hessians)
        result = all(np.any(frequencies < 0) for frequencies in vibs)
        assert result, f"expected {True}, got {result}"


# ===========================================================================
# --- freq_and_modes_extraction_tunnex - normal behaviour ---
# ===========================================================================
class TestFreqModesExtTUNNormal:

    def test_returns_zpve_frequencies_and_modes(self, water_like):
        """ The function should return ZPVE, frequencies, and mode vectors """
        coordinates, masses = water_like
        hessian = np.eye(3 * len(masses))
        zpve, vibs, modes = freq_and_modes_extraction_tunnex(coordinates, masses, hessian)
        assert isinstance(zpve, float), f"expected {float}, got {zpve}; test #1"
        assert vibs.ndim == 1, f"expected {vibs.ndim} == {1}; test #2"
        assert modes.ndim == 2, f"expected {modes.ndim} == {2}; test #3"
        result = modes.shape[1]
        expected = 3 * len(masses)
        assert result == expected, f"expected {result} == {expected}; test #4"

    def test_different_hessians_produce_different_frequencies(self, water_like):
        """ Different Hessians should produce different frequencies """
        coordinates, masses = water_like
        n = 3 * len(masses)
        i = 0
        hessian_1 = np.eye(n)
        hessian_2 = 2.0 * np.eye(n)
        _, vibs_1, _ = freq_and_modes_extraction_tunnex(coordinates, masses, hessian_1)
        _, vibs_2, _ = freq_and_modes_extraction_tunnex(coordinates, masses, hessian_2)
        assert not np.allclose(vibs_1, vibs_2), \
            f"frequencies for {i} and {i+1} Hessians should not match"

    def test_negative_hessian_produces_negative_frequencies(self, water_like):
        """ A negative Hessian should produce negative projected frequencies """
        coordinates, masses = water_like
        n = 3 * len(masses)
        hessian = -np.eye(n)
        _, vibs, _ = freq_and_modes_extraction_tunnex(coordinates, masses, hessian)
        result = np.all(vibs < 0)
        assert result, f"expected {True}, got {result}"


# ===========================================================================
# --- proj_freq_extraction_tunnex - input validation ---
# ===========================================================================
class TestProjFreqExtTUNValidation:

    def test_mismatched_number_of_hessians_raises(self, four_point_trajectory):
        """
        The number of Hessians must match the number of geometries
        and the number of IRC tangents. Expecting a ValueError
        """
        # Hessians and geometries match
        structures, masses = four_point_trajectory
        hessians = [np.eye(6)] * 3
        with pytest.raises(ValueError, match="must match"):
            proj_freq_extraction_tunnex(DUMMY_FILE, np.array(structures), masses, hessians)
        # IRC tangents and geometries match
        struct_coordinates = [np.zeros((1, 3)), np.ones((1, 3)), np.ones((1, 3)) * 2]
        atom_masses = np.array([1.0])
        hess_val_matrices = np.zeros((3, 3, 3))
        fake_tangents = [np.zeros(3), np.zeros(3)]
        with patch("tunnex_2.irc_computations.ancillary_functions.irc_helpers._find_tangent",
        return_value=fake_tangents,):
            with pytest.raises(ValueError, match="Number of geometries and IRC tangents must match"):
                proj_freq_extraction_tunnex("dummy.xyz", struct_coordinates,atom_masses,
                hess_val_matrices)


# ===========================================================================
# --- vibrations_format_transform - normal behaviour ---
# ===========================================================================
class TestVibFTransNormal:

    def test_normalization(self):
        """ Check vector normalization """
        frequencies = np.array([100.0])
        modes = np.array([[1.0, 0.0, 0.0, 0.0, 2.0, 0.0]])
        masses = np.array([1.0, 1.0])
        result = vibrations_format_transform(DUMMY_FILE, frequencies, modes, masses)
        result_list = [[float(x) for x in row] for row in result]
        expected = [[100.0, 1/np.sqrt(5), 0.0, 0.0, 0.0, 2/np.sqrt(5), 0.0]]
        result = np.allclose(result_list, expected)
        assert result, f"expected {True}, got {result}"

    def test_mass_unweighting_and_normalization(self):
        """ Check mass unweighting and vector normalization """
        frequencies = np.array([100.0])
        modes = np.array([[1.0, 0.0, 0.0, 0.0, 2.0, 0.0]])
        masses = np.array([1.0, 4.0])
        result = vibrations_format_transform(DUMMY_FILE, frequencies, modes, masses)
        result_list = [[float(x) for x in row] for row in result]
        expected = [[100.0, 1/np.sqrt(2), 0.0, 0.0, 0.0, 1/np.sqrt(2), 0.0]]
        result = np.allclose(result_list, expected)
        assert result, f"expected {True}, got {result}"
    
    def test_frequency_below_tolerance_is_filtered(self):
        """ Check the filtering of the translations and rotations """
        frequencies = np.array([1e-3, 1e-2, 1e-1])
        modes = np.eye(3)
        masses = np.array([1.0])
        result = vibrations_format_transform(DUMMY_FILE, frequencies, modes, masses, freq_tol=1e-2)
        assert len(result) == 2, f"expected {len(result)} == {2}; test #1"
        result = np.allclose([row[0] for row in result], [1e-2, 1e-1])
        assert result, f"expected {True}, got {result}; test #2"
    
    def test_negative_frequencies_are_preserved(self):
        """ Check that imaginary frequency correspond to the negative eigenvalue """
        frequencies = np.array([-100.0, 50.0])
        modes = np.eye(2, 3)
        masses = np.array([1.0])
        result = vibrations_format_transform(DUMMY_FILE, frequencies, modes, masses)
        assert len(result) == 2, f"expected {len(result)} == {2}; test #1"
        result = np.allclose([row[0] for row in result], [-100.0, 50.0])
        assert result, f"expected {True}, got {result}; test #2"


# ===========================================================================
# --- vibrations_format_transform - input validation ---
# ===========================================================================
class TestVibFTransValidation:

    def test_zero_norm_mode_raises(self):
        """ No zero-length modes are allowed because of the zero-division error. Expecting a ValueError """
        frequencies = np.array([100.0])
        modes = np.array([[0.0, 0.0, 0.0]])
        masses = np.array([1.0])
        with pytest.raises(ValueError, match="Zero-norm modes found"):
            vibrations_format_transform(DUMMY_FILE, frequencies, modes, masses)