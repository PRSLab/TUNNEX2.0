# --- Modules ---
import numpy as np
from pathlib import Path
from scipy.linalg import null_space

from tunnex_2.constants_and_dataclasses.constants_and_settings import ( # type: ignore
ANGSTROM_TO_BOHR_RADIUS,
CM_M1_TO_HARTREE,
CONVERSION_FACTOR_HESSIAN_VALUES)

from tunnex_2.irc_computations.ancillary_functions.interface import setup_logger # type: ignore


# --- Defining the logging function ---
logger = setup_logger()


# --- Defining the function to correct the geometries according to their center of masses ---
def _com_correction(geometry: np.ndarray, atom_masses: np.ndarray) -> np.ndarray:

    if geometry.ndim != 2 or geometry.shape[1] != 3:
        raise ValueError(f"Geometry must have shape (n_atoms, 3), got {geometry.shape}. Please check your data.")

    if atom_masses.ndim != 1:
        raise ValueError(f"Atom masses must have shape (n_atoms,), got {atom_masses.shape}. Please check your data.")

    if geometry.shape[0] != atom_masses.size:
        raise ValueError(f"Number of atoms in geometry ({geometry.shape[0]}) does not match number of masses ({atom_masses.size}). "
                         f"Please check your data.")

    center_of_mass = (np.sum(geometry * atom_masses[:, None], axis=0) / np.sum(atom_masses))

    return geometry - center_of_mass


# --- Defining the function to account for the geometry rotation correction ---
# Find R such that geometry @ R.T ≈ reference_geometry
def _rot_correction(reference_geometry: np.ndarray, geometry: np.ndarray,
    atom_masses: np.ndarray, filename: str | Path) -> np.ndarray:

    if reference_geometry.shape != geometry.shape:
        raise ValueError(f"Geometry shapes do not match: {reference_geometry.shape} vs {geometry.shape}. Please check your data.")

    if reference_geometry.ndim != 2 or reference_geometry.shape[1] != 3:
        raise ValueError(f"Geometries must have shape (n_atoms, 3), got {reference_geometry.shape}. Please check your data.")

    if atom_masses.ndim != 1:
        raise ValueError(f"Atom masses must have shape (n_atoms,), got {atom_masses.shape}. Please check your data.")

    if reference_geometry.shape[0] != atom_masses.size:
        raise ValueError(f"Number of atoms ({reference_geometry.shape[0]}) does not match number of masses ({atom_masses.size}). " 
                         f"Please check your data.")

    # --- Computing the mass-weighted covariance matrix ---
    path = Path(filename)
    H = geometry.T @ (atom_masses[:, None] * reference_geometry)

    # --- SVD decomposition ---
    U, s, Vt = np.linalg.svd(H)

    if np.isclose(s[-1], 0.0):
        n_zero = np.sum(np.isclose(s, 0.0))
        if n_zero == 2:
            mol_type = "linear geometry (rank 1)"
        elif n_zero == 1:
            mol_type = "planar geometry (rank 2)"
        else:
            mol_type = "fully degenerate (point) geometry (rank 0)"

        logger.warning("Degenerate rotation problem (%s), singular values: %s. "
        "Check '%s' if this is unexpected.", mol_type, s, path)
        # Rotation is not uniquely defined. This can occur for linear or otherwise degenerate geometries.

    # --- Eliminating the reflection.
    # If det < 0, apply an additional reflection to obtain a proper rotation ---
    det = np.linalg.det(Vt.T @ U.T)
    d = 1.0 if det >= 0.0 else -1.0
    D = np.diag([1.0, 1.0, d])

    # --- Building the optimal rotation matrix ---
    R = Vt.T @ D @ U.T

    # --- Rotating the geometry ---
    geometry_rotated = geometry @ R.T

    return geometry_rotated

# --- Defining the function to compute the distance between the structures (as amu ** 0.5 * bohr) ---
def _struct_dist(reference_geometry: np.ndarray, geometry: np.ndarray, atom_masses: np.ndarray) -> float:
    structural_distance = np.sqrt(np.sum(atom_masses[:, None] * (geometry - reference_geometry) ** 2)) * ANGSTROM_TO_BOHR_RADIUS
    return structural_distance


# --- Defining the function to find the IRC tangent ---
def _find_tangent(struct_coordinates: list[np.ndarray], atom_masses: np.ndarray, irc_file_xyz_file: str | Path) -> list[np.ndarray]:

    # --- Computing the square roots of atomic masses and defining the rotationally aligned geometries and the IRC steps ---
    if len(struct_coordinates) < 2:
        raise ValueError(f"At least two structures are required to compute an IRC tangent. Got {len(struct_coordinates)}.")

    irc_file_xyz_path = Path(irc_file_xyz_file)

    sqrt_atom_masses = np.sqrt(atom_masses)[:, None]

    com_corrected = [_com_correction(geometry, atom_masses) for geometry in struct_coordinates]

    rot_corrected_next = [_rot_correction(com_corrected[i], com_corrected[i + 1], atom_masses, irc_file_xyz_path)
                          for i in range(len(com_corrected) - 1)]
    rot_corrected_prev = [_rot_correction(com_corrected[i], com_corrected[i - 1], atom_masses, irc_file_xyz_path)
                          for i in range(1, len(com_corrected))]

    irc_steps = [_struct_dist(com_corrected[i], rot_corrected_next[i], atom_masses) for i in range(len(com_corrected) - 1)]
    # IRC step: amu ** 0.5 * bohr

    for i, step in enumerate(irc_steps):
        if np.isclose(step, 0.0):
            raise ValueError(f"Zero IRC step encountered at IRC point {i}. Please check '{irc_file_xyz_path}'.")

    # --- Computing the tangent data ---
    tangent_data = []
    
    for i in range(len(com_corrected)):

        if i == 0:
            tangent = rot_corrected_next[i] - com_corrected[i]

        elif i == len(com_corrected) - 1:
            tangent = com_corrected[i] - rot_corrected_prev[i - 1] 

        else: # 0 < i < len(com_corrected) - 1, an internal point

            # --- Computing delta_IRC- and delta_IRC+ ---
            h_prev = irc_steps[i - 1]
            h_next = irc_steps[i]

            # --- Retrieving the rotationally aligned neighboring geometries ---
            prev_aligned = rot_corrected_prev[i - 1]
            next_aligned = rot_corrected_next[i]
            curr_aligned = com_corrected[i]

            # --- Computing the tangent using the three-point non-uniform-grid formula ---
            func_der_prev = -h_next / (h_prev * (h_prev + h_next)) * prev_aligned
            func_der_curr = (h_next - h_prev) / (h_prev * h_next) * curr_aligned
            func_der_next = h_prev / (h_next * (h_prev + h_next)) * next_aligned
            tangent = (func_der_prev + func_der_curr + func_der_next)
            # Three-point first-derivative formula for a non-uniform grid,
            # obtained by differentiating a quadratic Lagrange interpolant.
            # See, e.g., Fornberg, Math. Comput., 51(184), 699-706 (1988).

        # --- Going to the mass-weighted space ---
        tangent *= sqrt_atom_masses
        # Tangent before normalization: Angstrom * amu ** 0.5 (edge points) | dimensionless (inner points)

        # --- Normalizing the data since we are interested only in the direction of the vector but not its lengths ---
        tangent_norm = np.linalg.norm(tangent)
        if np.isclose(tangent_norm, 0.0):
            raise ValueError(f"Zero tangent encountered at IRC point {i}. " 
                             f"Please check the geometries in '{irc_file_xyz_path}'.")
        tangent /= tangent_norm
        # Tangent after normalization: dimensionless

        # --- Flattening the [Nx3] matrix into a [3Nx1] vector for convenience ---
        tangent = tangent.ravel()

        # --- Storing the data ---
        tangent_data.append(tangent)

    return tangent_data


# --- Mass-scaling Hessian values ---
def _hessian_mass_weighting(hessian: np.ndarray, atom_masses: np.ndarray) -> np.ndarray:
    expected_size = 3 * atom_masses.size
    if hessian.shape != (expected_size, expected_size):
        raise ValueError(f"Hessian shape {hessian.shape} is incompatible with "
                         f"{atom_masses.size} atomic masses. Expected ({expected_size}, {expected_size}).")
    atom_masses_coordinates = np.repeat(atom_masses, 3)
    hessian_mass_weighted = (hessian / np.sqrt(atom_masses_coordinates[:, None] * atom_masses_coordinates[None,:]))
    return hessian_mass_weighted


# --- Defining the function to build a subspace of translations and rotations ---
def _build_transl_rot_space(coordinates: np.ndarray, atom_masses: np.ndarray) -> np.ndarray:

    # --- Correcting the center of mass position (it also validates the input parameters) ---
    com_coordinates = _com_correction(coordinates, atom_masses)
    # it also checks that coordinates.shape[0] == atom_masses.size

    # --- Square roots of atomic masses ---
    sqrt_atom_masses = np.sqrt(atom_masses)

    # --- Allocate translation and rotation matrix ---
    n_atoms = coordinates.shape[0]
    TR = np.zeros((3 * n_atoms, 6), dtype=float)

    # --- Translation vectors ---
    for i, sqrt_atom_mass in enumerate(sqrt_atom_masses):
        j = 3 * i
        TR[j:j + 3, 0:3] = np.eye(3) * sqrt_atom_mass

    # --- Rotation vectors ---
    rotation_axes = np.eye(3)

    for i, (coord, sqrt_mass) in enumerate(zip(com_coordinates, sqrt_atom_masses)):
        j = 3 * i
        for axis in range(3):
            TR[j:j + 3, 3 + axis] = (sqrt_mass * np.cross(rotation_axes[axis], coord))

    return TR


# --- Defining the function to project out translations, rotations, and, optionally, an IRC coordinate.
# It returns (IRC-projected) ZPVE, vibrational wavenumbers, and mode vectors ---
def _proj_modes_out(hessian_mass_weighted: np.ndarray, transl_rot_matrix: np.ndarray,
    irc_tangent: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:

    # --- Validation of the input parameters ---
    if (hessian_mass_weighted.ndim != 2 or hessian_mass_weighted.shape[0] != hessian_mass_weighted.shape[1]):
        raise ValueError(f"Mass-weighted Hessian must be a square matrix. Got {hessian_mass_weighted.shape}.")

    if hessian_mass_weighted.shape[0] == 0:
        raise ValueError(f"Mass-weighted Hessian must not be empty. Got {hessian_mass_weighted.shape}.")

    n_coordinates = hessian_mass_weighted.shape[0]

    if (transl_rot_matrix.ndim != 2 or transl_rot_matrix.shape[0] != n_coordinates):
        raise ValueError(f"Translation and rotation matrix must have shape ({n_coordinates}, n), got {transl_rot_matrix.shape}.")

    if irc_tangent is not None:
        if (irc_tangent.ndim != 1 or irc_tangent.shape[0] != n_coordinates):
            raise ValueError(f"IRC tangent must have shape ({n_coordinates},), got {irc_tangent.shape}.")

    # --- Adding (or not) the IRC tangent ---
    if irc_tangent is not None:
        project_out_matrix = np.column_stack((transl_rot_matrix, irc_tangent))
    else:
        project_out_matrix = transl_rot_matrix

    # --- Constructing an orthonormal basis (basis of the vibrational subspace)
    # for the null space of project_out_matrix using SVD ---
    Q_vib = null_space(project_out_matrix.T)

    # --- Projecting Hessian, reducing the number of dimensions in the matrix ---
    hessian_vib = Q_vib.T @ hessian_mass_weighted @ Q_vib

    # --- Extracting the eigenvalues and eigenvectors ---
    eigenvalues_vib, eigenvectors_vib = np.linalg.eigh(hessian_vib)

    # --- Going back to full 3N space for eigenvectors ---
    eigenvectors_full = Q_vib @ eigenvectors_vib
    eigenvectors_modes = eigenvectors_full.T

    # --- Extracting the eigenvalues and eigenvectors ---
    wavenums = (np.sign(eigenvalues_vib) * CONVERSION_FACTOR_HESSIAN_VALUES * np.sqrt(np.abs(eigenvalues_vib)))

    # --- Computing ZPVE ---
    positive_sum_zpve = float(0.5 * sum(wavenum for wavenum in wavenums if wavenum > 0) * CM_M1_TO_HARTREE)

    return positive_sum_zpve, wavenums, eigenvectors_modes


# --- Defining the function to compute the IRC steps ---
def irc_steps_comp(irc_file_xyz_file: str | Path, struct_coordinates: np.ndarray, atom_masses) -> list[float]:

    # --- Performing the center of mass (CoM) correction ---
    irc_file_xyz_path = Path(irc_file_xyz_file)
    if len(struct_coordinates) < 2:
        raise ValueError(f"At least two structures are required to compute an IRC tangent. Got {len(struct_coordinates)}.")
    
    com_corrected = [_com_correction(geometry, atom_masses) for geometry in struct_coordinates]

    # --- Performing the rotation correction ---
    rot_corrected = [_rot_correction(com_corrected[i], com_corrected[i + 1], atom_masses, irc_file_xyz_path)
                     for i in range(len(com_corrected) - 1)]

    # --- Computing the IRC steps ---
    irc_steps = [_struct_dist(com_corrected[i], rot_corrected[i], atom_masses) for i in range(len(com_corrected) - 1)]

    return irc_steps


# --- Defining the function to compute the projected frequencies ---
def proj_freq_extraction_tunnex(irc_file_xyz_file: str | Path, struct_coordinates: np.ndarray, atom_masses: np.ndarray,
    hess_val_matrices: np.ndarray) -> tuple[tuple[float, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:

    # --- Checking the data ---
    irc_xyz_path = Path(irc_file_xyz_file)
    if len(struct_coordinates) != len(hess_val_matrices):
        raise ValueError(f"Number of geometries and Hessians must match. "
                         f"Got {len(struct_coordinates)} and {len(hess_val_matrices)}.")
    
    # --- Finding the IRC tangents ---
    irc_tangents = _find_tangent(struct_coordinates, atom_masses, irc_xyz_path)
    if len(struct_coordinates) != len(irc_tangents):
        raise ValueError(f"Number of geometries and IRC tangents must match. "
                         f"Got {len(struct_coordinates)} and {len(irc_tangents)}.")

    # --- Mass-scaling Hessian values ---
    hess_mw_val_matrices = [_hessian_mass_weighting(hess_value_matrix, atom_masses)
                            for hess_value_matrix in hess_val_matrices]

    # --- Building a subspace of translations and rotations ---
    transl_rot_spaces = [_build_transl_rot_space(coordinates, atom_masses)
                         for coordinates in struct_coordinates]

    # --- Projecting out translations, rotations, and an IRC coordinate ---
    results = [_proj_modes_out(hess, space, tangent)
               for hess, space, tangent in zip(hess_mw_val_matrices, transl_rot_spaces, irc_tangents)]

    # --- Computing ZPVE values and returning ZPVEs, vibrations, and vibrational mode vectors ---
    zpves, vibs, modes = zip(*results)

    return zpves, vibs, modes


# --- Defining the function to compute the projected frequencies ---
def freq_and_modes_extraction_tunnex(struct_coordinates: np.ndarray, atom_masses: np.ndarray,
    hess_val_matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:

    # --- Mass-scaling Hessian values ---
    hess_mw_val_matrix = _hessian_mass_weighting(hess_val_matrix, atom_masses)

    # --- Building a subspace of translations and rotations ---
    transl_rot_space = _build_transl_rot_space(struct_coordinates, atom_masses)

    # --- Projecting out translations and rotations ---
    zpve, vibs, modes = _proj_modes_out(hess_mw_val_matrix, transl_rot_space, None)

    return zpve, vibs, modes


# --- Defining the function to transform the frequencies and vectors to the Carthesian space ---
def vibrations_format_transform(hess_file_out: Path, frequencies: np.ndarray, modes: np.ndarray, atom_masses: np.ndarray,
    freq_tol: float = 1e-2, zero_tol: float = 1e-12) -> list[list[float]]:

    # --- Going to the Carthesian space ---
    modes_cartesian = modes / np.sqrt(np.repeat(atom_masses, 3))[None,:]
    norms = np.linalg.norm(modes_cartesian, axis=1, keepdims=True)

    if np.any(norms <= zero_tol):
        bad_modes = np.flatnonzero(norms <= zero_tol)
        raise ValueError(f"Zero-norm modes found: {bad_modes.tolist()}. Please check '{hess_file_out}'.")

    mode_vectors = modes_cartesian / norms
    
    # --- Collecting the results ---
    freq_and_modes = [[freq, *mode] for freq, mode in zip(frequencies, mode_vectors, strict=True) if abs(freq) >= freq_tol]

    return freq_and_modes