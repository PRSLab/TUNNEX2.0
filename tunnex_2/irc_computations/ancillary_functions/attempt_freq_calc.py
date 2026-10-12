# --- Modules ---
import math  # To be sure that in some cases we are working only with numbers, not arrays
import numpy as np
from pathlib import Path

from tunnex_2.constants_and_dataclasses.constants_and_settings import CM_M1_TO_HARTREE # type: ignore
from tunnex_2.constants_and_dataclasses.dataclasses import IRCData, EnergyPointsData # type: ignore

from tunnex_2.irc_computations.ancillary_functions.interface import setup_logger, file_check # type: ignore

from tunnex_2.irc_computations.gaussian.gaussian_file_parsing import gauss_mode_coordinate_reader # type: ignore

from tunnex_2.irc_computations.orca.orca_file_parsing import orca_mode_coordinate_reader # type: ignore


# --- Defining the logging function ---
logger = setup_logger()


# --- Scaling the coordinates of the vibrational mode vectors ---
def _vib_modes_mass_scaling(modes: list[list[float]], masses: tuple[float, ...]) -> tuple[float, list[tuple[float, float]]]:
    
    modes_scaled = []

    if not modes:
        raise ValueError(f"Expected at least one mode. Got {len(modes)}")

    for row in modes:
        coordinates = row[1:]
        if (not coordinates or len(coordinates) != 3 * len(masses)):
            raise ValueError(f"Expected {3 * len(masses)} coordinates for {len(masses)} atoms, "
                             f"got {len(coordinates)}")
        scaled_coordinates = [coordinate * math.sqrt(masses[i // 3]) for i, coordinate in enumerate(coordinates)]
        modes_scaled_row = [row[0]] + scaled_coordinates
        modes_scaled.append(modes_scaled_row)

    return modes_scaled


# --- Defining the function to determine the attemt frequency ---
def attempt_freq_calc(ts_file_out: str | Path, minimum_file_out: str | Path, species: str, prog_key: str = 'gaussian',
    hess_parsing: bool = False, sum_tol: float = 1e-2,
    tol: float = 1e-12) -> tuple[float, list[list[float]], tuple[float, float]]:

    if prog_key not in ('gaussian', 'orca'):
        raise ValueError(f"Unsupported program key ({prog_key!r}). Expected {('gaussian', 'orca')}.")

    if species not in ('reactant', 'product'):
        raise ValueError(f"Unsupported species type ({species!r}). Expected {('reactant', 'product')}.")

    ts_output_file_path = file_check(ts_file_out)
    minimum_output_file_path = file_check(minimum_file_out)

    # --- Reading the negative transiton state frequency (reaction coordinate) and normalizing it ---
    if prog_key == 'gaussian':
        ts_modes, ts_masses = gauss_mode_coordinate_reader(ts_output_file_path, hess_parsing)
    else:
        ts_modes, ts_masses = orca_mode_coordinate_reader(ts_output_file_path, hess_parsing)

    ts_modes_scaled = _vib_modes_mass_scaling(ts_modes, ts_masses)
    ts_mode_scaled = ts_modes_scaled[0]
    ts_mode_norm = math.sqrt(sum(x ** 2 for x in ts_mode_scaled[1:]))
    if ts_mode_norm <= tol:
        raise ValueError(f"The frequency normalization factor is smaller than {tol}. Please check '{ts_output_file_path}'.")

    correlations = []

    # --- Reading the frequencies and normal mode shift vectors at the minimum ---
    if prog_key == 'gaussian':
        minimum_modes, minimum_masses = gauss_mode_coordinate_reader(minimum_output_file_path, hess_parsing)
    else:
        minimum_modes, minimum_masses = orca_mode_coordinate_reader(minimum_output_file_path, hess_parsing)
    
    minimum_modes_scaled = _vib_modes_mass_scaling(minimum_modes, minimum_masses)

    for minimum_mode_scaled in minimum_modes_scaled:

        freq = minimum_mode_scaled[0]

        # --- Normalizing the mode shift-vectors at the minimum ---
        minimum_mode_norm = math.sqrt(sum(x ** 2 for x in minimum_mode_scaled[1:]))
        if minimum_mode_norm <= tol:
            raise ValueError(f"The frequency normalization factor is smaller than {tol}. Please check '{minimum_output_file_path}'.")
        
        # --- Computing the scalar products of minimum normal mode shift-vectors and one mode of the transition state ---
        if len(ts_mode_scaled[1:]) != len(minimum_mode_scaled[1:]):
            raise ValueError(f"Transition state and minimum normal modes have different numbers of coordinates. "
                             f"Please check '{ts_output_file_path}' and '{minimum_output_file_path}'.")
        
        cosine_vectors = sum(x * y for x, y in zip(ts_mode_scaled[1:], minimum_mode_scaled[1:])) / (ts_mode_norm * minimum_mode_norm)
        correlation = cosine_vectors ** 2
        correlations.append((freq, correlation))

    if not correlations:
        raise ValueError(f"No valid normal modes were found. Please check {minimum_output_file_path}.")

    # --- Picking the frequency ---
    # Computing the correlation-weighted average attempt frequency and the list of correlations
    att_freq_aver = math.sqrt(sum(freq ** 2 * corr for freq, corr in correlations))
    sum_corrs = sum(corr for _, corr in correlations)

    # Picking the attemp frequency with the maximum correlation
    att_freq_max_corr = max(correlations, key=lambda pair: pair[1])

    if abs(1.0 - sum_corrs) >= sum_tol:
        logger.warning("Expected sum of correlation factors of 1.000. Got %.3f. Please check your data.", sum_corrs)
    # Weighting the freq ** 2 since we are working in the space of Hessian eigenvalues, which are proportional to freq ** 2.
    # Owing to this reason, correlations are also defined as cosine_vectors ** 2. Sum of correlation weights should be close to 1.0.

    logger.info("For %s the attempt frequency is %.2f cm-1. Correlation weights sum = %.3f. " \
    "Please check the frequency.", species, att_freq_aver, sum_corrs)

    return att_freq_aver, correlations, att_freq_max_corr


# ===========================================================================
# --- Comment to attempt_freq_calc ---
# ===========================================================================
# What I did not test here, but separately (the data were not stored) is the effect of hess_parsing
# on the attemp frequency. For Gaussian, there are several sources of errors if we either parse the frequencies and vectors
# from the transition state, reactant, and product files or compute these data via parsing the Hessians, atom masses, 
# and structures.
# 1. Gaussian did not project out translations and rotations completely, but diagonalize
# the Hessian in such a manner to make the corresponding frequencies as low as possible (typically < 10 cm-1).
# For TUNNEX 2.0, the corresponding values are strictly 0.00 cm-1, which slightly affect both
# vibrational vector components and (to a lesser extent) vibrational frequencies.
# 2. Gaussian round up the vibrational vector components with the 0.01 precision before printing while
# vectors computed in TUNNEX 2.0 are not rounded for the attemp frequency computations.
# It may result in a significant difference for the correlation-weighted average attempt frequency for Gaussian,
# while attempt frequency with max correlation should be very similar for these two methods.
# ORCA has neither of these two problems (it projects out translations and vibrations completely and does not
# round the vector components) and for this program hess_parsing flag barely makes any difference (the frequencies 
# typically differ by < 1 cm-1 because physical constant precision is TUNNEX 2.0 and ORCA seems to be a bit different).
# However, ORCA rounds up the atom mass up to 0.001 (while in fact it uses a more accurate value) which may result
# in some insignificant divergences between the control result and TUNNEX 2.0 output for an attemt frequency, correlations,
# and IRC steps. Gaussian prints the atomic mass with a better precision, but the two problems mentioned above may result
# in some divergences for the isotopic-substituted samples between hess_parsing=False and hess_parsing=True.


# --- Defining the function to write the correlation of reactant (product) frequencies with the IRC ---
def write_freq_corr(filename: str | Path, att_freq_aver: float,
    attempt_freq_species_corr: list[tuple[float, float]], att_freq_max_corr: tuple[float, float]) -> None:

    filename_path = file_check(filename)
    corr_file = filename_path.with_stem(filename_path.stem + "_corr").with_suffix(".txt")
    att_freq_max, corr_max = att_freq_max_corr

    with open(corr_file, "w", encoding="utf-8") as f_out:
        f_out.write(f"{'Frequency, cm-1':>15} | {'Correlation with the IRC':>25}\n")
        f_out.write("-" * 15 + "-+-" + "-" * 25 + "\n")
        for frequency, correlation in attempt_freq_species_corr:
            f_out.write(f"{frequency:15.2f} | {correlation:25.5f}\n")
        f_out.write(f"\nThe correlation-weighted average attempt frequency is {att_freq_aver:.2f} cm-1\n")
        f_out.write(f"\nThe attempt frequency with max correlation ({corr_max:.5f}) is {att_freq_max:.2f} cm-1\n")


# --- Defining the function to extract the energies to compute the number of vibrational levels ---
def _extract_energies_to_compute_number_of_vib_levels(irc_data: IRCData,
    energy_points: EnergyPointsData, direction: str = 'reactant') -> tuple[float, float]:
    
    allowed_directions = ("reactant", "product")
    if direction not in allowed_directions:
        raise ValueError(f"Expected {allowed_directions} as direction. Got '{direction}'.")

    if irc_data.electronic_energies is None:
        raise ValueError(f"Expected an array of electronic energies. Got '{irc_data.electronic_energies}'.")
    
    _, ts_el_energy = max(irc_data.electronic_energies, key=lambda point: point[1])
    
    if irc_data.zpve_energies_forward is not None:
        _, ts_zpve = energy_points.transition_state_zpve
    else:
        ts_zpve = 0.0

    ts_total_energy = ts_el_energy + ts_zpve

    if direction == 'reactant':
        _, react_energy = min(irc_data.electronic_energies, key=lambda point: point[0])
        if irc_data.zpve_energies_reverse is not None:
            _, react_zpve = min(irc_data.zpve_energies_reverse, key=lambda point: point[0])
        else:
            react_zpve = 0.0

        point_total_energy = react_energy + react_zpve
        
    else:
        _, prod_energy = max(irc_data.electronic_energies, key=lambda point: point[0])
        if irc_data.zpve_energies_forward is not None:
            _, prod_zpve = max(irc_data.zpve_energies_forward, key=lambda point: point[0])
        else:
            prod_zpve = 0.0

        point_total_energy = prod_energy + prod_zpve

    return ts_total_energy, point_total_energy


# --- Defining the function to compute the number of vibrational levels for temperature averaging ---
def compute_number_of_vib_levels(frequency: float, irc_data: IRCData, energy_points: EnergyPointsData,
    potential_scaling_factor: float = 1.0, direction: str = 'reactant') -> int:

    if frequency <= 0:
        raise ValueError(f"Frequency must be positive. Got {frequency:.2f} cm-1.")
    
    ts_energy, point_energy = _extract_energies_to_compute_number_of_vib_levels(irc_data, energy_points, direction)

    # --- Because the number of gaps is N, the number of level should be N + 1.
    # But the first level is started from 0.5 * freq, so, N + 1 - 0.5 = N + 0.5 ---
    number_of_levels = math.ceil(potential_scaling_factor * (ts_energy - point_energy) / (frequency * CM_M1_TO_HARTREE) + 0.5)

    return max(0, number_of_levels - 1) # a) This value cannot be negative. b) Since we adding 1 after, here we subtract 1.


# --- Defining the function to check electronic energies or ZPVE data for outliers ---
# Radius specifies how many neighboring datapoints are considered on each side (default: 5).
# The default absolute modified Z-score threshold of 3.5 is based on the recommendation by:
# Iglewicz, B. and Hoaglin, D. C., "How to Detect and Handle Outliers", Quality Press, 1993, p. 12.
def detect_outliers_local(irc: np.ndarray, energy: np.ndarray, mode: str, radius: int = 5, threshold: float = 3.5) -> None:

    # --- Checking the input data ---
    if irc.size == 0:
        raise ValueError(f"No {mode} data points provided. Please check your data.")
    
    if len(irc) != len(energy):
        raise ValueError(f"For {mode} data points 'irc' and 'energy' must have the same length. "
                            f"Got {len(irc)} and {len(energy)}.")

    if radius < 1:
        raise ValueError(f"The radius must be at least 1. Got {radius}.")

    if threshold <= 0:
        raise ValueError(f"The threshold must be positive. Got {threshold}.")
    
    # --- Ordering the input data and checking the data for duplicates ---
    order = np.argsort(irc)
    x_s, y_s = irc[order], energy[order]

    if len(np.unique(x_s)) < len(x_s):
        raise ValueError(f"Duplicate IRC values found in {mode} data. Please check your data.") 

    # --- Building the outlier mask ---
    n = len(x_s)
    outlier_mask = np.zeros(n, dtype=bool)
    skipped = []

    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        neighbors_y = np.concatenate([y_s[lo:i], y_s[i+1:hi]])

        # --- Skipping points with too few neighbors to reliably estimate the MAD (median absolute deviation).
        # At least 3 neighboring points are required ---
        if len(neighbors_y) < 3:
            skipped.append(x_s[i])
            continue

        med = np.median(neighbors_y)
        mad = np.median(np.abs(neighbors_y - med))

        # --- Handling zero MAD. In this case, any value that differs from the local median is a potential outlier ---
        if np.isclose(mad, 0.0):
            outlier_mask[i] = not np.isclose(y_s[i], med)
            continue

        # --- Flagging points whose modified Z-score exceeds the threshold.
        # (0.6745: consistency factor mapping MAD to sigma for normally distributed data) ---
        z = 0.6745 * np.abs(y_s[i] - med) / mad
        outlier_mask[i] = z > threshold

    # --- Logging the outlier warnings ---
    if skipped:
        logger.warning("Insufficient neighboring %s data points to estimate MAD for %d point(s) "
            "(IRC values: %s). Please check your data for outliers manually.",
            mode, len(skipped), ", ".join(f"{val:.3f}" for val in skipped))
   
    for idx in np.flatnonzero(outlier_mask):
        logger.warning(
            "A potential %s outlier was found at IRC=%.3f. "
            "Outlier value = %.3g. Please check your data.", mode, x_s[idx], y_s[idx])