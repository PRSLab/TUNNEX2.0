# --- Modules ---
import math # To be sure that in some cases we are working only with numbers, not arrays
import numpy as np
from pathlib import Path

from tunnex_2.constants_and_dataclasses.constants_and_settings import ( # type: ignore
    KJ_MOL_M1_TO_HARTREE,
    CM_M1_TO_HARTREE,
    SLURM_MODE_KEY,
    ENVR)
from tunnex_2.constants_and_dataclasses.dataclasses import ( # type: ignore
    IRCData,
    EnergyPointsData,
    AttemptFreq,
    TunnexInputSettings)

from tunnex_2.irc_computations.ancillary_functions.attempt_freq_calc import ( # type: ignore
    attempt_freq_calc,
    write_freq_corr,
    compute_number_of_vib_levels,
    detect_outliers_local)
from tunnex_2.irc_computations.ancillary_functions.eckart_potential_functions import ( # type: ignore
    eckart_potential_parameters,
    eckart_potential_data)
from tunnex_2.irc_computations.ancillary_functions.interface import( # type: ignore
    run_software,
    gauss_error_check,
    gauss_out_filename,
    orca_error_check,
    orca_out_filename,
    file_check,
    setup_logger)
from tunnex_2.irc_computations.ancillary_functions.tunnex_input import ( # type: ignore
    write_input_head,
    writing_el_energy,
    writing_zpve_energy)

from tunnex_2.irc_computations.gaussian.gaussian_file_parsing import ( # type: ignore
    reading_gauss_irc,
    reading_gauss_irc_tunnex,
    reading_gauss_struct_file,
    gauss_mode_coordinate_reader)
from tunnex_2.irc_computations.gaussian.gaussian_file_writing import ( # type: ignore
    create_gauss_input,
    create_gauss_irc_input,
    create_gauss_react_prod_input_from_irc)
from tunnex_2.irc_computations.gaussian import gaussian_patterns # type: ignore

from tunnex_2.irc_computations.orca.orca_file_parsing import ( # type: ignore
    reading_orca_irc_tunnex,
    reading_orca_struct_file,
    orca_mode_coordinate_reader)
from tunnex_2.irc_computations.orca.orca_file_writing import ( # type: ignore
    create_orca_input,
    create_orca_irc_input,
    create_orca_react_prod_input_from_irc)
from tunnex_2.irc_computations.orca import orca_patterns # type: ignore


# --- Defining the logging function ---
logger = setup_logger()


# --- Defining the function for the optimization of the transition state geometry ---
def ts_optimization(ts_guess_input: str | Path, command_file: str | Path, mode: str,
    run_software_key: bool = True, filename_out_key: bool=False) -> Path:

    allowed_modes = ('gaussian', 'orca')
    if mode not in allowed_modes:
        raise ValueError(f"Expected {allowed_modes} as program mode. Got '{mode}'.")
    
    ts_guess_input_path = Path(ts_guess_input)
    command_path = Path(command_file)
    
    if mode == 'gaussian':
        ts_input_file = create_gauss_input(ts_guess_input_path, None, gaussian_patterns.METHOD_TAIL_TS, "_ts")
    else:
        ts_input_file = create_orca_input(ts_guess_input_path, None, orca_patterns.BLOCK_TAIL_TS, "_ts")

    if run_software_key:
        run_software(ts_input_file, command_path, filename_out_key, SLURM_MODE_KEY, ENVR)

    if mode == 'gaussian':
        ts_output_file = gauss_out_filename(ts_input_file)
        gauss_error_check(ts_output_file)
    else:
        ts_output_file = orca_out_filename(ts_input_file, ".out")
        orca_error_check(ts_output_file)

    return ts_output_file


# --- Defining the function for the IRC computations ---
def irc_computations(ts_guess_input: str | Path, ts_output_file: Path, command_file: str | Path, mode: str,
    run_software_key: bool = True, proj_freq: bool=True,
    calc_all: bool=True, filename_out_key: bool=False) -> Path:

    allowed_modes = ('gaussian', 'orca')
    if mode not in allowed_modes:
        raise ValueError(f"Expected {allowed_modes} as program mode. Got '{mode}'.")
    
    ts_guess_input_path = Path(ts_guess_input)
    ts_output_path = Path(ts_output_file)
    command_path = Path(command_file)
    
    if mode == 'gaussian':
        irc_input_file = create_gauss_irc_input(ts_guess_input_path, ts_output_path, proj_freq, calc_all)
    else:
        irc_input_file = create_orca_irc_input(ts_guess_input_path, ts_output_path)

    if run_software_key:
        run_software(irc_input_file, command_path, filename_out_key, SLURM_MODE_KEY, ENVR)

    if mode == 'gaussian':
        irc_output_file = gauss_out_filename(irc_input_file)
        gauss_error_check(irc_output_file)
    else:
        irc_output_file = orca_out_filename(irc_input_file, ".xyz")
        orca_error_check(irc_output_file)

    return irc_output_file


# --- Defining the function for analyzing the heat effect of the reaction and checking for outliers ---
def _heat_check_func(irc_data: IRCData, outlier_check: bool = True) -> None:

    if irc_data.electronic_energies is None:
        raise ValueError(f"Expected an array of (IRC, electronic energy). Got {irc_data.electronic_energies}.")
    
    # --- Checking the data ---
    if outlier_check:
        irc, energy = np.array(irc_data.electronic_energies).T
        detect_outliers_local(irc, energy, 'electronic energy')
        if irc_data.zpve_energies_reverse is not None and irc_data.zpve_energies_forward is not None:
            zpve_data = (irc_data.zpve_energies_reverse + irc_data.zpve_energies_forward)
            irc, zpve = np.array(zpve_data).T
            detect_outliers_local(irc, zpve, 'ZPVE')

    react_energy = min(irc_data.electronic_energies, key=lambda x: x[0])[1]
    prod_energy = max(irc_data.electronic_energies, key=lambda x: x[0])[1]
    electronic_energy_difference = (react_energy - prod_energy) / KJ_MOL_M1_TO_HARTREE
    # The exothermic reaction means that the energy of a product is smaller than the energy of a reactant;
    # the energy difference (react_energy - prod_energy) should be positive for an exothermic reaction
    # considering only electronic energy here, it should be a reasonably good approximation for most of the cases
    
    if math.isclose(electronic_energy_difference, 0.0, abs_tol=1e-1):
        logger.info("Please note that the reaction is thermoneutral (%.1f kJ mol-1).",
        abs(electronic_energy_difference))
    elif electronic_energy_difference > 0:
        logger.info("Please note that the forward reaction is exothermic by %.1f kJ mol-1.",
        abs(electronic_energy_difference))
    else:  # electronic_energy_difference < 0
        logger.info("Please note that the forward reaction is endothermic by %.1f kJ mol-1.",
        abs(electronic_energy_difference))


# --- Defining the function for IRC output file parsing ---
def irc_file_parsing(irc_output_file: str | Path, ts_output_file: Path, ts_guess_input_file: str | Path,
    command_file: str | Path, mode: str, proj_freq: bool = True, hess_parsing: bool = False, run_software_key: bool = True,
    outlier_check: bool = True, filename_out_key: bool = False) -> IRCData:

    allowed_modes = ('gaussian', 'orca')
    if mode not in allowed_modes:
        raise ValueError(f"Expected {allowed_modes} as program mode. Got '{mode}'.")

    irc_output_path = Path(irc_output_file)
    ts_output_path = Path(ts_output_file)
    command_file_path = Path(command_file)

    if mode == 'gaussian':
        if not hess_parsing:
            irc_data = reading_gauss_irc(irc_output_path, proj_freq)
        else:
            irc_data = reading_gauss_irc_tunnex(irc_output_path, ts_output_path, proj_freq)
    else:
        ts_guess_input_path = Path(ts_guess_input_file)
        irc_data = reading_orca_irc_tunnex(irc_output_path, ts_output_path, ts_guess_input_path,
        command_file_path, proj_freq, run_software_key, filename_out_key)

    if irc_data.electronic_energies is None:
        raise ValueError(f"Expected an array of (IRC, electronic energy). Got {irc_data.electronic_energies}.")

    _heat_check_func(irc_data, outlier_check)

    return irc_data


# --- Defining the function for the optimization of the reactant and product geometries ---
def react_prod_optimization(ts_guess_input: str | Path, irc_output_file: str | Path, command_file: str | Path,
    mode: str, run_software_key: bool = True, filename_out_key: bool = False) -> tuple[Path, Path]:

    allowed_modes = ('gaussian', 'orca')
    if mode not in allowed_modes:
        raise ValueError(f"Expected {allowed_modes} as program mode. Got '{mode}'.")

    ts_guess_path = Path(ts_guess_input)
    irc_output_path = Path(irc_output_file)
    command_file_path = Path(command_file)

    if mode == 'gaussian':
        react_input_file, prod_input_file = create_gauss_react_prod_input_from_irc(irc_output_path, ts_guess_path)
    else:
        react_input_file, prod_input_file = create_orca_react_prod_input_from_irc(irc_output_path, ts_guess_path)

    if run_software_key:
        run_software(react_input_file, command_file_path, filename_out_key, SLURM_MODE_KEY, ENVR)
        run_software(prod_input_file, command_file_path, filename_out_key, SLURM_MODE_KEY, ENVR)

    if mode == 'gaussian':
        react_output_file = gauss_out_filename(react_input_file)
        prod_output_file = gauss_out_filename(prod_input_file)
        gauss_error_check(react_output_file)
        gauss_error_check(prod_output_file)
    else:
        react_output_file = orca_out_filename(react_input_file, ".out")
        prod_output_file = orca_out_filename(prod_input_file, ".out")
        orca_error_check(react_output_file)
        orca_error_check(prod_output_file)

    return react_output_file, prod_output_file


# --- Defining the function for output file parsing for the optimized transition state, reactant, and product energies ---
def opt_file_parsing(ts_output_file: str | Path, react_output_file: str | Path, prod_output_file: str | Path,
    attempt_freq: AttemptFreq, mode: str, hess_parsing: bool = False) -> EnergyPointsData:

    # --- Checking the input data ---
    allowed_modes = ('gaussian', 'orca')
    if mode not in allowed_modes:
        raise ValueError(f"Expected {allowed_modes} as program mode. Got '{mode}'.")

    ts_output_path = Path(ts_output_file)
    react_output_path = Path(react_output_file)
    prod_output_path = Path(prod_output_file)

    # --- Reading the corresponding files ---
    if mode == 'gaussian':
        ts_el_energy, ts_zpve_energy = reading_gauss_struct_file(ts_output_path, 'ts', hess_parsing)
        react_el_energy, react_zpve_energy = reading_gauss_struct_file(react_output_path, 'react', hess_parsing)
        prod_el_energy, prod_zpve_energy = reading_gauss_struct_file(prod_output_path, 'prod', hess_parsing)
    else:
        ts_el_energy, ts_zpve_energy = reading_orca_struct_file(ts_output_path, 'ts', hess_parsing)
        react_el_energy, react_zpve_energy = reading_orca_struct_file(react_output_path, 'react', hess_parsing)
        prod_el_energy, prod_zpve_energy = reading_orca_struct_file(prod_output_path, 'prod', hess_parsing)
    
    # --- Removing an attempt frequency from ZPVEs of the reactant and product.
    # It is needed to correct the corresponding ZPVEs, because we are dealing with the projected ZPVEs ---
    react_zpve_energy = (react_zpve_energy[0],
    react_zpve_energy[1] - 0.5 * attempt_freq.att_freq_react * CM_M1_TO_HARTREE)
    prod_zpve_energy = (prod_zpve_energy[0],
    prod_zpve_energy[1] - 0.5 * attempt_freq.att_freq_prod * CM_M1_TO_HARTREE)

    # --- Writing the data ---
    energy_points = EnergyPointsData(
    transition_state_el_energy=ts_el_energy,
    transition_state_zpve=ts_zpve_energy,
    reactant_el_energy=react_el_energy,
    reactant_zpve=react_zpve_energy,
    product_el_energy=prod_el_energy,
    product_zpve=prod_zpve_energy)
    
    return energy_points


# --- Defining the function for the computation of the attempt frequencies
# (attempt_freq_xxx_corr is extra information which could be useful during the attempt frequency analysis) ---
def attempt_freq_func(ts_output_file: str | Path, react_output_file: str | Path, prod_output_file: str | Path,
    corr_analysis: bool, mode: str, hess_parsing: bool = False, pick_freq_max_corr: bool = False) -> AttemptFreq:

    ts_output_path = Path(ts_output_file)
    react_output_path = Path(react_output_file)
    prod_output_path = Path(prod_output_file)
    
    attempt_freq_react_aver, attempt_freq_react_corr, attempt_freq_react_max_corr = attempt_freq_calc(ts_output_path,
    react_output_path, 'reactant', mode, hess_parsing)
    attempt_freq_prod_aver, attempt_freq_product_corr, attempt_freq_prod_max_corr = attempt_freq_calc(ts_output_path,
    prod_output_path, 'product', mode, hess_parsing)

    if corr_analysis:
        write_freq_corr(react_output_path, attempt_freq_react_aver, attempt_freq_react_corr, attempt_freq_react_max_corr)
        write_freq_corr(prod_output_path, attempt_freq_prod_aver, attempt_freq_product_corr, attempt_freq_prod_max_corr)

    # Choosing either the frequency with the highest correlation, or the correlation-weight-averaged value
    # (the latter is default)
    if pick_freq_max_corr:
        attempt_freq_react = attempt_freq_react_max_corr[0]
        attempt_freq_prod = attempt_freq_prod_max_corr[0]
    else:
        attempt_freq_react = attempt_freq_react_aver
        attempt_freq_prod = attempt_freq_prod_aver

    attempt_freq_and_levels_data = AttemptFreq(
    att_freq_react = attempt_freq_react,
    att_freq_react_corr = attempt_freq_react_corr,
    att_freq_prod = attempt_freq_prod,
    att_freq_prod_corr = attempt_freq_product_corr)

    return attempt_freq_and_levels_data


# --- Defining the function for the computation of the number of vibrational levels ---
def number_of_levels_func(irc_data: IRCData, energy_points: EnergyPointsData,
    attempt_freq_react: float, attempt_freq_prod: float, potential_scaling_factor_react: float,
    potential_scaling_factor_prod: float) -> tuple[int, int]:
    number_of_levels_react = compute_number_of_vib_levels(attempt_freq_react, irc_data, energy_points,
    potential_scaling_factor_react, 'reactant')
    number_of_levels_product = compute_number_of_vib_levels(attempt_freq_prod, irc_data, energy_points,
    potential_scaling_factor_prod, 'product')
    return number_of_levels_react, number_of_levels_product


# --- Defining the function for the potential scaling factor computations ---
def potential_scal_factor_comp(irc_data: IRCData, energy_points: EnergyPointsData,
    with_zpve: bool = True) -> tuple[float, float]:

    zpve_factor = 1.0 if with_zpve else 0.0
    
    irc_el_energy_react = min(irc_data.electronic_energies, key=lambda irc: irc[0])[1]
    irc_el_energy_ts = min(irc_data.electronic_energies, key=lambda irc: abs(irc[0]))[1]
    irc_el_energy_prod = max(irc_data.electronic_energies, key=lambda irc: irc[0])[1]

    if irc_data.zpve_energies_reverse is not None:
        irc_zpve_energy_react = zpve_factor * min(irc_data.zpve_energies_reverse, key=lambda x: x[0])[1]
        irc_zpve_energy_ts = zpve_factor * energy_points.transition_state_zpve[1]
        # Neither Gaussian nor ORCA computes this value during the IRC scan
    else:
        irc_zpve_energy_react = 0.0
        irc_zpve_energy_ts = 0.0

    if irc_data.zpve_energies_forward is not None:
        irc_zpve_energy_prod = zpve_factor * max(irc_data.zpve_energies_forward, key=lambda x: x[0])[1]
    else:
        irc_zpve_energy_prod = 0.0

    energy_points_react = (energy_points.transition_state_el_energy[1] + zpve_factor * energy_points.transition_state_zpve[1]
        - energy_points.reactant_el_energy[1] - zpve_factor * energy_points.reactant_zpve[1])
    irc_react = (irc_el_energy_ts + irc_zpve_energy_ts - irc_el_energy_react - irc_zpve_energy_react)
    
    if not math.isclose(irc_react, 0.0, abs_tol=1e-10):
        potential_scal_factor_react = abs(energy_points_react / irc_react)
        # Potential_scal_factor cannot be negative, thus, using abs()
    else:
        potential_scal_factor_react = 1.0 # Not changing the result if the barrier is very flat
        logger.warning("Very small barrier (%.3g Hartree) for a forward reaction. "
        "Setting potential_scal_factor_react to %s.",
        irc_react, potential_scal_factor_react)

    energy_points_prod = (energy_points.transition_state_el_energy[1] + zpve_factor * energy_points.transition_state_zpve[1]
        - energy_points.product_el_energy[1] - zpve_factor * energy_points.product_zpve[1])
    irc_prod = (irc_el_energy_ts + irc_zpve_energy_ts - irc_el_energy_prod - irc_zpve_energy_prod)
    
    if not math.isclose(irc_prod, 0.0, abs_tol=1e-10):
        potential_scal_factor_prod = abs(energy_points_prod / irc_prod)
        # Potential_scal_factor cannot be negative, thus, using abs()
    else:
        potential_scal_factor_prod = 1.0
        # Not changing the result if the barrier is very flat
        logger.warning("Very small barrier (%.3g Hartree) for a backward reaction. "
        "Setting potential_scal_factor_prod to %s.",
        irc_prod, potential_scal_factor_prod)

    return potential_scal_factor_react, potential_scal_factor_prod


# --- Defining the function for writing the input files for the TUNNEX 2.0 computations ---
def tunnex_input_writing(ts_guess_input: str | Path, irc_data: IRCData, energy_points: EnergyPointsData,
    attempt_freq: AttemptFreq, num_lev_react: int, num_lev_prod: int,
    potential_scal_factor_react: float = 1.0, potential_scal_factor_prod: float = 1.0) -> tuple[Path, Path]:

    ts_guess_input_path = file_check(ts_guess_input)

    tunnex_settings_react = TunnexInputSettings(
    freq=attempt_freq.att_freq_react,
    E0=energy_points.reactant_el_energy[1],
    ZPVE0=energy_points.reactant_zpve[1],
    number_of_levels=num_lev_react,
    potential_scaling_factor = potential_scal_factor_react)

    tunnex_settings_prod = TunnexInputSettings(
    freq=attempt_freq.att_freq_prod,
    E0=energy_points.product_el_energy[1],
    ZPVE0=energy_points.product_zpve[1],
    number_of_levels=num_lev_prod,
    potential_scaling_factor = potential_scal_factor_prod)
    # the default temperature averaging mode for tunneling probabilities is 'finite_sum'
    # both for the reactant and product

    tunnex_input_forward_file = write_input_head('_forward', ts_guess_input_path, tunnex_settings_react)
    tunnex_input_backward_file = write_input_head('_backward', ts_guess_input_path, tunnex_settings_prod)

    writing_el_energy(tunnex_input_forward_file, irc_data.electronic_energies, reverse_key=False)
    writing_el_energy(tunnex_input_backward_file, irc_data.electronic_energies, reverse_key=True)

    writing_zpve_energy(tunnex_input_forward_file, irc_data.electronic_energies, irc_data.zpve_energies_forward,
                        irc_data.zpve_energies_reverse, energy_points.transition_state_zpve, reverse_key=False)
    writing_zpve_energy(tunnex_input_backward_file, irc_data.electronic_energies, irc_data.zpve_energies_forward,
                        irc_data.zpve_energies_reverse, energy_points.transition_state_zpve, reverse_key=True)

    return tunnex_input_forward_file, tunnex_input_backward_file


# --- Defining the function for the optimization of the reactant and product geometries ---
def eckart_react_prod_optimization(react_guess_input: str | Path, prod_guess_input: str | Path, command_file: str | Path,
    mode: str, run_software_key: bool = True, filename_out_key: bool = False) -> tuple[Path, Path]:

    allowed_modes = ('gaussian', 'orca')
    if mode not in allowed_modes:
        raise ValueError(f"Expected {allowed_modes} as program mode. Got '{mode}'.")

    react_guess_path = Path(react_guess_input)
    prod_guess_path = Path(prod_guess_input)
    command_file_path = Path(command_file)
    
    if mode == 'gaussian':
        react_input_file = create_gauss_input(react_guess_path, None, gaussian_patterns.METHOD_TAIL_REACT_PROD, "_react")
        prod_input_file = create_gauss_input(prod_guess_path, None, gaussian_patterns.METHOD_TAIL_REACT_PROD, "_prod")
    else:
        react_input_file = create_orca_input(react_guess_path, None, orca_patterns.LINE_TAIL_REACT_PROD, "_react")
        prod_input_file = create_orca_input(prod_guess_path, None, orca_patterns.LINE_TAIL_REACT_PROD, "_prod")

    if run_software_key:
        run_software(react_input_file, command_file_path, filename_out_key, SLURM_MODE_KEY, ENVR)
        run_software(prod_input_file, command_file_path, filename_out_key, SLURM_MODE_KEY, ENVR)

    if mode == 'gaussian':
        react_output_file = gauss_out_filename(react_input_file)
        gauss_error_check(react_output_file)
        prod_output_file = gauss_out_filename(prod_input_file)
        gauss_error_check(prod_output_file)
    else:
        react_output_file = orca_out_filename(react_input_file, ".out")
        orca_error_check(react_output_file)
        prod_output_file = orca_out_filename(prod_input_file, ".out")
        orca_error_check(prod_output_file)

    return react_output_file, prod_output_file


# --- Defining the function to store the datapoints of the Eckart potential function ---
def eckart_potential_results(ts_output_file: str | Path, energy_points: EnergyPointsData, mode: str,
    hess_parsing: bool = False, outlier_check: bool = True) -> IRCData:

    allowed_modes = ('gaussian', 'orca')
    if mode not in allowed_modes:
        raise ValueError(f"Expected {allowed_modes} as program mode. Got '{mode}'.")

    ts_output_path = Path(ts_output_file)
    
    # --- Parsing the transition state frequency (imaginary one) ---
    if mode == 'gaussian':
        freq_and_modes, _ = gauss_mode_coordinate_reader(ts_output_path, hess_parsing)
    else:
        freq_and_modes, _ = orca_mode_coordinate_reader(ts_output_path, hess_parsing)

    ts_freq_cm_m1 = min(freq_and_modes, key=lambda freq: freq[0])[0]

    # --- Computing the barrier heights for Eckart potential ---
    ts_el_energy = energy_points.transition_state_el_energy[1]
    ts_zpve = energy_points.transition_state_zpve[1]

    react_el_energy = energy_points.reactant_el_energy[1]
    react_zpve = energy_points.reactant_zpve[1]

    prod_el_energy = energy_points.product_el_energy[1]
    prod_zpve = energy_points.product_zpve[1]

    dV1_Hartree = (ts_el_energy + ts_zpve) - (react_el_energy + react_zpve)
    dV2_Hartree = (ts_el_energy + ts_zpve) - (prod_el_energy + prod_zpve)

    # --- Computing the intermediate parameters for Eckart potential ---
    A_Hartree, B_Hartree, L_sqrt_amu_bohr = eckart_potential_parameters(ts_freq_cm_m1, dV1_Hartree, dV2_Hartree)

    # --- Computing Eckart potential ---
    irc_data = eckart_potential_data(A_Hartree, B_Hartree, L_sqrt_amu_bohr)

    if irc_data.electronic_energies is None:
        raise ValueError(f"Expected an array of (IRC, electronic energy). Got {irc_data.electronic_energies}.")

    _heat_check_func(irc_data, outlier_check)

    return irc_data