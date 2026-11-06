# --- Modules ---
from pathlib import Path

from tunnex_2.constants_and_dataclasses.constants_and_settings import ( # type: ignore
    COMMAND_FILE_PATH,
    RUN_SOFTWARE_KEY_MARKER,
    CORR_ANALYSIS_BOOL,
    POTENTIAL_SCAL_FACTOR_DEFAULT,
    PICK_MAX_FREQ_FLAG)
from tunnex_2.constants_and_dataclasses.dataclasses import FindIRCConfig # type: ignore

from tunnex_2.irc_computations.ancillary_functions.interface import file_check # type: ignore
from tunnex_2.irc_computations.pipeline.irc_computations_pipeline import ( # type: ignore
    ts_optimization,
    irc_computations,
    irc_file_parsing,
    react_prod_optimization,
    opt_file_parsing,
    attempt_freq_func,
    number_of_levels_func,
    potential_scal_factor_comp,
    tunnex_input_writing,
    eckart_react_prod_optimization,
    eckart_potential_results)


# --- Defining the function to compute the IRC data using Gaussian or ORCA (or Eckart potential) ---
# --- For Eckart potential formulas please check:
# Johnston, H. S., & Heicklen, J. (1962). J. Phys. Chem., 66(3), 532-533 ---
def irc_computations_main(configs: FindIRCConfig) -> tuple[Path, Path]:

    # --- Validation of the parameters ---
    allowed_prog_modes = ('orca', 'gaussian')
    if configs.prog_mode not in allowed_prog_modes:
        raise ValueError(f"Expected {allowed_prog_modes}. Got {configs.prog_mode}.")

    if configs.ts_input_file is None:
        raise FileNotFoundError(f"TS input file is required for the computations. Got {configs.ts_input_file}.")
    
    if configs.eckart:
        if configs.react_input_file is None:
            raise FileNotFoundError(f"Reactant input file is required for Eckart potential computations. "
                             f"Got {configs.react_input_file}.")

        if configs.prod_input_file is None:
            raise FileNotFoundError(f"Product input file is required for Eckart potential computations. "
                             f"Got {configs.prod_input_file}.")

    # --- Optimization of the transition state geometry ---
    if configs.ts_computed_file is None:
        ts_output_file = ts_optimization(configs.ts_input_file, COMMAND_FILE_PATH, configs.prog_mode, RUN_SOFTWARE_KEY_MARKER)
    else:
        ts_output_file = file_check(configs.ts_computed_file)

    # --- Computation of the IRC pathway in both directions ---
    if not configs.eckart:
        if configs.irc_computed_file is None:
            irc_output_file = irc_computations(configs.ts_input_file, ts_output_file, COMMAND_FILE_PATH,
            configs.prog_mode, RUN_SOFTWARE_KEY_MARKER, configs.proj_freq, configs.calc_all)
        else:
            irc_output_file = file_check(configs.irc_computed_file)
        
        # --- IRC output file parsing and analyzing the heat effect of the reaction ---
        irc_data = irc_file_parsing(irc_output_file, ts_output_file, configs.ts_input_file, COMMAND_FILE_PATH,
                configs.prog_mode, configs.proj_freq, configs.hess_parsing_flag, RUN_SOFTWARE_KEY_MARKER)

    # --- Optimization of the reactant and product geometries ---
    if configs.react_computed_file is None or configs.prod_computed_file is None:
        if not configs.eckart:
            react_output_file, prod_output_file = react_prod_optimization(configs.ts_input_file, irc_output_file,
            COMMAND_FILE_PATH, configs.prog_mode, RUN_SOFTWARE_KEY_MARKER)
        else:
            react_output_file, prod_output_file = eckart_react_prod_optimization(configs.react_input_file,
            configs.prod_input_file, COMMAND_FILE_PATH, configs.prog_mode, RUN_SOFTWARE_KEY_MARKER)
    else:
        react_output_file = file_check(configs.react_computed_file)
        prod_output_file = file_check(configs.prod_computed_file)

    
    # --- Computation of the attempt frequencies ---
    attempt_freq = attempt_freq_func(ts_output_file, react_output_file, prod_output_file,
    CORR_ANALYSIS_BOOL, configs.prog_mode, configs.hess_parsing_flag, PICK_MAX_FREQ_FLAG)

    # --- Output file parsing for the optimized transition state, reactant, and product ---
    energy_points = opt_file_parsing(ts_output_file, react_output_file, prod_output_file, attempt_freq,
        configs.prog_mode, configs.hess_parsing_flag)
    
    # --- IRC path computation and analyzing the heat effect of the reaction (Eckart potential) ---
    if configs.eckart:
        irc_data = eckart_potential_results(ts_output_file, energy_points, configs.prog_mode, configs.hess_parsing_flag)

    # --- Determining the potential scaling factor (available for a hybrid mode, default value is 1.0) ---
    if configs.hybrid_mode and not configs.eckart:
        potential_scal_factor_react, potential_scal_factor_prod = potential_scal_factor_comp(irc_data, energy_points)
    else: # The default value is the only available option for Eckart potential
        potential_scal_factor_react = POTENTIAL_SCAL_FACTOR_DEFAULT
        potential_scal_factor_prod = POTENTIAL_SCAL_FACTOR_DEFAULT

    # --- Computation of the number of vibrational levels for the temperature averaging ---
    num_levels_react, num_levels_prod = number_of_levels_func(irc_data,
    energy_points, attempt_freq.att_freq_react, attempt_freq.att_freq_prod,
    potential_scal_factor_react, potential_scal_factor_prod)
    
    # --- Writing the input files for the TUNNEX 2.0 computations ---
    tunnex_forward, tunnex_backward = tunnex_input_writing(configs.ts_input_file, irc_data, energy_points,
    attempt_freq, num_levels_react, num_levels_prod, potential_scal_factor_react, potential_scal_factor_prod)
    
    return tunnex_forward, tunnex_backward