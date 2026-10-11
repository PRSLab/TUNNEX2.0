# --- Modules ---
from pathlib import Path

from tunnex_2.irc_computations.ancillary_functions.interface import file_check # type: ignore

from tunnex_2.irc_computations.gaussian.gaussian_create_input_file import create_gauss_input_file  # type: ignore
from tunnex_2.irc_computations.gaussian.gaussian_file_parsing import (  # type: ignore
    gauss_irc_file_split, gauss_extract_geom_from_opt_file, gauss_extract_geom_from_irc_point)
from tunnex_2.irc_computations.gaussian import gaussian_patterns # type: ignore


# --- Defining the function to write the input file for a transition state, reactant or product ---
def create_gauss_input(start_file: str | Path, species_geometry: str | None, method_tail: str, suffix: str) -> Path:
    
    start_path = file_check(start_file)
    
    allowed_suffixes = ("_ts", "_react", "_prod")
    if suffix not in allowed_suffixes:
        raise ValueError(f"Expected suffix {allowed_suffixes}. Got {suffix}.")
    
    input_file = start_path.with_stem(start_path.stem + suffix)
    create_gauss_input_file(start_path, input_file, species_geometry, method_tail)

    return input_file
    

# --- Defining the function to create the IRC input files ---
def create_gauss_irc_input(gauss_ts_guess_input: str | Path, gauss_optimized_ts_file: str | Path,
    proj_freq: bool=True, calc_all: bool=True, max_points: int=50, max_cycle: int=40, step_size: float=-10)  -> Path:
    
    # --- Creating the filename of a new file ---
    input_file = file_check(gauss_ts_guess_input)
    gauss_optimized_ts_path = file_check(gauss_optimized_ts_file)
    gauss_irc_input = input_file.with_stem(input_file.stem + "_irc")

    # --- Specifies that the force constants be computed at every point ('calcall') or only at the first point ('calcfc').
    # The first one (default) is slower, but more accurate for projected frequencies ---
    calc_type = "calcall" if calc_all else "calcfc"
    METHOD_TAIL_IRC = f"irc=({calc_type},maxpoints={max_points},maxcycle={max_cycle},stepsize={step_size})"

    # --- Choosing whether one should compute the projected frequencies. Computing them by default ---
    if proj_freq:
        METHOD_TAIL_IRC += " iop(1/73=2)"

    # --- Opening the output files and reading the optimized geometry ---
    ts_geometry = gauss_extract_geom_from_opt_file(gauss_optimized_ts_path)
    create_gauss_input_file(input_file, gauss_irc_input, ts_geometry, METHOD_TAIL_IRC)

    return gauss_irc_input


# --- Defining the function to create the file for the geometry optimization of a reactant (product) ---
def create_gauss_react_prod_input_from_irc(irc_file_out: str | Path, gauss_ts_guess_input: str | Path) -> tuple[Path, Path]:

    # --- Creating the filenames of new files ---
    ts_guess_path = file_check(gauss_ts_guess_input)
    gauss_react_input = ts_guess_path.with_stem(ts_guess_path.stem + "_react")
    gauss_prod_input = ts_guess_path.with_stem(ts_guess_path.stem + "_prod")

    # --- Taking the reactant and product geometries ---
    irc_file_path = file_check(irc_file_out)
    forward, reverse = gauss_irc_file_split(irc_file_path, gaussian_patterns.IRC_SPLIT_MARKER_FORWARD)
    
    react_geom = gauss_extract_geom_from_irc_point(irc_file_path, reverse, reactant_key=True)
    prod_geom = gauss_extract_geom_from_irc_point(irc_file_path, forward, reactant_key=False)
        
    # --- Writing the input files for the reactant and product ---
    create_gauss_input_file(ts_guess_path, gauss_react_input, react_geom, gaussian_patterns.METHOD_TAIL_REACT_PROD)
    create_gauss_input_file(ts_guess_path, gauss_prod_input, prod_geom, gaussian_patterns.METHOD_TAIL_REACT_PROD)

    return gauss_react_input, gauss_prod_input