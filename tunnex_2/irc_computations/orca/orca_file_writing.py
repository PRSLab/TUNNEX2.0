# --- Modules ---
from pathlib import Path

from tunnex_2.irc_computations.ancillary_functions.interface import file_check # type: ignore

from tunnex_2.irc_computations.orca.orca_create_input_file import create_orca_input_file # type: ignore
from tunnex_2.irc_computations.orca.orca_file_parsing import ( # type: ignore
    orca_extract_geom_from_opt_file, orca_extract_geom_from_irc_point)
from tunnex_2.irc_computations.orca import orca_patterns # type: ignore


# --- Defining the function to write the input file for a transition state, reactant or product ---
def create_orca_input(start_file: str | Path, species_geometry: str | None, method_tail: str, suffix: str) -> Path:
    start_path = file_check(start_file)

    allowed_suffixes = ("_ts", "_react", "_prod")
    if suffix not in allowed_suffixes:
        raise ValueError(f"Expected suffix {allowed_suffixes}. Got {suffix}.")

    input_file = start_path.with_stem(start_path.stem + suffix)
    create_orca_input_file(start_path, input_file, species_geometry, method_tail)

    return input_file


# --- Defining the function to create the IRC input files
# (No method of computing projected frequencies is available in ORCA by now [01.10.2026]) ---
def create_orca_irc_input(orca_ts_guess_input: str | Path, orca_optimized_ts_file: str | Path)  -> Path:
    
    # --- Creating the filename of a new file ---
    ts_guess_input_path = file_check(orca_ts_guess_input)
    orca_irc_input = ts_guess_input_path.with_stem(ts_guess_input_path.stem + "_irc")
    
    # --- Opening the output files and reading the optimized geometry ---
    ts_optimized_path = file_check(orca_optimized_ts_file)
    ts_geometry = orca_extract_geom_from_opt_file(ts_optimized_path)
    create_orca_input_file(ts_guess_input_path, orca_irc_input, ts_geometry, orca_patterns.BLOCK_TAIL_IRC)

    return orca_irc_input


# --- Defining the function to create the file for the geometry optimization of the reactant (product) ---
def create_orca_react_prod_input_from_irc(irc_output_file: str | Path, orca_ts_guess_input: str | Path) -> tuple[Path, Path]:

    # --- Creating the filenames of new files ---
    ts_guess_path = file_check(orca_ts_guess_input)
    orca_react_input = ts_guess_path.with_stem(ts_guess_path.stem + "_react")
    orca_prod_input = ts_guess_path.with_stem(ts_guess_path.stem + "_prod")

    # --- Taking the reactant and product geometries ---
    irc_output_path = file_check(irc_output_file)
    react_irc_name = irc_output_path.with_name(ts_guess_path.stem + "_irc_IRC_B.xyz")
    react_geom = orca_extract_geom_from_irc_point(react_irc_name)
    prod_irc_name = irc_output_path.with_name(ts_guess_path.stem + "_irc_IRC_F.xyz")
    prod_geom = orca_extract_geom_from_irc_point(prod_irc_name)

    # --- Writing new files ---
    create_orca_input_file(ts_guess_path, orca_react_input, react_geom, orca_patterns.LINE_TAIL_REACT_PROD)
    create_orca_input_file(ts_guess_path, orca_prod_input, prod_geom, orca_patterns.LINE_TAIL_REACT_PROD)
          
    return orca_react_input, orca_prod_input