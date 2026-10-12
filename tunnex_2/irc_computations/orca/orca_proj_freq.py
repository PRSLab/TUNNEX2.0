# --- Modules ---
import numpy as np
from pathlib import Path

from tunnex_2.constants_and_dataclasses.constants_and_settings import ( # type: ignore
    SLURM_MODE_KEY,
    ENVR)

from tunnex_2.irc_computations.ancillary_functions.irc_helpers import proj_freq_extraction_tunnex # type: ignore
from tunnex_2.irc_computations.ancillary_functions.interface import ( # type: ignore
    run_software,
    orca_error_check,
    orca_out_filename)

from tunnex_2.irc_computations.orca.orca_create_input_file import create_orca_hess_comp # type: ignore
from tunnex_2.irc_computations.orca import orca_patterns # type: ignore


# --- Collecting Hessian values (ORCA) ---
def orca_hessian_reader(filename: str | Path) -> np.ndarray:

    file_path = Path(filename)

    # --- Reading a file and searching for Hessian lines ---
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    hess_block = orca_patterns.PATTERN_HESSIAN_BLOCK.search(text)
    if not hess_block:
        raise ValueError(f"A Hessian was not found. Please check '{file_path}'.")

    n_coordinates, hessian_text = hess_block.groups()
    n = int(n_coordinates)
    
    if not hessian_text:
        raise ValueError(f"Lines of hessian were not found. Please check '{file_path}'.")

    # --- Creating an empty Hessian matrix ---
    hessian_values = np.full((n, n), np.nan, dtype=float)

    columns = None
    rows_found = set()

    for line in hessian_text.splitlines():
        fields = line.split()
        if not fields:
            continue

        # --- Reading the headers ---
        if all(field.isdigit() for field in fields):
            columns = [int(field) for field in fields]
            if any(not 0 <= column < n for column in columns):
                raise ValueError(f"Invalid Hessian column index in '{file_path}'.")
            continue

        # --- Reading the data ---
        if fields[0].isdigit() and columns is not None:
            row = int(fields[0])

            if not 0 <= row < n:
                raise ValueError(f"Invalid Hessian row index {row} in '{file_path}'.")

            try:
                values = [float(x) for x in fields[1:]]
            except ValueError as e:
                raise ValueError(f"Invalid numeric Hessian value in row {row} of '{file_path}'.") from e

            if len(values) != len(columns):
                raise ValueError(f"Unexpected number of Hessian values in row {row} of '{file_path}'.")

            hessian_values[row, columns] = values
            rows_found.add(row)

    # --- Checking that the whole Hessian matrix was read ---
    if len(rows_found) != n:
        missing_rows = sorted(set(range(n)) - rows_found)
        raise ValueError(f"Incomplete Hessian in '{file_path}'. Missing rows: {missing_rows}.")

    # --- Checking that all matrix elements were read and the matrix is symmetric ---
    if np.isnan(hessian_values).any():
        raise ValueError(f"Incomplete Hessian in '{file_path}'. Some matrix elements were not found.")
    if not np.allclose(hessian_values, hessian_values.T):
        raise ValueError(f"Hessian matrix is not symmetric in '{file_path}'.")

    return hessian_values


# --- Defining the function to collect the projected ZPVE values (ORCA) ---
def orca_collect_proj_zpves(orca_ts_guess_input: str | Path, irc_file_xyz: str | Path, command_file: str | Path,
    data_irc_el_energy: np.ndarray, coordinates: np.ndarray, structures_text: list[str], atom_masses: np.ndarray,
    run_software_key: bool = False, filename_out_key: bool = False) -> tuple[float, ...]:

    orca_ts_guess_input_path = Path(orca_ts_guess_input)
    irc_file_xyz_path = Path(irc_file_xyz)
    command_path = Path(command_file)
    
    # --- Writing input-files ---
    n_structures = len(data_irc_el_energy)

    if len(coordinates) != n_structures:
        raise ValueError(f"Number of electronic energies and geometries (array) must match. "
                         f"Got {n_structures} and {len(coordinates)}.")
    
    if len(structures_text) != n_structures:
        raise ValueError(f"Number of electronic energies and geometries (text) must match. "
                         f"Got {n_structures} and {len(structures_text)}.")

    hess_input_files = []

    for i in range(n_structures):

        hess_input_file = create_orca_hess_comp(orca_ts_guess_input_path, structures_text[i], data_irc_el_energy[i], i)
        hess_input_files.append(hess_input_file)

    # --- Computing Hessians for each IRC point ---
    if run_software_key:
        for hess_input_file in hess_input_files:
            run_software(hess_input_file, command_path, filename_out_key, SLURM_MODE_KEY, ENVR)
        
    # --- Storing the names of the Hessians ---
    hess_output_files = []
    for hess_input_file in hess_input_files:
        hess_output_file = orca_out_filename(hess_input_file, ".hess")
        orca_error_check(hess_output_file)
        hess_output_files.append(hess_output_file)

    # --- Storing the values of the Hessians ---
    hess_matrices = []
    for hess_output_file in hess_output_files:
        hess_matrix = orca_hessian_reader(hess_output_file)
        hess_matrices.append(hess_matrix)

    # --- Computing the projected ZPVEs ---
    zpve_values, _, _ = proj_freq_extraction_tunnex(irc_file_xyz_path, coordinates, atom_masses, hess_matrices)

    return zpve_values