# --- Modules ---
import numpy as np
from pathlib import Path

from tunnex_2.irc_computations.ancillary_functions.irc_helpers import proj_freq_extraction_tunnex # type: ignore

from tunnex_2.irc_computations.gaussian import gaussian_patterns # type: ignore


# --- Extracting Hessian values (Gaussian) ---
def _gauss_hessian_values_extract(filename: str | Path, text: str) -> np.ndarray:

    file_path = Path(filename)
    
    # --- First pass: parsing Hessian blocks ---
    lines = text.splitlines()
    parsed_rows = []
    columns = None

    for line in lines:
        tokens = line.split()
        if not tokens:
            continue

        # --- Parsing block headers. Example: 1  2  3  4  5 ---
        if all(token.isdigit() for token in tokens):
            columns = [int(token) - 1 for token in tokens]

            #if len(columns) == 0:
                #raise ValueError(f"Empty Hessian column block was found. Please check '{file_path}'.")
                # the ValueError cannot be reached in this code configuration,
                # but it would be useful to keep it for later use

            if len(set(columns)) != len(columns):
                raise ValueError(f"Duplicate Hessian column index was found. Please check '{file_path}'.")
            
            continue

        # --- Parsing Hessian rows. Example: 10  0.711018D-01  0.581355D-01 ---
        if columns is not None and tokens[0].isdigit():
            row = int(tokens[0]) - 1
            try:
                values = [float(value.replace("D", "E")) for value in tokens[1:]]
            except ValueError as exc:
                raise ValueError(f"Invalid Hessian value in row {row + 1} was found. "
                                 f"Please check '{file_path}'.") from exc

            expected_values = min(row + 1, max(columns) + 1) - min(columns)
            if len(values) > expected_values:
                raise ValueError(f"Expected {expected_values} values, got {len(values)} in row {row + 1}. "
                                 f"Please check '{file_path}'.")         
            
            parsed_rows.append((row, columns.copy(), values))

    # --- Check that something was found ---
    if not parsed_rows:
        raise ValueError(f"Hessian lines were not found. Please check '{file_path}'.")

    # --- Determine matrix size and filling it ---
    n = max(max(row, max(columns)) for row, columns, _ in parsed_rows) + 1
    matrix = np.full((n, n), np.nan, dtype=float)

    # --- Keep track of rows and columns that were actually found ---
    rows_found = set()
    columns_found = set()

    # --- Fill Hessian symmetrically ---
    for row, columns, values in parsed_rows:

        #if not 0 <= row < n:
            #raise ValueError(f"Invalid Hessian row index {row + 1} was found. Please check '{file_path}'.")
        # the ValueError cannot be reached in this code configuration,
        # but it would be useful to keep it for later use
        rows_found.add(row)

        for column, value in zip(columns, values):

            if not 0 <= column < n:
                raise ValueError(f"Invalid Hessian column index {column + 1} was found. Please check '{file_path}'.")
            columns_found.add(column)

            matrix[row, column] = value
            matrix[column, row] = value

    # --- Check that all rows are present ---
    expected_indices = set(range(n))

    if rows_found != expected_indices:
        missing_rows = sorted(index + 1 for index in expected_indices - rows_found)
        raise ValueError(f"Incomplete Hessian in '{file_path}'. Missing rows: {missing_rows}.")

    # --- Check that all columns are present ---
    if columns_found != expected_indices:
        missing_columns = sorted(index + 1 for index in expected_indices - columns_found)
        raise ValueError(f"Incomplete Hessian in '{file_path}'. Missing columns: {missing_columns}.")

    # --- Checking that all matrix elements were read and the matrix is symmetric ---
    if np.isnan(matrix).any():
        missing_elements = np.argwhere(np.isnan(matrix))
        raise ValueError(f"Incomplete Hessian in '{file_path}'. Some matrix elements were not found. "
                         f"First missing element: ({missing_elements[0, 0] + 1}, {missing_elements[0, 1] + 1}).")
    
    #if not np.allclose(matrix, matrix.T):
        #raise ValueError(f"Hessian matrix is not symmetric in '{file_path}'.")
        # it should never occur since Gaussian stores only the low triangle of a Hessian,
        # but it would be useful to keep it for later use

    return matrix


# --- Collecting Hessian values (Gaussian) ---
def gauss_hessian_reader(filename: str | Path, mode: str = 'irc') -> np.ndarray | list[np.ndarray]:

    allowed_modes = ('opt', 'irc')
    if mode not in allowed_modes:
        raise ValueError(f"Expected {allowed_modes} as mode. Got {mode}.")

    file_path = Path(filename)
    
    with open(file_path, encoding="utf-8") as f_irc:
        text = f_irc.read()

    if mode == 'irc':
        hessian_matches = gaussian_patterns.PATTERN_HESSIAN_IN_IRC_BLOCK.findall(text)

        if not hessian_matches:
            raise ValueError(f"A Hessian was not found. Please check '{file_path}'.")
        
        hessian_matrices = [_gauss_hessian_values_extract(file_path, hessian_match)
            for hessian_match in hessian_matches]
        return hessian_matrices

    else:
        hessian_matches = gaussian_patterns.PATTERN_HESSIAN_IN_OPT_BLOCK.findall(text)
        
        if not hessian_matches:
            raise ValueError(f"A Hessian was not found. Please check '{file_path}'.")

        last_match = hessian_matches[-1]
        hessian_matrix = _gauss_hessian_values_extract(file_path, last_match)
        return hessian_matrix


# --- Defining the function to collect the projected ZPVE values (Gaussian) ---
def gauss_collect_proj_zpves(irc_file_out: str | Path, irc_and_structures: list[tuple[float, np.ndarray]],
    atom_masses: np.ndarray) -> tuple[float, ...]:

    irc_file_path = Path(irc_file_out)
    
    for item in irc_and_structures:
        if len(item) != 2:
            raise ValueError(f"Each IRC structure entry must contain exactly two values. "
                             f"Got {len(item)}.")  
    
    # --- Storing the values of the Hessians ---
    hess_matrices = gauss_hessian_reader(irc_file_path, 'irc')
    if len(hess_matrices) != len(irc_and_structures):
        raise ValueError(f"Number of Hessians ({len(hess_matrices)}) and structures ({len(irc_and_structures)}) must match. "
                         f"Please check '{irc_file_path}'.")

    # --- Sorting Hessians and structures in the increasing IRC order ---
    data = [(irc, structure, hessian) for (irc, structure), hessian in zip(irc_and_structures, hess_matrices)]
    data.sort(key=lambda x: x[0])

    coordinates = [coords for _, coords, _ in data]
    hess_matrices_sorted = [hess for _, _, hess in data]

    # --- Computing the projected ZPVEs ---
    zpve_values, _, _ = proj_freq_extraction_tunnex(irc_file_path, coordinates, atom_masses, hess_matrices_sorted)

    return zpve_values