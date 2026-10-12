# --- Modules ---
import numpy as np
from pathlib import Path

from tunnex_2.constants_and_dataclasses.dataclasses import IRCData # type: ignore

from tunnex_2.irc_computations.ancillary_functions.irc_helpers import ( # type: ignore
 irc_steps_comp,
 freq_and_modes_extraction_tunnex,
 vibrations_format_transform)
from tunnex_2.irc_computations.ancillary_functions.interface import file_check # type: ignore

from tunnex_2.irc_computations.orca import orca_patterns # type: ignore
from tunnex_2.irc_computations.orca.orca_proj_freq import orca_hessian_reader, orca_collect_proj_zpves # type: ignore


# --- Defining the function to extract the atomic mass (it supports only xxx.out files) ---
def _orca_mass_extraction(input_file: str | Path, mass_filename: str = 'orca_atom_masses.out') -> np.ndarray:

    input_path = Path(input_file)

    # --- Checking if the atom masses were already stored ---
    if input_path.suffix.lower() != ".out":
        raise ValueError(f"Expected an .out file, got '{input_path}'.")

    mass_file = input_path.parent / mass_filename

    if mass_file.is_file():
        filename = mass_file
    else:
        filename = input_path

    with open(filename, encoding="utf-8") as f_in:
        text = f_in.read()

    # --- Reading atom masses ---
    match = orca_patterns.PATTERN_MASS_BLOCK.search(text)

    if match is None:
        raise ValueError(f"Cannot find the atomic mass block in '{filename}'. Please check '{filename}'.")

    mass_block = match.group(1)

    atom_masses = np.array([float(fields[4]) for line in mass_block.splitlines()
    if (fields := line.split())], dtype=float)
    # atom mass is the line.split()[4] according to the standard ORCA output

    if atom_masses.size == 0:
        raise ValueError(f"Atomic mass block was found, but no atomic masses could be extracted from '{filename}'. "
                         f"Please check '{filename}'.")
    # it is a protection from an empty output

    if np.any(atom_masses <= 0):
        raise ValueError(f"Atom masses must be positive. Got {atom_masses} atom masses.")
    
    # --- Storing the atom masses for later use if they were not stored before ---
    if not mass_file.is_file():
        with open(mass_file, "w", encoding="utf-8") as f_out:
            f_out.write(match.group(0))

    return atom_masses


# --- Defining the function to read the optimized geometry (assuming that the optimization is converged) ---
def orca_extract_geom_from_opt_file(filename: str | Path, species: str = 'species',
    as_array: bool = False) -> str | np.ndarray:

    file_path = Path(filename)
    
    with open(file_path, encoding="utf-8") as f_in:
        text = f_in.read()

    geometry_blocks = orca_patterns.PATTERN_OPTIMIZED_SPECIES_GEOM_BLOCK.findall(text)
    if not geometry_blocks:
        raise ValueError(f"The geometry of the {species} was not found. Please check '{file_path}'.")

    last_geometry = geometry_blocks[-1]

    geometry_matches = orca_patterns.PATTERN_GEOMETRY_LINE_ORCA.findall(last_geometry)
    if not geometry_matches:
        raise ValueError(f"No atom coordinates were found. Please check '{file_path}'.")

    if not as_array:
        return "\n".join(" ".join(row) for row in geometry_matches)

    try:
        coordinates = [[float(x), float(y), float(z)] for _, x, y, z in geometry_matches]
    except ValueError as exc:
        raise ValueError(f"Invalid coordinates in geometry. Please check '{file_path}'.") from exc

    return np.array(coordinates, dtype=float)


# --- Defining the function to take the reactant (product) geometry ---
def orca_extract_geom_from_irc_point(filename: str | Path, species: str = 'species') -> str:

    file_path = Path(filename)

    with open(file_path, encoding="utf-8") as f_in:
        text = f_in.read()

    geometry_matches = orca_patterns.PATTERN_GEOMETRY_LINE_ORCA.findall(text)
    if not geometry_matches:
        raise ValueError(f"The geometry of the {species} was not found. Please check {file_path}.")

    return "\n".join(" ".join(row) for row in geometry_matches)


# --- Defining the function to read the IRC geometries and electronic energies from the IRC file ---
def _orca_struct_extraction_irc(irc_file_xyz: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:

    irc_file_xyz_path = Path(irc_file_xyz)
    
    with open(irc_file_xyz_path, encoding="utf-8") as f_irc:
        text = f_irc.read()

    geometry_blocks = orca_patterns.PATTERN_IRC_COORDINATES_BLOCK.findall(text)
    if not geometry_blocks:
        raise ValueError(f"No geometries were found in '{irc_file_xyz_path}'. "
                         f"Please check '{irc_file_xyz_path}'.")

    energies = []
    geometries_array = []
    geometries_text = []

    expected_n_atoms = None

    for energy, geometry in geometry_blocks:

        coordinates = (orca_patterns.PATTERN_GEOMETRY_LINE_ORCA.findall(geometry))
        if not coordinates:
            raise ValueError(f"No atomic coordinates found for IRC point with energy {energy}. "
                             f"Please check '{irc_file_xyz_path}'.")

        try:
            geometry_coordinates = [[float(x) for x in row[1:]] for row in coordinates]
        except ValueError as exc:
            raise ValueError(f"Invalid Cartesian coordinates type for IRC point with energy {energy}. "
                             f"Please check '{irc_file_xyz_path}'.") from exc
        
        if any(len(atom) != 3 for atom in geometry_coordinates):
            raise ValueError(f"Invalid Cartesian coordinates length for IRC point with energy {energy}. "
                             f"Please check '{irc_file_xyz_path}'.")

        if expected_n_atoms is None:
            expected_n_atoms = len(geometry_coordinates)
        elif len(geometry_coordinates) != expected_n_atoms:
            raise ValueError(f"Inconsistent number of atoms between IRC geometries. "
                             f"Please check '{irc_file_xyz_path}'.")

        geometry_text = "\n".join(" ".join(row) for row in coordinates)
        
        energies.append(float(energy))
        geometries_array.append(geometry_coordinates)
        geometries_text.append(geometry_text)

    energies = np.array(energies, dtype=float)
    geometries_array = np.array(geometries_array, dtype=float)

    return energies, geometries_array, geometries_text


# --- Defining the function to read the IRC coordinates, electronic energies, and ZPVEs from the IRC output file ---
def reading_orca_irc_tunnex(irc_file_xyz: str | Path, ts_output_file: str | Path, ts_guess_input: str | Path,
    command_file: str | Path, proj_freq: bool = True,
    run_software_key: bool = True, filename_out_key: bool = False) -> IRCData:

    irc_file_xyz_path = file_check(irc_file_xyz)
    ts_output_file_path = file_check(ts_output_file)
    ts_guess_input_path = file_check(ts_guess_input)
    command_path = Path(command_file)

    # --- Extracting the atom masses and other data ---
    atom_masses = _orca_mass_extraction(ts_output_file_path)
    data_irc_el_energy, struct_coordinates, struct_text = _orca_struct_extraction_irc(irc_file_xyz_path)

    # --- Computing the IRC steps ---
    irc_steps = irc_steps_comp(irc_file_xyz_path, struct_coordinates, atom_masses)

    # --- Accumulating the IRC steps along the path ---
    data_irc = np.concatenate([[0.0], np.cumsum(irc_steps)])

    # --- Merging IRC coordinates with the electronic energies ---
    data_el_energy = np.column_stack((data_irc, data_irc_el_energy))

    # --- Correcting the IRC coordinates in order to make the TS at IRC = 0.0
    # and transforming the data to list[tuple[float, float]] ---
    ts_index = np.argmax(data_el_energy[:, 1])
    ts_irc_value = data_el_energy[ts_index, 0]
    data_el_energy[:, 0] -= ts_irc_value
    electronic_energies = [(float(irc), float(energy)) for irc, energy in data_el_energy]

    # --- Extracting ZPVEs (No method of computing projected frequencies is available in ORCA by now [01.10.2026]).
    # It is my own (experimental) feature ---
    if proj_freq:
        zpve_values = orca_collect_proj_zpves(ts_guess_input_path, irc_file_xyz_path, command_path,
            data_irc_el_energy, struct_coordinates, struct_text, atom_masses, run_software_key, filename_out_key)

        # --- Merging IRC coordinates with the ZPVEs and correcting the IRC coordinates
        # in order to make the TS at IRC = 0.0 ---
        data_zpves = np.column_stack((data_irc, zpve_values))

        data_zpves[:, 0] -= ts_irc_value
        zpve_energies_forward = [(float(irc), float(zpve)) for irc, zpve in data_zpves if irc > 0]
        zpve_energies_reverse = [(float(irc), float(zpve)) for irc, zpve in data_zpves if irc < 0]

    else:
        zpve_energies_forward = None
        zpve_energies_reverse = None

    result = IRCData(electronic_energies=electronic_energies,
    zpve_energies_forward=zpve_energies_forward,
    zpve_energies_reverse=zpve_energies_reverse)

    return result


# --- Defining the function to collect the frequencies and modes using direct Hessian reading (ORCA) ---
def _orca_collect_freq_and_modes_tunnex(opt_file_out: str | Path, hess_file_out: str | Path,
    species: str = 'species'):
    opt_file_path = Path(opt_file_out)
    hess_file_path = Path(hess_file_out)
    atom_masses = _orca_mass_extraction(opt_file_path)
    opt_structure = orca_extract_geom_from_opt_file(opt_file_path, species, True)
    hessian_matrix = orca_hessian_reader(hess_file_path)
    zpve, vibs, modes = freq_and_modes_extraction_tunnex(opt_structure, atom_masses, hessian_matrix)
    return zpve, vibs, modes, atom_masses


# --- Defining the function to store the frequencies of the transition state and minima (reactant, product) ---
def reading_orca_struct_file(opt_file_out: str | Path, mode: str = 'ts',
    hess_parsing: bool = False) -> tuple[tuple[float, float], tuple[float, float]]:

    # --- Checking if we correctly defined the mode ---
    opt_file_out_path = file_check(opt_file_out)
    
    mode_positions = {"ts": 0.0, "react": float("-inf"), "prod": float("inf")}
    if mode not in mode_positions.keys():
        raise ValueError(f"The inserted mode ('{mode}') is incorrect. "
                         f"Please use {mode_positions.keys()}.")
    position = mode_positions[mode]

    # --- Opening the file and reading the data ---
    with open(opt_file_out_path, encoding="utf-8") as f_in:
        text = f_in.read()

    # --- Searching for the electronic energy and ZPVE ---
    matches = orca_patterns.PATTERN_EL_ENERGY_ZPVE_SINGLE_POINT.findall(text)
    if not matches:
        raise ValueError(f"The electronic energy of the species was not found. "
                         f"Please check '{opt_file_out_path}'.")
    
    last_match = matches[-1]
    
    el_energy = (position, float(last_match[0]))

    if not hess_parsing:
        zpve_energy = (position, float(last_match[1]))
    else:
        hess_file_out = opt_file_out_path.with_suffix(".hess")
        hess_file_out_path = file_check(hess_file_out)
        
        zpve, _, _, _ = _orca_collect_freq_and_modes_tunnex(opt_file_out_path, hess_file_out_path)
        zpve_energy = (position, zpve)

    return el_energy, zpve_energy


# --- Collecting frequencies and normal mode vectors (ORCA) ---
def orca_mode_coordinate_reader(filename: str | Path, hess_parsing: bool = False,
    freq_tol: float = 1e-2) -> tuple[list[list[float]], tuple[float, ...]]:

    filename_path = file_check(filename)

    if not hess_parsing:

        # --- Reading a file and searching for frequencies and normal modes ---
        with open(filename_path, encoding="utf-8") as f:
            text = f.read()

        atom_masses = _orca_mass_extraction(filename_path)

        vibr_blocks = orca_patterns.PATTERN_VIBR_BLOCK.findall(text)
        if not vibr_blocks:
            raise ValueError(f"Frequencies were not found. Please check '{filename_path}'.")
        
        last_vibr_block = vibr_blocks[-1]
        freqs = [float(x) for x in orca_patterns.PATTERN_VIBR_FREQUENCIES.findall(last_vibr_block)]
        if not freqs:
            raise ValueError(f"Vibrational frequencies were not found in the frequency block. "
                             f"Please check '{filename_path}'.")

        normal_mode_matches = orca_patterns.PATTERN_NORMAL_MODES_BLOCK.findall(text)
        if not normal_mode_matches:
            raise ValueError(f"Normal modes block was not found. Please check '{filename_path}'.")
        
        normal_modes_last_match = normal_mode_matches[-1]

        # --- Filling the values of normal modes ---
        matrix_mode_coordinates = []
        columns = None

        for line in normal_modes_last_match.splitlines():
            fields = line.split()
            if not fields:
                continue

            # --- Reading the heades ---
            if all(field.isdigit() for field in fields):
                columns = [int(field) for field in fields]
                # in this code realization columns is never empty if it exists
                while len(matrix_mode_coordinates) <= max(columns):
                    matrix_mode_coordinates.append([])
                continue

            # --- Reading the data ---
            if fields[0].isdigit() and columns is not None:
                
                try:
                    values = [float(x) for x in fields[1:]]
                except ValueError as exc:
                    raise ValueError(f"Invalid normal mode vector value. "
                                     f"Please check '{filename_path}'.") from exc

                if len(columns) != len(values):
                    raise ValueError(f"Numbers of columns and normal mode vectors do not match. "
                                     f"Please check '{filename_path}'.")
                
                for column, value in zip(columns, values):
                    matrix_mode_coordinates[column].append(value)

        if columns is None:
            raise ValueError(f"Normal modes were not found. Please check '{filename_path}'.")
        
        # --- Collecting the results ---
        if len(freqs) != len(matrix_mode_coordinates):
            raise ValueError(f"Number of frequencies and normal mode vectors do not match. "
                             f"Please check '{filename_path}'.")
        
        freq_and_modes = [[freq, *mode]
        for freq, mode in zip(freqs, matrix_mode_coordinates) if abs(freq) >= freq_tol]

    else:
        ts_hess_filename = filename_path.with_suffix(".hess")
        ts_hess_path = file_check(ts_hess_filename)
        _, frequencies, modes, atom_masses = _orca_collect_freq_and_modes_tunnex(filename_path, ts_hess_path)
        freq_and_modes = vibrations_format_transform(ts_hess_path, frequencies, modes, atom_masses, freq_tol)  
   
    return freq_and_modes, tuple(atom_masses)