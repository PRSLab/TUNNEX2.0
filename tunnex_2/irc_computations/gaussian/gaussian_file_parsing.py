# --- Modules ---
import numpy as np
from pathlib import Path

from tunnex_2.constants_and_dataclasses.constants_and_settings import CM_M1_TO_HARTREE # type: ignore
from tunnex_2.constants_and_dataclasses.dataclasses import IRCData # type: ignore

from tunnex_2.irc_computations.ancillary_functions.interface import file_check # type: ignore
from tunnex_2.irc_computations.ancillary_functions.irc_helpers import ( # type: ignore
 irc_steps_comp,
 freq_and_modes_extraction_tunnex,
 vibrations_format_transform)

from tunnex_2.irc_computations.gaussian.gaussian_proj_freq import gauss_hessian_reader, gauss_collect_proj_zpves # type: ignore
from tunnex_2.irc_computations.gaussian import gaussian_patterns # type: ignore


# --- Defining the function to extract the atomic mass (it supports only xxx.out files, does not work with the IRC files) ---
def _gauss_mass_extraction(input_file: str | Path, mass_filename: str = 'gaussian_atom_masses.out') -> np.ndarray:

    input_path = Path(input_file)

    # --- Checking if the atom masses were already stored ---
    if input_path.suffix.lower() != ".out":
        raise ValueError(f"Expected an '.out' file, got '{input_path}'.")

    mass_file = input_path.parent / mass_filename

    if mass_file.is_file():
        filename = mass_file
    else:
        filename = input_path

    with open(filename, encoding="utf-8") as f_in:
        text = f_in.read()

    # --- Reading atom masses ---
    matches = gaussian_patterns.PATTERN_MASS_LINES_AND_VALUES.findall(text)

    if not matches:
        raise ValueError(f"The atom masses were not found. Please check '{filename}'.")
    # it is a protecting from an empty output

    mass_lines = [line for line, _ in matches]

    for _, mass in matches:
        if not mass:
            raise ValueError(f"Atom mass block was found, but no atomic masses could be extracted from '{filename}'. "
                             f"Please check '{filename}'.")

    atom_masses = np.array([float(mass) for _, mass in matches], dtype=float)

    if np.any(atom_masses <= 0):
        raise ValueError(f"Atom masses must be positive. Got {atom_masses} atom masses.")

    # --- Storing the atom masses for later use if they were not stored before ---
    if not mass_file.is_file():
        with open(mass_file, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(mass_lines))

    return atom_masses


# --- Defining the function to split the file to forward and reverse directions ---
def gauss_irc_file_split(irc_file_out: str | Path, marker: str) -> tuple[str, str]:

    irc_file_out_path = Path(irc_file_out)
    
    with open(irc_file_out_path, encoding="utf-8") as f_irc:
        text = f_irc.read()
        sections = text.split(marker, maxsplit=1)

        if len(sections) != 2:
            raise ValueError(f"Cannot find '{marker}' in '{irc_file_out_path}'. "
                             f"Please check '{irc_file_out_path}'.")
        
        forward, reverse = sections

        return forward, reverse


# --- Defining the function to read the optimized geometry (assuming that the optimization is converged) ---
def gauss_extract_geom_from_opt_file(filename: str | Path, species: str = 'species',
        as_array: bool = False) -> str | np.ndarray:

    file_path = Path(filename)
    
    with open(file_path, encoding="utf-8") as f_in:
        text = f_in.read()

    geometry_blocks = gaussian_patterns.PATTERN_OPTIMIZED_SPECIES_GEOMETRY_BLOCK.findall(text)
    
    if not geometry_blocks:
        raise ValueError(f"The geometry of the {species} was not found. Please check '{file_path}'.")

    last_geometry = geometry_blocks[-1]

    geometry_lines = [f"{atom} {x} {y} {z}"
                      for atom, x, y, z in gaussian_patterns.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall(last_geometry)]
    if not geometry_lines:
            raise ValueError(f"The {species} geometry block was found, but no geometry coordinates could be extracted. "
                             f"Please check '{file_path}'.")
    if not as_array:
        return "\n".join(geometry_lines)
    try:
        coordinates = np.array([list(map(float, line.split()[1:])) for line in geometry_lines], dtype=float)
    except ValueError as exc:
        raise ValueError(f"Invalid coordinates in geometry. Please check '{file_path}'.") from exc

    return coordinates


# --- Defining the function to take the reactant (product) geometry ---
def gauss_extract_geom_from_irc_point(file: str | Path, text: str, reactant_key: bool = True) -> str:
  
    file_path = Path(file)

    species = "reactant" if reactant_key else "product"
    
    last_match = None

    for geometry_match in gaussian_patterns.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer(text):
        last_match = geometry_match

    if last_match is None:
        raise ValueError(f"The geometry block of a {species} was not found. Please check '{file_path}'.")

    geometry_lines = [f"{atom} {x} {y} {z}"
                      for atom, x, y, z in gaussian_patterns.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall(last_match.group())]
    
    if not geometry_lines:
        raise ValueError(f"The geometry of a {species} was not found in the geometry block. "
                         f"Please check '{file_path}'.")

    return "\n".join(geometry_lines)


# --- Defining the function to extract IRC coordinates and ZPVEs from the IRC output file
# (ZPVEs are obtained via the summation of positive projected frequencies) ---
def _gauss_freq_extraction_irc(irc_direction_parse: str, filename: str | Path,
    positive: bool = True) -> list[tuple[float, float]]:

    file_path = Path(filename)
    sign = 1 if positive else -1

    zpve_energies_direction = []
    positive_frequency_sum = 0.0

    for line in irc_direction_parse.splitlines():
        
        if "Frequencies --" in line:
            positive_frequency_sum += sum(freq
            for freq in map(float, gaussian_patterns.PATTERN_FREQ_MODES_LINES.findall(line)) if freq > 0.0)
            continue

        elif "NET REACTION COORDINATE UP TO THIS POINT" in line:
            match = gaussian_patterns.PATTERN_NET_REACTION_COORDINATE.search(line)
            if match is None or match.group(1) is None:
                raise ValueError(f"Could not extract IRC coordinate from line: {line!r}. "
                                 f"Please check '{file_path}'.")
            irc_coord = sign * float(match.group(1))

            zpve_point = 0.5 * positive_frequency_sum * CM_M1_TO_HARTREE
            zpve_energies_direction.append((irc_coord, zpve_point))
            if np.isclose(positive_frequency_sum, 0.0, atol=1e-12):
                raise ValueError(f"No frequencies was found for the species. "
                f"Please check '{file_path}'.")
            positive_frequency_sum = 0.0

    if zpve_energies_direction == []:
        raise ValueError(f"No frequencies was found in the file. "
        f"Please check '{file_path}'.")
    
    return zpve_energies_direction


# --- Defining the function to read the IRC coordinates, electronic energies, and ZPVEs from the IRC output file ---
def reading_gauss_irc(irc_file_out: str | Path, proj_freq: bool=True) -> IRCData:

    # --- Opening the file and reading the data ---
    irc_file_out_path = file_check(irc_file_out)
    forward, reverse = gauss_irc_file_split(irc_file_out_path, gaussian_patterns.IRC_SPLIT_MARKER_FORWARD)

    # --- Extracting the transition state energy ---
    match = gaussian_patterns.PATTERN_TS_ENERGY_IRC.search(reverse)

    if match and match.group(1):
        ts_el_energy = float(match.group(1))
        text_irc_energies = reverse[match.end():]
    else:
        raise ValueError(f"The transition state energy was not found. Please check '{irc_file_out_path}'.")

    # --- Extracting the IRC values and electronic energies ---
    data_el_energy = [(float(irc), ts_el_energy + float(relative_el_energy))
                      for relative_el_energy, irc in gaussian_patterns.PATTERN_EL_ENERGY_IRC.findall(text_irc_energies)]

    # --- Extracting ZPVEs ---
    if proj_freq:
        zpve_energies_forward = _gauss_freq_extraction_irc(forward, irc_file_out_path, positive=True)
        zpve_energies_reverse = _gauss_freq_extraction_irc(reverse, irc_file_out_path, positive=False)
    else:
        zpve_energies_forward = None
        zpve_energies_reverse = None

    results = IRCData(electronic_energies=data_el_energy,
    zpve_energies_forward=zpve_energies_forward,
    zpve_energies_reverse=zpve_energies_reverse)

    return results


# --- Defining the function to collect the frequencies and modes using direct Hessian reading (Gaussian) ---
def _gauss_collect_freq_and_modes_tunnex(opt_file_out: str | Path,
    species: str = 'species') -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    opt_file_path = Path(opt_file_out)
    atom_masses = _gauss_mass_extraction(opt_file_path)
    opt_structure = gauss_extract_geom_from_opt_file(opt_file_path, species, as_array=True)
    hessian_matrix = gauss_hessian_reader(opt_file_path, 'opt')
    zpve, vibs, modes = freq_and_modes_extraction_tunnex(opt_structure, atom_masses, hessian_matrix)
    return zpve, vibs, modes, atom_masses


# --- Defining the function to store the frequencies of the transition state and minima (reactant, product) ---
def reading_gauss_struct_file(opt_file_out: str | Path, mode: str='ts',
    hess_parsing: bool = False) -> tuple[tuple[float, float], tuple[float, float]]:

    # --- Checking if we correctly defined the mode ---
    opt_file_out_path = file_check(opt_file_out)
    mode_positions = {"ts": 0.0, "react": float("-inf"), "prod": float("inf")}
    if mode not in mode_positions:
        raise ValueError(f"The inserted mode ('{mode}') is incorrect. Please use {list(mode_positions.keys())}.")
    position = mode_positions[mode]

    # --- Opening the file and reading the data ---
    with open(opt_file_out_path, encoding="utf-8") as f_in:

        el_energy, zpve_energy = None, None
        positive_frequency_sum = 0.0

        # --- Searching for the electronic energy and ZPVE ---
        for line in f_in:
            match_el_energy = gaussian_patterns.PATTERN_EL_ENERGY_SINGLE_POINT.search(line)

            if match_el_energy:
                el_energy = (position, float(match_el_energy.group(1)))

            elif "Frequencies --" in line:
                positive_frequency_sum += sum(freq for freq in map(float,
                    gaussian_patterns.PATTERN_FREQ_MODES_LINES.findall(line)) if freq > 0.0)

            elif "- Thermochemistry -" in line:
                if positive_frequency_sum == 0.0:
                    raise ValueError(f"No frequencies was found for the species. "
                    f"Please check '{opt_file_out_path}'.")
                zpve_energy = (position, 0.5 * positive_frequency_sum * CM_M1_TO_HARTREE)
                positive_frequency_sum = 0.0

    if el_energy is None:
        raise ValueError(f"The electronic energy of the species was not found. "
        f"Please check '{opt_file_out_path}'.")

    if zpve_energy is None:
        raise ValueError(f"The ZPVE of a the species was not found. "
        f"Please check '{opt_file_out_path}'.")
    
    if hess_parsing:
        zpve, _, _, _ = _gauss_collect_freq_and_modes_tunnex(opt_file_out_path, 'species')
        zpve_energy = (position, zpve)
    
    return el_energy, zpve_energy


# --- Collecting frequencies and normal mode vectors (Gaussian) ---
def gauss_mode_coordinate_reader(filename: str | Path, hess_parsing: bool = False,
    freq_tol: float = 1e-2) -> tuple[list[list[float]], tuple[float, ...]]:

    filename_path = file_check(filename)

    # --- Reading a file and searching for frequencies and masses ---
    if not hess_parsing:
        with open(filename_path, encoding="utf-8") as f:
            text = f.read()

        block_match = gaussian_patterns.PATTERN_FREQ_MODES_BLOCK.search(text)
        if block_match is None:
            raise ValueError(f"The normal mode vectors block was not found. "
                             f"Please check '{filename_path}'.")

        atom_masses = _gauss_mass_extraction(filename_path)

        freq_and_modes = []

        # --- Reading frequencies and normal mode vectors ---
        lines = iter(block_match.group(1).splitlines())

        for line in lines:

            if "Frequencies --" not in line:
                continue

            nums = gaussian_patterns.PATTERN_FREQ_MODES_LINES.findall(line)
            if not nums:
                raise ValueError(f"Frequencies were not found. Please check '{filename_path}'.")
            freqs = [[float(freq)] for freq in nums]

            # --- Skipping the lines to line 'Atom  AN...' and going to the next line ---
            coordinate_table = False

            for line in lines:
                if "Atom  AN" in line:
                    coordinate_table = True
                    break

            if not coordinate_table:
                raise ValueError(f"Normal mode vector table was not found. Please check '{filename_path}'.")

            coordinates_found = False

            # --- Saving normal mode vectors ---
            for line in lines:
                nums = gaussian_patterns.PATTERN_FREQ_MODES_LINES.findall(line)
                # it ignores the first two digit in the  "1 6 0.00 0.00 0.11 -0.09 -0.20 0.00 0.00 0.00 -0.07" line
                # Atom  AN      X      Y      Z        X      Y      Z        X      Y      Z"

                if not nums:
                    break

                coordinates = [float(value) for value in nums]

                expected_coordinates = 3 * len(freqs)

                if len(coordinates) != expected_coordinates:
                    raise ValueError(f"Expected {expected_coordinates} normal mode coordinates. "
                    f"Got {len(coordinates)}. Please check '{filename_path}'.")

                coordinates_found = True

                for freq_index, freq in enumerate(freqs):
                    start = 3 * freq_index
                    freq.extend(coordinates[start:start + 3])

            if not coordinates_found:
                raise ValueError(f"Normal mode vectors were not found. Please check '{filename_path}'.")

            freq_and_modes.extend(freqs)
     
        if not freq_and_modes:
            raise ValueError(f"Frequencies and modes were not found. Please check '{filename_path}'.")
        
    else:
        _, frequencies, modes, atom_masses = _gauss_collect_freq_and_modes_tunnex(filename_path, 'species')
        freq_and_modes = vibrations_format_transform(filename_path, frequencies, modes, atom_masses, freq_tol)

    return freq_and_modes, tuple(atom_masses)


# --- Defining the function to search for the IRC point structures in the IRC file ---
def _gauss_struct_irc_direction(irc_file_out: str | Path, direction: str,
    forward_key: bool = True) -> tuple[list[float], list[np.ndarray]]:

    # --- Determining the direction of the IRC ---
    irc_file_out_path = Path(irc_file_out)
    
    sign = 1 if forward_key else -1
    direction_name = "forward" if forward_key else "reverse"

    irc_coords = []
    list_geometries = []

    # --- Searching for the IRC coordinates and corresponding structures ---
    structure_matches = list(gaussian_patterns.PATTERN_IRC_CURRENT_STRUCTURE_BLOCK.finditer(direction))
    if not structure_matches:
        raise ValueError(f"The geometry blocks of the {direction_name} IRC path were not found. "
                         f"Please check '{irc_file_out_path}'.")

    for structure_match in structure_matches:
        geom_block = structure_match.group(0)

        irc_coord_match = (gaussian_patterns.PATTERN_NET_REACTION_COORDINATE.search(geom_block))
        if irc_coord_match is None:
            raise ValueError(f"The IRC coordinate was not found. "
                             f"Please check '{irc_file_out_path}'.")

        irc_coord = sign * float(irc_coord_match.group(1))

        geometry_matches = (gaussian_patterns.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall(geom_block))
        if not geometry_matches:
            raise ValueError(f"The geometry of the IRC point={irc_coord} was not found. "
                             f"Please check '{irc_file_out_path}'.")

        irc_point_geom = np.array([[float(x), float(y), float(z)] for _, x, y, z in geometry_matches])

        irc_coords.append(irc_coord)
        list_geometries.append(irc_point_geom)

    return irc_coords, list_geometries


# --- Defining the function to read the IRC geometries from the IRC file ---
def _gauss_struct_extraction(irc_file_out: str | Path) -> list[float, np.ndarray]:

    # --- Opening the file and reading the data ---
    irc_file_out_path = Path(irc_file_out)
    forward, reverse = gauss_irc_file_split(irc_file_out_path, gaussian_patterns.IRC_SPLIT_MARKER_FORWARD)

    irc_coordinates = []
    irc_geometries = []

    # --- Searching for the transition state geometry ---
    # It is the first 'Input orientation' block
    irc_coord = 0.0
    ts_geom_match = gaussian_patterns.PATTERN_IRC_INPUT_ORIENTATION_BLOCK.search(forward)

    if ts_geom_match is None:
            raise ValueError(f"The geometry block of the IRC point={irc_coord} was not found. "
                             f"Please check '{irc_file_out_path}'.")

    ts_geom_matches = gaussian_patterns.PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS.findall(ts_geom_match.group(0))

    if not ts_geom_matches:
        raise ValueError(f"The geometry of the IRC point={irc_coord} was not found in the geometry block. "
                         f"Please check '{irc_file_out_path}'.")
    
    ts_geom = np.array([[float(x), float(y), float(z)] for _, x, y, z in ts_geom_matches])
    
    irc_coordinates.append(irc_coord)
    irc_geometries.append(ts_geom)

    # --- Searching for the forward and reverse path coordinates and geometries ---
    forw_coords, forw_geometries = _gauss_struct_irc_direction(irc_file_out_path, forward, forward_key=True)
    rev_coords, rev_geometries = _gauss_struct_irc_direction(irc_file_out_path, reverse, forward_key=False)

    irc_coordinates.extend(forw_coords)
    irc_geometries.extend(forw_geometries)
    irc_coordinates.extend(rev_coords)
    irc_geometries.extend(rev_geometries)

    # --- Merging the geometries with their IRC coordinates ---
    irc_data_coord_struct = list(zip(irc_coordinates, irc_geometries))

    return irc_data_coord_struct


# --- Defining the function to read the IRC coordinates, electronic energies, and ZPVEs from the IRC output file ---
def reading_gauss_irc_tunnex(irc_file_out: str | Path, ts_output_file: str | Path, proj_freq: bool = True) -> IRCData:
   
    irc_file_out_path = file_check(irc_file_out)
    ts_output_file_path = file_check(ts_output_file)

    # --- Extracting the atom masses and other data ---
    atom_masses = _gauss_mass_extraction(ts_output_file_path)
    irc_and_structures = _gauss_struct_extraction(irc_file_out_path)

    # --- Sorting the data according to the IRC coordinate ---
    irc_and_structures_sorted = sorted(irc_and_structures, key=lambda x: x[0])
    struct_coordinates_sorted = [struct for _, struct in irc_and_structures_sorted]

    # --- Computing the IRC steps ---
    irc_steps = irc_steps_comp(irc_file_out_path, struct_coordinates_sorted, atom_masses)

    # --- Accumulating the IRC steps along the path ---
    data_irc = np.concatenate([[0.0], np.cumsum(irc_steps)])

    # --- Opening the file and reading the data ---
    _, reverse = gauss_irc_file_split(irc_file_out_path, gaussian_patterns.IRC_SPLIT_MARKER_FORWARD)

    # --- Extracting the transition state energy ---
    match = gaussian_patterns.PATTERN_TS_ENERGY_IRC.search(reverse)

    if match:
        ts_el_energy = float(match.group(1))
        text_irc_energies = reverse[match.end():]
    else:
        raise ValueError(f"The transition state energy was not found. Please check '{irc_file_out_path}'.")

    # --- Extracting the IRC values and electronic energies ---
    el_energies = [ts_el_energy + float(relative_el_energy)
                   for relative_el_energy, _ in gaussian_patterns.PATTERN_EL_ENERGY_IRC.findall(text_irc_energies)]

    # --- Merging IRC coordinates with the electronic energies ---
    data_el_energy = np.column_stack((data_irc, el_energies))

    # --- Correcting the IRC coordinates in order to make the TS at IRC = 0.0
    # and transforming the data to list[tuple[float, float]] ---
    ts_index = np.argmax(data_el_energy[:, 1])
    ts_irc_value = data_el_energy[ts_index, 0]
    data_el_energy[:, 0] -= ts_irc_value
    electronic_energies = [(float(irc), float(energy)) for irc, energy in data_el_energy]

    # --- Extracting ZPVEs ---
    if proj_freq:
        zpve_values = gauss_collect_proj_zpves(irc_file_out_path, irc_and_structures, atom_masses)

        # --- Merging IRC coordinates with the ZPVEs and correcting
        # the IRC coordinates in order to make the TS at IRC = 0.0 ---
        if len(zpve_values) != len(data_irc):
            raise ValueError(f"Number of ZPVEs ({len(zpve_values)}) and IRC coordinates ({len(data_irc)}) must match. "
                             f"Please check '{irc_file_out_path}'.")
      
        data_zpves = np.column_stack((data_irc, zpve_values))
        data_zpves[:, 0] -= ts_irc_value
  
        zpve_energies_forward = [(float(irc), float(zpve)) for irc, zpve in data_zpves if irc > 0]
        zpve_energies_reverse = [(float(irc), float(zpve)) for irc, zpve in data_zpves if irc < 0]
    else:
        zpve_energies_forward = None
        zpve_energies_reverse = None

    results = IRCData(electronic_energies=electronic_energies,
    zpve_energies_forward=zpve_energies_forward,
    zpve_energies_reverse=zpve_energies_reverse)

    return results