
# --- Modules ---
from pathlib import Path


from tunnex_2.irc_computations.ancillary_functions.interface import file_check # type: ignore

from tunnex_2.irc_computations.orca import orca_patterns # type: ignore


# --- Defining the function to write the input file for a reactant (product) or the IRC input file ---
def create_orca_input_file(orca_input_file: str | Path, orca_output_file: str | Path, species_geometry: str | None,
    method_tail: str) -> None:

    orca_input_path = Path(orca_input_file)
    orca_output_path = Path(orca_output_file)
    
    try:
        with open(orca_input_path, "r", encoding="utf-8") as f_in, \
            open(orca_output_path, "w", encoding="utf-8") as f_species_out:

            geometry_found = species_geometry is None
            method_found = False
            inside_geom_block = False

            for line in f_in:
                stripped = line.strip()
                if stripped:

                    if stripped == r'%geom':
                        inside_geom_block = True
                        continue
                
                    if inside_geom_block:
                        if stripped == "end":
                            inside_geom_block = False
                        continue

                    match_method = orca_patterns.PATTERN_COMPUTATIONAL_METHOD_ORCA.match(stripped)

                    if match_method:
                        if method_found:
                            raise ValueError(f"Multiple computational method lines were found. "
                                             f"Please check the {orca_input_path} file.")
                        method_found = True
                        method, basis = match_method.groups()
                        f_species_out.write(f"! {method} {basis} {method_tail}\n")
                        continue

                    if species_geometry is not None:
                        match_geometry_line = orca_patterns.PATTERN_GEOMETRY_LINE_ORCA.match(stripped)
                        if match_geometry_line:
                            geometry_found = True
                            break

                    f_species_out.write(line)

            if not geometry_found:
                raise ValueError(f"The geometry of the species was not found. Please check the {orca_input_path} file.")
            if not method_found:
                raise ValueError(f"The computational method was not found. Please check the {orca_input_path} file.")
            if inside_geom_block:
                raise ValueError(f"The %geom block was not closed with 'end'. Please check the {orca_input_path} file.")

            if species_geometry is not None:
                f_species_out.write(species_geometry)
                f_species_out.write("\n*\n\n\n")
            else:
                f_species_out.write("\n\n")

    except ValueError:
        orca_output_path.unlink(missing_ok=True)
        raise


# --- Defining the function to write an input file for hessian computation,
# it is needed for the IRC-projected frequencies (ORCA) ---
def create_orca_hess_comp(orca_ts_guess_input: str | Path, geometry: str, el_energy: float, irc_step_number: int) -> Path:

    # --- Creating the filenames of new files ---
    ts_guess_path = file_check(orca_ts_guess_input)
    orca_hess_comp = ts_guess_path.with_stem(ts_guess_path.stem + f"_hess_irc_{irc_step_number}")

    # --- Writing a new file ---
    method_tail = orca_patterns.LINE_HESS_COMP.format(el_energy)
    create_orca_input_file(ts_guess_path, orca_hess_comp, geometry, method_tail)
          
    return orca_hess_comp