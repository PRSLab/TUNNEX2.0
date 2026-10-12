# --- Modules ---
from pathlib import Path

from tunnex_2.irc_computations.gaussian import gaussian_patterns # type: ignore


# --- Defining the function to write the input file for a transition state, reactant (product) or the IRC input file ---
def create_gauss_input_file(gauss_ts_guess_input: str | Path, gauss_species_output: str | Path,
    species_geometry: str | None, method_tail: str) -> None:

    gauss_ts_guess_input_path = Path(gauss_ts_guess_input)
    gauss_species_output_path = Path(gauss_species_output)
    
    try:
        with open(gauss_ts_guess_input_path, "r", encoding="utf-8") as f_in, \
            open(gauss_species_output_path, "w", encoding="utf-8") as f_species_out:

            geometry_found = species_geometry is None
            method_found = False

            for line in f_in:
                stripped = line.strip()
                if stripped:

                    match_method = gaussian_patterns.PATTERN_COMPUTATIONAL_METHOD_GAUSS.match(stripped)

                    if match_method:
                        if method_found:
                            raise ValueError(f"Multiple computational method lines were found. "
                                             f"Please check '{gauss_ts_guess_input_path}'.")
                        method_found = True
                        prefix, method_and_basis = match_method.groups()
                        f_species_out.write(f"{prefix} {method_and_basis} {method_tail}\n\n")
                        continue

                    if species_geometry is not None:
                        match_geometry_line = gaussian_patterns.PATTERN_GEOMETRY_LINE_INPUT_GAUSS.match(stripped)
                        if match_geometry_line:
                            geometry_found = True
                            break

                    f_species_out.write(line)

                    if gaussian_patterns.PATTERN_COMMENT in stripped:
                        f_species_out.write("\n")

            if not geometry_found:
                raise ValueError(f"The geometry of the species was not found. "
                                 f"Please check '{gauss_ts_guess_input_path}'.")
            if not method_found:
                raise ValueError(f"The computational method was not found. "
                                 f"Please check '{gauss_ts_guess_input_path}'.")
            
            if species_geometry is not None:
                f_species_out.write(species_geometry)
                f_species_out.write("\n\n\n\n")
            else:
                f_species_out.write("\n\n")

    except ValueError:
        gauss_species_output_path.unlink(missing_ok=True)
        raise