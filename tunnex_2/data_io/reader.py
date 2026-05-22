from __future__ import annotations
from pathlib import Path
import numpy as np

from tunnex_2.models.input import InputParams, IRCData  # type: ignore


# --- Defining the supplementary functions ---
def _parse_line(
    line: str, line_num: int
) -> dict:  # Parsing the string of type 'key=value, key=value, ..., key=value'
    result = {}
    try:
        pairs = line.strip().split(";")
        for item in pairs:
            key, value = item.split("=")
            result[key.strip()] = float(value)
    except Exception:
        raise ValueError(
            f"[Line {line_num}] Invalid key=value format: '{line.strip()}'"
        )
    return result


def _validate_positive(
    name: str, value: float, allow_zero=False
):  # Validating the input arguments
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0 (got {value})")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be > 0 (got {value})")


# --- Defining the function to read the IRC_matrices [Nx2], [Kx2] and putting them in two separate arrays ---
def _read_irc_matrix(lines: str, start_tag: str, end_tag: str) -> np.array:
    reading = False
    x_vals = []
    y_vals = []

    first_x = -np.inf

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()

        if line == start_tag:
            reading = True
            continue

        if line == end_tag:
            break

        if reading:
            if not line:
                continue

            try:
                x_str, y_str = line.split(";")
                x = float(x_str)
                y = float(y_str)
            except Exception:
                raise ValueError(f"[Line {i+1}] Invalid IRC row: '{line}'")

            if x <= first_x:
                raise ValueError(
                    f"[Line {i+1}] IRC values must be strictly increasing"
                    f"(got {x} after {first_x})"
                )

            first_x = x
            x_vals.append(x)
            y_vals.append(y)

    if not x_vals:
        raise ValueError(f"Block '{start_tag}' is empty or missing")

    return np.array(x_vals), np.array(y_vals)


# --- Defining the main function to read the input file ---
def read_input(path: str | Path) -> tuple[InputParams, IRCData]:
    path = Path(path)  # Defining the path for the input file

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    lines = path.read_text().splitlines()  # Opening an input-file and read all lines

    if len(lines) < 3:
        raise ValueError("Input file must contain at least 3 header lines")

    # --- Parsing 3 header lines ---
    lines_param = [_parse_line(lines[i], i + 1) for i in range(3)]

    # --- Checking if all the parameters required for the computations are given ---
    required_keys = [
        {"freq", "T"},
        {"E0", "ZPVE0", "potential_scaling_factor", "N"},
        {"T_arrhenius_min", "T_arrhenius_max", "T_step"},
    ]

    for i, line in enumerate(lines_param):
        for key in line:
            if key not in required_keys[i]:
                raise ValueError(f"Missing '{key}' in line {i}")

    # --- Checking if all the parameters required for the computations are given ---
    freq = lines_param[0]["freq"]
    T = lines_param[0]["T"]

    E0 = lines_param[1]["E0"]
    ZPVE0 = lines_param[1]["ZPVE0"]
    potential_scaling_factor = lines_param[1]["potential_scaling_factor"]
    the_upper_level_to_compute = int(lines_param[1]["N"])

    T_arrhenius_min = lines_param[2]["T_arrhenius_min"]
    T_arrhenius_max = lines_param[2]["T_arrhenius_max"]
    T_arrhenius_step = lines_param[2]["T_step"]

    # --- Validating the input parameters ---
    _validate_positive("freq", freq)
    _validate_positive("temperature", T)
    _validate_positive("scaling", potential_scaling_factor)
    _validate_positive("max_level", the_upper_level_to_compute, allow_zero=True)

    _validate_positive("T_min", T_arrhenius_min)
    _validate_positive("T_max", T_arrhenius_max)
    _validate_positive("T_step", T_arrhenius_step)

    if T_arrhenius_step > T_arrhenius_max:
        raise ValueError("T_arrhenius_min cannot be greater than T_arrhenius_max")

    # --- Reading the IRC data ---
    E_x, E_y = _read_irc_matrix(lines, "IRC; E", "END; E")
    ZPVE_x, ZPVE_y = _read_irc_matrix(lines, "IRC; ZPVE", "END; ZPVE")

    # --- Checking if both matrices start and end with the same IRC values ---
    if not np.allclose(E_x[0], ZPVE_x[0], atol=1e-6):
        raise ValueError("Initial IRC values for E and ZPVE must match")
    if not np.allclose(E_x[-1], ZPVE_x[-1], atol=1e-6):
        raise ValueError("Final IRC values for E and ZPVE must match")

    # --- Storing the parameters and IRC matrices ---
    params = InputParams(
        freq=freq,
        T=T,
        E0=E0,
        ZPVE0=ZPVE0,
        potential_scaling_factor=potential_scaling_factor,
        the_upper_level_to_compute=the_upper_level_to_compute,
        T_arrhenius_min=T_arrhenius_min,
        T_arrhenius_max=T_arrhenius_max,
        T_arrhenius_step=T_arrhenius_step,
    )

    irc = IRCData(
        E_x=E_x,
        E_y=E_y,
        ZPVE_x=ZPVE_x,
        ZPVE_y=ZPVE_y,
    )

    return params, irc
