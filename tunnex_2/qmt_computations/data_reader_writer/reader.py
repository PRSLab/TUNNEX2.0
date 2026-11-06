# --- Modules ---
import numpy as np
from pathlib import Path

from tunnex_2.constants_and_dataclasses.constants_and_settings import TUNN_PROB_AVERAGING_ALLOWED_MODES  # type: ignore
from tunnex_2.constants_and_dataclasses.dataclasses import TunnexInputSettings, IRCDataQMT  # type: ignore


# --- Defining the function to parse the header lines ---
def _parse_line(line: str, line_num: int) -> dict:
    # Parsing the string of type 'key=value, key=value, ..., key=value'
    result = {}
    pairs = line.strip().split(";")
    
    for item in pairs:
        if item:
            try:
                key, value = item.split("=")
                key = key.strip()
                value = value.strip()
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError # it catches NaN, -inf, and +inf
            except ValueError:
                raise ValueError(f"[Line {line_num}] Invalid key=value format: '{line.strip()}'.")
            
            if key in result:
                raise ValueError(f"[Line {line_num}]: duplicate key '{key}'.")
            
            result[key] = value

    return result


# --- Defining the function to validate the input arguments ---
def _validate_positive(name: str, value: float, allow_zero: bool = False) -> None:
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0 (got {value}).")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be > 0 (got {value}).")


# --- Defining the function to read the IRC [Nx2] matrix and putting it in two separate [Nx1], [Nx1] arrays ---
def _read_irc_matrix(lines: list[str], start_tag: str, end_tag: str) -> tuple[np.ndarray, np.ndarray]:
    
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
                if not np.isfinite(x) or not np.isfinite(y):
                    raise ValueError # it catches NaN, -inf, and +inf
            except Exception:
                raise ValueError(f"[Line {i+1}] Invalid IRC row: '{line}'.")

            if x <= first_x:
                raise ValueError(f"[Line {i+1}] IRC values must be strictly increasing "
                                 f"(got {x} after {first_x}).")

            first_x = x
            x_vals.append(x)
            y_vals.append(y)

    if not x_vals:
        raise ValueError(f"Block '{start_tag}' is empty or missing.")

    if len(x_vals) < 2:
        raise ValueError(f"Expected at least 2 points for a {start_tag} array. "
                         f"Got {len(x_vals)}")

    return np.array(x_vals), np.array(y_vals)


# --- Defining the main function to read the input file ---
def read_input(filename: str | Path) -> tuple[TunnexInputSettings, IRCDataQMT]:
    
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}.")

    lines = path.read_text().splitlines()

    header_lines_number = 3
    if len(lines) < header_lines_number:
        raise ValueError(f"Input file must contain at least {header_lines_number} header lines. "
                         f"Got {len(lines)} lines.")

    # --- Parsing 3 header lines ---
    lines_param = [_parse_line(lines[i], i + 1) for i in range(header_lines_number)]

    # --- Checking if all the parameters required for the computations are given ---
    required_keys = [{"freq", "T"},
        {"E0", "corr-ZPVE0", "potential_scaling_factor", "N"},
        {"T_arrhenius_min", "T_arrhenius_max", "T_step", "tunn_prob_aver"}]

    for i, line in enumerate(lines_param):
        required = required_keys[i]
        missing = required - line.keys()
        extra = line.keys() - required
        if missing:
            raise ValueError(f"Missing required keys {sorted(missing)} in line {i + 1}.")
        if extra:
            raise ValueError(f"Unrecognized keys {sorted(extra)} in line {i + 1}.")

    # --- Checking if all the parameters required for the computations are given ---
    freq = lines_param[0]["freq"]
    T = lines_param[0]["T"]

    E0 = lines_param[1]["E0"]
    ZPVE0 = lines_param[1]["corr-ZPVE0"]
    potential_scaling_factor = lines_param[1]["potential_scaling_factor"]
    N = lines_param[1]["N"]

    T_arrhenius_min = lines_param[2]["T_arrhenius_min"]
    T_arrhenius_max = lines_param[2]["T_arrhenius_max"]
    T_arrhenius_step = lines_param[2]["T_step"]
    tunn_prob_averaging_val = lines_param[2]["tunn_prob_aver"]

    if tunn_prob_averaging_val not in TUNN_PROB_AVERAGING_ALLOWED_MODES.keys():
        raise ValueError(f"Expected {TUNN_PROB_AVERAGING_ALLOWED_MODES.keys()} key as a temperature averaging mode key. "
                         f"Got {tunn_prob_averaging_val}")
    
    tunn_prob_averaging = TUNN_PROB_AVERAGING_ALLOWED_MODES[tunn_prob_averaging_val]

    # --- Validating the input parameters ---
    _validate_positive("freq", freq)
    _validate_positive("temperature", T)

    _validate_positive("scaling", potential_scaling_factor)
    _validate_positive("max_level", N, allow_zero=True)

    if not N.is_integer():
        raise ValueError(f"N must be an integer. Got N={N}") 
    the_upper_level_to_compute = int(lines_param[1]["N"])

    _validate_positive("T_min", T_arrhenius_min)
    _validate_positive("T_max", T_arrhenius_max)
    _validate_positive("T_step", T_arrhenius_step)

    if T_arrhenius_min >= T_arrhenius_max:
        raise ValueError(f"T_arrhenius_min={T_arrhenius_min} cannot be >= than "
                         f"T_arrhenius_max={T_arrhenius_max}.")
  
    # --- Reading the IRC data ---
    E_x, E_y = _read_irc_matrix(lines, "IRC; E", "END; E")
    ZPVE_x, ZPVE_y = _read_irc_matrix(lines, "IRC; ZPVE", "END; ZPVE")

    # --- Checking if both matrices start and end with the same IRC values ---
    # The lengths of the array can be different, since they are interpolated separately
    if not np.allclose(E_x[0], ZPVE_x[0], atol=1e-8):
        raise ValueError(f"Initial IRC values for E and ZPVE must match. "
                         f"Got {E_x[0]} and {ZPVE_x[0]}.")
    if not np.allclose(E_x[-1], ZPVE_x[-1], atol=1e-8):
        raise ValueError(f"Final IRC values for E and ZPVE must match. "
                         f"Got {E_x[-1]} and {ZPVE_x[-1]}.")

    # --- Storing the parameters and IRC matrices ---
    params = TunnexInputSettings(freq=freq, T=T, E0=E0, ZPVE0=ZPVE0,
        potential_scaling_factor=potential_scaling_factor,
        number_of_levels=the_upper_level_to_compute,
        T_arrhenius_min=T_arrhenius_min,
        T_arrhenius_max=T_arrhenius_max,
        T_step=T_arrhenius_step,
        prob_aver_mode_qmt=tunn_prob_averaging)

    irc = IRCDataQMT(E_x=E_x, E_y=E_y, ZPVE_x=ZPVE_x, ZPVE_y=ZPVE_y,)

    return params, irc
