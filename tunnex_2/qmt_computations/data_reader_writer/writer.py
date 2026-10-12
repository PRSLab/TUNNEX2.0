# --- Modules ---
import numpy as np
from pathlib import Path

from tunnex_2.constants_and_dataclasses.constants_and_settings import KJ_MOL_M1_TO_HARTREE  # type: ignore


# --- Parameters of the output-file format ---
CHAR_NUM_VAL = 28   # Number of characters per section (in general)
CHAR_NUM_T = 4      # Number of characters per section (temperature)
CHAR_NUM_OFFSET = 8 # Number of characters per section (offset)
ROUND_NUM_VAL = 6   # Number of digits for rounding (in general)
ROUND_NUM_T = 2     # Number of digits for rounding (temperature)


# --- Checking the format of the input-value ---
def _fmt(val: float, char_num: int, round_num: int, is_int: bool = False) -> str:
    if is_int:
        return f"{val:^{char_num}.2f}"
    elif val is None:
        return f"{'No turning point':^{char_num}}"
    elif np.isinf(val):
        return f"{'+inf' if val > 0 else '-inf':^{char_num}}"
    else:
        return f"{val:^{char_num}.{round_num}e}"


# --- Writing the vibrational level indexes and relative parameters (part 1) ---
def _section_level_results_1(data: object) -> str:

    lines = []
    lines.append("#" * 165)
    lines.append("The vibrational level indexes, energies, corresponding turning points (tp), and WKB integrals".center(165))
    lines.append("#" * 165)
    lines.append("")

    header = (f"{'N':^{CHAR_NUM_VAL}} | {'Vibrational energy, kJ mol-1':^{CHAR_NUM_VAL}} | "
        f"{'tp (left), sqrt(amu) * bohr':^{CHAR_NUM_VAL}} | {'tp (right), sqrt(amu) * bohr':^{CHAR_NUM_VAL}} | "
        f"{'WKB integral, sqrt(Hartree) * amu * bohr':^{CHAR_NUM_VAL}}")
    lines.append(header)
    lines.append("-" * 165)

    for level in data:
        lines.append(f"{level.vib_index:^{CHAR_NUM_VAL}} | "
            f"{level.vib_energy / KJ_MOL_M1_TO_HARTREE:^{CHAR_NUM_VAL}.{ROUND_NUM_VAL}e} | "
            f"{_fmt(level.turning_point_left, CHAR_NUM_VAL, ROUND_NUM_VAL)} | "
            f"{_fmt(level.turning_point_right, CHAR_NUM_VAL, ROUND_NUM_VAL)} | "
            f"{_fmt(level.wkb, CHAR_NUM_VAL, ROUND_NUM_VAL)}")
    lines.append("")

    return "\n".join(lines)


# --- Writing the vibrational level indexes and relative parameters (part 2) ---
def _section_level_results_2(data: object) -> str:

    lines = []
    lines.append("#" * 242)
    lines.append("The vibrational level indexes, energies, corresponding transmission probabilities (P), reaction rate (k), "
    "and tunneling half-lives (t_1/2)".center(242))
    lines.append("#" * 242)
    lines.append("")

    header = (f"{'N':^{CHAR_NUM_VAL}} | {'Vibrational energy, kJ mol-1':^{CHAR_NUM_VAL}} | "
        f"{'P':^{CHAR_NUM_VAL}} | {'k, s-1':^{CHAR_NUM_VAL}} | "
        f"{'t_1/2, seconds':^{CHAR_NUM_VAL}} | {'t_1/2, hours':^{CHAR_NUM_VAL}} | "
        f"{'t_1/2, days':^{CHAR_NUM_VAL}} | {'t_1/2, years':^{CHAR_NUM_VAL}}")
    lines.append(header)
    lines.append("-" * 242)

    for level in data:
        lines.append(f"{level.vib_index:^{CHAR_NUM_VAL}} | "
            f"{level.vib_energy / KJ_MOL_M1_TO_HARTREE:^{CHAR_NUM_VAL}.{ROUND_NUM_VAL}e} | "
            f"{_fmt(level.transmission_probability, CHAR_NUM_VAL, ROUND_NUM_VAL)} | "
            f"{_fmt(level.reaction_rate, CHAR_NUM_VAL, ROUND_NUM_VAL)} | "
            f"{_fmt(level.half_s, CHAR_NUM_VAL, ROUND_NUM_VAL)} | "
            f"{_fmt(level.half_h, CHAR_NUM_VAL, ROUND_NUM_VAL)} | "
            f"{_fmt(level.half_d, CHAR_NUM_VAL, ROUND_NUM_VAL)} | "
            f"{_fmt(level.half_y, CHAR_NUM_VAL, ROUND_NUM_VAL)}")
    lines.append("")

    return "\n".join(lines)


# --- Writing the temperature-averaged parameters ---
def _section_temperature_aver_results(data: object, temperature: float) -> str:

    lines = []
    lines.append("#" * 242)
    lines.append(f"The average transmission probabilities (P), reaction rate (k), and tunneling half-lives (t_1/2) "
        f"for T = {temperature:^{CHAR_NUM_T}.{ROUND_NUM_T}f} K".center(242))
    lines.append("#" * 242)
    lines.append("")

    header = (f"{'Temperature':^{CHAR_NUM_VAL}} | {'T, K':^{CHAR_NUM_VAL}} | {'P':^{CHAR_NUM_VAL}} | {'k, s-1':^{CHAR_NUM_VAL}} | "
        f"{'t_1/2, seconds':^{CHAR_NUM_VAL}} | {'t_1/2, hours':^{CHAR_NUM_VAL}} | "
        f"{'t_1/2, days':^{CHAR_NUM_VAL}} | {'t_1/2, years':^{CHAR_NUM_VAL}}")
    lines.append(header)
    lines.append("-" * 242)

    lines.append(f"{'averaged data':^{CHAR_NUM_VAL}} | "
        f"{temperature:^{CHAR_NUM_VAL}.{ROUND_NUM_T}f} | "
        f"{_fmt(data.transmission_probability, CHAR_NUM_VAL, ROUND_NUM_VAL)} | "
        f"{_fmt(data.reaction_rate, CHAR_NUM_VAL, ROUND_NUM_VAL)} | "
        f"{_fmt(data.half_s, CHAR_NUM_VAL, ROUND_NUM_VAL)} | "
        f"{_fmt(data.half_h, CHAR_NUM_VAL, ROUND_NUM_VAL)} | "
        f"{_fmt(data.half_d, CHAR_NUM_VAL, ROUND_NUM_VAL)} | "
        f"{_fmt(data.half_y, CHAR_NUM_VAL, ROUND_NUM_VAL)}")
    lines.append("")

    return "\n".join(lines)


# --- Writing the Arrhenius plot data (tunneling only!) ---
def _section_arrhenius(data: object) -> str:
    lines = []

    lines.append("#" * 90)
    lines.append("Arrhenius plot data (tunneling only!)".center(90))
    lines.append("#" * 90)
    lines.append("")

    header = f"{'T, K':^{CHAR_NUM_VAL}} | {'1/T, K':^{CHAR_NUM_VAL}} | {'ln(k), s-1':^{CHAR_NUM_VAL}}"
    lines.append(header)
    lines.append("-" * 90)

    for row in data:
        lines.append(f"{_fmt(row.arrhenius_temperature, CHAR_NUM_VAL, ROUND_NUM_VAL)} | "
            f"{_fmt(row.arrhenius_temperature_inv, CHAR_NUM_VAL, ROUND_NUM_VAL)} | "
            f"{_fmt(row.log_reaction_rate, CHAR_NUM_VAL, ROUND_NUM_VAL)}")
    lines.append("")

    return "\n".join(lines)


# --- Writing the interpolated and ZPVE-corrected IRC-curve ---
def _section_irc_interpol(data: object) -> str:
    lines = []

    lines.append("#" * 101)
    lines.append(
        "The ZPVE-corrected IRC-surface approximated by a cubic spline "
        f"after offsetting and potential scaling".center(101))
    lines.append("#" * 101)
    lines.append("")

    header = f"{'IRC, sqrt(amu) * bohr':^{CHAR_NUM_VAL}} | {'Energy, kJ mol-1':^{CHAR_NUM_VAL}}"
    lines.append(header)
    lines.append("-" * 60)

    for row in data:
        lines.append(f"{row['irc_value']:^{CHAR_NUM_VAL}.{ROUND_NUM_VAL}f} | "
            f"{row['energy_value'] / KJ_MOL_M1_TO_HARTREE:^{CHAR_NUM_VAL}.{ROUND_NUM_VAL}f}")
    lines.append("")

    return "\n".join(lines)


# Writing the offset, defined as E(IRC_min) + ZPVE(IRC_min) − E0 − ZPVE0
def _section_offset(offset: float) -> str:
    
    lines = []

    lines.append("#" * 80)
    lines.append("Offset is defined as E(IRC_min) + ZPVE(IRC_min) − E0 − ZPVE0")
    lines.append("#" * 80)
    lines.append("")
    lines.append(f"{f'offset = {offset / KJ_MOL_M1_TO_HARTREE:^{CHAR_NUM_OFFSET}.{ROUND_NUM_VAL}f} kJ mol-1':^80}")
    lines.append("")

    return "\n".join(lines)


def build_report(data_level: object, data_temperature_averaged: object, temperature: float, data_arrhenius: object,
    data_irc: object, offset: float) -> str:
    
    parts = []

    parts.append(_section_level_results_1(data_level))
    parts.append(_section_level_results_2(data_level))
    parts.append("")
    parts.append(_section_temperature_aver_results(data_temperature_averaged, temperature))
    parts.append("")
    parts.append(_section_arrhenius(data_arrhenius))
    parts.append("")
    parts.append(_section_irc_interpol(data_irc))
    parts.append(_section_offset(offset))

    return "\n\n".join(parts)


# --- Opening a new file in a write mode and writing the data ---
def write_output(filename: str | Path, report: str) -> None:
    path = Path(filename)
    path.write_text(report, encoding="utf-8")