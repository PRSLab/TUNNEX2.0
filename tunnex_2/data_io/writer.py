from tunnex_2.constants import Hartree_to_kJ_mol_m1  # type: ignore
from pathlib import Path
import numpy as np


# --- Parameters of the output-file format: number of characters per section (n) and number of digits for rounding (k) in general,
# for rounding a temperature (l), and for characters per section for temperature and offset (m and o), respectively ---
n, k, l, m, o = 28, 6, 2, 4, 8


# --- Opening a new file in a 'w' (write) mode and writing the data ---
def write_output(path: Path, report: str) -> None:
    path.write_text(report, encoding="utf-8")


# --- Checking the format of the input-value ---
def _fmt(val: float, n: int, k: int, is_int=False) -> object:
    if is_int:
        return f"{val:^{n}.2f}"
    elif val is None:
        return f"{'No turning point':^{n}}"
    elif np.isinf(val):
        return f"{'+inf':^{n}}"
    else:
        return f"{val:^{n}.{k}e}"


# --- Printing the vibrational level indexes and relative parameters (part 1) ---
def _section_level_results_1(data: list) -> str:

    lines = []
    lines.append("#" * 165)
    lines.append(
        "The vibrational level indexes, energies, corresponding turning points (tp), and WKB integrals".center(
            165
        )
    )
    lines.append("#" * 165)
    lines.append("")

    header = (
        f"{'N':^{n}} | {'Vibrational energy, kJ mol-1':^{n}} | "
        f"{'tp (left), sqrt(amu) * bohr':^{n}} | {'tp (right), sqrt(amu) * bohr':^{n}} | {'WKB integral, sqrt(Hartree) * amu * bohr':^{n}}"
    )
    lines.append(header)
    lines.append("-" * 165)

    for level in data:

        lines.append(
            f"{level.vib_index:^{n}} | "
            f"{level.vib_energy * Hartree_to_kJ_mol_m1:^{n}.{k}e} | "
            f"{_fmt(level.turning_point_left, n, k)} | "
            f"{_fmt(level.turning_point_right, n, k)} | "
            f"{_fmt(level.wkb, n, k)}"
        )

    lines.append("")

    return "\n".join(lines)


# --- Printing the vibrational level indexes and relative parameters (part 2) ---
def _section_level_results_2(data: list) -> str:

    lines = []
    lines.append("#" * 242)
    lines.append(
        "The vibrational level indexes, energies, corresponding transmission probabilities (P), reaction rate (k), and tunneling half-lives (t_1/2)".center(
            242
        )
    )
    lines.append("#" * 242)
    lines.append("")

    header = (
        f"{'N':^{n}} | {'Vibrational energy, kJ mol-1':^{n}} | "
        f"{'P':^{n}} | {'k, s-1':^{n}} | "
        f"{'t_1/2, seconds':^{n}} | {'t_1/2, hours':^{n}} | "
        f"{'t_1/2, days':^{n}} | {'t_1/2, years':^{n}}"
    )
    lines.append(header)
    lines.append("-" * 242)

    for level in data:

        lines.append(
            f"{level.vib_index:^{n}} | "
            f"{level.vib_energy * Hartree_to_kJ_mol_m1:^{n}.{k}e} | "
            f"{_fmt(level.transmission_probability, n, k)} | "
            f"{_fmt(level.reaction_rate, n, k)} | "
            f"{_fmt(level.half_s, n, k)} | "
            f"{_fmt(level.half_h, n, k)} | "
            f"{_fmt(level.half_d, n, k)} | "
            f"{_fmt(level.half_y, n, k)}"
        )

    lines.append("")

    return "\n".join(lines)


# --- Printing the temperature-averaged parameters ---
def _section_temperature_aver_results(data: list, temperature: int) -> str:

    lines = []
    lines.append("#" * 217)
    lines.append(
        f"The average transmission probabilities (P), reaction rate (k), and tunneling half-lives (t_1/2) for T = {temperature:^{m}.{l}f} K".center(
            217
        )
    )
    lines.append("#" * 217)
    lines.append("")

    header = (
        f"{'T, K':^{n}} | {'P':^{n}} | {'k, s-1':^{n}} | "
        f"{'t_1/2, seconds':^{n}} | {'t_1/2, hours':^{n}} | "
        f"{'t_1/2, days':^{n}} | {'t_1/2, years':^{n}}"
    )
    lines.append(header)
    lines.append("-" * 217)

    lines.append(
        f"{temperature:^{n}.{l}f} | "
        f"{_fmt(data.transmission_probability, n, k)} | "
        f"{_fmt(data.reaction_rate, n, k)} | "
        f"{_fmt(data.half_s, n, k)} | "
        f"{_fmt(data.half_h, n, k)} | "
        f"{_fmt(data.half_d, n, k)} | "
        f"{_fmt(data.half_y, n, k)}"
    )
    lines.append("")

    return "\n".join(lines)


# --- Printing the Arrhenius plot data (tunneling only!) ---
def _section_arrhenius(data: list) -> str:
    lines = []

    lines.append("#" * 90)
    lines.append("Arrhenius plot data (tunneling only!)".center(90))
    lines.append("#" * 90)
    lines.append("")

    header = f"{'T, K':^{n}} | {'1/T, K':^{n}} | {'ln(k), s-1':^{n}}"
    lines.append(header)
    lines.append("-" * 90)

    for row in data:
        lines.append(
            f"{_fmt(row.arrhenius_temperature, n, k, is_int=True)} | "
            f"{_fmt(row.arrhenius_temperature_inv, n, k)} | "
            f"{_fmt(row.log_reaction_rate, n, k)}"
        )

    lines.append("")

    return "\n".join(lines)


# --- Printing the interpolated and ZPVE-corrected IRC-curve ---
def _section_irc_interpol(data: object) -> str:
    lines = []

    lines.append("#" * 101)
    lines.append(
        "The ZPVE-corrected IRC-surface approximated by a cubic spline after offsetting and potential scaling".center(
            101
        )
    )
    lines.append("#" * 101)
    lines.append("")

    header = f"{'IRC, sqrt(amu) * bohr':^{n}} | {'Energy, kJ mol-1':^{n}}"
    lines.append(header)
    lines.append("-" * 60)

    for row in data:
        lines.append(
            f"{row['irc_value']:^{n}.{k}f} | {row['energy_value'] * Hartree_to_kJ_mol_m1:^{n}.{k}f}"
        )

    lines.append("")

    return "\n".join(lines)


# Printing the offset, defined as E(IRC_min) + ZPVE(IRC_min) − E0 − ZPVE0
def _section_offset(offset: float) -> str:
    lines = []

    lines.append("#" * 80)
    lines.append("Offset is defined as E(IRC_min) + ZPVE(IRC_min) − E0 − ZPVE0")
    lines.append("#" * 80)
    lines.append("")
    lines.append(
        f"{f'offset = {offset * Hartree_to_kJ_mol_m1:^{o}.{k}f} kJ mol-1':^80}"
    )
    lines.append("")

    return "\n".join(lines)


def build_report(
    data_level: object,
    data_temperature_averaged: object,
    temperature: float,
    data_arrhenius: object,
    data_irc: object,
    offset: float,
) -> str:
    parts = []

    parts.append(_section_level_results_1(data_level))
    parts.append(_section_level_results_2(data_level))
    parts.append("")
    parts.append(
        _section_temperature_aver_results(data_temperature_averaged, temperature)
    )
    parts.append("")
    parts.append(_section_arrhenius(data_arrhenius))
    parts.append("")
    parts.append(_section_irc_interpol(data_irc))
    parts.append(_section_offset(offset))

    return "\n\n".join(parts)
