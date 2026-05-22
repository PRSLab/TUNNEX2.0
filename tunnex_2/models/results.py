from dataclasses import dataclass


@dataclass
class LevelResult:
    vib_index: int  # Vibrational level index [dimentionless]
    vib_energy: float  # Vibrational level energy [Hartree]
    turning_point_left: float | None  # Left turning point IRC value [amu^0.5 * bohr]
    turning_point_right: float | None  # Right turning point IRC value [amu^0.5 * bohr]
    wkb: float  # WKB integral [Hartree^0.5 * amu * bohr]
    transmission_probability: float  # Transmission probability [dimentionless]
    reaction_rate: float  # Reaction rate in seconds-1
    half_s: float  # Half-life in seconds
    half_h: float  # Half-life in hours
    half_d: float  # Half-life in days
    half_y: float  # Half-life in years


@dataclass
class TemperatureAverResult:
    transmission_probability: float  # Transmission probability [dimentionless]
    reaction_rate: float  # Reaction rate in seconds-1
    half_s: float  # Half-life in seconds
    half_h: float  # Half-life in hours
    half_d: float  # Half-life in days
    half_y: float  # Half-life in years


@dataclass
class ArrheniusResult:
    arrhenius_temperature: float  # Temperature in K
    arrhenius_temperature_inv: float  # Inverse temperature in K-1
    log_reaction_rate: float  # LN of Reaction rate in seconds-1
