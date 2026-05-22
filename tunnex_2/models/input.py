from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class InputParams:
    freq: float  # The attempt frequency [freq, in cm-1]
    T: float  # Temperature [T; K]
    E0: float  # The reagent's electronic energy [E0, Hartree]
    ZPVE0: float  # The reagent's zero-point vibrational Energy [ZPVE0, Hartree]
    potential_scaling_factor: (
        float  # Potential_scaling_factor [potential_scaling_factor]
    )
    the_upper_level_to_compute: int  # The_upper_level_to_compute [N]
    T_arrhenius_min: float  # Minimum temperature [T; K] for an Arrhenius plot
    T_arrhenius_max: float  # Maximum temperature [T; K] for an Arrhenius plot
    T_arrhenius_step: float  # Step [T; K] for an Arrhenius plot


@dataclass
class IRCData:
    E_x: np.ndarray  # IRC value [IRC, amu^0.5 * bohr] - E
    E_y: Any  # Electronic energy [E; Hartree] value
    ZPVE_x: Any  # IRC value [IRC, amu^0.5 * bohr] - ZPVE
    ZPVE_y: Any  # Zero-point vibrational Energy [ZPVE; Hartree] value