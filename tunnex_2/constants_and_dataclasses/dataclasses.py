# --- Modules ---
from dataclasses import dataclass
import numpy as np
from pathlib import Path


# --- Dataclasses (IRC module) ---
# Input data
@dataclass # It stores the initial configurations for irc_computations
class FindIRCConfig:
    # Calculation settings
    prog_mode: str = "gaussian" # The mode to compute the IRC path (IRC module)
    proj_freq: bool = True # Should the projected frequencies be computed or not
    calc_all: bool = True # Should a Hessian be computed analytically every IRC step (only Gaussian)
    hybrid_mode: bool = False # Should the program use the hybrid mode or not
    eckart: bool = False # Should the program use Eckart potential or not
    qmt_module: bool = True # Should the program use also the QMT module or not
    hess_parsing_flag: bool = False # Should the program parse Hessians and compute frequencies or parse frequencies directly
    prob_aver_mode_qmt: str = "finite_sum" # The mode to perform the temperature averaging of computed probabilities (QMT module)
    # Input files
    ts_input_file: str | Path | None = None # The input file for the transition state
    react_input_file: str | Path | None = None # The input file for the reactant (Eckart only)
    prod_input_file: str | Path | None = None # The input file for the product (Eckart only)
    # Already computed files
    ts_computed_file: str | Path | None = None # The computed file for the transition state (optimization+frequencies)
    irc_computed_file: str | Path | None = None # The computed file of the IRC path (IRC+energy)
    react_computed_file: str | Path | None = None # The computed file for the reactant (optimization+frequencies)
    prod_computed_file: str | Path | None = None # The computed file for the product (optimization+frequencies)
    # Tunnex input file
    tunnex_input: str | Path | None = None # The input file for the QMT module


# Results
@dataclass # It stores the IRC data
class IRCData:
    electronic_energies: list[tuple[float, float]]  # The list of electronic energies (IRC, Energy)
    zpve_energies_forward: list[tuple[float, float]] | None # The list of forward ZPVE energies (IRC, ZPVE)
    zpve_energies_reverse: list[tuple[float, float]] | None # The list of reverse ZPVE energies (IRC, ZPVE)

@dataclass # It stores the electronic energies and ZPVEs for the transition state, reactant, and product
class EnergyPointsData:
    transition_state_el_energy: tuple[float, float] # The tuple for TS electronic energy (IRC, Energy)
    transition_state_zpve: tuple[float, float] # The tuple for TS ZPVE (IRC, ZPVE)
    reactant_el_energy: tuple[float, float] # The tuple for reactant electronic energy (IRC, Energy)
    reactant_zpve: tuple[float, float] # The tuple for reactant ZPVE (IRC, ZPVE)
    product_el_energy: tuple[float, float] # The tuple for product electronic energy (IRC, Energy)
    product_zpve: tuple[float, float] # The tuple for product ZPVE (IRC, ZPVE)

@dataclass # It stores the attempt frequencies and correlations
class AttemptFreq:
    att_freq_react: float # The attempt frequency for reactant [freq, cm-1]
    att_freq_react_corr: list[list[float, float]] # The tuple of frequencies for reactant [freq, correlation]
    att_freq_prod: float # The attempt frequency for product [freq, cm-1]
    att_freq_prod_corr: list[list[float, float]] # The tuple of frequencies for product [freq, correlation]

@dataclass # It stores the important import parameters for the TUNNEX 2.0 computations
# (it is also the input data for the QMT module)
class TunnexInputSettings:
    freq: float  # The attempt frequency [freq, in cm-1]
    E0: float # The reagent's electronic energy [E0, Hartree]
    ZPVE0: float # The reagent's zero-point vibrational Energy [ZPVE0, Hartree]
    T: float = 10 # Temperature [T; K]
    potential_scaling_factor: float = 1.0 # Potential_scaling_factor [potential_scaling_factor]
    number_of_levels: int = 10 # The upper level to compute [N]
    T_arrhenius_min: float = 5 # Minimum temperature [T; K] for an Arrhenius plot
    T_arrhenius_max: float = 1000 # Maximum temperature [T; K] for an Arrhenius plot
    T_step: float = 5 # Step [T; K] for an Arrhenius plot
    prob_aver_mode_qmt: int = 1 # The mode to perform the temperature averaging of computed probabilities (QMT module)
    # prob_aver_mode_qmt specifies the method used for temperature averaging of tunneling probabilities
    # it should be set to one of the following number values: 1 = finite_sum, 2 = infinite_sum, 3 = integral


# --- Dataclasses (QMT module) ---
# Input data
@dataclass
class IRCDataQMT: # It stores the IRC input data
    E_x: np.ndarray  # IRC value [IRC, amu^0.5 * bohr] - for E_y
    E_y: np.ndarray  # Electronic energy [E; Hartree] value
    ZPVE_x: np.ndarray  # IRC value [IRC, amu^0.5 * bohr] - for ZPVE_y
    ZPVE_y: np.ndarray  # Zero-point vibrational Energy [ZPVE; Hartree] value


# Results
@dataclass
class LevelResult: # It stores the QMT data for different vibrational levels
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
class TemperatureAverResult: # It stores the temeprature averaged data
    transmission_probability: float  # Transmission probability [dimentionless]
    reaction_rate: float  # Reaction rate in seconds-1
    half_s: float  # Half-life in seconds
    half_h: float  # Half-life in hours
    half_d: float  # Half-life in days
    half_y: float  # Half-life in years

@dataclass
class ArrheniusResult:  # It stores the Arrhenius plot results
    arrhenius_temperature: float  # Temperature in K
    arrhenius_temperature_inv: float  # Inverse temperature in K-1
    log_reaction_rate: float  # LN of Reaction rate in seconds-1