# --- Modules ---
import numpy as np
from pathlib import Path


# --- Settings (how to run software) ---
COMMAND_FILE_PATH = Path("command_settings.txt") # Where to store the command line (Path)
RUN_SOFTWARE_KEY_MARKER = False # The key defining whether to compute the files or not
SLURM_MODE_KEY = False # SLURM option used by run_software (run SLURM or not)
ENVR = None # environment used by run_software


# --- Settings (how to scale the potential) ---
POTENTIAL_SCAL_FACTOR_DEFAULT = 1.0 # Default potential scaling factor


# --- Settings (attempt frequencies and correlations) ---
CORR_ANALYSIS_BOOL = True # Useful tool for checking the correlation
# between the reaction coordinate and reactant frequencies.
# It is helpful to determine the attempt frequency.
# This parameter determines whether to print it or not
PICK_MAX_FREQ_FLAG = False # It determines which frequency is automatically set as an attempt frequency:
# one with the maximum IRC correlation (True) or the correlation-weighted average value (False, default)


# --- Settings (temperature averaging of the tunneling probability) ---
TUNN_PROB_AVERAGING_ALLOWED_MODES = {1 : "finite_sum", 2 : "infinite_sum", 3 : "integral"}
# this list defines the way how the program will do the temperature averaging
# of the tunneling probability across the energy levels of a molecule


# --- Constants (IRC module) --- 
CM_M1_TO_HARTREE = 1 / 219474.63136320  # cm-1 to Hartree
KJ_MOL_M1_TO_HARTREE = 1 / 2625.49963948  # kJ mol-1 to Hartree
CM_M1_TO_HZ = 2.99793e+10  # cm-1 to Hz (or sec-1)
HARTREE_TO_J = 4.3597447222e-18  # Hartree to J
BOHR_RADIUS_TO_M = 5.29177210544e-11  # Bohr radius to meter
ANGSTROM_TO_BOHR_RADIUS = 1.8897259886  # Angstrom to Bohr radius
AMU_TO_KG = 1.66053906893e-27  # Atomic mass unit (amu, also known as Dalton) to kg
LIGHT_SPEED = 299792458  # light speed in m s-1


# --- Conversion factors (IRC module) ---
CONVERSION_FACTOR_HESSIAN_VALUES = 1 / (2 * np.pi * LIGHT_SPEED * 100) * (
    np.sqrt(HARTREE_TO_J / (AMU_TO_KG * BOHR_RADIUS_TO_M ** 2)))
# A conversion factor between the sqrt(hessian_values) and wavenumbers in cm-1
CONVERSION_FACTOR_F_STAR_ECKART = CM_M1_TO_HZ ** 2 * AMU_TO_KG * BOHR_RADIUS_TO_M ** 2 / HARTREE_TO_J
# A conversion factor between the F_star (an intermediate variable connecting the TS imaginary frequency (cm-1)
# and the width of the Eckart potential). It converts F_star from (cm-1)^2 to Hartree / (amu * bohr^2)


# --- Constants (QMT module) ---
REVELO = 1822.8886259874  # A unified atomic mass unit (Dalton) in electron mass (atomic units)
K_B = 3.166811563e-6  # Boltzmann constant in Hartree K-1
GRID_SIZE = 101  # Size of a grid to store the interpolated IRC-curves
INTEGRAL_RTOL = 1e-4 # The relative tolerance of the scipy.integrate.quad integral computations
HOUR = 1 / 3600  # 1 second in hours
DAY = 1 / 86400  # 1 second in days
YEAR = 1 / 31536000  # 1 second in years