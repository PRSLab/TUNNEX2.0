# --- Modules ---
import math # To be sure that in some cases we are working only with numbers, not arrays
import numpy as np

from tunnex_2.constants_and_dataclasses.constants_and_settings import CONVERSION_FACTOR_F_STAR_ECKART # type: ignore
from tunnex_2.constants_and_dataclasses.dataclasses import IRCData # type: ignore


# --- Defining the function to compute the Eckart potential for a given coordinate ---
def _eckart_potential_value(coordinate_sqrt_amu_bohr: float, A_Hartree: float, B_Hartree: float, L_sqrt_amu_bohr: float) -> float:

    # --- Eckart potential intermediate function ---
    y = - math.exp(2 * math.pi * coordinate_sqrt_amu_bohr / L_sqrt_amu_bohr)

    # --- Eckart potential ---
    potential = - A_Hartree * y / (1 - y) - B_Hartree * y / (1 - y) ** 2
    
    return potential
# --- For formulas please check: Johnston, H. S., & Heicklen, J. (1962). J. Phys. Chem., 66(3), 532-533 ---


# --- Defining the function to compute the Eckart potential parameters  ---
def eckart_potential_parameters(ts_freq_cm_m1: float, dV1_Hartree: float, dV2_Hartree: float) -> tuple[float, float, float]:

    if ts_freq_cm_m1 >= 0:
            raise ValueError(f"Expected negative transition state frequency to compute Eckart potential. " 
                             f"Got {ts_freq_cm_m1:.3g} cm-1.")
    
    if dV1_Hartree <= 0 or dV2_Hartree <= 0:
        raise ValueError(f"Expected both positive barrier ('dV1') and reverse barrier ('dV2') to compute Eckart potential. "
                         f"Got {dV1_Hartree:.3g} and {dV2_Hartree:.3g} Hartrees, respectively.")

    # --- Eckart potential second derivative at the maximum ---
    F_star_cm_m2 = - (2 * math.pi * ts_freq_cm_m1) ** 2
    # The reference implementation mass-scales F_star, i.e. F_star = - mass * (2 * math.pi * ts_freq_cm_m1) ** 2 in a referense.
    # We intentionally omit that step because the IRC coordinate used here is expressed in sqrt(amu) * bohr rather than bohr.
    # Reference: [Johnston, H. S., & Heicklen, J. (1962). J. Phys. Chem., 66(3), 532-533 / equation number 10].
    F_star_Hartree__amu_m1__bohr_m2 = CONVERSION_FACTOR_F_STAR_ECKART * F_star_cm_m2
    # Convert cm^-2 to Hartree amu^-1 bohr^-2

    # --- Eckart barrier width ---
    denominator = (1 / math.sqrt(dV1_Hartree) + 1 / math.sqrt(dV2_Hartree))
    L_sqrt_amu_bohr = 2 * math.pi * math.sqrt((- 2 / F_star_Hartree__amu_m1__bohr_m2)) / denominator
    
    # --- Eckart asymmetry parameter ---
    A_Hartree = dV1_Hartree - dV2_Hartree
    
    # --- Eckart barrier-shape parameter ---
    B_Hartree = (math.sqrt(dV1_Hartree) + math.sqrt(dV2_Hartree)) ** 2

    return A_Hartree, B_Hartree, L_sqrt_amu_bohr
# --- For formulas please check: Johnston, H. S., & Heicklen, J. (1962). J. Phys. Chem., 66(3), 532-533 ---


# --- Defining the function to store the computed values as grid ---
def eckart_potential_data(A_Hartree: float, B_Hartree: float, L_sqrt_amu_bohr: float,
    border_factor: float = 2.7, grid_size: int = 101) -> IRCData:

    # --- Checking if the border_factor and L_sqrt_amu_bohr are valid ---
    if border_factor <= 0:
        raise ValueError(f"Expected a positive border factor. Got {border_factor}.")
    if L_sqrt_amu_bohr <= 0:
        raise ValueError(f"Expected a positive Eckart barrier width. Got {L_sqrt_amu_bohr:.3g} sqrt(amu) * bohr.")
    if B_Hartree <= 0:
            raise ValueError(f"Expected a positive Eckart barrier-shape parameter. Got {B_Hartree:.3g} Hartree.")

    grid_limit = border_factor * L_sqrt_amu_bohr

    # Quick numerical check showed that for border_factor of 2.7 the values (V(-grid_limit) - V(-inf)) / dV1
    # and (V(+grid_limit) - V(+inf)) / dV1 are smaller than 0.1 % and it should be valid approximately up to dV2/dV1 ~ 10000).
    # Here dV1 and dV2 are forward and reverse barrier heights, respectively.
    # Setting border_factor = 2.7 as a default value which should cover almost all needs of chemistry.

    # Here are the results of the dependence of left and right border factors for the 0.1 % tolerance
    # on the dV2/dV1 ratio (this ratio cannot be negative):
    # No dV2/dV1 border_factor_left border_factor_right
    # 1   0e+00        -1.21               0.55
    # 2   1e-02        -1.23               0.86
    # 3   1e-01        -1.26               1.07
    # 4   1e+00        -1.32               1.32
    # 5   1e+01        -1.44               1.62
    # 6   1e+02        -1.60               1.96
    # 7   1e+03        -1.77               2.31
    # 8   1e+04        -1.95               2.68     (2.7 is a default value)
    # 9   1e+05        -2.13               3.05
    # 10  1e+06        -2.31               3.41
    
    # --- Checking if the grid_size is not smaller than 3 and odd (odd grid_size is needed to include IRC = 0.0) ---
    if grid_size < 3:
        raise ValueError(f"Expected grid_size >= 3. Got {grid_size}.")  
    if grid_size % 2 == 0:
        raise ValueError(f"Expected an odd grid_size. Got {grid_size}.")

    # --- Creating the IRC grid and computing Eckart potential and shifting it
    # so that the Eckart barrier maximum is at IRC = 0.0 ---
    potential_max_irc = (L_sqrt_amu_bohr / (2 * math.pi)) * math.log((B_Hartree + A_Hartree) / (B_Hartree - A_Hartree))
    irc_grid = list(np.linspace(- grid_limit, grid_limit, grid_size))
    data_el_energy = [(irc, _eckart_potential_value(irc + potential_max_irc, A_Hartree, B_Hartree, L_sqrt_amu_bohr))
                      for irc in irc_grid]
    # Please note that different variables (irc and irc + potential_max_irc) are used intentionally for the x-axis shift
    # to make the potential maximum value at IRC = 0.0

    return IRCData(electronic_energies=data_el_energy, zpve_energies_forward=None, zpve_energies_reverse=None)
# --- For formulas please check: Johnston, H. S., & Heicklen, J. (1962). J. Phys. Chem., 66(3), 532-533 ---