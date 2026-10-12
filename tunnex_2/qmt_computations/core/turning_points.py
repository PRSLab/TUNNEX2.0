# --- Modules ---
import math
from scipy.optimize import brentq


# --- Defining the function which computes the turning points ---
def find_turning_points(level_energy: float, potential: object, a_tol: float = 1e-12) -> dict:
    
    # --- Defining the coordinates and energies ---
    irc_reagent = potential.irc_x_min
    energy_reagent = potential(irc_reagent)
    irc_product = potential.irc_x_max
    energy_product = potential(irc_product)
    irc_ts = potential.irc_x_ts
    energy_ts = potential(irc_ts)

    # --- Checking the negative level energies.
    # Because the reagent's energy is set to be zero after offset,
    # and non-positive vibrational energy levels
    # fail the earlier check, this error should never occur ---
    if level_energy < energy_reagent:
        raise ValueError(f"The energy level ({level_energy}) is below the reagent's energy ({energy_reagent}). "
            f"Please check your data.")

    # --- The energy level is exactly at the barrier ---
    if math.isclose(level_energy, energy_ts, abs_tol=a_tol):
        result = {"left": None, "right": None, "regime": "OVER THE BARRIER", "OVER_STATUS": "EXACT_MATCH"}
        
        return result
    
    # --- The energy level is above the barrier. The resulting integral should be negative ---
    if level_energy > energy_ts:

        # --- Reflecting the level energy over the barrier height,
        # but not allowing it to be negative ---
        level_energy_corrected = max(0, 2 * energy_ts - level_energy)
        func_target_corr = lambda x: potential(x) - level_energy_corrected

        # --- For a quasiendothermic reaction, computing the maximum possible integral
        # and then scale it linearly if level_energy > 2 * energy_ts - energy_product ---
        if level_energy_corrected < energy_product:
            func_target_prod = lambda x: potential(x) - energy_product
            quasi_turning_point_left = brentq(func_target_prod, irc_reagent, irc_ts)
            quasi_turning_point_right = brentq(func_target_prod, irc_ts, irc_product)
            
            # --- Zero protection procedure ---
            energy_sum = 2 * energy_ts - energy_product
            if math.isclose(energy_sum, 0.0, abs_tol=a_tol):
                energy_sum = a_tol
            
            # --- Estimating the scaling factor (it cannot be smaller than 1) ---
            scale_factor = max(1, abs(level_energy / (energy_sum)))
            result = {"left": None, "right": None, "regime": "OVER THE BARRIER",
                      "OVER_STATUS": "QUASI_ENDOTHERMIC_REACTION",
                      "quasi_left": quasi_turning_point_left, "quasi_right": quasi_turning_point_right,
                      "QUASI_LEVEL": energy_product, "SCALE_FACTOR": scale_factor}
            
            return result

        # --- For a quasinormal reaction, computing the maximum possible integral
        # and then scale it linearly if level_energy > 2 * energy_ts - energy_reagent ---
        quasi_turning_point_left = brentq(func_target_corr, irc_reagent, irc_ts)
        quasi_turning_point_right = brentq(func_target_corr, irc_ts, irc_product)

        # --- Zero protection procedure ---
        energy_sum = 2 * energy_ts - energy_reagent
        if math.isclose(energy_sum, 0.0, abs_tol=a_tol):
            energy_sum = a_tol
        
        # --- Estimating the scaling factor (it cannot be smaller than 1) ---
        scale_factor = max(1, abs(level_energy / (energy_sum)))
        result = {"left": None, "right": None, "regime": "OVER THE BARRIER", "OVER_STATUS": "QUASI_NORMAL",
                  "quasi_left": quasi_turning_point_left, "quasi_right": quasi_turning_point_right,
                  "QUASI_LEVEL": level_energy_corrected, "SCALE_FACTOR": scale_factor}
        
        return result

    func_target = lambda x: potential(x) - level_energy

    turning_point_left = brentq(func_target, irc_reagent, irc_ts)

    # --- The energy is bigger than the reagent's energy, but smaller than the product's energy.
    # From this level, tunneling is not possible, but the vibrational excitation can make it possible ---
    if level_energy < energy_product:
        result = {"left": turning_point_left, "right": None, "regime": "ENDOTHERMIC REACTION"}
        
        return result

    # --- In a normal case, two turning points should exist ---
    turning_point_right = brentq(func_target, irc_ts, irc_product)
    result = {"left": turning_point_left, "right": turning_point_right, "regime": "NORMAL"}
    
    return result