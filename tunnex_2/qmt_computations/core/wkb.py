# --- Modules ---
import numpy as np
from scipy.integrate import quad

from tunnex_2.constants_and_dataclasses.constants_and_settings import INTEGRAL_RTOL, REVELO  # type: ignore

from tunnex_2.irc_computations.ancillary_functions.interface import setup_logger # type: ignore


# --- Defining the logging function ---
logger = setup_logger()


# --- Defining the function for the integration ---
def _integrand_wkb(x: float, potential: object, level_value: float) -> float:
    result = np.sqrt(np.maximum(0, 2 * REVELO * (potential(x) - level_value)))
    return result
    # Implementation of the wkb_function function implies that V(x) - E >= 0 for a given x.
    # However, for turning points, the binary search could converge to V(x) slightly smaller than E
    # (the default turning point search tolerance is tol=2e-12).
    # Thus, negative values are clamped to zero for the cases where V(x) - E < 0.


# --- Defining the function to log the integration problems ---
def int_convergence_warning(int_result: tuple, logger: object, info: str) -> bool:
    if len(int_result) > 3:
        logger.warning("Integration problem occurred (%s): %s | estimated absolute error: %g | integral %g",
        info, int_result[3].strip(), int_result[1], int_result[0])
        return True
    return False


# --- Defining the function to compute the WKB-integral ---
def compute_wkb(turning_points: dict, level: float, potential: object, zero_tol: float = 1e-12) -> float:

    # --- Input data validation ---
    regime = turning_points["regime"]
    
    allowed_regimes = ("NORMAL", "ENDOTHERMIC REACTION", "OVER THE BARRIER")

    if regime not in allowed_regimes:
        raise ValueError(f"Expected tunneling regimes: {allowed_regimes}. Got {regime}.")

    if regime == "OVER THE BARRIER":
          
        try:
            over_status = turning_points["OVER_STATUS"]
        except KeyError:
            raise KeyError(f"The input dictionary must contain key {"OVER_STATUS"}. "
                           f"Got {turning_points.keys()}")
        
        allowed_over_the_barrier = ("QUASI_NORMAL", "QUASI_ENDOTHERMIC_REACTION", "EXACT_MATCH")

        if over_status not in allowed_over_the_barrier:
            raise ValueError(f"Expected over the barrier tunneling regimes: {allowed_over_the_barrier}. "
                             f"Got {over_status}.")
        
        # --- The energy level is exactly the barrier, the resulting integral is zero ---
        if over_status == "EXACT_MATCH":
            return 0.0

        # --- The energy level is over the barrier, the resulting integral is negative ---
        integrand_func_over = lambda x: _integrand_wkb(x, potential, turning_points["QUASI_LEVEL"])
        result = quad(integrand_func_over, turning_points["quasi_left"], turning_points["quasi_right"], full_output=1)
        int_convergence_warning(result, logger, f"WKB for level {turning_points["QUASI_LEVEL"]:.2e} Hartree")
        
        # --- Dealing with the very flat barriers ---
        if np.isclose(result[0], 0.0, atol=zero_tol):
            ans = -turning_points["SCALE_FACTOR"] * zero_tol
        else:
            ans = -turning_points["SCALE_FACTOR"] * result[0]
            # The result gives an integrated value (result[0]) and a bunch of integration details.
            # Storing only the former value
        return ans
    
    # --- regime == "ENDOTHERMIC REACTION, the resulting integral is +inf ---
    elif regime == "ENDOTHERMIC REACTION":
        return np.inf
    
    # --- regime == "ENDOTHERMIC REACTION, the resulting integral is positive ---
    else:
        integrand_func_normal = lambda x: _integrand_wkb(x, potential, level)
        result = quad(integrand_func_normal, turning_points["left"], turning_points["right"],
            epsabs=0.0, epsrel=INTEGRAL_RTOL, full_output=1)
        int_convergence_warning(result, logger, f"WKB for level {level:.2e} Hartree")
        ans = result[0]
        # The result gives an integrated value (result[0]) and a bunch of integration details.
        # Storing only the former value
    return ans
