from tunnex_2.constants import revelo  # type: ignore
import numpy as np
from scipy.integrate import quad


# --- Defining the function to compute the WKB-integral ---
def compute_wkb(turning_points: dict, level: float, potential: object) -> float:
    if turning_points["regime"] != 'NORMAL':
        if turning_points["regime"] == 'OVER THE BARRIER':
            return 0.0  # The energy level is over the barrier, thus, returning zero
        return (
            np.inf
        )  # The only option here is tunneling not possible, thus, returning "-inf"

    def integrand(x: float) -> float:
        return np.sqrt(np.maximum(0, 2 * revelo * (potential(x) - level)))

    # Implementation of the wkb_function function implies that V(x) - E >= 0 for a given x.
    # However, for turning points, the binary search could converge to V(x) slightly smaller than E
    # (the default turning point search tolarance is tol=1e-12)
    # Thus, zero is returned for the cases where wkb_function_square <= 0

    result = quad(
        integrand,
        turning_points["left"],
        turning_points["right"],
        full_output=1,
    )
    # The default tolerance of integration is tol=1.49e-8. However, the other limiting factor,
    # the maximum number of steps is only max_iter=50. It is often exceeded, but I turned off the notifications about this
    return result[0]
