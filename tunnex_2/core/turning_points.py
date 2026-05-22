from scipy.optimize import brentq


# --- Defining the function which computes the turning points ---
def find_turning_points(level_energy: float, potential: object) -> tuple:
    energy_reagent = potential(potential.irc_x_min)
    energy_product = potential(potential.irc_x_max)
    energy_ts = potential(potential.irc_x_ts)

    if level_energy < energy_reagent:
        raise ValueError(
            "The energy level is below the reagent's energy. Please check your data"
        )
    # Because the reagent's energy is set to be zero after offset, and vibrational energy levels
    # are positive, this error means the problem with your data

    if level_energy < energy_product:
        func_target = lambda x: potential(x) - level_energy
        turning_point_left = brentq(
            func_target, potential.irc_x_min, potential.irc_x_ts
        )
        return {
            "left": turning_point_left,
            "right": None,
            "regime": "ENDOTHERMIC REACTION",
        }

    # It means that the energy is not smaller than the reagent's
    # energy, but smaller than the product's energy. From this level, tunneling is not possible, but the vibrational excitation can
    # make it possible

    if level_energy >= energy_ts:
        return {"left": None, "right": None, "regime": "OVER THE BARRIER"}
    # The vibrational level's energy is bigger than the energy of the transition state in this case.
    # The resulting integral should be zero

    # In other cases, we should have two turning points
    func_target = lambda x: potential(x) - level_energy

    turning_point_left = brentq(func_target, potential.irc_x_min, potential.irc_x_ts)
    turning_point_right = brentq(func_target, potential.irc_x_ts, potential.irc_x_max)

    return {
        "left": turning_point_left,
        "right": turning_point_right,
        "regime": "NORMAL",
    }
