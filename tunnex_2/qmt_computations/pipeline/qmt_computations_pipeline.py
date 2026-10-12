# --- Modules ---
import numpy as np

from tunnex_2.constants_and_dataclasses.constants_and_settings import ( # type: ignore
    CM_M1_TO_HARTREE,
    GRID_SIZE,
    HOUR,
    DAY,
    YEAR)
from tunnex_2.constants_and_dataclasses.dataclasses import LevelResult, TemperatureAverResult  # type: ignore

from tunnex_2.qmt_computations.core.kinetics import (  # type: ignore
    transmission_prob,
    reaction_rate,
    half_life,
    temperature_averaging_tunneling,
    arrhenius_rate)
from tunnex_2.qmt_computations.core.turning_points import find_turning_points  # type: ignore
from tunnex_2.qmt_computations.core.wkb import compute_wkb  # type: ignore


# --- Defining the pipeline function to compute the QMT half-lives ---
def qmt_computations_pipeline(params: object, potential: object) -> tuple[np.ndarray, list[object], object, list[object]]:
    
    # --- Storing the interpolated ZPVE-corrected IRC  ---
    x_grid = np.linspace(potential.irc_x_min, potential.irc_x_max, GRID_SIZE)
    dtype = np.dtype([("irc_value", float), ("energy_value", float)])
    interpolated_irc_grid = np.array(list(zip(x_grid, potential(x_grid))), dtype=dtype)

    # --- Filling the matrix of the results for the given vibrational levels ---
    level_result_data = []

    for i in range(params.number_of_levels + 1):
        # --- Computing vibrational energy levels ---
        level_energy = (i + 0.5) * params.freq * CM_M1_TO_HARTREE

        # --- Defining the function which computes the turning points ---
        turning_points = find_turning_points(level_energy, potential)

        # --- Computing the WKB-integrals ---
        wkb = compute_wkb(turning_points, level_energy, potential)

        # --- Computing the transmission probabilities ---
        prob = transmission_prob(wkb)

        # --- Computing the reaction rate in seconds-1 ---
        reaction_rate_value = reaction_rate(params.freq, prob)

        # --- Computing the half-life in seconds ---
        half_life_value = half_life(reaction_rate_value)

        # --- Taking the computed vibrational energy level indexes and levels
        # and appending them with the computed parameters ---
        level_result_data.append(LevelResult(i, level_energy,
            turning_points["left"], turning_points["right"], wkb,
            prob, reaction_rate_value,
            half_life_value, half_life_value * HOUR,
            half_life_value * DAY, half_life_value * YEAR))

    # --- Computing the temperature-averaged transmission probabilities ---
    prob_aver = temperature_averaging_tunneling(params.freq, level_result_data, potential,
        params.T, params.prob_aver_mode_qmt)
    
    # --- Computing the temperature-averaged reaction rate in seconds^-1 ---
    reaction_rate_aver = reaction_rate(params.freq, prob_aver)

    # --- Computing the temparature-averaged half-life in seconds ---
    half_life_aver = half_life(reaction_rate_aver)

    # --- Storing the temparature-averaged parameters ---
    results_aver = TemperatureAverResult(prob_aver, reaction_rate_aver, half_life_aver,
    half_life_aver * HOUR, half_life_aver * DAY, half_life_aver * YEAR)

    # --- Storing the arrhenius graph parameters (tunneling only!) ---
    arrhenius_rate_vals = arrhenius_rate(level_result_data, params.freq, params.T_arrhenius_min, params.T_arrhenius_max,
    params.T_step, potential, params.prob_aver_mode_qmt)

    return interpolated_irc_grid, level_result_data, results_aver, arrhenius_rate_vals


# --- Comment ---
# By the convention suggesting that the reagent is on the left hand side, and the product is on the right hand side
# Also suggesting that in the input file, IRCs increase from top to bottom, both for the electronic energy and ZPVE
# If it is not the case, please fix your input file, or own the errors and weird results

# The program also uses the input electronic energy IRC matrix to estimate the IRC coordinate of the transition state
# (it should have a maximum of an electronic energy)