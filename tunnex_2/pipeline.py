from tunnex_2.constants import cm_m1_to_Hartree, hour, day, year, grid_size  # type: ignore
from tunnex_2.models.results import LevelResult, TemperatureAverResult  # type: ignore
from tunnex_2.core.turning_points import find_turning_points  # type: ignore
from tunnex_2.core.wkb import compute_wkb  # type: ignore
from tunnex_2.core.kinetics import (  # type: ignore
    transmission_prob,
    reaction_rate,
    half_life,
    boltzmann_data_extraction,
    arrhenius_rate,
)

import numpy as np


# --- Defining the function to compute main properties ---
def run_pipeline(params, potential):

    # --- Storing the interpolated ZPVE-corrected IRC  ---
    x_grid = np.linspace(potential.irc_x_min, potential.irc_x_max, grid_size)
    dtype = np.dtype([("irc_value", float), ("energy_value", float)])
    interpolated_irc_grid = np.array(list(zip(x_grid, potential(x_grid))), dtype=dtype)

    # --- Filling the matrix of the results for the given vibrational levels ---
    level_result_data = []

    for i in range(params.the_upper_level_to_compute + 1):
        # --- Computing vibrational energy levels ---
        level_energy = (i + 0.5) * params.freq * cm_m1_to_Hartree

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

        # --- Taking the computed vibrational energy level indexes and levels and appending them with the computed parameters ---
        level_result_data.append(
            LevelResult(
                i,
                level_energy,
                turning_points["left"],
                turning_points["right"],
                wkb,
                prob,
                reaction_rate_value,
                half_life_value,
                half_life_value * hour,
                half_life_value * day,
                half_life_value * year,
            )
        )

    # --- Computing the temperature-averaged transmission probabilities ---
    prob_aver = boltzmann_data_extraction(level_result_data, params.T)

    # --- Computing the temperature-averaged reaction rate in seconds-1 ---
    reaction_rate_aver = reaction_rate(params.freq, prob_aver)

    # --- Computing the temparature-averaged half-life in seconds ---
    half_life_aver = half_life(reaction_rate_aver)

    # --- Storing the temparature-averaged parameters ---
    results_aver = TemperatureAverResult(
        prob_aver,
        reaction_rate_aver,
        half_life_aver,
        half_life_aver * hour,
        half_life_aver * day,
        half_life_aver * year,
    )

    # --- Storing the arrhenius graph parameters (tunneling only!) ---
    arrhenius_rate_vals = arrhenius_rate(
        level_result_data,
        params.freq,
        params.T_arrhenius_min,
        params.T_arrhenius_max,
        params.T_arrhenius_step,
    )

    return interpolated_irc_grid, level_result_data, results_aver, arrhenius_rate_vals
