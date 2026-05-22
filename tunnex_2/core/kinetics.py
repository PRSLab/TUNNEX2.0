from tunnex_2.constants import light_speed, k_b  # type: ignore
from tunnex_2.models.results import ArrheniusResult  # type: ignore
import numpy as np


# --- Defining the function to compute the transmission probability ---
def transmission_prob(wkb: float) -> float:
    if np.isinf(wkb):
        return 0.0
    prob = 1 / (1 + np.exp(2 * wkb))
    return prob  # A transmission probability [dimentionsless]


# --- Defining the function to compute the reaction rate ---
def reaction_rate(freq: float, prob: float) -> float:
    return freq * light_speed * prob  # A reaction rate in seconds-1


# --- Defining the function to compute the half-life ---
def half_life(k) -> float:
    if k == 0:
        return np.inf
    return np.log(2) / k  # A half-life in seconds


# --- Defining the average transmission probability ---
def _boltzmann_average(
    energies: np.array,
    probabilities: np.array,
    temperature: float,
) -> float:

    E0 = energies[0]

    boltzmann_factor = np.exp(-(energies - E0) / (k_b * temperature))

    sum_boltzmann = np.sum(boltzmann_factor)

    if sum_boltzmann == 0:
        return 0.0

    return np.sum(boltzmann_factor * probabilities) / sum_boltzmann


# --- Extraction data from the list for boltzmann_average() ---
def boltzmann_data_extraction(levels: list, temperature: float) -> object:
    vib_energies = np.array([level.vib_energy for level in levels])
    transmission_probs = np.array([level.transmission_probability for level in levels])
    return _boltzmann_average(vib_energies, transmission_probs, temperature)


# --- Defining the function to compute the average reaction rate for different temperatures ---
def arrhenius_rate(
    levels: list, freq: float, T_min: float, T_max: float, T_step: float
) -> list:
    arrhenius_matrix = []

    for arrhenius_T in range(int(T_max), int(T_min) - int(T_step), -int(T_step)):
        if arrhenius_T <= 0:
            break
        else:
            arrhenius_prob = boltzmann_data_extraction(levels, arrhenius_T)
            arrhenius_rate = reaction_rate(freq, arrhenius_prob)
            if arrhenius_rate > 0:
                log_arrhenius_rate = np.log(arrhenius_rate)
            else:
                log_arrhenius_rate = np.inf

            arrhenius_matrix.append(
                ArrheniusResult(arrhenius_T, 1 / arrhenius_T, log_arrhenius_rate)
            )

    return arrhenius_matrix
