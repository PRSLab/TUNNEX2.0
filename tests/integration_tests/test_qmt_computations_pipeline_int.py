# --- Test: test_qmt_computations_pipeline_int. Integration tests for qmt_computations_pipeline.py
# Run with: pytest test_qmt_computations_pipeline_int.py -v ---

# --- Modules ---
import pytest  # type: ignore
from contextlib import ExitStack
from unittest.mock import patch
pytestmark = pytest.mark.integration  # type: ignore

import numpy as np
from scipy.integrate import quad
from dataclasses import astuple
from types import SimpleNamespace

from tunnex_2.constants_and_dataclasses.constants_and_settings import ( # type: ignore
    CM_M1_TO_HARTREE, REVELO, LIGHT_SPEED, K_B,
    GRID_SIZE,
    HOUR, DAY, YEAR)
from tunnex_2.constants_and_dataclasses.dataclasses import (  # type: ignore
    LevelResult, TemperatureAverResult, ArrheniusResult)

from tunnex_2.qmt_computations.core.kinetics import ( # type: ignore
    transmission_prob,
    reaction_rate,
    half_life,
    temperature_averaging_tunneling,
    arrhenius_rate)
from tunnex_2.qmt_computations.core.turning_points import find_turning_points # type: ignore
from tunnex_2.qmt_computations.core.wkb import compute_wkb # type: ignore


# --- Module to test ---
KINETICS_MODULE = "tunnex_2.qmt_computations.core.kinetics"
KINETICS_LOGGER = f"{KINETICS_MODULE}.logger"
from tunnex_2.qmt_computations.core.kinetics import ( # type: ignore
    _reaching_target_probability,
    _temperature_averaging_infinite_sum,
    _integration_over_weighted_probs,
    _temperature_averaging_integral)

QMT_PIPELINE_MODULE = "tunnex_2.qmt_computations.pipeline.qmt_computations_pipeline"
from tunnex_2.qmt_computations.pipeline.qmt_computations_pipeline import qmt_computations_pipeline # type: ignore


# --- Helpers and shared data ---
PROB_AVER_MODE = "finite_sum"


LVL_INDEX, LVL_ENERGY, LVL_LEFT, LVL_RIGHT, LVL_WKB, LVL_PROB, LVL_RATE, \
    LVL_HL_S, LVL_HL_H, LVL_HL_D, LVL_HL_Y = range(11)
AVR_PROB, AVR_RATE, AVR_HL_S, AVR_HL_H, AVR_HL_D, AVR_HL_Y = range(6)
# Positions of the fields in the (positional) LevelResult and TemperatureAverResult dataclasses

RES_GRID, RES_LEVELS, RES_AVER, RES_ARRHENIUS = range(4)
# Positions of the returned items


SPIED_FUNCS = {"find_turning_points": find_turning_points,
               "compute_wkb": compute_wkb,
               "transmission_prob": transmission_prob,
               "reaction_rate": reaction_rate,
               "half_life": half_life,
               "temperature_averaging_tunneling": temperature_averaging_tunneling,
               "arrhenius_rate": arrhenius_rate}

ENERGIES = [0.0, 0.01, 0.02, 0.03]
# Energies to test for the symmetric parabolic barrier
TP_RTOL, WKB_RTOL = 1e-8, 1e-4
# Relative tolerance for the turning point search and WKB integral computations

TEMPERATURES = [100.0, 300.0, 1e3, 1e4, 1e5]
# A list of temperatures for the tests
BORDER_FACTOR = 92.2
# The default border_factor of _integration_over_weighted_probs

SEARCH_MODES = ["infinite_sum", "integral"]
# The modes which call _reaching_target_probability

STUB_PROBS, STUB_ZERO_LEVEL, STUB_TARGET_ENERGY = [0.1, 0.5, 0.9], 0.002, 0.05
# The values returned by the stubbed _reaching_target_probability


class _SyntheticEckhartPotential:
    """ 
    Asymmetric Eckart potential (Hartree), reactant on the left (V -> 0),
    product on the right (V -> A)
    """
    def __init__(self, A: float = -0.085, B: float = 0.375, L: float = 3.275, half_range: float = 8.0):
        # --- Defining the main parameters ---
        self.asymmetry_A = A
        self.shape_B = B
        self.width_L = L
        self.irc_x_min = -half_range
        self.irc_x_max = half_range
        
        # --- Defining the parameters which may help ---
        self.irc_x_ts = (self.width_L / (2 * np.pi) * 
            np.log((self.shape_B + self.asymmetry_A) / (self.shape_B - self.asymmetry_A)))
        self.energy_max = (self.asymmetry_A + self.shape_B) ** 2 / (4 * self.shape_B)
        
        # --- Defining the parameters to mimic the real class Potential fully ---
        self.potential_scaling_factor = 1.0
        self.offset = 0.0
        self.deviation = 0.0

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        y = - np.exp(2 * np.pi * x / self.width_L)
        potential = - self.asymmetry_A * y / (1 - y) - self.shape_B * y / (1 - y) ** 2
        return potential

class _SyntheticParabolicPotential:
    """ Symmetric parabolic potential (Hartree), y = C - A * x ** 2 """
    def __init__(self, A: float = 0.001, C: float = 0.062, half_range: float = 8.0):
        # --- Defining the main parameters ---
        self.width_A = A  # should be strictly > 0
        self.offset_C = C
        # Default values A and C [0.001, 0.062] give V(x=0) ~ 13600 cm-1
        # and V(x=8) ~ -440 cm-1
        self.irc_x_min = -half_range
        self.irc_x_max = half_range
        
        # --- Defining the parameters which may help ---
        self.irc_x_ts = 0.0
        self.energy_max = C
        
        # --- Defining the parameters to mimic the real class Potential fully ---
        self.potential_scaling_factor = 1.0
        self.offset = 0.0
        self.deviation = 0.0

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        potential = self.offset_C - self.width_A * x ** 2
        return potential


def _make_eckhart_potential(**overrides) -> _SyntheticEckhartPotential:
    """ Return the synthetic Eckhart potential; the parameters can be overridden """
    return _SyntheticEckhartPotential(**overrides)

def _make_parabolic_potential(**overrides) -> _SyntheticParabolicPotential:
    """ Return the synthetic parabolic potential; the parameters can be overridden """
    return _SyntheticParabolicPotential(**overrides)

def _analytic_integral_par_potential(parabolic_potential, energy_level):
    """
    Return the integral for a parabolic potential (y(x) = C - A * x **2)
    and energy_level (E); it is negative above the barrier
    """
    A = parabolic_potential.width_A
    if not A > 0:
        raise ValueError(f"Expected positive A for a parabolic potential. Got {A}")
    C = parabolic_potential.offset_C
    wkb_int = (C - energy_level) * np.pi * np.sqrt(REVELO) / np.sqrt(2 * A)
    # S = integral_{-x0}^{+x0} sqrt(2 * mass (V(x) - E)) dx = pi * np.sqrt(mass) * (C - E) / sqrt(2 * A)
    # mass is REVELO; in fact, it is a half-ellipse square
    turn_point = np.sqrt(abs(C - energy_level) / A)
    # turning points:  x = -+ x0,  x0 = sqrt((C - E) / A)
    if C - energy_level < 0:
        turn_point *= 1j
    # Making the number complex if the energy level is above the barrier
    return wkb_int, turn_point

def _analytic_probability_par_potential(parabolic_potential, energy_level):
    """ Return p = 1 / (1 + exp(2 * S)) for a parabolic potential """
    wkb_integral = _analytic_integral_par_potential(parabolic_potential, energy_level)[0]
    result = 1.0 / (1.0 + np.exp(2.0 * wkb_integral))
    return result

def _analytic_probabilities_until_target(parabolic_potential, freq, target, max_levels=1000):
    """ Reference of the loop of _reaching_target_probability built from the analytic level probabilities """
    probs = []
    i = 0
    prob = 0.0
    while prob < target and i < max_levels:
        energy = (i + 0.5) * freq * CM_M1_TO_HARTREE
        prob = _analytic_probability_par_potential(parabolic_potential, energy)
        probs.append(prob)
        i += 1
    return probs

def _analytic_number_of_levels_closed_form(parabolic_potential, freq, target):
    """ Closed-form number of the levels needed to reach the target """
    e_star = (parabolic_potential.offset_C
        + np.sqrt(parabolic_potential.width_A) 
        * np.log(target / (1.0 - target)) / (np.sqrt(2.0) * np.pi * REVELO))
    len_prob_list = max(0, np.ceil(e_star / (freq * CM_M1_TO_HARTREE) - 0.5)) + 1
    return len_prob_list
    # The closed form for energy is
    # E* = C + sqrt(A) * ln(P / (1 - P)) / (sqrt(2.0) * pi * sqrt(mass)), where mass is REVELO
    # The smallest index (i) with (i + 0.5) * freq >= E* is idx_min;
    # idx_min = math.ceil(E*/freq - 0.5), it cannot be negative
    # len_prob_list = idx_min + 1

def _analytic_levels(parabolic_potential, freq, target):
    """
    Return the analytic probabilities, the zero-level energy
    and the last level energy (the target energy)
    """
    probs = _analytic_probabilities_until_target(parabolic_potential, freq, target)
    zero_level = 0.5 * freq * CM_M1_TO_HARTREE
    target_energy = (len(probs) - 0.5) * freq * CM_M1_TO_HARTREE
    return probs, zero_level, target_energy

def _analytic_weighted_integral(parabolic_potential, zero_level, target_energy, temperature,
        border_factor=BORDER_FACTOR):
    """
    Should return integral [0, t_cut] (P(E0 + t * kT) * exp(-t)dt),
    where P is the analytic transmission probability
    """
    def integrand(t):
        wkb_integral = _analytic_integral_par_potential(parabolic_potential,
            zero_level + t * (K_B * temperature))[0]
        result = 1.0 / (1.0 + np.exp(2.0 * wkb_integral)) * np.exp(-t)
        return result
    # Defining the function for integration
    t_cut = min((target_energy - zero_level) / (K_B * temperature), border_factor)
    # Cutting the integration range (important at low temperatures)
    t_cross = (parabolic_potential.offset_C - zero_level) / (K_B * temperature)
    points = [t_cross] if 0.0 < t_cross < t_cut else None
    # The integral changes a sign at t_cross, so it would be better to enforce
    # the integration interval splitting there
    result = quad(integrand, 0.0, t_cut, points=points, epsabs=0.0, epsrel=WKB_RTOL)[0]
    return result
    
def _closed_form_integral_average(parabolic_potential, zero_level, target_energy, temperature, tail_target):
    """
    Return P_under + P_over, where P_under in the integral and
    P_over = 0.5 * (1 + reached_prob) * exp(-(target_energy - zero_level) / kT)
    """
    P_under = _analytic_weighted_integral(parabolic_potential, zero_level, target_energy, temperature)
    P_over = 0.5 * (1.0 + tail_target) * np.exp(-(target_energy - zero_level) / (K_B * temperature))
    return P_under + P_over

def _make_level_data(parabolic_potential, n_levels, freq):
    """ Return the level-like objects with the fields used by the finite_sum averaging """
    levels = []
    for i in range(n_levels + 1):
        energy = (i + 0.5) * freq * CM_M1_TO_HARTREE
        levels.append(SimpleNamespace(vib_energy=energy,
        transmission_probability=_analytic_probability_par_potential(parabolic_potential, energy)))
    return levels

def _level_data_for(mode, parabolic_potential, n_levels, freq):
    """ Return the levels for finite_sum and an empty list for the modes which do not use them """
    result = _make_level_data(parabolic_potential, n_levels, freq) if mode == "finite_sum" else []
    return result

def _expected_temp_average(mode, parabolic_potential, temperature, freq, n_levels, target=0.99):
    """ Return the closed-form temperature-averaged probability for the given mode """
    if mode == "finite_sum":
        levels = _make_level_data(parabolic_potential, n_levels, freq)
        energies = np.array([level.vib_energy for level in levels])
        probs = np.array([level.transmission_probability for level in levels])
        weights = np.exp(-(energies - energies[0]) / (K_B * temperature))
        result = float(np.dot(weights / np.sum(weights), probs))
        return result
    probs, zero_level, target_energy = _analytic_levels(parabolic_potential, freq, target)
    if mode == "infinite_sum":
        tail_target = probs[-1]
        q = np.exp(-freq * CM_M1_TO_HARTREE / (K_B * temperature))
        body = (1.0 - q) * sum(q ** i * p for i, p in enumerate(probs))
        return body + q ** len(probs) * 0.5 * (1.0 + tail_target)
    # mode = "integral"
    under = _analytic_weighted_integral(parabolic_potential, zero_level, target_energy, temperature)
    over = 0.5 * (1.0 + probs[-1]) * np.exp(-(target_energy - zero_level) / (K_B * temperature))
    return under + over


def _make_params(**overrides) -> SimpleNamespace:
    """ Return the synthetic parameters """
    params = dict(number_of_levels=3,
                  freq=1000.0,
                  T=300.0,
                  prob_aver_mode_qmt=PROB_AVER_MODE,
                  T_arrhenius_min=200.0,
                  T_arrhenius_max=400.0,
                  T_step=50.0)
    params.update(overrides)
    return SimpleNamespace(**params)

def _run_pipeline(params=None, potential=None):
    """ Run the pipeline with the real core functions """
    params = _make_params() if params is None else params
    potential = _make_eckhart_potential() if potential is None else potential
    return qmt_computations_pipeline(params, potential)

def _run_with_spies(params=None, potential=None):
    """ Run the pipeline with the real core functions wrapped into spies """
    params = _make_params() if params is None else params
    potential = _make_eckhart_potential() if potential is None else potential
    with ExitStack() as stack:
        spies = {name: stack.enter_context(patch(f"{QMT_PIPELINE_MODULE}.{name}", wraps=real))
            for name, real in SPIED_FUNCS.items()}
        result = qmt_computations_pipeline(params, potential)
    return result, spies

def _fields(dataclass_instance) -> tuple:
    """ Return the dataclass values as a tuple in the order of the fields """
    return astuple(dataclass_instance)

def _flat(dataclass_instance) -> np.ndarray:
    """ Return all the numeric values of the dataclass as one flat float array """
    result = np.concatenate([np.atleast_1d(_as_array(v)).ravel() for v in astuple(dataclass_instance)])
    return result

def _as_array(value) -> np.ndarray:
    """ Convert the value (a number, a list, an array etc.) to a float array """
    if isinstance(value, np.ndarray) and value.dtype.names:
        return np.array([value[name] for name in value.dtype.names], dtype=float)
    return np.asarray(value, dtype=float)

def _patch_helpers(stack, prob=0.5):
    """ Patch the averaging helpers and the probability search """
    mocks = {}
    for name, short in (("_temperature_averaging_finite_sum", "finite"),
                        ("_temperature_averaging_infinite_sum", "infinite"),
                        ("_temperature_averaging_integral", "integral")):
        mocks[short] = stack.enter_context(patch(f"{KINETICS_MODULE}.{name}", return_value=prob))
    mocks["reach"] = stack.enter_context(patch(f"{KINETICS_MODULE}._reaching_target_probability",
        return_value=(STUB_PROBS, STUB_ZERO_LEVEL, STUB_TARGET_ENERGY)))
    return mocks

def _arr_grid(T_min, T_max, T_step, freq):
    """ Return the Arrhenius temperatures for the given settings (the averaging is stubbed) """
    with ExitStack() as stack:
        _patch_helpers(stack)
        result = arrhenius_rate([], freq, T_min, T_max, T_step, None)
    return [data.arrhenius_temperature for data in result]


# ===========================================================================
# --- qmt_computations_pipeline - returned structure ---
# ===========================================================================
class TestReturnedStructure:

    def test_returns_four_items(self):
        """
        The pipeline returns the grid, the level results,
        the averaged results and the Arrhenius data
        """
        result = _run_pipeline()
        assert len(result) == 4, f"expected {4} returned items, got {len(result)}"

    def test_returned_types(self):
        """ Types of the returned items are the expected ones """
        grid, levels, aver, arrs = _run_pipeline()
        assert isinstance(grid, np.ndarray), f"expected {np.ndarray}, got {type(grid)}; test #1"
        assert isinstance(levels, list), f"expected {list}, got {type(levels)}; test #2"
        assert all(isinstance(level, LevelResult) for level in levels), \
        f"expected all items to be {LevelResult}, got {[type(level) for level in levels]}; test #3"
        assert isinstance(aver, TemperatureAverResult), \
        f"expected {TemperatureAverResult}, got {type(aver)}; test #4"
        assert isinstance(arrs, list), f"expected {list}, got {type(arrs)}; test #5"
        assert all(isinstance(arr, ArrheniusResult) for arr in arrs), \
        f"expected all items to be {ArrheniusResult}, got {[type(arr) for arr in arrs]}; test #6"

    def test_number_of_levels(self):
        """
        The number of the level results is number_of_levels + 1
        (it should include the ground state)
        """
        expected = 3
        params = _make_params(number_of_levels=expected)
        _, levels, _, _ = _run_pipeline(params)
        assert len(levels) == expected + 1, f"expected {expected + 1} levels, got {len(levels)}"

    def test_zero_levels_gives_ground_state_only(self):
        """ number_of_levels=0 gives the ground state only and the whole pipeline still works """
        expected = 0
        params = _make_params(number_of_levels=expected)
        _, levels, aver, _ = _run_pipeline(params)
        assert len(levels) == expected + 1, \
        f"expected {expected + 1} level, got {len(levels)}; test #1"
        assert isinstance(aver, TemperatureAverResult), \
        f"expected {TemperatureAverResult}, got {type(aver)}; test #2"


# ===========================================================================
# --- qmt_computations_pipeline - interpolated IRC grid ---
# ===========================================================================
class TestInterpolatedIRCGrid:

    def test_grid_size(self):
        """ The grid has GRID_SIZE points """
        grid, _, _, _ = _run_pipeline()
        assert grid.shape == (GRID_SIZE,), f"expected shape={(GRID_SIZE,)}, got {grid.shape}"

    def test_grid_field_names(self):
        """ The grid is a structured array with the irc_value and energy_value fields """
        grid, _, _, _ = _run_pipeline()
        assert grid.dtype.names == ("irc_value", "energy_value"), \
            f"expected {('irc_value', 'energy_value')}, got {grid.dtype.names}"

    def test_grid_covers_irc_range(self):
        """ The grid starts and ends at the boundaries of the IRC range """
        potential = _make_eckhart_potential(half_range=10.0)
        grid, _, _, _ = _run_pipeline(potential=potential)
        assert np.isclose(grid["irc_value"][0], potential.irc_x_min, rtol=1e-8), \
            f"expected x_min={potential.irc_x_min}, got {grid['irc_value'][0]}; test #1"
        assert np.isclose(grid["irc_value"][-1], potential.irc_x_max, rtol=1e-8), \
            f"expected x_max={potential.irc_x_max}, got {grid['irc_value'][-1]}; test #2"

    def test_grid_is_strictly_increasing(self):
        """ The IRC values of the grid are strictly increasing """
        grid, _, _, _ = _run_pipeline()
        assert np.all(np.diff(grid["irc_value"]) > 0), "expected strictly increasing irc_value"

    def test_grid_energies_match_potential(self):
        """ The grid energies are the potential evaluated at the grid points """
        potential = _make_eckhart_potential()
        grid, _, _, _ = _run_pipeline(potential=potential)
        expected = potential(grid["irc_value"])
        assert np.allclose(grid["energy_value"], expected, atol=1e-12), \
        f"expected energies from the potential, max deviation {np.max(np.abs(grid['energy_value'] - expected))}"

    def test_grid_contains_barrier(self):
        """ The grid contains the barrier: the maximum is above both asymptotes """
        potential = _make_eckhart_potential()
        grid, _, _, _ = _run_pipeline(potential=potential)
        e_max = np.max(grid["energy_value"])
        assert e_max > max(grid["energy_value"][0], grid["energy_value"][-1]), \
        f"expected barrier maximum above both ends, got max={e_max}, ends={grid['energy_value'][[0, -1]]}"


# ===========================================================================
# --- qmt_computations_pipeline - level_result_data ---
# ===========================================================================
class TestLevelResultsData:

    def test_level_indices(self):
        """ The level indices are range(number_of_levels + 1) """
        _, levels, _, _ = _run_pipeline()
        indices = [_fields(level)[LVL_INDEX] for level in levels]
        expected = [0, 1, 2, 3]  # expected 4 levels (default test parameter)
        assert indices == expected, f"expected {expected}, got {indices}"

    def test_level_energies(self):
        """ The level energies are (i + 0.5) * freq (converted to Hartree) """
        params = _make_params()
        _, levels, _, _ = _run_pipeline(params)
        energies = [_fields(level)[LVL_ENERGY] for level in levels]
        expected = [(i + 0.5) * params.freq * CM_M1_TO_HARTREE
            for i in range(params.number_of_levels + 1)]
        assert np.allclose(energies, expected, atol=1e-12), f"expected {expected}, got {energies}"

    def test_turning_points_are_ordered_around_barrier(self):
        """ The left turning point is on the left and the right one is on the right of the barrier top """
        _, levels, _, _ = _run_pipeline()
        for level in levels:
            fields = _fields(level)
            assert fields[LVL_LEFT] < fields[LVL_RIGHT], \
            f"level {fields[LVL_INDEX]}: expected tp_left < tp_right, got {fields[LVL_LEFT]} and {fields[LVL_RIGHT]}"
    # here and further it is assumed that the tested level energies are strictly under the barrier
    # and the reaction is not endothermic

    def test_turning_points_inside_irc_range(self):
        """ The turning points are inside the IRC range """
        potential = _make_eckhart_potential()
        _, levels, _, _ = _run_pipeline(potential=potential)
        for level in levels:
            fields = _fields(level)
            assert potential.irc_x_min <= fields[LVL_LEFT] and fields[LVL_RIGHT] <= potential.irc_x_max, \
                f"level {fields[LVL_INDEX]}: expected turning points inside " \
                f"[{potential.irc_x_min}, {potential.irc_x_max}], got {fields[LVL_LEFT]}, {fields[LVL_RIGHT]}"

    def test_potential_at_turning_points_equals_level_energy(self):
        """ The potential at both turning points equals the level energy """
        potential = _make_eckhart_potential()
        _, levels, _, _ = _run_pipeline(potential=potential)
        for level in levels:
            fields = _fields(level)
            for x in (fields[LVL_LEFT], fields[LVL_RIGHT]):
                assert np.isclose(potential(x), fields[LVL_ENERGY], atol=1e-8), \
                f"level {fields[LVL_INDEX]}: expected V({x})={fields[LVL_ENERGY]}, got {potential(x)}"

    def test_barrier_gets_thinner_with_level(self):
        """ The distance between the turning points decreases with the level """
        _, levels, _, _ = _run_pipeline()
        widths = [_fields(level)[LVL_RIGHT] - _fields(level)[LVL_LEFT] for level in levels]
        assert all(w2 < w1 for w1, w2 in zip(widths, widths[1:])), \
        f"expected decreasing widths, got {widths}"

    def test_integral_decreases_with_level(self):
        """ The integral decreases with the level but stays positive """
        _, levels, _, _ = _run_pipeline()
        wkb_list = [_fields(level)[LVL_WKB] for level in levels]
        assert all(wkb > 0 for wkb in wkb_list), \
        f"the WKB integrals should be positive under the barrier, got {wkb_list}; test #1"
        assert all(wkb2 < wkb1 for wkb1, wkb2 in zip(wkb_list, wkb_list[1:])), \
        f"expected decreasing integrals, got {wkb_list}; test #2"
    
    def test_probabilities_are_valid_and_grow_with_level(self):
        """ The transmission probabilities are in [0, 1] and do not decrease with the level """
        _, levels, _, _ = _run_pipeline()
        probs = [_fields(level)[LVL_PROB] for level in levels]
        expected = [0, 1]
        assert all(0.0 <= p <= 1.0 for p in probs), \
        f"expected probabilities in {expected}, got {probs}; test #1"
        assert all(p2 >= p1 for p1, p2 in zip(probs, probs[1:])), \
        f"expected non-decreasing probabilities, got {probs}; test #2"
        assert probs[-1] > probs[0], \
        f"expected the top level to tunnel better than the ground state, got {probs}; test #3"
        # the last test catches a [0.2, 0.2, 0.2, 0.2] case and similar

    def test_rates_and_half_lives_trends(self):
        """
        The rates are non-negative and grow with the level,
        the half-lives are positive and shrink with it
        """
        _, levels, _, _ = _run_pipeline()
        rates = [_fields(level)[LVL_RATE] for level in levels]
        half_lives = [_fields(level)[LVL_HL_S] for level in levels]
        assert all(r >= 0.0 for r in rates), f"expected non-negative rates, got {rates}; test #1"
        assert all(r2 >= r1 for r1, r2 in zip(rates, rates[1:])), f"expected non-decreasing rates, got {rates}; test #2"
        assert all(h > 0.0 for h in half_lives), f"expected positive half-lives, got {half_lives}; test #3"
        assert all(h2 <= h1 for h1, h2 in zip(half_lives, half_lives[1:])), \
        f"expected non-increasing half-lives, got {half_lives}; test #4"

    def test_chain_of_quantities_is_consistent_per_level(self):
        """
        Probability, rate and half-life stored for each level
        are the real core functions applied in a chain
        """
        params = _make_params()
        _, levels, _, _ = _run_pipeline(params)
        for level in levels:
            f = _fields(level)
            prob = transmission_prob(f[LVL_WKB])
            rate = reaction_rate(params.freq, f[LVL_PROB])
            hl = half_life(f[LVL_RATE])
            assert np.isclose(f[LVL_PROB], prob, rtol=1e-8), \
            f"level {f[LVL_INDEX]}: expected prob={prob}, got {f[LVL_PROB]}; test #1"
            assert np.isclose(f[LVL_RATE], rate, rtol=1e-8), \
            f"level {f[LVL_INDEX]}: expected rate={rate}, got {f[LVL_RATE]}; test #2"
            assert np.isclose(f[LVL_HL_S], hl, rtol=1e-8), \
            f"level {f[LVL_INDEX]}: expected half-life={hl}, got {f[LVL_HL_S]}; test #3"

    def test_time_unit_conversions_per_level(self):
        """ The half-lives in hours, days and years are the half-life are stored correctly """
        _, levels, _, _ = _run_pipeline()
        for level in levels:
            f = _fields(level)
            assert np.isclose(f[LVL_HL_H], f[LVL_HL_S] * HOUR, rtol=1e-8), \
            f"level {f[LVL_INDEX]}: expected hours={f[LVL_HL_S] * HOUR}, got {f[LVL_HL_H]}; test #1"
            assert np.isclose(f[LVL_HL_D], f[LVL_HL_S] * DAY, rtol=1e-8), \
            f"level {f[LVL_INDEX]}: expected days={f[LVL_HL_S] * DAY}, got {f[LVL_HL_D]}; test #2"
            assert np.isclose(f[LVL_HL_Y], f[LVL_HL_S] * YEAR, rtol=1e-8), \
            f"level {f[LVL_INDEX]}: expected years={f[LVL_HL_S] * YEAR}, got {f[LVL_HL_Y]}; test #3"

    def test_ground_state_matches_independent_core_chain(self):
        """ The ground-state result equals the one obtained by calling the core functions manually """
        params = _make_params()
        potential = _make_eckhart_potential()
        _, levels, _, _ = _run_pipeline(params, potential)
        f = _fields(levels[0])
        energy = 0.5 * params.freq * CM_M1_TO_HARTREE
        tp = find_turning_points(energy, potential)
        wkb = compute_wkb(tp, energy, potential)
        prob = transmission_prob(wkb)
        rate = reaction_rate(params.freq, prob)
        hl = half_life(rate)
        expected = [tp["left"], tp["right"], _as_array(wkb), prob, rate, hl]
        got = [f[LVL_LEFT], f[LVL_RIGHT], _as_array(f[LVL_WKB]), f[LVL_PROB], f[LVL_RATE], f[LVL_HL_S]]
        for name, exp, res in zip(("left", "right", "wkb", "prob", "rate", "half_life"), expected, got):
            assert np.allclose(res, exp, rtol=1e-8), f"{name}: expected {exp}, got {res}"


# ===========================================================================
# --- qmt_computations_pipeline - temperature-averaged results ---
# ===========================================================================
class TestTemperatureAveragedResults:

    def test_probability_is_valid_and_not_below_ground_state(self):
        """ The averaged probability is in [0, 1] and not below the ground-state one """
        _, levels, aver, _ = _run_pipeline()
        prob_aver = _fields(aver)[AVR_PROB]
        prob_ground = _fields(levels[0])[LVL_PROB]
        expected = [0, 1]
        comp_tol = 1e-8
        assert 0.0 <= prob_aver <= 1.0, f"expected probability in {expected}, got {prob_aver}; test #1"
        assert prob_aver >= prob_ground * (1.0 - comp_tol), f"expected >= {prob_ground}, got {prob_aver}; test #2"

    def test_probability_matches_direct_call(self):
        """ The averaged probability equals the direct call of temperature_averaging_tunneling on the same levels """
        params = _make_params()
        potential = _make_eckhart_potential()
        _, levels, aver, _ = _run_pipeline(params, potential)
        expected = temperature_averaging_tunneling(params.freq, levels, potential,
            params.T, params.prob_aver_mode_qmt)
        assert np.isclose(_fields(aver)[AVR_PROB], expected, rtol=1e-8), \
        f"expected {expected}, got {_fields(aver)[AVR_PROB]}"

    def test_rate_and_half_life_are_derived_from_averaged_probability(self):
        """ The averaged rate and half-life are computed from the averaged probability """
        params = _make_params()
        _, _, aver, _ = _run_pipeline(params)
        f = _fields(aver)
        rate = reaction_rate(params.freq, f[AVR_PROB])
        hl = half_life(rate)
        assert np.isclose(f[AVR_RATE], rate, rtol=1e-8), \
        f"expected rate={rate}, got {f[AVR_RATE]}; test #1"
        assert np.isclose(f[AVR_HL_S], hl, rtol=1e-8), \
        f"expected half-life={hl}, got {f[AVR_HL_S]}; test #2"

    def test_time_unit_conversions(self):
        """ The averaged half-lives in hours, days and years use HOUR, DAY and YEAR """
        _, _, aver, _ = _run_pipeline()
        f = _fields(aver)
        assert np.isclose(f[AVR_HL_H], f[AVR_HL_S] * HOUR, rtol=1e-12), \
        f"expected hours={f[AVR_HL_S] * HOUR}, got {f[AVR_HL_H]}; test #1"
        assert np.isclose(f[AVR_HL_D], f[AVR_HL_S] * DAY, rtol=1e-12), \
        f"expected days={f[AVR_HL_S] * DAY}, got {f[AVR_HL_D]}; test #2"
        assert np.isclose(f[AVR_HL_Y], f[AVR_HL_S] * YEAR, rtol=1e-12), \
        f"expected years={f[AVR_HL_S] * YEAR}, got {f[AVR_HL_Y]}; test #3"

    def test_probability_does_not_decrease_with_temperature(self):
        """ The averaged probability does not decrease when the temperature rises """
        probs = []
        comp_tol = 1e-8
        for temperature in (50.0, 150.0, 300.0, 600.0):
            _, _, aver, _ = _run_pipeline(_make_params(T=temperature))
            probs.append(_fields(aver)[AVR_PROB])
        assert all(p2 >= p1 * (1.0 - comp_tol) for p1, p2 in zip(probs, probs[1:])), \
        f"expected non-decreasing probabilities, got {probs}"

    def test_half_life_does_not_increase_with_temperature(self):
        """ The averaged half-life does not increase when the temperature rises """
        half_lives = []
        comp_tol = 1e-8
        for temperature in (50.0, 150.0, 300.0, 600.0):
            _, _, aver, _ = _run_pipeline(_make_params(T=temperature))
            half_lives.append(_fields(aver)[AVR_HL_S])
        assert all(h2 <= h1 * (1.0 + comp_tol) for h1, h2 in zip(half_lives, half_lives[1:])), \
        f"expected non-increasing half-lives, got {half_lives}"


# ===========================================================================
# --- qmt_computations_pipeline - Arrhenius data ---
# ===========================================================================
class TestArrhenius:

    def test_matches_direct_call(self):
        """ The Arrhenius data equal the direct call of arrhenius_rate on the same levels """
        params = _make_params()
        potential = _make_eckhart_potential()
        _, levels, _, arrhenius = _run_pipeline(params, potential)
        expected = arrhenius_rate(levels, params.freq, params.T_arrhenius_min, params.T_arrhenius_max,
        params.T_step, potential, params.prob_aver_mode_qmt)
        arrhenius_list = [astuple(data) for data in arrhenius]
        expected_list = [astuple(data) for data in expected]
        assert np.allclose(arrhenius_list, expected_list, rtol=1e-8), \
        f"expected {expected_list}, got {arrhenius_list}"

    def test_is_not_empty(self):
        """ The Arrhenius data are not empty """
        _, _, _, arrhenius = _run_pipeline()
        assert len(arrhenius) > 0, f"expected non-empty Arrhenius data, got {arrhenius}"

    def test_finer_temperature_step_gives_more_points(self):
        """ A smaller temperature step gives more points of the Arrhenius graph """
        _, _, _, coarse = _run_pipeline(_make_params(T_step=100.0))
        _, _, _, fine = _run_pipeline(_make_params(T_step=25.0))
        assert len(fine) > len(coarse), f"expected more than {len(coarse)} points, got {len(fine)}"

    def test_values_are_correct(self):
        """ The Arrhenius data contain only finite numbers (ln(k) can be -inf) """
        _, _, _, arrhenius = _run_pipeline()
        for arr in arrhenius:
            assert np.isfinite(arr.arrhenius_temperature), \
            f"expected only finite T, got {arr.arrhenius_temperature}; test #1"
            assert np.isfinite(arr.arrhenius_temperature_inv), \
            f"expected only finite 1/T, got {arr.arrhenius_temperature_inv}; test #2"
            result = np.isfinite(arr.log_reaction_rate) or arr.log_reaction_rate == -np.inf
            assert result, \
            f"expected either a finite or -inf ln(k), got {arr.log_reaction_rate}; test #3"


# ===========================================================================
# --- qmt_computations_pipeline - wiring between the stages ---
# ===========================================================================
class TestWiring:

    def test_per_level_functions_called_for_each_level(self):
        """
        The per-level functions are called once per level;
        the rate and half-life once more for the average
        """
        params = _make_params(number_of_levels=3)
        _, spies = _run_with_spies(params)
        n_levels = params.number_of_levels + 1
        expected_counts = {"find_turning_points": n_levels, "compute_wkb": n_levels,
                           "transmission_prob": n_levels, "reaction_rate": n_levels + 1,
                           "half_life": n_levels + 1,
                           "temperature_averaging_tunneling": 1, "arrhenius_rate": 1}
        for name, expected in expected_counts.items():
            assert spies[name].call_count == expected, \
            f"{name}: expected {expected} calls, got {spies[name].call_count}"

    def test_turning_points_flow_into_wkb(self):
        """
        The turning points of each level are passed to the WKB integral
        together with the same level energy
        """
        _, spies = _run_with_spies()
        for tp_call, wkb_call in zip(spies["find_turning_points"].call_args_list,
                spies["compute_wkb"].call_args_list):
            level_energy = tp_call.args[0]
            assert wkb_call.args[1] == level_energy, \
            f"expected level energy={level_energy}, got {wkb_call.args[1]}; test #1"
            assert isinstance(wkb_call.args[0], dict), \
            f"expected turning points as {dict}, got {type(wkb_call.args[0])}; test #2"

    def test_level_results_flow_into_averaging_and_arrhenius(self):
        """
        The same list of level results is passed to the averaging,
        to the Arrhenius stage and returned
        """
        params = _make_params()
        potential = _make_eckhart_potential()
        (_, levels, _, _), spies = _run_with_spies(params, potential)
        aver_args = spies["temperature_averaging_tunneling"].call_args.args
        arr_args = spies["arrhenius_rate"].call_args.args
        assert aver_args[1] is levels, \
        "expected the averaging to receive the returned list of the level results; test #1"
        assert arr_args[0] is levels, \
        "expected the Arrhenius stage to receive the returned list of the level results; test #2"
        assert aver_args[2] is potential and arr_args[5] is potential, \
        "expected the same potential object in both stages; test #3"

    def test_parameters_flow_into_averaging_and_arrhenius(self):
        """ The temperature settings of the parameters are passed to the averaging and Arrhenius stages """
        params = _make_params(T=250.0, T_arrhenius_min=100.0, T_arrhenius_max=500.0, T_step=40.0)
        _, spies = _run_with_spies(params)
        aver_args = spies["temperature_averaging_tunneling"].call_args.args
        arr_args = spies["arrhenius_rate"].call_args.args
        assert (aver_args[0], aver_args[3], aver_args[4]) == (params.freq, 250.0, PROB_AVER_MODE), \
        f"expected {(params.freq, 250.0, PROB_AVER_MODE)}, got {(aver_args[0], aver_args[3], aver_args[4])}; test #1"
        assert arr_args[1:5] == (params.freq, 100.0, 500.0, 40.0), \
        f"expected {(params.freq, 100.0, 500.0, 40.0)}, got {arr_args[1:5]}; test #2"
        assert arr_args[6] == PROB_AVER_MODE, \
        f"expected mode={PROB_AVER_MODE}, got {arr_args[6]}; test #3"

    def test_averaged_probability_flows_into_reaction_rate(self):
        """ The averaged probability is the one passed to the last reaction_rate call """
        (_, _, aver, _), spies = _run_with_spies()
        last_args = spies["reaction_rate"].call_args_list[-1].args
        expected = _fields(aver)[AVR_PROB]
        assert np.isclose(last_args[1], expected, rtol=1e-12), f"expected {expected}, got {last_args[1]}"


# ===========================================================================
# --- qmt_computations_pipeline - reproducibility and side effects ---
# ===========================================================================
class TestReproducibilityAndSideEffects:

    def test_repeated_runs_give_identical_results(self):
        """ Two runs with the same inputs give identical results """
        grid1, levels1, aver1, arr1 = _run_pipeline()
        grid2, levels2, aver2, arr2 = _run_pipeline()
        assert np.array_equal(grid1, grid2), "expected identical grids; test #1"
        assert len(levels1) == len(levels2), f"expected {len(levels1)} levels, got {len(levels2)}; test #2"
        for l1, l2 in zip(levels1, levels2):
            assert np.array_equal(_flat(l1), _flat(l2)), \
            f"expected identical level results, got {l1} and {l2}; test #3"
        assert np.array_equal(_flat(aver1), _flat(aver2)), \
        f"expected identical averaged results, got {aver1} and {aver2}; test #4"
        for a1, a2 in zip(arr1, arr2):
            assert np.array_equal(_flat(a1), _flat(a2)), \
            f"expected identical level results, got {a1} and {a2}; test #5"

    def test_inputs_are_not_modified(self):
        """ The parameters and the potential are not modified by the pipeline """
        params = _make_params()
        potential = _make_eckhart_potential()
        params_before = dict(vars(params))
        potential_before = dict(vars(potential))
        _run_pipeline(params, potential)
        assert vars(params) == params_before, \
        f"expected {params_before}, got {vars(params)}; test #1"
        assert vars(potential) == potential_before, \
        f"expected {potential_before}, got {vars(potential)}; test #2"

    def test_higher_barrier_gives_longer_half_life(self):
        """ A higher barrier gives a smaller averaged probability and a longer averaged half-life """
        _, _, aver_low, _ = _run_pipeline(potential=_make_eckhart_potential(B=0.375))
        _, _, aver_high, _ = _run_pipeline(potential=_make_eckhart_potential(B=0.750))
        low, high = _fields(aver_low), _fields(aver_high)
        assert high[AVR_PROB] < low[AVR_PROB], \
        f"expected prob < {low[AVR_PROB]}, got {high[AVR_PROB]}; test #1"
        assert high[AVR_HL_S] > low[AVR_HL_S], \
        f"expected half-life > {low[AVR_HL_S]}, got {high[AVR_HL_S]}; test #2"


# ===========================================================================
# --- qmt_computations_pipeline - missing inputs ---
# ===========================================================================
class TestMissingInputs:

    def test_potential_without_irc_range_raises(self):
        """ A potential without the IRC range raises an error. Expecting an AttributeError """
        potential = _make_eckhart_potential()
        del potential.irc_x_min
        with pytest.raises(AttributeError, match="irc_x_min"):
            _run_pipeline(potential=potential)

    @pytest.mark.parametrize("missing", ["number_of_levels", "freq"])
    def test_params_without_required_field_raises(self, missing):
        """ Parameters without number_of_levels or freq raise an error. Expecting an AttributeError """
        params = _make_params()
        delattr(params, missing)
        with pytest.raises(AttributeError, match=missing):
            _run_pipeline(params)


# =============================================================================
# --- Self-check of the analytic helpers (independent numerical quadrature) ---
# =============================================================================
class TestAnalyticHelpers:

    @pytest.mark.parametrize("energy", ENERGIES)
    def test_analytic_integral_matches_quadrature(self, energy):
        """ The closed-form integral equals a brute-force quadrature """
        pot = _make_parabolic_potential()
        expected, x0 = _analytic_integral_par_potential(pot, energy)
        t = np.linspace(-np.pi / 2, np.pi / 2, 200001)
        # replacement x = x0 * sin(t) removes the problems near the turning points
        integrand = np.sqrt(np.maximum(2.0 * REVELO * (pot(x0 * np.sin(t)) - energy), 0.0)) * x0 * np.cos(t)
        # dx = x0 * cos(t) * dt
        numeric = np.sum((integrand[1:] + integrand[:-1]) * np.diff(t)) / 2.0
        # integration via a trapezoidal rule
        assert np.isclose(numeric, expected, rtol=1e-8), f"expected {expected}, got {numeric}"

    @pytest.mark.parametrize("energy", ENERGIES)
    def test_potential_at_analytic_turning_points_equals_energy(self, energy):
        """ The potential evaluated at the analytic turning points equals the energy """
        pot = _make_parabolic_potential()
        _, x0 = _analytic_integral_par_potential(pot, energy)
        assert np.allclose(pot([-x0, x0]), energy, atol=1e-12), \
        f"expected V={energy}, got {pot([-x0, x0])}"
    
    @pytest.mark.parametrize("energy", [1e-6])
    def test_zero_a_raises(self, energy):
        """ The potential y(x) = C - A * x ** 2 should be an inverted parabola. Expecting a ValueError """
        wrong_pot = _make_parabolic_potential(A=0.0)
        with pytest.raises(ValueError, match="positive A"):
            _analytic_integral_par_potential(wrong_pot, energy)
    
    @pytest.mark.parametrize("energy", [1e-6])
    def test_negative_a_raises(self, energy):
        """ The potential y(x) = C - A * x ** 2 should be an inverted parabola """
        wrong_pot = _make_parabolic_potential(A=-0.001)
        with pytest.raises(ValueError, match="positive A"):
            _analytic_integral_par_potential(wrong_pot, energy)


# ===========================================================================
# --- find_turning_points is compared with the analytic solution ---
# ===========================================================================
class TestTurningPointsVsAnalytic:

    @pytest.mark.parametrize("energy", ENERGIES)
    def test_turning_points_match_analytic(self, energy):
        """ The numerical turning points match the analytic formula """
        pot = _make_parabolic_potential()
        tp = find_turning_points(energy, pot)
        _, x0 = _analytic_integral_par_potential(pot, energy)
        left, right = -x0, x0
        assert np.isclose(tp["left"], left, rtol=TP_RTOL), \
        f"E={energy}: expected left={left}, got {tp['left']}; test #1"
        assert np.isclose(tp["right"], right, rtol=TP_RTOL), \
        f"E={energy}: expected right={right}, got {tp['right']}; test #2"

    @pytest.mark.parametrize("energy", ENERGIES)
    def test_turning_points_are_symmetric(self, energy):
        """ The turning points of the symmetric barrier are symmetric with respect to the TS """
        pot = _make_parabolic_potential()
        tp = find_turning_points(energy, pot)
        assert np.isclose(tp["left"], -tp["right"], rtol=TP_RTOL),\
        f"expected left=-right, got {tp['left']}, {tp['right']}"

    def test_turning_points_move_inward_with_energy(self):
        """ The turning points approach each other when the energy grows """
        pot = _make_parabolic_potential()
        widths = []
        for energy in ENERGIES:
            tp = find_turning_points(energy, pot)
            widths.append(tp["right"] - tp["left"])
        assert all(w2 < w1 for w1, w2 in zip(widths, widths[1:])), \
        f"expected decreasing widths, got {widths}"

    def test_turning_points_scale_with_width_parameter(self):
        """ Quadrupling A halves the distance of the turning points from the TS """
        energy = 0.01
        tp_narrow = find_turning_points(energy, _make_parabolic_potential(A=0.001))
        tp_wide = find_turning_points(energy, _make_parabolic_potential(A=4*0.001))
        assert np.isclose(tp_narrow["right"] / tp_wide["right"], 2.0, rtol=TP_RTOL), \
            f"expected ratio={2}, got {tp_narrow['right'] / tp_wide['right']}"

    def test_turning_points_shift_with_offset(self):
        """ Raising the barrier moves the turning points outwards """
        energy = 0.03
        mult_factor = 1.5
        tp_low = find_turning_points(energy, _make_parabolic_potential(C=0.062))
        tp_high = find_turning_points(energy, _make_parabolic_potential(C=mult_factor*0.062))
        expected = np.sqrt((mult_factor*0.062 - energy) / 0.001)
        assert tp_high["right"] > tp_low["right"], \
        f"expected right > {tp_low['right']}, got {tp_high['right']}; test #1"
        assert np.isclose(tp_high["right"], expected, rtol=TP_RTOL), \
        f"expected {expected}, got {tp_high['right']}; test #2"


# ===========================================================================
# --- compute_wkb is compared with the analytic solution ---
# ===========================================================================
class TestWKBVsAnalytic:

    def _wkb(self, pot, energy):
        """ Run the real chain: turning points and WKB integral """
        tp = find_turning_points(energy, pot)
        wkb_int = compute_wkb(tp, energy, pot)
        result = float(np.asarray(wkb_int, dtype=float).ravel()[0])
        return result

    @pytest.mark.parametrize("energy", ENERGIES)
    def test_wkb_matches_analytic(self, energy):
        """ The numerical WKB integral matches the analytic formula """
        pot = _make_parabolic_potential()
        expected, _ = _analytic_integral_par_potential(pot, energy)
        got = self._wkb(pot, energy)
        assert np.isclose(got, expected, rtol=WKB_RTOL), \
        f"E={energy}: expected {expected}, got {got}"

    def test_wkb_is_linear_in_barrier_depth(self):
        """ WKB / (C - E) is constant for the parabolic barrier """
        pot = _make_parabolic_potential()
        ratios = [self._wkb(pot, energy) / (pot.offset_C - energy) for energy in ENERGIES]
        assert np.allclose(ratios, ratios[0], rtol=WKB_RTOL), f"expected a constant ratio, got {ratios}"

    def test_wkb_scales_as_inverse_sqrt_of_width_parameter(self):
        """ Quadrupling A halves the WKB integral """
        energy = 0.01
        wkb_narrow = self._wkb(_make_parabolic_potential(A=0.001), energy)
        wkb_wide = self._wkb(_make_parabolic_potential(A=4*0.001), energy)
        assert np.isclose(wkb_narrow / wkb_wide, 2.0, rtol=WKB_RTOL), \
        f"expected ratio={2}, got {wkb_narrow / wkb_wide}"

    def test_wkb_vanishes_at_barrier_top(self, tol_en=1e-6, tol_wkb=1e-3):
        """ The WKB integral tends to zero when the energy approaches the barrier top """
        pot = _make_parabolic_potential()
        energy = pot.offset_C * (1.0 - tol_en)
        got = self._wkb(pot, energy)
        assert 0.0 <= got < tol_wkb, f"expected a value close to {0.0}, got {got}"

    def test_wkb_is_positive_and_decreasing_with_energy(self):
        """ The WKB integral is positive and decreases with the energy """
        pot = _make_parabolic_potential()
        values = [self._wkb(pot, energy) for energy in ENERGIES]
        assert all(v > 0.0 for v in values), \
        f"expected positive values, got {values}; test #1"
        assert all(v2 < v1 for v1, v2 in zip(values, values[1:])), \
        f"expected decreasing values, got {values}; test #2"

    def test_wkb_does_not_depend_on_irc_range(self):
        """ Extending the IRC range beyond the turning points does not change the WKB integral """
        energy = 0.03
        wkb_short = self._wkb(_make_parabolic_potential(half_range=6.0), energy)
        wkb_long = self._wkb(_make_parabolic_potential(half_range=12.0), energy)
        assert np.isclose(wkb_short, wkb_long, rtol=WKB_RTOL), \
        f"expected {wkb_short}, got {wkb_long}"

    def test_wkb_becomes_negative_over_the_barrier(self):
        """ Over the barrier WKB becomes negative and wkb(e=barrier+C) = -wkb(e=barrier-C) """
        energies = (0.05, 0.07)
        pot = _make_parabolic_potential(C=0.06)
        wkb_under = self._wkb(pot, energies[0])
        wkb_over = self._wkb(pot, energies[1])
        assert wkb_over < 0, f"expected {wkb_over} < {0.0}; test #1"
        assert np.isclose(wkb_under, -wkb_over, rtol=WKB_RTOL), \
        f"expected {wkb_under} == - {wkb_over}; test #2"

    def test_wkb_becomes_zero_at_the_barrier(self):
        """ At the barrier WKB becomes 0.0 """
        energy = 0.06
        pot = _make_parabolic_potential(C=energy)
        wkb_exact = self._wkb(pot, energy)
        assert np.isclose(wkb_exact, 0.0, atol=WKB_RTOL), f"expected {0.0}, got {wkb_exact}"


# =====================================================================================
# --- Full pipeline on the parabolic barrier is compared with the analytic solution ---
# =====================================================================================
class TestPipelineVsAnalytic:

    def _levels(self, params, pot):
        """ Run the pipeline and return the level results as tuples """
        _, levels, _, _ = qmt_computations_pipeline(params, pot)
        return [astuple(level) for level in levels]

    def test_turning_points_of_all_levels(self):
        """ The turning points stored for each level match the analytic ones for the level energy """
        params = _make_params()
        pot = _make_parabolic_potential()
        for level in self._levels(params, pot):
            _, x0 = _analytic_integral_par_potential(pot, level[1])
            left, right = -x0, x0
            assert np.isclose(level[2], left, rtol=TP_RTOL), \
            f"level {level[0]}: expected left={left}, got {level[2]}; test #1"
            assert np.isclose(level[3], right, rtol=TP_RTOL), \
            f"level {level[0]}: expected right={right}, got {level[3]}; test #2"

    def test_wkb_of_all_levels(self):
        """ The WKB integral stored for each level matches the analytic one for the level energy """
        params = _make_params()
        pot = _make_parabolic_potential()
        for level in self._levels(params, pot):
            expected, _ = _analytic_integral_par_potential(pot, level[1])
            result = level[4]
            assert np.isclose(result, expected, rtol=WKB_RTOL), \
            f"level {level[0]}: expected {expected}, got {result}"

    def test_probabilities_of_all_levels_from_analytic_wkb(self, tol=1e-6):
        """ The stored probabilities equal transmission_prob applied to the analytic WKB integral """
        params = _make_params()
        pot = _make_parabolic_potential()
        for level in self._levels(params, pot):
            analytic_wkb, _ = _analytic_integral_par_potential(pot, level[1])
            expected_prob = 1 / (1 + np.exp(2.0 * analytic_wkb))
            expected_rate = params.freq * LIGHT_SPEED * 100 * expected_prob
            expected_half_life = np.log(2) / expected_rate
            assert np.isclose(level[5], expected_prob, rtol=tol), \
            f"level {level[0]}: expected prob={expected_prob}, got {level[5]}; test #1"
            assert np.isclose(level[6], expected_rate, rtol=tol), \
            f"level {level[0]}: expected rate={expected_rate}, got {level[6]}; test #2"
            assert np.isclose(level[7], expected_half_life, rtol=tol), \
            f"level {level[0]}: expected half-life={expected_half_life}, got {level[7]}; test #3"

    def test_analytic_probabilities_under_and_over_the_barrier_give_one(self, tol=1e-6):
        """ The sum of stored analytic probabilities under and over the barrier is 1.0 """
        params = _make_params()
        barrier = 0.06
        pot = _make_parabolic_potential(C=barrier)
        for level in self._levels(params, pot):
            analytic_wkb, _ = _analytic_integral_par_potential(pot, level[1])
            analytic_wkb_over, _ = _analytic_integral_par_potential(pot, 2 * barrier - level[1])
            expected_prob_under = 1 / (1 + np.exp(2.0 * analytic_wkb))
            expected_prob_over = 1 / (1 + np.exp(2.0 * analytic_wkb_over))
            expected_prob = expected_prob_under + expected_prob_over
            assert np.isclose(expected_prob, 1.0, rtol=tol), \
            (f"level {level[0]}: expected prob_sum={1.0}, "
             f"got {expected_prob_under} + {expected_prob_over} = {expected_prob}")

    def test_level_energy_matches_input_frequency(self):
        """ The level energies used by the pipeline are (i + 0.5) * freq in Hartree """
        params = _make_params(freq=1500.0)
        pot = _make_parabolic_potential()
        energies = [level[1] for level in self._levels(params, pot)]
        expected = [(i + 0.5) * 1500.0 * CM_M1_TO_HARTREE for i in range(params.number_of_levels + 1)]
        assert np.allclose(energies, expected, atol=1e-14), f"expected {expected}, got {energies}"


# ===========================================================================
# --- Closed-form temperature average (finite sum of the harmonic levels) on the parabolic barrier ---
# ===========================================================================
class TestClosedFormTemperatureAverageOnParabolicBarrier:

    @staticmethod
    def _closed_form_average(pot, freq, n_levels, temperature):
        """ Return the finite_sum averaging for the closed form """
        energies = np.array([(i + 0.5) * freq * CM_M1_TO_HARTREE for i in range(n_levels + 1)])
        probs = np.array([1.0 / (1.0 + np.exp(2.0 * _analytic_integral_par_potential(pot, e)[0])) for e in energies])
        weights = np.exp(-(energies - energies[0]) / (K_B * temperature))
        weights = weights / np.sum(weights)
        result = float(np.dot(weights, probs))
        return result

    @staticmethod
    def _aver_fields(params, pot):
        """ Run the pipeline and return the fields of the temperature-averaged result """
        _, _, aver, _ = qmt_computations_pipeline(params, pot)
        return astuple(aver)

    @pytest.mark.parametrize("temperature", [50.0, 300.0, 1000.0, 3000.0])
    def test_average_matches_closed_form(self, temperature):
        """ The averaged probability equals the Boltzmann-weighted sum of the analytic level probabilities """
        params = _make_params(T=temperature)
        pot = _make_parabolic_potential(A=0.002, C=0.03)
        # a thinner and lower barrier is needed for the decent tunneling probabilities
        got = self._aver_fields(params, pot)[0]
        expected = self._closed_form_average(pot, params.freq, params.number_of_levels, temperature)
        assert np.isclose(got, expected, rtol=1e-3), f"T={temperature}: expected {expected}, got {got}"

    def test_averaged_rate_and_half_life_match_closed_form(self):
        """ The averaged rate and half-life follow from the closed-form averaged probability """
        params = _make_params(T=600.0)
        pot = _make_parabolic_potential(A=0.002, C=0.03)
        # a thinner and lower barrier is needed for the decent tunneling probabilities
        fields = self._aver_fields(params, pot)
        prob = self._closed_form_average(pot, params.freq, params.number_of_levels, params.T)
        rate = params.freq * LIGHT_SPEED * 100.0 * prob
        assert np.isclose(fields[1], rate, rtol=1e-3), f"expected rate={rate}, got {fields[1]}; test #1"
        assert np.isclose(fields[2], np.log(2.0) / rate, rtol=5e-3), \
        f"expected half-life={np.log(2.0) / rate}, got {fields[2]}; test #2"

    def test_low_temperature_limit_is_ground_state(self):
        """ At a very low temperature only the ground state contributes """
        params = _make_params(T=1.0)
        pot = _make_parabolic_potential(A=0.002, C=0.03)
        # a thinner and lower barrier is needed for the decent tunneling probabilities
        got = self._aver_fields(params, pot)[0]
        energy0 = 0.5 * params.freq * CM_M1_TO_HARTREE
        expected = 1.0 / (1.0 + np.exp(2.0 * _analytic_integral_par_potential(pot, energy0)[0]))
        assert np.isclose(got, expected, rtol=1e-3), f"expected {expected}, got {got}"

    def test_high_temperature_limit_is_arithmetic_mean(self):
        """ At a very high temperature all the levels have the same weight """
        params = _make_params(T=1e8)
        pot = _make_parabolic_potential(A=0.002, C=0.03)
        # a thinner and lower barrier is needed for the decent tunneling probabilities
        got = self._aver_fields(params, pot)[0]
        energies = [(i + 0.5) * params.freq * CM_M1_TO_HARTREE for i in range(params.number_of_levels + 1)]
        expected = np.mean([1.0 / (1.0 + np.exp(2.0 * _analytic_integral_par_potential(pot, e)[0]))
            for e in energies])
        assert np.isclose(got, expected, rtol=1e-3), f"expected {expected}, got {got}"


# ============================================================================================
# --- _reaching_target_probability is compared with the analytic solution ---
# ============================================================================================
class TestReachingTargetProbabilityVsAnalytic:

    FREQ = 1000.0  # a model frequency for tests, in cm-1

    @pytest.mark.parametrize("target", [0.1, 0.4, 0.99])
    def test_closed_form_count_matches_reference_loop(self, target):
        """ Self-check: the closed-form number of the levels equals the one of the reference loop """
        pot = _make_parabolic_potential()
        loop = _analytic_probabilities_until_target(pot, self.FREQ, target)
        closed = _analytic_number_of_levels_closed_form(pot, self.FREQ, target)
        assert closed == len(loop), f"target={target}: expected {len(loop)} levels, got {closed}"

    @pytest.mark.parametrize("target", [0.1, 0.25, 0.4])
    def test_below_barrier_top_matches_analytic(self, target):
        """ For a target below 0.5 the result matches the analytic formula """
        pot = _make_parabolic_potential()
        probs, _, _ = _reaching_target_probability(self.FREQ, pot, target_probability=target)
        expected_lvl_num = _analytic_number_of_levels_closed_form(pot, self.FREQ, target)
        expected = _analytic_probabilities_until_target(pot, self.FREQ, target)
        assert len(probs) == expected_lvl_num, \
        f"target={target}: expected {expected_lvl_num} levels, got {len(probs)}; test #1"
        assert np.allclose(probs, expected, rtol=1e-3), \
        f"target={target}: expected {expected}, got {probs}; test #2"

    @pytest.mark.parametrize("target", [0.1, 0.25, 0.4])
    def test_zero_level_and_last_level_energy(self, target):
        """ The returned zero-level energy and the last level energy matchthe formula """
        pot = _make_parabolic_potential()
        probs, zero_level, level_energy = _reaching_target_probability(self.FREQ, pot, target_probability=target)
        freq = self.FREQ * CM_M1_TO_HARTREE
        assert np.isclose(zero_level, 0.5 * freq, rtol=1e-12), \
        f"expected zero level={0.5 * freq}, got {zero_level}; test #1"
        assert np.isclose(level_energy, (len(probs) - 0.5) * freq, rtol=1e-12), \
        f"expected last level={(len(probs) - 0.5) * freq}, got {level_energy}; test #2"

    @pytest.mark.parametrize("target", [0.1, 0.4, 0.99])
    def test_probabilities_grow_and_the_last_one_reaches_target(self, target):
        """ The probabilities do not decrease and only the last one reaches the target """
        pot = _make_parabolic_potential()
        probs, _, _ = _reaching_target_probability(self.FREQ, pot, target_probability=target)
        assert all(p2 >= p1 for p1, p2 in zip(probs, probs[1:])), \
        f"expected non-decreasing probabilities, got {probs}; test #1"
        assert probs[-1] >= target, \
        f"expected the last probability >= {target}, got {probs[-1]}; test #2"
        assert all(p < target for p in probs[:-1]), \
        f"expected all the previous probabilities < {target}; test #3"

    def test_default_target_crosses_barrier_top(self):
        """ The default target 0.99 needs levels above the barrier top """
        pot = _make_parabolic_potential()
        probs, _, level_energy = _reaching_target_probability(self.FREQ, pot)
        expected = _analytic_probabilities_until_target(pot, self.FREQ, 0.99)
        assert level_energy >= pot.offset_C, \
        f"expected the last level above the top {pot.offset_C}, got {level_energy}; test #1"
        assert len(probs) == len(expected), f"expected {len(expected)} levels, got {len(probs)}; test #2"
        assert np.allclose(probs, expected, rtol=1e-3), \
        f"expected ...{expected[-3:]}, got ...{probs[-3:]}; test #3"

    def test_max_levels_stops_the_loop_and_logs_warning(self):
        """ With max_levels smaller than needed the loop stops and a warning is logged """
        pot = _make_parabolic_potential()
        with patch(KINETICS_LOGGER) as logger:
            probs, _, level_energy = _reaching_target_probability(self.FREQ, pot, max_levels=10)
        expected = _analytic_probabilities_until_target(pot, self.FREQ, 0.99, max_levels=10)
        assert len(probs) == 10, f"expected {10} levels, got {len(probs)}; test #1"
        assert np.allclose(probs, expected, rtol=1e-3), f"expected {expected}, got {probs}; test #2"
        assert logger.warning.call_count == 1, f"expected {1} warning, got {logger.warning.call_count}; test #3"
        assert np.isclose(level_energy, 9.5 * self.FREQ * CM_M1_TO_HARTREE, rtol=1e-12), \
        f"expected last level={9.5 * self.FREQ * CM_M1_TO_HARTREE}, got {level_energy}; test #4"

    def test_no_warning_when_target_is_reached(self):
        """ No warning is logged when the target is reached before max_levels """
        pot = _make_parabolic_potential()
        with patch(KINETICS_LOGGER) as logger:
            _reaching_target_probability(self.FREQ, pot, target_probability=0.4)
        assert logger.warning.call_count == 0, f"expected {0} warnings, got {logger.warning.call_count}"


# ============================================================================================
# --- _temperature_averaging_infinite_sum is compared with the analytic solution ---
# ============================================================================================
class TestInfiniteSumAverageVsClosedForm:

    FREQ = 1000.0  # a model frequency for tests, in cm-1
    TARGET = 0.4  # a model target probabilty for tests, should belong to (0, 1)

    def _closed_form_infinite_sum(self, pot, temperature):
        """ The analytic probability averaging for the infinite_sum mode """
        probs = _analytic_probabilities_until_target(pot, self.FREQ, self.TARGET)
        n = len(probs)
        q = np.exp(-self.FREQ * CM_M1_TO_HARTREE / (K_B * temperature))
        body = (1.0 - q) * sum(q ** i * p for i, p in enumerate(probs))
        result = body + q ** n * 0.5 * (1.0 + probs[-1])
        return result
        # The formula used here is: P_aver = (1 - q) * sum(q ** i * P_i) + 0.5 * q ** n * (1 + P_last)

    @pytest.mark.parametrize("temperature", [300.0, 1e4, 1e5])
    def test_matches_closed_form(self, temperature):
        """ The averaged probability equals the closed form """
        pot = _make_parabolic_potential()
        probs, _, _ = _reaching_target_probability(self.FREQ, pot, target_probability=self.TARGET)
        got = _temperature_averaging_infinite_sum(self.FREQ, probs, temperature)
        expected = self._closed_form_infinite_sum(pot, temperature)
        assert np.isclose(got, expected, rtol=1e-3), f"T={temperature}: expected {expected}, got {got}"
        # The tail should be significant at high temperatures

    def test_low_temperature_limit_is_ground_state(self):
        """ At a very low temperature the average is the ground-state probability """
        pot = _make_parabolic_potential()
        temperature = 1.0
        probs, _, _ = _reaching_target_probability(self.FREQ, pot, target_probability=self.TARGET)
        got = _temperature_averaging_infinite_sum(self.FREQ, probs, temperature)
        expected = _analytic_probability_par_potential(pot, 0.5 * self.FREQ * CM_M1_TO_HARTREE)
        assert np.isclose(got, expected, rtol=1e-3), f"expected {expected}, got {got}"

    def test_high_temperature_limit_is_tail_probability(self):
        """ At a very high temperature the infinite tail dominates """
        pot = _make_parabolic_potential()
        temperature = 1e12
        probs, _, _ = _reaching_target_probability(self.FREQ, pot, target_probability=self.TARGET)
        got = _temperature_averaging_infinite_sum(self.FREQ, probs, temperature)
        expected = 0.5 * (1.0 + probs[-1])
        assert np.isclose(got, expected, rtol=1e-3), f"expected {expected}, got {got}"

    def test_average_grows_with_temperature(self):
        """ The averaged probability does not decrease with the temperature """
        pot = _make_parabolic_potential()
        probs, _, _ = _reaching_target_probability(self.FREQ, pot, target_probability=self.TARGET)
        values = [_temperature_averaging_infinite_sum(self.FREQ, probs, t)
            for t in (100.0, 300.0, 1e3, 1e4, 1e5, 1e6)]
        assert all(v2 >= v1 for v1, v2 in zip(values, values[1:])), \
        f"expected non-decreasing values, got {values}"


# ===============================================================================
# --- _integration_over_weighted_probs is compared with the analytic solution ---
# ===============================================================================
class TestIntegrationOverWeightedProbsVsAnalytic:

    FREQ = 1000.0  # a model frequency for tests, in cm-1
    ZERO_LEVEL = 0.5 * FREQ * CM_M1_TO_HARTREE
    # The zero-point level of the model frequency, in Hartree
    TARGET_ENERGY = 0.124
    # A target energy above the top of the default parabolic barrier (C = 0.062), in Hartree
    
    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_matches_analytic_integral(self, temperature):
        """ The integral equals the quadrature of the weighted analytic probability """
        pot = _make_parabolic_potential()
        got = _integration_over_weighted_probs(self.ZERO_LEVEL, self.TARGET_ENERGY, pot, temperature)
        expected = _analytic_weighted_integral(pot, self.ZERO_LEVEL, self.TARGET_ENERGY, temperature)
        assert np.isclose(got, expected, rtol=1e-3), f"T={temperature}: expected {expected}, got {got}"

    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_result_is_bounded_by_probabilities_at_the_ends(self, temperature, tol=1e-3):
        """
        The probability integral should be within P(zero_level) * w
        and P(target_energy) * w range, w = 1 - exp(-t_max) is the weight of the [0, t_max] interval
        (the weight of the [t_max, +inf] interval is exp(-t_max), respectively)
        """
        pot = _make_parabolic_potential()
        got = _integration_over_weighted_probs(self.ZERO_LEVEL, self.TARGET_ENERGY, pot, temperature)
        t_max = min((self.TARGET_ENERGY - self.ZERO_LEVEL) / (K_B * temperature), BORDER_FACTOR)
        weight = -np.expm1(-t_max)  # np.expm1(x) = e(x) - 1
        p_low = _analytic_probability_par_potential(pot, self.ZERO_LEVEL)
        p_high = _analytic_probability_par_potential(pot, self.TARGET_ENERGY)
        assert got >= p_low * weight * (1.0 - tol), \
        f"T={temperature}: expected >= {p_low * weight}, got {got}; test #1"
        assert got <= p_high * weight * (1.0 + tol), \
        f"T={temperature}: expected <= {p_high * weight}, got {got}; test #2"

    @pytest.mark.parametrize("border", [0.5, 2.0, 10.0, BORDER_FACTOR])
    def test_border_factor_sets_the_upper_limit(self, border):
        """ The upper limit of the integration is min(t_max, border_factor) """
        pot = _make_parabolic_potential()
        temperature = 300.0
        got = _integration_over_weighted_probs(self.ZERO_LEVEL, self.TARGET_ENERGY, pot,
            temperature, border_factor=border)
        expected = _analytic_weighted_integral(pot, self.ZERO_LEVEL, self.TARGET_ENERGY,
            temperature, border_factor=border)
        assert np.isclose(got, expected, rtol=1e-3), f"border={border}: expected {expected}, got {got}"

    def test_integral_does_not_decrease_with_border_factor(self, tol=1e-8):
        """ The integrand is positive, so a larger border_factor cannot decrease the integral """
        pot = _make_parabolic_potential()
        borders = [1.0, 5.0, 25.0, BORDER_FACTOR]
        values = [_integration_over_weighted_probs(self.ZERO_LEVEL, self.TARGET_ENERGY,
            pot, 300.0, border_factor=b) for b in borders]
        assert all(v2 >= v1 * (1.0 - tol) for v1, v2 in zip(values, values[1:])), \
        f"expected non-decreasing values, got {values}; test #1"
        assert values[0] < values[-1], f"expected the integral to grow, got {values}; test #2"

    @pytest.mark.parametrize("temperature, expected_calls", [(300.0, 4), (3000.0, 3), (1e4, 2), (1e5, 1)])
    def test_number_of_intervals_depends_on_upper_limit(self, temperature, expected_calls):
        """
        The integration is split by the points 0.0, 1.0, 5.0, 25.0, and t_cut;
        t_cut is defined as min(t_max, border_factor), the default border_factor is 92.2
        (t_max is about 128.1, 12.8, 3.8 and 0.38 for TARGET_ENERGY (0.124 Hartree) at these temperatures)
        """
        pot = _make_parabolic_potential()
        with patch(f"{KINETICS_MODULE}.int_convergence_warning") as warn:
            _integration_over_weighted_probs(self.ZERO_LEVEL, self.TARGET_ENERGY, pot, temperature)
        assert warn.call_count == expected_calls, \
        f"T={temperature}: expected {expected_calls} intervals, got {warn.call_count}; test #1"
        assert f"{temperature:.2f} K" in warn.call_args_list[0].args[2], \
        f"expected the temperature in the message, got {warn.call_args_list[0].args[2]}; test #2"


# ==============================================================================
# --- _temperature_averaging_integral is compared with the analytic solution ---
# ==============================================================================
class TestTemperatureAveragingIntegralVsAnalytic:

    FREQ = 1000.0  # a model frequency for tests, in cm-1
    
    @pytest.mark.parametrize("target", [0.4, 0.99])
    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_matches_closed_form(self, temperature, target):
        """
        The averaged probability equals the analytic integral plus the tail
        built from the last probability
        """
        pot = _make_parabolic_potential()
        probs, zero_level, target_energy = _analytic_levels(pot, self.FREQ, target)
        got = _temperature_averaging_integral(zero_level, target_energy, probs, pot, temperature)
        expected = _closed_form_integral_average(pot, zero_level, target_energy, temperature,
            tail_target=probs[-1])
        assert np.isclose(got, expected, rtol=1e-3), \
        f"T={temperature}, target={target}: expected {expected}, got {got}"

    def test_result_is_valid_and_grows_with_temperature(self, tol=1e-6):
        """ The averaged probability is in [P(zero_level), 1] and does not decrease with the temperature """
        pot = _make_parabolic_potential()
        probs, zero_level, target_energy = _analytic_levels(pot, self.FREQ, 0.99)
        temperatures = [10.0, 100.0, 300.0, 1e3, 1e4, 1e5, 1e6]
        values = [_temperature_averaging_integral(zero_level, target_energy, probs, pot, t)
            for t in temperatures]
        p_zero_level = _analytic_probability_par_potential(pot, zero_level)
        assert all(p_zero_level * (1.0 - tol) <= v <= 1.0 for v in values), \
        f"expected values in [{p_zero_level, 1.0}], got {values}; test #1"
        assert all(v2 >= v1 * (1.0 - tol) for v1, v2 in zip(values, values[1:])), \
        f"expected non-decreasing values, got {values}; test #2"

    def test_high_temperature_limit_is_tail_probability(self):
        """ At a very high temperature the tail dominates """
        pot = _make_parabolic_potential()
        probs, zero_level, target_energy = _analytic_levels(pot, self.FREQ, 0.99)
        got = _temperature_averaging_integral(zero_level, target_energy, probs, pot, 1e12)
        expected = 0.5 * (1.0 + probs[-1])
        assert np.isclose(got, expected, rtol=1e-3), f"expected {expected}, got {got}"


# ===========================================================================
# --- temperature_averaging_tunneling - wiring between the stages ---
# ===========================================================================
class TestAveragingWiring:

    FREQ = 1000.0  # a model frequency for tests, in cm-1
    
    @pytest.mark.parametrize("mode", SEARCH_MODES)
    def test_search_receives_target_and_max_levels(self, mode):
        """ The probability search gets the frequency, the potential, target_probability and max_levels """
        potential = object()
        with ExitStack() as stack:
            mocks = _patch_helpers(stack)
            temperature_averaging_tunneling(self.FREQ, [], potential, 300.0, prob_aver_mode=mode,
                target_probability=0.7, max_levels=50)
        mocks["reach"].assert_called_once_with(self.FREQ, potential, target_probability=0.7, max_levels=50)
        # it should be called once

    def test_infinite_sum_receives_search_output(self):
        """ The infinite_sum helper gets the frequency, the found probabilities and the temperature """
        with ExitStack() as stack:
            mocks = _patch_helpers(stack)
            temperature_averaging_tunneling(self.FREQ, [], object(), 300.0, prob_aver_mode="infinite_sum")
        mocks["infinite"].assert_called_once_with(self.FREQ, STUB_PROBS, 300.0)
        # it should be called once

    def test_integral_receives_search_output(self):
        """ The integral helper gets the zero level, the target energy, the probabilities, the potential, T """
        potential = object()
        with ExitStack() as stack:
            mocks = _patch_helpers(stack)
            temperature_averaging_tunneling(self.FREQ, [], potential, 300.0, prob_aver_mode="integral")
        mocks["integral"].assert_called_once_with(STUB_ZERO_LEVEL, STUB_TARGET_ENERGY, STUB_PROBS,
        potential, 300.0)
        # it should be called once

    @pytest.mark.parametrize("mode", ["finite_sum", "infinite_sum", "integral"])
    def test_returns_the_helper_value(self, mode):
        """ The value of the selected helper is returned as it is """
        with ExitStack() as stack:
            _patch_helpers(stack, prob=0.123)
            result = temperature_averaging_tunneling(self.FREQ, [], object(), 300.0, prob_aver_mode=mode)
        assert result == 0.123, f"expected {0.123}, got {result}"

    def test_finite_sum_ignores_search_parameters(self):
        """ The finite_sum mode does not use the search parameters, so invalid ones do not raise """
        with ExitStack() as stack:
            _patch_helpers(stack)
            result = temperature_averaging_tunneling(self.FREQ, [], object(), 300.0,
            prob_aver_mode="finite_sum", target_probability=1.5, max_levels=1)
        assert result == 0.5, f"expected {0.5}, got {result}"
    
    def test_invalid_mode_calls_nothing(self):
        """ An invalid mode is rejected before any helper is called. Expecting a ValueError """
        with ExitStack() as stack:
            mocks = _patch_helpers(stack)
            with pytest.raises(ValueError, match="Expected prob_aver_mode"):
                temperature_averaging_tunneling(self.FREQ, [], object(), 300.0, prob_aver_mode="bogus")
        assert not any(mock.called for mock in mocks.values()), "expected no helper calls"

    @pytest.mark.parametrize("mode", SEARCH_MODES)
    @pytest.mark.parametrize("kwargs", [{"target_probability": 1.5}, {"max_levels": 3}])
    def test_search_errors_propagate(self, mode, kwargs):
        """ Invalid search parameters raise an error in the search modes. Expecting a ValueError """
        with pytest.raises(ValueError): # the empty list of energy levels data shoul call an error
            temperature_averaging_tunneling(self.FREQ, [], _make_parabolic_potential(), 300.0,
            prob_aver_mode=mode, **kwargs)


# ===========================================================================
# --- arrhenius_rate - wiring between the stages ---
# ===========================================================================
class TestArrheniusWiring:

    FREQ = 1000.0  # a model frequency for tests, in cm-1
    
    @pytest.mark.parametrize("mode", SEARCH_MODES)
    def test_search_is_called_once_with_search_parameters(self, mode):
        """ The expensive probability search is performed once, not for each temperature """
        potential = object()
        with ExitStack() as stack:
            mocks = _patch_helpers(stack)
            arrhenius_rate([], self.FREQ, 200.0, 300.0, 50.0, potential, prob_aver_mode=mode,
            target_probability=0.7, max_levels=50)
        mocks["reach"].assert_called_once_with(self.FREQ, potential, target_probability=0.7, max_levels=50)
        # it should be called once

    def test_finite_sum_receives_levels_and_each_temperature(self):
        """ The finite_sum helper gets the levels and each temperature of the grid """
        levels = [object()]
        with ExitStack() as stack:
            mocks = _patch_helpers(stack)
            arrhenius_rate(levels, self.FREQ, 200.0, 300.0, 50.0, None)
        calls = [(c.args[0], c.args[1]) for c in mocks["finite"].call_args_list]
        expected = [(levels, 300.0), (levels, 250.0), (levels, 200.0)]
        assert calls == expected, f"expected {expected}, got {calls}"

    def test_infinite_sum_receives_search_output_and_each_temperature(self):
        """ The infinite_sum helper gets the frequency, the found probabilities and each temperature """
        with ExitStack() as stack:
            mocks = _patch_helpers(stack)
            arrhenius_rate([], self.FREQ, 200.0, 300.0, 50.0, None, prob_aver_mode="infinite_sum")
        calls = [c.args for c in mocks["infinite"].call_args_list]
        expected = [(self.FREQ, STUB_PROBS, t) for t in (300.0, 250.0, 200.0)]
        assert calls == expected, f"expected {expected}, got {calls}"

    def test_integral_receives_search_output_and_each_temperature(self):
        """
        The integral helper gets the zero level, the target energy,
        the probabilities, the potential, and T
        """
        potential = object()
        with ExitStack() as stack:
            mocks = _patch_helpers(stack)
            arrhenius_rate([], self.FREQ, 200.0, 300.0, 50.0, potential, prob_aver_mode="integral")
        calls = [c.args for c in mocks["integral"].call_args_list]
        expected = [(STUB_ZERO_LEVEL, STUB_TARGET_ENERGY, STUB_PROBS, potential, t)
            for t in (300.0, 250.0, 200.0)]
        assert calls == expected, f"expected {expected}, got {calls}"

    def test_zero_rate_does_not_stop_the_loop(self):
        """
        A zero rate gives -inf only for its own temperature,
        the next temperatures are still computed
        """
        with ExitStack() as stack:
            mocks = _patch_helpers(stack)
            mocks["finite"].side_effect = [0.0, 0.5, 0.6]
            result = arrhenius_rate([], self.FREQ, 200.0, 300.0, 50.0, None)
        logs = [data.log_reaction_rate for data in result]
        assert np.isneginf(logs[0]), f"expected {-np.inf}, got {logs[0]}; test #1"
        assert np.all(np.isfinite(logs[1:])), f"expected finite values, got {logs[1:]}; test #2"


# ===========================================================================
# --- arrhenius_rate - properties of the temperature grid ---
# ===========================================================================
class TestArrheniusGrid:

    FREQ = 1000.0  # a model frequency for tests, in cm-1
    
    @pytest.mark.parametrize("T_min, T_max, T_step", [
    (200.0, 400.0, 33.3), (273.15, 373.15, 10.0), (1.0, 1.7, 0.1), (50.0, 300.0, 0.7)])
    def test_grid_properties(self, T_min, T_max, T_step, tol=1e-9):
        """ The grid starts at T_max, strictly decreases, ends at T_min and has no duplicates """
        temperatures = _arr_grid(T_min, T_max, T_step, self.FREQ)
        expected_len = int(np.ceil((T_max - T_min) / T_step - tol)) + 1
        assert temperatures[0] == T_max, f"expected first={T_max}, got {temperatures[0]}; test #1"
        assert np.all(np.diff(temperatures) < 0), \
        f"expected strictly decreasing grid, got {temperatures}; test #2"
        assert np.isclose(temperatures[-1], T_min, rtol=1e-5), \
        f"expected last={T_min}, got {temperatures[-1]}; test #3"
        assert len(temperatures) == expected_len, \
        f"expected {expected_len} points, got {len(temperatures)}; test #4"

    def test_equal_limits_give_one_point(self):
        """ T_min == T_max gives the single point """
        temperatures = _arr_grid(300.0, 300.0, 50.0, self.FREQ)
        assert temperatures == [300.0], f"expected {[300.0]}, got {temperatures}"

    def test_zero_t_min_is_not_computed(self):
        """ The temperature of 0 K is skipped (1 / T is not defined) """
        temperatures = _arr_grid(0.0, 100.0, 50.0, self.FREQ)
        assert temperatures == [100.0, 50.0], f"expected {[100.0, 50.0]}, got {temperatures}"


# ================================================================================
# --- temperature_averaging_tunneling - is compared with the analytic solution ---
# ================================================================================
class TestAveragingVsAnalytic:

    FREQ = 1000.0  # a model frequency for tests, in cm-1
    N_LEVELS = 3  # a typical number of energy levels
    
    @pytest.mark.parametrize("temperature", [100.0, 300.0, 1e4])
    def test_finite_sum_matches_closed_form(self, temperature):
        """ The finite_sum mode equals the weighted sum of the analytic level probabilities """
        pot = _make_parabolic_potential()
        got = temperature_averaging_tunneling(self.FREQ,
            _make_level_data(pot, self.N_LEVELS, self.FREQ), pot, temperature)
        expected = _expected_temp_average("finite_sum", pot, temperature, self.FREQ, self.N_LEVELS)
        assert np.isclose(got, expected, rtol=1e-12), f"T={temperature}: expected {expected}, got {got}"

    @pytest.mark.parametrize("target", [0.4, 0.99])
    @pytest.mark.parametrize("temperature", [100.0, 300.0, 1e4])
    @pytest.mark.parametrize("mode", SEARCH_MODES)
    def test_search_modes_match_closed_form(self, mode, temperature, target):
        """ The infinite_sum and integral modes equal the analytic closed forms """
        pot = _make_parabolic_potential()
        got = temperature_averaging_tunneling(self.FREQ, [], pot, temperature, prob_aver_mode=mode,
            target_probability=target)
        expected = _expected_temp_average(mode, pot, temperature, self.FREQ, self.N_LEVELS, target=target)
        assert np.isclose(got, expected, rtol=1e-3), \
        f"mode={mode}, T={temperature}, target={target}: expected {expected}, got {got}"

    def test_low_temperature_all_modes_give_ground_state(self):
        """ At a very low temperature all the modes give the ground-state probability """
        pot = _make_parabolic_potential()
        expected = _analytic_probability_par_potential(pot, 0.5 * self.FREQ * CM_M1_TO_HARTREE)
        for mode in ["finite_sum", "infinite_sum", "integral"]:
            got = temperature_averaging_tunneling(self.FREQ,
            _level_data_for(mode, pot, self.N_LEVELS, self.FREQ), pot, 0.5, prob_aver_mode=mode)
            assert np.isclose(got, expected, rtol=1e-3), f"mode={mode}: expected {expected}, got {got}"

    @pytest.mark.parametrize("mode", SEARCH_MODES)
    def test_max_levels_flows_into_search(self, mode):
        """ A small max_levels stops the search and the result stays a valid probability """
        pot = _make_parabolic_potential()
        with patch(KINETICS_LOGGER) as logger:
            got = temperature_averaging_tunneling(self.FREQ, [], pot, 300.0, prob_aver_mode=mode, max_levels=10)
        assert logger.warning.call_count == 1, f"expected {1} warning, got {logger.warning.call_count}; test #1"
        assert 0.0 <= got <= 1.0, f"expected a probability in {[0, 1]}, got {got}; test #2"


# ===========================================================================
# --- arrhenius_rate - the real chain is compared with the analytic solution ---
# ===========================================================================
class TestArrheniusVsAnalytic:

    FREQ = 1000.0  # a model frequency for tests, in cm-1
    N_LEVELS = 3  # a typical number of energy levels
    TEMPERATURES = [400.0, 350.0, 300.0, 250.0, 200.0]
    # a list of temperatures for the tests

    @pytest.mark.parametrize("mode", ["finite_sum", "infinite_sum", "integral"])
    def test_log_rate_matches_closed_form(self, mode):
        """ The temperatures, 1/T and ln(k) match the closed-form averaged probability for each mode """
        pot = _make_parabolic_potential()
        result = arrhenius_rate(_level_data_for(mode, pot, self.N_LEVELS, self.FREQ),
        self.FREQ, 200.0, 400.0, 50.0, pot, prob_aver_mode=mode)
        temperatures = [data.arrhenius_temperature for data in result]
        inverse = [data.arrhenius_temperature_inv for data in result]
        logs = [data.log_reaction_rate for data in result]
        expected_logs = [np.log(self.FREQ * LIGHT_SPEED * 100.0 *
        _expected_temp_average(mode, pot, t, self.FREQ, self.N_LEVELS)) for t in self.TEMPERATURES]
        assert np.allclose(temperatures, self.TEMPERATURES), \
        f"expected {self.TEMPERATURES}, got {temperatures}; test #1"
        assert np.allclose(inverse, 1.0 / np.array(self.TEMPERATURES)), \
        f"expected the reciprocals of {self.TEMPERATURES}, got {inverse}; test #2"
        assert np.allclose(logs, expected_logs, rtol=0.0, atol=1e-3), \
        f"mode={mode}: expected {expected_logs}, got {logs}; test #3"

    @pytest.mark.parametrize("mode", ["finite_sum", "infinite_sum", "integral"])
    def test_log_rate_does_not_increase_towards_low_temperature(self, mode, comp_tol=1e-6):
        """ The grid goes from high to low T, so ln(k) should not increase along it """
        pot = _make_parabolic_potential()
        result = arrhenius_rate(_level_data_for(mode, pot, self.N_LEVELS, self.FREQ),
        self.FREQ, 100.0, 1000.0, 100.0, pot, prob_aver_mode=mode)
        logs = [data.log_reaction_rate for data in result]
        assert all(l2 <= l1 + comp_tol for l1, l2 in zip(logs, logs[1:])), \
        f"expected non-increasing, got {logs}"

    @pytest.mark.parametrize("mode", ["finite_sum", "infinite_sum"])
    def test_low_temperature_plateau_is_ground_state_rate(self, mode):
        """ At low temperatures ln(k) becomes temperature independent and equals the ground-state one """
        pot = _make_parabolic_potential()
        result = arrhenius_rate(_level_data_for(mode, pot, self.N_LEVELS, self.FREQ),
        self.FREQ, 5.0, 20.0, 5.0, pot, prob_aver_mode=mode)
        logs = [data.log_reaction_rate for data in result]
        p_zero_level = _analytic_probability_par_potential(pot, 0.5 * self.FREQ * CM_M1_TO_HARTREE)
        expected = np.log(self.FREQ * LIGHT_SPEED * 100.0 * p_zero_level)
        assert np.allclose(logs, expected, rtol=0.0, atol=1e-6), f"expected {expected}, got {logs}"
        # for the "integral" mode there is no plato at the 5-20 K within the given accuracy,
        # so this test is skipped