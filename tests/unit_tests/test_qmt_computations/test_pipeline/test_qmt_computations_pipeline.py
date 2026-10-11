# --- Test: test_qmt_computations_pipeline. Unit tests for qmt_computations_pipeline.py
# Run with: pytest test_qmt_computations_pipeline.py -v ---


# --- Modules ---
import pytest # type: ignore
from unittest.mock import MagicMock, patch
from contextlib import ExitStack

import numpy as np

from tunnex_2.constants_and_dataclasses.constants_and_settings import ( # type: ignore
    CM_M1_TO_HARTREE,
    GRID_SIZE,
    HOUR, DAY, YEAR)


# --- Module to test ---
from tunnex_2.qmt_computations.pipeline.qmt_computations_pipeline import qmt_computations_pipeline # type: ignore


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.qmt_computations.pipeline.qmt_computations_pipeline"


def _make_params(upper_level=2, freq=1500.0, T=298.15,
        T_min=200.0, T_max=400.0, T_step=10.0, prob_aver_mode="finite_sum"):
    """ Should build a mock params object with the attributes the pipeline reads """
    params = MagicMock()
    params.number_of_levels = upper_level
    params.freq = freq
    params.T = T
    params.T_arrhenius_min = T_min
    params.T_arrhenius_max = T_max
    params.T_step = T_step
    params.prob_aver_mode_qmt = prob_aver_mode
    return params

def _make_potential(x_min=-1.0, x_max=1.0, energy_value=0.5):
    """ Should build a mock potential callable with irc_x_min and irc_x_max bounds """
    potential = MagicMock()
    potential.irc_x_min = x_min
    potential.irc_x_max = x_max
    potential.side_effect = lambda x_grid: np.full_like(x_grid, energy_value)
    return potential

def _run_pipeline(params, potential, turning_points=None,
    wkb_value=1.0, prob_value=0.1, rate_value=10.0,
    half_life_value=0.05, prob_aver_value=0.2,
    arrhenius_value="arrhenius_result"):
    """ Should patch every collaborator of the pipeline and run it """
    turning_points = turning_points if turning_points is not None else {"left": -0.5, "right": 0.5}
    mocks = {}
    with ExitStack() as stack:
        mocks["find_turning_points"] = stack.enter_context(
            patch(FUNC_PATH + ".find_turning_points", return_value=turning_points))
        mocks["compute_wkb"] = stack.enter_context(
            patch(FUNC_PATH + ".compute_wkb", return_value=wkb_value))
        mocks["transmission_prob"] = stack.enter_context(
            patch(FUNC_PATH + ".transmission_prob", return_value=prob_value))
        mocks["reaction_rate"] = stack.enter_context(
            patch(FUNC_PATH + ".reaction_rate", return_value=rate_value))
        mocks["half_life"] = stack.enter_context(
            patch(FUNC_PATH + ".half_life", return_value=half_life_value))
        mocks["temperature_averaging_tunneling"] = stack.enter_context(
            patch(FUNC_PATH + ".temperature_averaging_tunneling", return_value=prob_aver_value))
        mocks["arrhenius_rate"] = stack.enter_context(
            patch(FUNC_PATH + ".arrhenius_rate", return_value=arrhenius_value))
        mocks["LevelResult"] = stack.enter_context(
            patch(FUNC_PATH + ".LevelResult", side_effect=lambda *a, **kw: ("LevelResult", a, kw)))
        mocks["TemperatureAverResult"] = stack.enter_context(
            patch(FUNC_PATH + ".TemperatureAverResult",
            side_effect=lambda *a, **kw: ("TemperatureAverResult", a, kw)))
        result = qmt_computations_pipeline(params, potential)
    return mocks, result


# ===========================================================================
# --- qmt_computations_pipeline - normal behaviour and input validation ---
# ===========================================================================
class TestQmtComputationsPipeline:

    def test_returns_structured_array_with_correct_fields(self):
        """
        Verify that the interpolated IRC grid is a structured array
        with irc_value and energy_value fields
        """
        params = _make_params(upper_level=0)
        potential = _make_potential()
        _, (grid, _, _, _) = _run_pipeline(params, potential)
        expected = ("irc_value", "energy_value")
        assert grid.dtype.names == expected, f"expected {expected}, got {grid.dtype.names}"

    def test_grid_has_correct_length_and_bounds(self):
        """ Verify that the interpolated IRC grid spans irc_x_min to irc_x_max with GRID_SIZE points """
        params = _make_params(upper_level=0)
        potential = _make_potential(x_min=-2.0, x_max=2.0)
        _, (grid, _, _, _) = _run_pipeline(params, potential)
        assert len(grid) == GRID_SIZE, f"expected {GRID_SIZE}, got {len(grid)}; test #1"
        assert grid["irc_value"][0] == -2.0, f"expected {-2.0}, got {grid['irc_value'][0]}; test #2"
        assert grid["irc_value"][-1] == 2.0, f"expected {2.0}, got {grid['irc_value'][-1]}; test #3"

    def test_grid_energy_values_come_from_potential_call(self):
        """ Verify that the energy values in the grid come from calling the potential on the x grid """
        params = _make_params(upper_level=0)
        potential = _make_potential(energy_value=3.5)
        _, (grid, _, _, _) = _run_pipeline(params, potential)
        assert np.allclose(grid["energy_value"], 3.5), f"expected all {3.5}, got {grid['energy_value']}"

    def test_computes_one_level_result_per_vibrational_level(self):
        """ Verify that the number of level results equals number_of_levels + 1 """
        params = _make_params(upper_level=3)
        potential = _make_potential()
        _, (_, level_results, _, _) = _run_pipeline(params, potential)
        assert len(level_results) == 4, f"expected {4}, got {len(level_results)}"

    def test_calls_find_turning_points_for_each_level_with_correct_energy(self):
        """ Verify that find_turning_points is invoked once per level with the correct level energy """
        params = _make_params(upper_level=2, freq=1000.0)
        potential = _make_potential()
        mocks, _ = _run_pipeline(params, potential)
        expected_energies = [(i + 0.5) * 1000.0 * CM_M1_TO_HARTREE for i in range(3)]
        actual_energies = [c.args[0] for c in mocks["find_turning_points"].call_args_list]
        assert np.allclose(actual_energies, expected_energies), f"expected {expected_energies}, "
        f"got {actual_energies}"

    def test_calls_compute_wkb_with_turning_points_and_potential(self):
        """ Verify that compute_wkb receives the turning points, level energy and potential """
        params = _make_params(upper_level=0, freq=500.0)
        potential = _make_potential()
        turning_points = {"left": -0.7, "right": 0.7}
        mocks, _ = _run_pipeline(params, potential, turning_points=turning_points)
        args = mocks["compute_wkb"].call_args.args
        assert args[0] == turning_points, f"expected {turning_points}, got {args[0]}; test #1"
        assert args[2] is potential, f"expected {potential}, got {args[2]}; test #2"

    def test_calls_transmission_prob_with_wkb(self):
        """ Verify that transmission_prob receives the wkb value """
        params = _make_params(upper_level=0, freq=500.0)
        potential = _make_potential()
        wkb_value_int = 2.0
        mocks, _ = _run_pipeline(params, potential, wkb_value=wkb_value_int)
        args = mocks["transmission_prob"].call_args.args
        assert args[0] == wkb_value_int, f"expected {wkb_value_int}, got {args[0]}"

    def test_calls_reaction_rate_with_freq_and_probability(self):
        """
        Verify that reaction_rate is called with
        the vibrational frequency and transmission probability
        """
        params = _make_params(upper_level=0, freq=1234.0)
        potential = _make_potential()
        mocks, _ = _run_pipeline(params, potential, prob_value=0.42)
        mocks["reaction_rate"].assert_any_call(1234.0, 0.42)
        # it should be called

    def test_calls_half_life_with_reaction_rate_value(self):
        """ Verify that half_life is called with the reaction rate value returned earlier """
        params = _make_params(upper_level=0)
        potential = _make_potential()
        mocks, _ = _run_pipeline(params, potential, rate_value=99.0)
        mocks["half_life"].assert_any_call(99.0)
        # it should be called

    def test_level_result_constructed_with_index_and_energy(self):
        """ Verify that LevelResult is constructed with the vibrational index and computed level energy """
        params = _make_params(upper_level=1, freq=1000.0)
        potential = _make_potential()
        mocks, _ = _run_pipeline(params, potential)
        first_call_args = mocks["LevelResult"].call_args_list[0].args
        expected_energy = 0.5 * 1000.0 * CM_M1_TO_HARTREE
        assert first_call_args[0] == 0, f"expected {0}, got {first_call_args[0]}; test #1"
        assert np.isclose(first_call_args[1], expected_energy), f"expected {expected_energy}, "
        f"got {first_call_args[1]}; test #2"

    def test_calls_temperature_averaging_tunneling_with_correct_arguments(self):
        """
        Verify that temperature_averaging_tunneling receives
        freq, level results, potential, and temperature
        """
        params = _make_params(upper_level=1, freq=800.0, T=310.0)
        potential = _make_potential()
        mocks, _ = _run_pipeline(params, potential)
        args = mocks["temperature_averaging_tunneling"].call_args.args
        assert args[0] == 800.0, f"expected {800.0}, got {args[0]}; test #1"
        assert args[2] is potential, f"expected {potential}, got {args[2]}; test #2"
        assert args[3] == 310.0, f"expected {310.0}, got {args[3]}; test #3"

    def test_computes_averaged_reaction_rate_from_prob_aver(self):
        """
        Verify that reaction_rate is also called
        with the temperature-averaged probability
        """
        params = _make_params(upper_level=0, freq=600.0)
        potential = _make_potential()
        mocks, _ = _run_pipeline(params, potential, prob_aver_value=0.33)
        mocks["reaction_rate"].assert_any_call(600.0, 0.33)
        # it should be called

    def test_returns_temperature_averaged_result_object(self):
        """ Verify that the averaged results are wrapped in a TemperatureAverResult dataclass """
        params = _make_params(upper_level=0)
        potential = _make_potential()
        _, (_, _, aver, _) = _run_pipeline(params, potential)
        expected = "TemperatureAverResult"
        assert aver[0] == expected, f"expected {expected}, got {aver[0]}"

    def test_calls_arrhenius_rate_with_temperature_range_and_mode(self):
        """ Verify that arrhenius_rate receives the temperature range, potential, and probability mode """
        params = _make_params(upper_level=0, freq=900.0, T_min=150.0, T_max=350.0, T_step=25.0)
        potential = _make_potential()
        mocks, _ = _run_pipeline(params, potential)
        args = mocks["arrhenius_rate"].call_args.args
        assert args[1] == 900.0, f"expected {900.0}, got {args[1]}; test #1"
        assert args[2] == 150.0, f"expected {150.0}, got {args[2]}; test #2"
        assert args[3] == 350.0, f"expected {350.0}, got {args[3]}; test #3"
        assert args[4] == 25.0, f"expected {25.0}, got {args[4]}; test #4"
        assert args[5] is potential, f"expected {potential}, got {args[5]}; test #5"
        assert args[6] == "finite_sum", f"expected {'finite_sum'}, got {args[6]}; test #6"

    def test_returns_arrhenius_rate_value_unmodified(self):
        """ Verify that the value returned by arrhenius_rate is passed through as the final pipeline output """
        params = _make_params(upper_level=0)
        potential = _make_potential()
        expected = "my_arrhenius_data"
        _, (_, _, _, arrhenius) = _run_pipeline(params, potential, arrhenius_value=expected)
        assert arrhenius == expected, f"expected {expected}, got {arrhenius}"

    def test_returns_a_four_element_tuple(self):
        """ Verify that the pipeline returns exactly four values """
        params = _make_params(upper_level=0)
        potential = _make_potential()
        _, result = _run_pipeline(params, potential)
        assert len(result) == 4, f"expected {4}, got {len(result)}"

# ===========================================================================
# --- qmt_computations_pipeline - additional coverage ---
# ===========================================================================
class TestQmtComputationsPipelineAdditional:

    def test_temperature_averaging_tunneling_receives_prob_aver_mode_from_params(self):
        """
        Verify that temperature_averaging_tunneling receives
        the probability-averaging mode from params
        """
        expected = "integral"
        params = _make_params(upper_level=0, prob_aver_mode=expected)
        potential = _make_potential()
        mocks, _ = _run_pipeline(params, potential)
        args = mocks["temperature_averaging_tunneling"].call_args.args
        assert args[4] == expected, f"expected {expected}, got {args[4]}"

    def test_arrhenius_rate_receives_custom_prob_aver_mode_from_params(self):
        """
        Verify that arrhenius_rate is forwarded the same custom
        probability-averaging mode as the temperature averaging
        """
        expected = "infinite_sum"
        params = _make_params(upper_level=0, prob_aver_mode=expected)
        potential = _make_potential()
        mocks, _ = _run_pipeline(params, potential)
        args = mocks["arrhenius_rate"].call_args.args
        assert args[6] == expected, f"expected {expected}, got {args[6]}"

    def test_level_result_includes_half_life_conversions(self):
        """
        Verify that LevelResult is constructed with the half-life
        converted to hours, days, and years
        """
        params = _make_params(upper_level=0)
        potential = _make_potential()
        mocks, _ = _run_pipeline(params, potential, half_life_value=2.0)
        call_args = mocks["LevelResult"].call_args_list[0].args
        assert call_args[7] == 2.0, f"expected {2.0}, got {call_args[7]}; test #1"
        assert np.isclose(call_args[8], 2.0 * HOUR), f"expected {2.0 * HOUR}, "
        f"got {call_args[8]}; test #2"
        assert np.isclose(call_args[9], 2.0 * DAY), f"expected {2.0 * DAY}, "
        f"got {call_args[9]}; test #3"
        assert np.isclose(call_args[10], 2.0 * YEAR), f"expected {2.0 * YEAR}, "
        f"got {call_args[10]}; test #4"

    def test_calls_compute_wkb_with_level_energy_as_second_argument(self):
        """
        Verify that compute_wkb receives the correct level energy
        for each vibrational level, not just turning points
        """
        params = _make_params(upper_level=1, freq=1000.0)
        potential = _make_potential()
        mocks, _ = _run_pipeline(params, potential)
        calls = mocks["compute_wkb"].call_args_list
        expected_energies = [(i + 0.5) * 1000.0 * CM_M1_TO_HARTREE for i in range(2)]
        actual_energies = [c.args[1] for c in calls]
        assert np.allclose(actual_energies, expected_energies), f"expected {expected_energies}, "
        f"got {actual_energies}"