# --- Test: test_wkb. Unit tests for wkb.py
# Run with: pytest test_wkb.py -v ---


# --- Modules ---
import pytest # type: ignore
from unittest.mock import MagicMock, patch

import numpy as np


# --- Module to test ---
from tunnex_2.qmt_computations.core.wkb import ( # type: ignore
    _integrand_wkb,
    int_convergence_warning,
    compute_wkb)


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.qmt_computations.core.wkb"


def _make_potential(return_value=1.0):
    """ Should build a potential that returns a fixed value regardless of x """
    potential = MagicMock(return_value=return_value)
    return potential

def _make_turning_points(regime, **overrides):
    """ Should build a turning_points dict for the given regime """
    base = {"regime": regime}
    base.update(overrides)
    return base


# ===========================================================================
# --- _integrand_wkb - normal behaviour and input validation ---
# ===========================================================================
class TestIntegrandWKB:

    def test_matches_formula_when_potential_exceeds_level(self):
        """ Verify that _integrand_wkb matches the formula when the difference is positive """
        potential = _make_potential(return_value=5.0)
        with patch(FUNC_PATH + ".REVELO", new=2.0):
             result = _integrand_wkb(0.0, potential, level_value=1.0)
        expected = np.sqrt(2 * 2.0 * (5.0 - 1.0))
        assert np.isclose(result, expected), f"expected {expected}, got {result}"

    def test_returns_zero_when_potential_equals_level(self):
        """ Verify that _integrand_wkb returns exactly zero when potential(x) equals level_value """
        potential = _make_potential(return_value=3.0)
        with patch(FUNC_PATH + ".REVELO", new=1.0):
             result = _integrand_wkb(0.0, potential, level_value=3.0)
        assert result == 0.0, f"expected {0.0}, got {result}"

    def test_clamps_negative_difference_to_zero(self):
        """ Verify that a slightly negative difference is clamped to zero """
        potential = _make_potential(return_value=0.999999999999)
        with patch(FUNC_PATH + ".REVELO", new=1.0):
             result = _integrand_wkb(0.0, potential, level_value=1.0)
        assert result == 0.0, f"expected {0.0}, got {result}"

    def test_calls_potential_with_the_given_x(self):
        """ Verify that _integrand_wkb evaluates the potential at the given x-value """
        potential = _make_potential(return_value=10.0)
        with patch(FUNC_PATH + ".REVELO", new=1.0):
             _integrand_wkb(0.42, potential, level_value=0.0)
        potential.assert_called_once_with(0.42)
        # it should be called once


# ===========================================================================
# --- int_convergence_warning - normal behaviour and input validation ---
# ===========================================================================
class TestIntConvergenceWarning:

    def test_returns_false_and_does_not_log_without_problem(self):
        """ Verify that the function does not log without a problem """
        logger = MagicMock()
        result = int_convergence_warning((1.0, 1e-14, {}), logger, "info")
        assert result is False, f"expected {False}, got {result}"
        logger.warning.assert_not_called()
        # it should not be called

    def test_logs_and_returns_true_on_problem(self):
        logger = MagicMock()
        res = (1.0, 0.5, {}, "  The maximum number of subdivisions (50) has been achieved.\n")
        result = int_convergence_warning(res, logger, "info")
        assert result is True, f"expected {True}, got {result}; test #1"
        logger.warning.assert_called_once()
        # it should be called once
        _, _, message, err, integral = logger.warning.call_args.args
        assert message == message.strip(), f"expected {message.strip()}, got {message}; test #2"
        expected = "maximum number of subdivisions"
        assert expected in message, f"expected {expected} in {message}; test #3"
        assert err == 0.5, f"expected {0.5}, got {err}; test #4"
        assert integral == 1.0, f"expected {1.0}, got {integral}; test #5"


# ===========================================================================
# --- compute_wkb - normal behaviour ---
# ===========================================================================
class TestComputeWkbNormal:

    def test_exact_match_returns_zero(self):
        """
        Verify that the EXACT_MATCH over-the-barrier status
        returns exactly 0.0 without integrating
        """
        turning_points = _make_turning_points("OVER THE BARRIER", OVER_STATUS="EXACT_MATCH")
        with patch(FUNC_PATH + ".quad") as mock_quad:
             result = compute_wkb(turning_points, 1.0, _make_potential())
        assert result == 0.0, f"expected {0.0}, got {result}"
        mock_quad.assert_not_called() # it should not be called

    def test_endothermic_reaction_returns_positive_infinity(self):
        """
        Verify that the ENDOTHERMIC REACTION regime
        returns +inf without integrating
        """
        turning_points = _make_turning_points("ENDOTHERMIC REACTION", left=-0.5, right=None)
        with patch(FUNC_PATH + ".quad") as mock_quad:
             result = compute_wkb(turning_points, 0.3, _make_potential())
        assert result == np.inf, f"expected {np.inf}, got {result}"
        mock_quad.assert_not_called() # it should not be called

    def test_integrates_between_left_and_right_turning_points(self):
        """
        Verify that quad is called with the left and right
        turning points as integration bounds
        """
        turning_points = _make_turning_points("NORMAL", left=-0.7, right=0.7)
        with patch(FUNC_PATH + ".quad", return_value=(1.23, {})) as mock_quad:
             compute_wkb(turning_points, 0.5, _make_potential())
        args = mock_quad.call_args.args
        assert args[1] == -0.7, f"expected {-0.7}, got {args[1]}; test #1"
        assert args[2] == 0.7, f"expected {0.7}, got {args[2]}; test #2"

    def test_returns_the_integrated_value(self):
        """ Verify that the function returns the first element of the quad result unchanged """
        turning_points = _make_turning_points("NORMAL", left=-0.5, right=0.5)
        with patch(FUNC_PATH + ".quad", return_value=(4.56, {})):
             result = compute_wkb(turning_points, 0.2, _make_potential())
        assert result == 4.56, f"expected {4.56}, got {result}"

    def test_uses_the_given_level_value_in_the_integrand_wkb(self):
        """
        Verify that the integrand built for the NORMAL regime
        uses the given level, not a turning-point field
        """
        turning_points = _make_turning_points("NORMAL", left=-0.5, right=0.5)
        captured_func = {}
        def _capture_quad(func, a, b, **kwargs):
            captured_func["func"] = func
            return (0.0, {})
        potential = _make_potential(return_value=2.0)
        with patch(FUNC_PATH + ".quad", side_effect=_capture_quad):
             compute_wkb(turning_points, 0.75, potential)
        with patch(FUNC_PATH + ".REVELO", new=1.0):
             integrand_value = captured_func["func"](0.0)
        expected = np.sqrt(2 * 1.0 * (2.0 - 0.75))
        assert np.isclose(integrand_value, expected), f"expected {expected}, got {integrand_value}"

    def test_integrates_between_quasi_left_and_quasi_right(self):
        """ Verify that quad is called with quasi_left and quasi_right as integration bounds """
        turning_points = _make_turning_points("OVER THE BARRIER", OVER_STATUS="QUASI_NORMAL",
                                            quasi_left=-0.3, quasi_right=0.4,
                                            QUASI_LEVEL=0.8, SCALE_FACTOR=1.0)
        with patch(FUNC_PATH + ".quad", return_value=(0.5, {})) as mock_quad:
            compute_wkb(turning_points, 1.5, _make_potential())
        args = mock_quad.call_args.args
        assert args[1] == -0.3, f"expected {-0.3}, got {args[1]}; test #1"
        assert args[2] == 0.4, f"expected {0.4}, got {args[2]}; test #2"

    def test_result_is_negative_scale_factor_times_integral(self):
        """
        Verify that the returned value equals
        -(SCALE_FACTOR * the integrated value) when it is not near zero """
        turning_points = _make_turning_points("OVER THE BARRIER", OVER_STATUS="QUASI_NORMAL",
                                            quasi_left=-0.3, quasi_right=0.4,
                                            QUASI_LEVEL=0.8, SCALE_FACTOR=2.0)
        with patch(FUNC_PATH + ".quad", return_value=(1.5, {})):
             result = compute_wkb(turning_points, 1.5, _make_potential())
        expected = -2.0 * 1.5
        assert np.isclose(result, expected), f"expected {expected}, got {result}"

    def test_quasi_endothermic_reaction_uses_same_formula(self):
        """
        Verify that the QUASI_ENDOTHERMIC_REACTION status uses
        the same -SCALE_FACTOR * integral formula
        """
        turning_points = _make_turning_points("OVER THE BARRIER", OVER_STATUS="QUASI_ENDOTHERMIC_REACTION",
                                            quasi_left=-0.6, quasi_right=0.9,
                                            QUASI_LEVEL=0.6, SCALE_FACTOR=1.5)
        with patch(FUNC_PATH + ".quad", return_value=(2.0, {})):
             result = compute_wkb(turning_points, 1.8, _make_potential())
        expected = -1.5 * 2.0
        assert np.isclose(result, expected), f"expected {expected}, got {result}"

    def test_uses_zero_protection_when_integral_is_near_zero(self):
        """
        Verify that a near-zero integral is replaced by zero_tol
        before scaling, instead of using the raw value
        """
        turning_points = _make_turning_points("OVER THE BARRIER", OVER_STATUS="QUASI_NORMAL",
                                            quasi_left=-0.1, quasi_right=0.1,
                                            QUASI_LEVEL=0.9, SCALE_FACTOR=3.0)
        with patch(FUNC_PATH + ".quad", return_value=(1e-15, {})):
            result = compute_wkb(turning_points, 1.5, _make_potential(), zero_tol=1e-12)
        expected = -3.0 * 1e-12
        assert np.isclose(result, expected), f"expected {expected}, got {result}"

    def test_uses_the_quasi_level_in_the_integrand_wkb(self):
        """
        Verify that the integrand built for OVER THE BARRIER
        uses QUASI_LEVEL, not the passed-in level
        """
        turning_points = _make_turning_points("OVER THE BARRIER", OVER_STATUS="QUASI_NORMAL",
                                            quasi_left=-0.3, quasi_right=0.4,
                                            QUASI_LEVEL=0.8, SCALE_FACTOR=1.0)
        captured_func = {}
        def _capture_quad(func, a, b, **kwargs):
            captured_func["func"] = func
            return (0.0, {})
        potential = _make_potential(return_value=5.0)
        with patch(FUNC_PATH + ".quad", side_effect=_capture_quad):
            compute_wkb(turning_points, 999.0, potential) 
        with patch(FUNC_PATH + ".REVELO", new=1.0):
            integrand_value = captured_func["func"](0.0)
        expected = np.sqrt(2 * 1.0 * (5.0 - 0.8))
        assert np.isclose(integrand_value, expected), f"expected {expected}, got {integrand_value}"

    def test_no_warning_on_clean_integration(self):
        """ Verify that the clean integration raises no warnings """
        tp = _make_turning_points("NORMAL", left=-0.5, right=0.5)
        with patch(FUNC_PATH + ".quad", return_value=(1.0, 1e-14, {})), \
             patch(FUNC_PATH + ".logger") as mock_logger:
             compute_wkb(tp, 0.2, _make_potential())
        mock_logger.warning.assert_not_called()
        # it should not be called


# ===========================================================================
# --- compute_wkb - input validation ---
# ===========================================================================
class TestComputeWkbValidation:

    def test_raises_on_unknown_regime(self):
        """ Verify that an unrecognized regime raises an error. Expecting a ValueError """
        turning_points = _make_turning_points("SOMETHING_ELSE")
        with pytest.raises(ValueError, match = "Expected tunneling regimes"):
             compute_wkb(turning_points, 0.5, _make_potential())

    def test_raises_on_missing_over_the_barrier_status(self):
        """ Verify that a missing OVER_STATUS raises an error. Expecting a KeyError """
        turning_points = _make_turning_points("OVER THE BARRIER")
        with pytest.raises(KeyError, match = "The input dictionary must contain key"):
             compute_wkb(turning_points, 1.5, _make_potential())

    def test_raises_on_unknown_over_the_barrier_status(self):
        """ Verify that an unrecognized OVER_STATUS raises an error. Expecting a ValueError """
        turning_points = _make_turning_points("OVER THE BARRIER", OVER_STATUS="BOGUS_STATUS")
        with pytest.raises(ValueError, match = "Expected over the barrier tunneling regimes"):
             compute_wkb(turning_points, 1.5, _make_potential())