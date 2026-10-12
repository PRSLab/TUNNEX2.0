# --- Test: test_eckart_potential_functions.py. Unit tests for eckart_potential.py
# Run with: pytest test_eckart_potential_functions.py -v ---

# --- Modules ---
import pytest # type: ignore

import math

# --- Module to test ---
from tunnex_2.irc_computations.ancillary_functions.eckart_potential_functions import ( # type: ignore
    _eckart_potential_value,
    eckart_potential_parameters,
    eckart_potential_data)


# --- Helpers and shared data ---
TS_FREQ = -1500.0 # cm-1; typical imaginary frequency for a broad range of chemical reactions
DV1 = 0.03 # Hartree (~78.8 kJ mol-1)
DV2 = 0.05 # Hartree (~131.3 kJ mol-1)
DV_SYMM = 0.04 # Hartree (~105.0 kJ mol-1; symmetric, i.e., dV1 == dV2)


# ===========================================================================
# --- _eckart_potential_value - normal behaviour ---
# ===========================================================================
class TestEckartPotentialValueNormal:

    def _get_params(self, freq=TS_FREQ, dv1=DV1, dv2=DV2):
        """ Getting the Eckart potential parameters from the given input data """
        A, B, L = eckart_potential_parameters(freq, dv1, dv2)
        return A, B, L

    def _get_irc_max(self, A, B, L):
        """ Getting the IRC coordinate and value of the Eckart potential maximum """
        irc_max = (L / (2 * math.pi)) * math.log((B + A) / (B - A))
        v_max = (A + B) ** 2 / (4 * B)
        return irc_max, v_max

    def test_returns_float(self):
        """ The function should return float """
        A, B, L = self._get_params()
        result = _eckart_potential_value(0.0, A, B, L)
        assert isinstance(result, float), f"expected {float}, got {type(result)}"

    def test_maximum_at_shifted_origin(self, tol=1e-12):
        """ The potential maximum should occur at the coordinate irc_max """
        A, B, L = self._get_params(TS_FREQ, DV1, DV2)
        irc_max, v_max = self._get_irc_max(A, B, L) # coordinate and value of the raw maximum
        v_max_func = _eckart_potential_value(irc_max, A, B, L)
        v_left = _eckart_potential_value(irc_max - 0.01, A, B, L) # evaluate slightly to the left
        v_right = _eckart_potential_value(irc_max + 0.01, A, B, L) # evaluate slightly to the right
        assert not math.isclose(irc_max, 0.0, abs_tol=tol), f"irc_max {irc_max} should not be {0.0} "
        " for a non-symmetrical potential; test #1"
        assert math.isclose(v_max, v_max_func, abs_tol=tol), f"v_max {v_max} and v_max_func {v_max_func} should be equal; test #2"
        # Use weak inequalities because the Eckart potential can have a flat maximum (at specific A, B, and L),
        # so numerical precision may not distinguish the maximum from nearby values
        assert v_max_func >= v_left, f"v_max should be a true maximum, i.e. v_max ({v_max_func}) >= v_left ({v_left}); test #3"
        assert v_max_func >= v_right, f"v_max should be a true maximum, i.e. v_max ({v_max_func}) >= v_right ({v_right}); test #4"

    def test_potential_approaches_zero_far_left(self):
        """ V(x) should be close to zero at the reactant side """
        A, B, L = self._get_params()
        irc_max, _ = self._get_irc_max(A, B, L)
        v_far_left = _eckart_potential_value(irc_max - 50 * L, A, B, L)
        assert math.isclose(v_far_left, 0.0, abs_tol=1e-12), f"V(x) -> {0.0} as x -> -inf (reactant side), got {v_far_left}"

    def test_potential_approaches_minus_A_far_right(self):
        """ V(x) should be close to A at the product side """
        A, B, L = self._get_params()
        irc_max, _ = self._get_irc_max(A, B, L)
        v_far_right = _eckart_potential_value(irc_max + 50 * L, A, B, L)
        assert math.isclose(v_far_right, A, abs_tol=1e-12), f"V(x) -> {A} as x -> +inf (product side), got {v_far_right}"

    def test_maximum_value_close_to_dV1(self):
        """ The barrier height as seen from the reactant side should equal dV1 """
        A, B, L = self._get_params(TS_FREQ, DV1, DV2)
        irc_max, _ = self._get_irc_max(A, B, L)
        v_max_func = _eckart_potential_value(irc_max, A, B, L)
        v_minus_inf = 0.0 # reactant asymptote
        difference = v_max_func - v_minus_inf
        assert math.isclose(difference, DV1, rel_tol=1e-12), f"V_max - V_react = {DV1} as expected, got {difference}"

    def test_maximum_value_close_to_dV2_from_product_side(self):
        """ Barrier height from the product side should equal dV2 """
        A, B, L = self._get_params(TS_FREQ, DV1, DV2)
        irc_max, _ = self._get_irc_max(A, B, L)
        v_max_func = _eckart_potential_value(irc_max, A, B, L)
        v_plus_inf = A # product asymptote
        difference = v_max_func - v_plus_inf
        assert math.isclose(difference, DV2, rel_tol=1e-12), f"V_max - V_prod = {DV2} as expected, got {difference}"

    def test_symmetric_barrier_a_parameter(self, tol_a=1e-12):
        """ For a symmetric barrier A = 0, V -> 0 on both sides """
        A, B, L = eckart_potential_parameters(TS_FREQ, DV_SYMM, DV_SYMM)
        assert math.isclose(A, 0.0, abs_tol=tol_a), f"A should be {0.0} for a symmetric potential, got {A}"
        
    def test_symmetric_barrier_irc_max(self, tol_a=1e-12):
        """ For a symmetric barrier irc_max should be at 0.0 """
        A, B, L = eckart_potential_parameters(TS_FREQ, DV_SYMM, DV_SYMM)       
        irc_max, _ = self._get_irc_max(A, B, L)
        assert math.isclose(irc_max, 0.0, abs_tol=tol_a), f"irc_max should be {0.0} for a symmetric potential, "
        f"got {irc_max}"
        
    def test_symmetric_barrier_v_max(self, tol_r=1e-12):
        """ For a symmetric barrier v_max should be DV_SYMM """
        A, B, L = eckart_potential_parameters(TS_FREQ, DV_SYMM, DV_SYMM)
        irc_max, v_max = self._get_irc_max(A, B, L)     
        v_max_func = _eckart_potential_value(irc_max, A, B, L)
        assert math.isclose(v_max, DV_SYMM, rel_tol=tol_r), f"v_max should be {DV_SYMM} for a symmetric potential, "
        f" got {v_max}; test #1"
        assert math.isclose(v_max_func, DV_SYMM, rel_tol=tol_r), f"v_max_func should be {DV_SYMM} for a symmetric potential, "
        f" got {v_max_func}; test #2"
        
    def test_symmetric_barrier_reactant_asymptote(self, tol_a=1e-12):
        """ For a symmetric barrier V(x) x -> -inf should be 0.0 """
        A, B, L = eckart_potential_parameters(TS_FREQ, DV_SYMM, DV_SYMM)    
        irc_max, _ = self._get_irc_max(A, B, L)
        v_far_left = _eckart_potential_value(irc_max - 50 * L, A, B, L)
        assert math.isclose(v_far_left, 0.0, abs_tol=tol_a), f"V(x) -> {0.0} as x -> -inf (symmetric potential), "
        f"got {v_far_left}"
        
    def test_symmetric_barrier_product_asymptote(self, tol_a=1e-12):
        """ For a symmetric barrier V(x) x -> +inf should be 0.0 """
        A, B, L = eckart_potential_parameters(TS_FREQ, DV_SYMM, DV_SYMM)    
        irc_max, _ = self._get_irc_max(A, B, L)        
        v_far_right = _eckart_potential_value(irc_max + 50 * L, A, B, L)
        assert math.isclose(v_far_right, 0.0, abs_tol=tol_a), f"V(x) -> {0.0} as x -> +inf (symmetric potential), "
        f"got {v_far_right}"
        
    def test_symmetric_barrier_product_asymptote(self, tol_a=1e-12):
        """ For a symmetric barrier V(0.0) should be a true maximum """
        A, B, L = eckart_potential_parameters(TS_FREQ, DV_SYMM, DV_SYMM)    
        irc_max, _ = self._get_irc_max(A, B, L)       
        v_max_func = _eckart_potential_value(irc_max, A, B, L)
        v_left = _eckart_potential_value(irc_max - 0.01, A, B, L) # evaluate slightly to the left
        v_right = _eckart_potential_value(irc_max + 0.01, A, B, L) # evaluate slightly to the right
        # Use weak inequalities because the Eckart potential can have a flat maximum (at specific A, B, and L),
        # so numerical precision may not distinguish the maximum from nearby values
        assert v_max_func >= v_left, f"v_max should be a true maximum, i.e. v_max >= v_left, "
        f"got {v_max_func} and {v_left}; test #1"
        assert v_max_func >= v_right, "v_max should be a true maximum, i.e. v_max >= v_right, "
        f"got {v_max_func} and {v_right}; test #2"


# ===========================================================================
# --- eckart_potential_parameters - normal behaviour ---
# ===========================================================================
class TestEckartPotentialParametersNormal:

    def test_returns_three_floats(self):
        """ The function should return exactly 3 float values """
        result = eckart_potential_parameters(TS_FREQ, DV1, DV2)
        assert len(result) == 3, f"the function should return exactly 3 values, got {len(result)}; test #1"
        assert all(isinstance(v, float) for v in result), f"expected exactly 3 {float} values, "
        f"got {[type(v) for v in result]}; test #2"

    def test_A_equals_dV1_minus_dV2(self):
        """ Checking the formula for A """
        A, _, _ = eckart_potential_parameters(TS_FREQ, DV1, DV2)
        assert math.isclose(A, DV1 - DV2, rel_tol=1e-12), f"A = dV1 - dV2, got {A} and {DV1 - DV2}"

    def test_A_sign_flips_when_dV1_dV2_swapped(self):
        """ The sign of A should change when dV1 and dV2 are swapped """
        A1, _, _ = eckart_potential_parameters(TS_FREQ, DV1, DV2)
        A2, _, _ = eckart_potential_parameters(TS_FREQ, DV2, DV1)
        assert math.isclose(A1, -A2, rel_tol=1e-12), f"expected a sign change for A, got {A1} and {A2}"

    def test_B_equals_sum_of_sqrt_squared(self):
        """ Checking the formula for B """
        _, B, _ = eckart_potential_parameters(TS_FREQ, DV1, DV2)
        expected = (math.sqrt(DV1) + math.sqrt(DV2)) ** 2
        assert math.isclose(B, expected, rel_tol=1e-12), "B = (math.sqrt(DV1) + math.sqrt(DV2)) ** 2, "
        f"expected {expected}, got {B}"

    def test_B_is_positive(self):
        """ B should always be positive """
        _, B, _ = eckart_potential_parameters(TS_FREQ, DV1, DV2)
        assert B > 0.0, f"B > {0.0}, got {B}"

    def test_B_unchanged_when_dV1_dV2_swapped(self):
        """ The sign of B should not change when dV1 and dV2 are swapped """
        _, B1, _ = eckart_potential_parameters(TS_FREQ, DV1, DV2)
        _, B2, _ = eckart_potential_parameters(TS_FREQ, DV2, DV1)
        assert math.isclose(B1, B2, rel_tol=1e-12), f"did not expect a change of B, got {B1:.3g} and {B2:.3g}"

    def test_L_equals_the_literature_formula(self):
        """ L is computed according to the literature formula """
        freqs = [-1500, -1800, -2100]
        L_expected = [2.972131801116901, 2.47677650093075, 2.1229512865120723]  # freqs = [-1500, -1800, -2100]
        for i, freq in enumerate(freqs):
            _, _, L = eckart_potential_parameters(freq, DV1, DV2)
            assert math.isclose(L_expected[i], L, rel_tol=1e-12), \
            f"L is not computed according to the literature formula, freq={freq}, expected {L_expected[i]}, "
            f"got {L}"
        # The reference implementation mass-scales F_star, i.e. F_star = - mass * (2 * math.pi * ts_freq_cm_m1) ** 2 in a referense.
        # We intentionally omit that step because the IRC coordinate used here is expressed in sqrt(amu) * bohr rather than bohr.
        # Reference: [Johnston, H. S., & Heicklen, J. (1962). J. Phys. Chem., 66(3), 532-533 / equation number 10].

    def test_L_is_positive(self):
        """ L should always be positive """
        _, _, L = eckart_potential_parameters(TS_FREQ, DV1, DV2)
        assert L > 0.0, f"L > {0.0}, got {L}"

    def test_symmetric_barrier_A(self):
        """ For a symmetric barrier A = 0 """
        A, _, _ = eckart_potential_parameters(TS_FREQ, DV_SYMM, DV_SYMM)
        assert math.isclose(A, 0.0, abs_tol=1e-12), f"A should be zero for a symmetric barrier, got {A}"

    def test_symmetric_barrier_B(self):
        """ For a symmetric barrier B = 4dV """
        A, B, _ = eckart_potential_parameters(TS_FREQ, DV_SYMM, DV_SYMM)
        B_expected = 4.0 * DV_SYMM
        assert math.isclose(B, B_expected, rel_tol=1e-12), f"B should be 4dV ({B_expected}) for a symmetric barrier, "
        f"got {B}"

    def test_larger_freq_value_gives_narrower_barrier(self):
        """ Larger frequency value gives narrower barrier (i.e., smaller L) """
        _, _, L_low = eckart_potential_parameters(-1500.0, DV1, DV2)
        _, _, L_high = eckart_potential_parameters(-3000.0, DV1, DV2)
        assert L_high < L_low, f"expected smaller L for larger frequency, i.e. {L_high} < {L_low}"

    def test_larger_barrier_gives_wider_barrier(self):
        """ Higher barrier means bigger L for the same frequency """
        _, _, L_low = eckart_potential_parameters(TS_FREQ, DV1, DV2)
        _, _, L_medium = eckart_potential_parameters(TS_FREQ, 10 * DV1, DV2)
        _, _, L_high = eckart_potential_parameters(TS_FREQ, 10 * DV1, 10 * DV2)
        assert L_medium > L_low, f"expected bigger L for larger barrier, i.e. {L_medium} > {L_low}, dV1; test #1"
        assert L_high > L_medium, f"expected bigger L for larger barrier, i.e. {L_high} > {L_medium}, dV2; test #2"


# ===========================================================================
# eckart_potential_parameters - input validation ---
# ===========================================================================
class TestEckartPotentialParametersValidation:

    def test_positive_freq_raises(self):
        """ The transition state frequency should not be positive. Expecting a ValueError """
        with pytest.raises(ValueError, match="negative"):
            eckart_potential_parameters(1500.0, DV1, DV2)

    def test_zero_freq_raises(self):
        """ The transition state frequency should not be zero. Expecting a ValueError """
        with pytest.raises(ValueError, match="negative"):
            eckart_potential_parameters(0.0, DV1, DV2)

    def test_zero_dV1_raises(self):
        """ The dV1 should not be zero. Expecting a ValueError """
        with pytest.raises(ValueError, match="positive"):
            eckart_potential_parameters(TS_FREQ, 0.0, DV2)

    def test_negative_dV1_raises(self):
        """ The dV1 should not be negative. Expecting a ValueError """
        with pytest.raises(ValueError, match="positive"):
            eckart_potential_parameters(TS_FREQ, -0.01, DV2)

    def test_zero_dV2_raises(self):
        """ The dV2 should not be zero. Expecting a ValueError """
        with pytest.raises(ValueError, match="positive"):
            eckart_potential_parameters(TS_FREQ, DV1, 0.0)

    def test_negative_dV2_raises(self):
        """ The dV2 should not be negative. Expecting a ValueError """
        with pytest.raises(ValueError, match="positive"):
            eckart_potential_parameters(TS_FREQ, DV1, -0.01)


# ===========================================================================
# eckart_potential_data - normal behaviour ---
# ===========================================================================
class TestEckartPotentialDataNormal:

    def _get_params(self, freq=TS_FREQ, dv1=DV1, dv2=DV2):
        """ Getting the Eckart potential parameters from the given input data """
        A, B, L = eckart_potential_parameters(freq, dv1, dv2)
        return A, B, L

    def _data(self, **kwargs):
        """ Getting the Eckart potential IRC curve from the given input data """
        A, B, L = self._get_params()
        return eckart_potential_data(A, B, L, **kwargs)

    def test_returns_irc_data_with_correct_length(self):
        """ Should return N points for the N-size (odd) grid. ZPVE should not play role in this mode """
        grid_size = 101
        result = self._data(grid_size=grid_size)
        n_points = len(result.electronic_energies)
        assert n_points == grid_size, f"expected {grid_size} results for {grid_size} points, got {n_points}"

    def test_returns_irc_data_with_correct_length_zpve_forward_field_is_none(self):
        """ Should return N points for the N-size (odd) grid. ZPVE should not play role in this mode """
        grid_size = 101
        result = self._data(grid_size=grid_size)
        assert result.zpve_energies_forward is None, f"expected {None} for forward ZPVEs, "
        f"got {result.zpve_energies_forward}"

    def test_returns_irc_data_with_correct_length_zpve_reverse_field_is_none(self):
        """ Should return N points for the N-size (odd) grid. ZPVE should not play role in this mode """
        grid_size = 101
        result = self._data(grid_size=grid_size)
        assert result.zpve_energies_reverse is None, f"expected {None} for reverse ZPVEs, "
        f" got {result.zpve_energies_reverse}"

    def test_irc_grid_is_symmetric_around_zero(self):
        """ IRC grid should be symmetric around zero """
        result = self._data(grid_size=101)
        ircs = [x[0] for x in result.electronic_energies]
        difference = ircs[0] - (-ircs[-1])
        assert math.isclose(difference, 0.0, abs_tol=1e-12), f"expected a symmetric grid, "
        f"got end point difference={difference:.3g}"

    def test_irc_midpoint_is_zero(self):
        """The middle element of an odd grid must be exactly at IRC = 0.0 """
        grid_size = 101
        result = self._data(grid_size=grid_size)
        mid = grid_size // 2
        irc_mid = result.electronic_energies[mid][0]
        assert math.isclose(irc_mid, 0.0, abs_tol=1e-12), f"expected the middle element to be at IRC = {0.0}, "
        f"got {irc_mid:.3g}"

    def test_potential_maximum_at_irc_zero(self):
        """ After the x-axis shift, V(0) should be the maximum value """
        result = self._data(grid_size=201)
        ircs = [x for x, _ in result.electronic_energies]
        energies = [e for _, e in result.electronic_energies]
        idx_zero = min(range(len(ircs)), key=lambda i: abs(ircs[i]))
        v_at_zero = energies[idx_zero]
        assert v_at_zero == max(energies), f"expected maximum at IRC = {0.0}, got {v_at_zero}"

    def test_potential_decreases_away_from_maximum(self):
        """ Expecting V(x) to decrease as we move away from the center """
        result = self._data(grid_size=201)
        energies = [e for _, e in result.electronic_energies]
        mid = len(energies) // 2
        # energy should decrease as we move away from the centre
        assert energies[mid] > energies[mid - 10], f"expecting E_mig > E_left, "
        f"got {energies[mid]:.3g} and {energies[mid - 10]:.3g}; test #1"
        assert energies[mid] > energies[mid + 10], f"expecting E_mig > E_right, "
        f"got {energies[mid]:.3g} and {energies[mid + 10]:.3g}; test #2"

    def test_electronic_energies_are_list_of_tuples(self):
        """ Expecting a list[tuple(float, float)] format for electronic energies """
        result = self._data(grid_size=11)
        assert isinstance(result.electronic_energies, list), f"expected {list[tuple(float, float)]}, "
        f"got {type(result.electronic_energies)}"

    def test_electronic_energies_irc_is_float(self):
        """ Expecting a list[tuple(irc - float, float)] format for electronic energies """
        result = self._data(grid_size=11)
        irc, _ = result.electronic_energies[0]
        assert isinstance(irc, float), f"expected irc to be {float}, got {type(irc)}"

    def test_electronic_energies_energy_is_float(self):
        """ Expecting a list[tuple(float, energy - float)] format for electronic energies """
        result = self._data(grid_size=11)
        _, energy = result.electronic_energies[0]
        assert isinstance(energy, float), f"expected energy to be {float}, got {type(energy)}"

    def test_larger_grid_size_same_endpoints(self, tol=1e-12):
        """ Multiplying the grid size should not change the IRC range endpoints """
        grid_size_1 = 51
        grid_size_2 = 201
        r1 = self._data(grid_size=grid_size_1)
        r2 = self._data(grid_size=grid_size_2)
        assert math.isclose(r1.electronic_energies[0][0], r2.electronic_energies[0][0], rel_tol=tol), \
        f"left edge points of the datasets at {grid_size_1} and {grid_size_2} grid sizes do not match; test #1"
        assert math.isclose(r1.electronic_energies[-1][0], r2.electronic_energies[-1][0], rel_tol=tol), \
        f"right edge points of the datasets at {grid_size_1} and {grid_size_2} grid sizes do not match; test #2"
        step_1 = abs(r1.electronic_energies[-1][0] - r1.electronic_energies[-2][0])
        step_2 = abs(r2.electronic_energies[-1][0] - r2.electronic_energies[-2][0])
        difference_step = step_1 - 4.0 * step_2
        assert math.isclose(step_1, 4.0 * step_2, rel_tol=1e-12), \
        f"zero difference is expected, got {difference_step}; test #3"

    def test_custom_border_factor_scales_range(self):
        """ The range should be scaled when the border factor is scaled """
        A, B, L = self._get_params()
        r1 = eckart_potential_data(A, B, L, border_factor=1.0, grid_size=11)
        r2 = eckart_potential_data(A, B, L, border_factor=2.0, grid_size=11)
        range_1 = abs(r1.electronic_energies[-1][0] - r1.electronic_energies[0][0])
        range_2 = abs(r2.electronic_energies[-1][0] - r2.electronic_energies[0][0])
        difference_range = range_2 - 2.0 * range_1
        assert math.isclose(range_2, 2.0 * range_1, rel_tol=1e-12), \
        f"zero difference is expected, got {difference_range}"
        
    def test_custom_border_factor_scales_step(self):
        """ The step should be scaled when the border factor is scaled """
        A, B, L = self._get_params()
        r1 = eckart_potential_data(A, B, L, border_factor=1.0, grid_size=11)
        r2 = eckart_potential_data(A, B, L, border_factor=2.0, grid_size=11) 
        step_1 = abs(r1.electronic_energies[-1][0] - r1.electronic_energies[-2][0])
        step_2 = abs(r2.electronic_energies[-1][0] - r2.electronic_energies[-2][0])
        difference_step = step_2 - 2.0 * step_1
        assert math.isclose(step_2, 2.0 * step_1, rel_tol=1e-12), \
        f"zero difference is expected, got {difference_step}"


# ===========================================================================
# eckart_potential_data - input validation ---
# ===========================================================================
class TestEckartPotentialDataValidation:

    def _get_params(self, freq=TS_FREQ, dv1=DV1, dv2=DV2):
        """ Getting the Eckart potential parameters from the given input data """
        A, B, L = eckart_potential_parameters(freq, dv1, dv2)
        return A, B, L

    def test_negative_border_factor_raises(self):
        """ The border factor should not be negative. Expecting a ValueError """
        A, B, L = self._get_params()
        with pytest.raises(ValueError, match="positive border factor"):
            eckart_potential_data(A, B, L, border_factor=-1.0)

    def test_zero_border_factor_raises(self):
        """ The border factor should not be zero. Expecting a ValueError """
        A, B, L = self._get_params()
        with pytest.raises(ValueError, match="positive border factor"):
            eckart_potential_data(A, B, L, border_factor=0.0)

    def test_negative_L_raises(self):
        """ The Eckart barrier width should not be negative. Expecting a ValueError """
        A, B, _ = self._get_params()
        with pytest.raises(ValueError, match="positive Eckart barrier width"):
            eckart_potential_data(A, B, -1.0)

    def test_zero_L_raises(self):
        """ The Eckart barrier width should not be zero. Expecting a ValueError """
        A, B, _ = self._get_params()
        with pytest.raises(ValueError, match="positive Eckart barrier width"):
            eckart_potential_data(A, B, 0.0)

    def test_zero_B_raises(self):
        """ The B value should not be negative or zero. Expecting a ValueError """
        A, _, L = self._get_params()
        with pytest.raises(ValueError, match="Eckart barrier-shape parameter"):
            eckart_potential_data(A, 0.0, L)

    def test_negative_B_raises(self):
        """ The B value should not be negative or zero. Expecting a ValueError """
        A, _, L = self._get_params()
        with pytest.raises(ValueError, match="Eckart barrier-shape parameter"):
            eckart_potential_data(A, -0.1, L)

    def test_grid_size_less_than_3_raises(self):
        """ The grid size not be less than 3. Expecting a ValueError """
        A, B, L = self._get_params()
        with pytest.raises(ValueError, match="grid_size >= 3"):
            eckart_potential_data(A, B, L, grid_size=2)

    def test_non_integer_or_even_grid_size_raises(self):
        """ The grid size should be an integer and should be odd, not even. Expecting a ValueError """
        A, B, L = self._get_params()
        with pytest.raises(ValueError, match="odd grid_size"):
            eckart_potential_data(A, B, L, grid_size=100)