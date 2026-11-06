# --- Test: test_dataclasses. Unit tests for dataclasses.py
# Run with: test_dataclasses.py.py -v ---

# --- Modules ---
import pytest # type: ignore

import numpy as np
from pathlib import Path
import dataclasses


# --- Module to test ---
from tunnex_2.constants_and_dataclasses.dataclasses import ( # type: ignore
    FindIRCConfig,
    IRCData,
    EnergyPointsData,
    AttemptFreq,
    TunnexInputSettings,
    IRCDataQMT,
    LevelResult,
    TemperatureAverResult,
    ArrheniusResult)


# ===========================================================================
# --- FindIRCConfig - normal behaviour and input validation ---
# ===========================================================================
class TestFindIRCConfig:

    def test_default_construction(self):
        """ Verify that the default configuration values are set correctly """
        cfg = FindIRCConfig()
        assert cfg.prog_mode == "gaussian", f"expected {"gaussian"}, got {cfg.prog_mode}; test #1"
        assert cfg.proj_freq is True, f"expected {True}, got {cfg.proj_freq}; test #2"
        assert cfg.calc_all is True, f"expected {True}, got {cfg.calc_all}; test #3"
        assert cfg.hybrid_mode is False, f"expected {False}, got {cfg.hybrid_mode}; test #4"
        assert cfg.eckart is False, f"expected {False}, got {cfg.eckart}; test #5"
        assert cfg.qmt_module is True, f"expected {True}, got {cfg.qmt_module}; test #6"
        assert cfg.hess_parsing_flag is False, f"expected {False}, got {cfg.hess_parsing_flag}; test #7"
        assert cfg.prob_aver_mode_qmt == "finite_sum", f"expected {"finite_sum"}, got {cfg.prob_aver_mode_qmt}; test #8"

    def test_all_file_fields_default_to_none(self):
        """ Verify that all file fields default to None """
        cfg = FindIRCConfig()
        assert cfg.ts_input_file is None, f"expected {None}, got {cfg.ts_input_file}; test #1"
        assert cfg.react_input_file is None, f"expected {None}, got {cfg.react_input_file}; test #2"
        assert cfg.prod_input_file is None, f"expected {None}, got {cfg.prod_input_file}; test #3"
        assert cfg.ts_computed_file is None, f"expected {None}, got {cfg.ts_computed_file}; test #4"
        assert cfg.irc_computed_file is None, f"expected {None}, got {cfg.irc_computed_file}; test #5"
        assert cfg.react_computed_file is None, f"expected {None}, got {cfg.react_computed_file}; test #6"
        assert cfg.prod_computed_file is None, f"expected {None}, got {cfg.prod_computed_file}; test #7"
        assert cfg.tunnex_input is None, f"expected {None}, got {cfg.tunnex_input}; test #8"

    def test_override_prog_mode(self):
        """ Verify that the program mode can be overridden """
        expected = "orca"
        cfg = FindIRCConfig(prog_mode=expected)
        assert cfg.prog_mode == "orca", f"expected {expected}, got {cfg.prog_mode}"

    def test_accepts_string_paths(self):
        """ Verify that string file paths are accepted """
        expected = "mol_ts.gjf"
        cfg = FindIRCConfig(ts_input_file=expected)
        assert cfg.ts_input_file == expected, f"expected {expected}, got {cfg.ts_input_file}"

    def test_accepts_path_objects(self):
        """ Verify that Path objects are accepted as file paths (Path) """
        expected = Path("mol_ts.gjf")
        cfg = FindIRCConfig(ts_input_file=expected)
        assert cfg.ts_input_file == expected, f"expected {expected}, got {cfg.ts_input_file}"

    def test_equality_same_values(self):
        """ Verify that configurations with different values are not equal """
        a = FindIRCConfig(prog_mode="orca", proj_freq=False)
        b = FindIRCConfig(prog_mode="orca", proj_freq=False)
        assert a == b, f"expected {a} == {b}"

    def test_inequality_different_values(self):
        """ Verify that configurations with different values are not equal """
        a = FindIRCConfig(prog_mode="gaussian")
        b = FindIRCConfig(prog_mode="orca")
        assert a != b, f"expected {a} != {b}"

    def test_all_fields_settable(self):
        """ Verify that all configuration fields can be set """
        cfg = FindIRCConfig(
            prog_mode="orca",
            proj_freq=False,
            calc_all=False,
            hybrid_mode=True,
            eckart=True,
            qmt_module=False,
            hess_parsing_flag=True,
            prob_aver_mode_qmt="finite_sum",
            ts_input_file=Path("ts.inp"),
            react_input_file=Path("react.inp"),
            prod_input_file=Path("prod.inp"),
            ts_computed_file=Path("ts.out"),
            irc_computed_file=Path("irc.out"),
            react_computed_file=Path("react.out"),
            prod_computed_file=Path("prod.out"),
            tunnex_input=Path("tunnex_input.txt"))
        assert cfg.prog_mode == "orca", f"expected {"orca"}, got {cfg.prog_mode}; test #1"
        assert cfg.tunnex_input == Path("tunnex_input.txt"), f"expected {Path("tunnex_input.txt")}, "
        f" got {cfg.tunnex_input}; test #2"


# ===========================================================================
# --- IRCData - normal behaviour and input validation ---
# ===========================================================================
class TestIRCData:

    def _make(self, with_zpve=True):
        """ Should make a sample object for the IRCData tests """
        el = [(-1.0, -153.10), (-0.5, -153.08), (0.0, -153.05), (0.5, -153.12), (1.0, -153.15)]
        fwd = [(0.05, 0.03), (1.0, 0.04)] if with_zpve else None
        rev = [(-1.0, 0.02), (-0.05, 0.03)] if with_zpve else None
        return IRCData(electronic_energies=el, zpve_energies_forward=fwd, zpve_energies_reverse=rev)

    def test_construction_with_zpve(self):
        """ Verify that IRCData can be constructed with ZPVE data """
        data = self._make(with_zpve=True)
        assert data.electronic_energies is not None, f"expected not {None}, got {data.electronic_energies}; test #1"
        assert data.zpve_energies_forward is not None, f"expected not {None}, got {data.zpve_energies_forward}; test #2"
        assert data.zpve_energies_reverse is not None, f"expected not {None}, got {data.zpve_energies_reverse}; test #3"

    def test_construction_without_zpve(self):
        """ Verify that IRCData can be constructed without ZPVE data """
        data = self._make(with_zpve=False)
        assert data.zpve_energies_forward is None, f"expected {None}, got {data.zpve_energies_forward}; test #1"
        assert data.zpve_energies_reverse is None, f"expected {None}, got {data.zpve_energies_reverse}; test #2"

    def test_electronic_energies_stored_correctly(self):
        """ Verify that electronic energies are stored correctly """
        el = [(-1.0, -153.10), (1.0, -153.09)]
        data = IRCData(electronic_energies=el, zpve_energies_forward=None, zpve_energies_reverse=None)
        assert data.electronic_energies == el, f"expected {el}, got {data.electronic_energies}"

    def test_equality(self):
        """ Verify that IRCData objects with the same values are equal """
        el = [(0.0, -153.0)]
        a = IRCData(el, None, None)
        b = IRCData(el, None, None)
        assert a == b, f"expected {a} == {b}"

    def test_inequality(self):
        """ Verify that IRCData objects with different values are not equal """
        a = IRCData([(0.0, -153.0)], None, None)
        b = IRCData([(1.0, -153.0)], None, None)
        assert a != b, f"expected {a} != {b}"

    def test_electronic_energies_required(self):
        """ Verify that electronic energies are required. Expecting a TypeError """
        with pytest.raises(TypeError):
            IRCData()  # missing required fields (electronic energy and ZPVEs)

    def test_all_fields_required(self):
        """ Verify that all IRCData fields are required. Expecting a TypeError """
        with pytest.raises(TypeError):
            IRCData(electronic_energies=[])  # missing ZPVE fields


# ===========================================================================
# --- EnergyPointsData - normal behaviour and input validation ---
# ===========================================================================
class TestEnergyPointsData:

    def _make(self):
        """ Should make a sample object for the EnergyPointsData tests """
        return EnergyPointsData(
            transition_state_el_energy=(0.0, -153.05),
            transition_state_zpve=(0.0, 0.05),
            reactant_el_energy=(0.0, -153.15),
            reactant_zpve=(0.0, 0.02),
            product_el_energy=(0.0, -153.20),
            product_zpve=(0.0, 0.02))

    def test_construction(self):
        """ Verify that EnergyPointsData is constructed with the expected values """
        ep = self._make()
        assert ep.transition_state_el_energy == (0.0, -153.05), f"expected {ep.transition_state_el_energy} == " 
        f"{(0.0, -153.05)}, test #1"
        assert ep.reactant_el_energy == (0.0, -153.15), f"expected {ep.reactant_el_energy} == " 
        f"{(0.0, -153.15)}, test #2"
        assert ep.product_el_energy == (0.0, -153.20), f"expected {ep.product_el_energy} == " 
        f"{(0.0, -153.10)}, test #3"

    def test_has_six_fields(self):
        """ Verify that EnergyPointsData contains six fields """
        assert len(dataclasses.fields(EnergyPointsData)) == 6, f"expected {len(dataclasses.fields(EnergyPointsData))} "
        f"== {6}"

    def test_equality(self):
        """ Verify that EnergyPointsData objects with the same values are equal """
        a = self._make()
        b = self._make()
        assert a == b, f"expected {a} == {b}"

    def test_inequality(self):
        """ Verify that EnergyPointsData objects with different values are not equal """
        a = self._make()
        b = EnergyPointsData(
            transition_state_el_energy=(0.0, -999.0),
            transition_state_zpve=(0.0, 0.05),
            reactant_el_energy=(0.0, -153.15),
            reactant_zpve=(0.0, 0.02),
            product_el_energy=(0.0, -153.10),
            product_zpve=(0.0, 0.02))
        assert a != b, f"expected {a} != {b}"

    def test_zpve_tuples_accessible(self):
        """ Verify that ZPVE tuple values can be accessed correctly """
        ep = self._make()
        _, ts_zpve_value = ep.transition_state_zpve
        assert ts_zpve_value == 0.05, f"expected {ts_zpve_value} == {0.05}"

    def test_all_fields_required(self):
        """ Verify that all EnergyPointsData fields are required. Expecting a TypeError """
        with pytest.raises(TypeError):
            EnergyPointsData() # missing required fields (electronic energy and ZPVEs)

    def test_missing_one_field_raises(self):
        """ Verify that construction fails when a required field is missing. Expecting a TypeError """
        with pytest.raises(TypeError):
            EnergyPointsData(
                transition_state_el_energy=(0.0, -153.05),
                transition_state_zpve=(0.0, 0.05),
                reactant_el_energy=(0.0, -153.15),
                reactant_zpve=(0.0, 0.02),
                product_el_energy=(0.0, -153.10),
                # product_zpve is missing
                )


# ===========================================================================
# --- AttemptFreq - normal behaviour and input validation ---
# ===========================================================================
class TestAttemptFreq:

    def _make(self):
        """ Should make a sample object for the AttemptFreq tests """
        return AttemptFreq(
            att_freq_react=1234.5,
            att_freq_react_corr=[(1234.5, 0.9), (500.0, 0.1)],
            att_freq_prod=987.6,
            att_freq_prod_corr=[(987.6, 0.85), (400.0, 0.15)])

    def test_construction(self):
        """ Verify that AttemptFreq is constructed with the expected values """
        af = self._make()
        assert af.att_freq_react == 1234.5, f"expected {1234.5}, got {af.att_freq_react}; test #1"
        assert af.att_freq_prod == 987.6, f"expected {987.6}, got {af.att_freq_prod}; test #2"
        expected = ([(1234.5, 0.9), (500.0, 0.1)], [(987.6, 0.85), (400.0, 0.15)])
        assert af.att_freq_react_corr == expected[0], f"expected {expected[0]}, got {af.att_freq_react_corr}; test #3"
        assert af.att_freq_prod_corr  == expected[1], f"expected {expected[1]}, got {af.att_freq_prod_corr}; test #4"

    def test_corr_lists_stored_correctly(self):
        """ Verify that frequency-correlation lists are stored correctly """
        af = self._make()
        assert len(af.att_freq_react_corr) == 2, f"expected {2}, got {len(af.att_freq_react_corr)}; test #1"
        assert len(af.att_freq_prod_corr)  == 2, f"expected {2}, got {af.att_freq_prod_corr}; test #2"

    def test_dataclass_len_and_dataclass_equality(self):
        """ Verify that AttemptFreq objects have length of four and objects with the same values are equal """
        a = self._make()
        b = self._make()
        assert len(dataclasses.fields(AttemptFreq)) == 4, f"expected {len(dataclasses.fields(AttemptFreq))} == "
        f"{4}; test #1"
        assert a == b, f"expected {a} == {b}; test #2"

    def test_inequality_different_freq(self):
        """ Verify that AttemptFreq objects with different frequencies are not equal """
        a = self._make()
        b = AttemptFreq(
            att_freq_react=999.0,
            att_freq_react_corr=[],
            att_freq_prod=987.6,
            att_freq_prod_corr=[])
        assert a != b, f"expected {a} != {b}"

    def test_all_fields_required(self):
        """ Verify that all AttemptFreq fields are required. Expecting a TypeError """
        with pytest.raises(TypeError):
            AttemptFreq() # missing required fields (frequencies and correlations)


# ===========================================================================
# --- TunnexInputSettings - normal behaviour and input validation ---
# ===========================================================================
class TestTunnexInputSettings:

    def test_required_fields_only(self):
        """ Verify that TunnexInputSettings can be created with only required fields """
        s = TunnexInputSettings(freq=1000.0, E0=-153.0, ZPVE0=0.05)
        assert s.freq == 1000.0, f"expected {s.freq} == {1000}; test #1"
        assert s.E0 == -153.0, f"expected {s.E0} == {-153.0}; test #2"
        assert s.ZPVE0 == 0.05, f"expected {s.ZPVE0} == {0.05}; test #3"

    def test_default_temperature(self):
        """ Verify that the default temperature is set correctly """
        s = TunnexInputSettings(freq=1000.0, E0=-153.0, ZPVE0=0.05)
        expected = 10
        assert s.T == expected, f"expected {s.number_of_levels} == {expected}"

    def test_default_potential_scaling_factor(self):
        """ Verify that the default potential scaling factor is set correctly """
        s = TunnexInputSettings(freq=1000.0, E0=-153.0, ZPVE0=0.05)
        expected = 1.0
        assert s.potential_scaling_factor == expected, f"expected {s.number_of_levels} == {expected}"

    def test_default_number_of_levels(self):
        """ Verify that the default number of levels and probability averaging are set correctly """
        s = TunnexInputSettings(freq=1000.0, E0=-153.0, ZPVE0=0.05)
        expected_num = 10
        expected_prob_aver = 1
        assert s.number_of_levels == expected_num, f"expected {s.number_of_levels} == "
        f"{expected_num}; test #1"
        assert s.prob_aver_mode_qmt == expected_prob_aver, f"expected {s.prob_aver_mode_qmt} == "
        f"{expected_prob_aver}; test #2"

    def test_default_T_arrhenius_min(self):
        """ Verify that the default minimum Arrhenius temperature is set correctly """
        s = TunnexInputSettings(freq=1000.0, E0=-153.0, ZPVE0=0.05)
        expected = 5
        assert s.T_arrhenius_min == expected, f"expected {s.T_arrhenius_min} == {expected}"

    def test_default_T_arrhenius_max(self):
        """ Verify that the default maximum Arrhenius temperature is set correctly """
        s = TunnexInputSettings(freq=1000.0, E0=-153.0, ZPVE0=0.05)
        expected = 1000
        assert s.T_arrhenius_max == 1000, f"expected {s.T_arrhenius_max} == {expected}"

    def test_default_T_step(self):
        """ Verify that the default Arrhenius temperature step is set correctly """
        s = TunnexInputSettings(freq=1000.0, E0=-153.0, ZPVE0=0.05)
        expected = 5
        assert s.T_step == 5, f"expected {s.T_arrhenius_max} == {expected}"

    def test_override_defaults(self):
        """ Verify that default settings can be overridden """
        s = TunnexInputSettings(
            freq=2000.0,
            E0=-100.0,
            ZPVE0=0.01,
            T=300,
            potential_scaling_factor=1.5,
            number_of_levels=20,
            T_arrhenius_min=100,
            T_arrhenius_max=500,
            T_step=10,
            prob_aver_mode_qmt = "integral")
        assert s.T == 300, f"expected {s.T} == {300}; test #1"
        assert s.potential_scaling_factor == 1.5, f"expected {s.potential_scaling_factor} == "
        f"{1.5}; test #2"
        assert s.number_of_levels == 20, f"expected {s.number_of_levels} == {20}; test #3"
        assert s.T_arrhenius_min == 100, f"expected {s.T_arrhenius_min} == {100}; test #4"
        assert s.T_arrhenius_max == 500, f"expected {s.T_arrhenius_max} == {500}; test #5"
        assert s.T_step == 10, f"expected {s.T_step} == {10}; test #6"
        assert s.prob_aver_mode_qmt == "integral", f"expected {s.prob_aver_mode_qmt} == "
        f"{"integral"}; test #7"

    def test_equality(self):
        """ Verify that settings with the same values are equal """
        a = TunnexInputSettings(freq=1000.0, E0=-153.0, ZPVE0=0.05)
        b = TunnexInputSettings(freq=1000.0, E0=-153.0, ZPVE0=0.05)
        assert a == b, f"expected {a} == {b}"

    def test_inequality_different_freq(self):
        """ Verify that settings with different frequencies are not equal """
        a = TunnexInputSettings(freq=1000.0, E0=-153.0, ZPVE0=0.05)
        b = TunnexInputSettings(freq=2000.0, E0=-153.0, ZPVE0=0.05)
        assert a != b, f"expected {a} != {b}"

    def test_T_arrhenius_range_sensible_by_default(self):
        """ Verify that the default Arrhenius temperature range is valid """
        s = TunnexInputSettings(freq=1000.0, E0=-153.0, ZPVE0=0.05)
        assert s.T_arrhenius_max > s.T_arrhenius_min, f"expected {s.T_arrhenius_max} > {s.T_arrhenius_min}"

    def test_has_ten_fields(self):
        """ Verify that TunnexInputSettings contains ten fields """
        expected = 10
        assert len(dataclasses.fields(TunnexInputSettings)) == expected, f"expected {expected}, "
        f"got {len(dataclasses.fields(TunnexInputSettings))}"

    def test_missing_required_field_raises(self):
        """ Verify that construction fails when a required field is missing. Expecting a TypeError """
        with pytest.raises(TypeError):
            TunnexInputSettings(freq=1000.0, E0=-153.0)  # ZPVE0 is missing


# ===========================================================================
# --- IRCDataQMT - normal behaviour and input validation ---
# ===========================================================================
class TestIRCDataQMTNormal:

    def _make(self, e_x=None, e_y=None, zpve_x=None, zpve_y=None):
        """ Should build an IRCDataQMT instance with default or given numpy array fields """
        e_x = e_x if e_x is not None else np.array([1.0, 2.0, 3.0])
        e_y = e_y if e_y is not None else np.array([-10.0, -11.0, -12.0])
        zpve_x = zpve_x if zpve_x is not None else np.array([1.0, 2.0, 3.0])
        zpve_y = zpve_y if zpve_y is not None else np.array([0.01, 0.02, 0.03])
        result = IRCDataQMT(e_x, e_y, zpve_x, zpve_y)
        return result

    def test_fields_store_given_arrays(self):
        """ Verify that all four fields store the arrays passed to the constructor """
        e_x = np.array([1.0, 2.0])
        e_y = np.array([-1.0, -2.0])
        zpve_x = np.array([1.0, 2.0])
        zpve_y = np.array([0.1, 0.2])
        result = IRCDataQMT(e_x, e_y, zpve_x, zpve_y)
        assert np.array_equal(result.E_x, e_x), f"expected {e_x}, got {result.E_x}; test #1"
        assert np.array_equal(result.E_y, e_y), f"expected {e_y}, got {result.E_y}; test #2"
        assert np.array_equal(result.ZPVE_x, zpve_x), f"expected {zpve_x}, "
        f"got {result.ZPVE_x}; test #3"
        assert np.array_equal(result.ZPVE_y, zpve_y), f"expected {zpve_y}, "
        f"got {result.ZPVE_y}; test #4"

    def test_fields_accept_numpy_ndarray_type(self):
        """ Verify that the stored fields remain numpy ndarrays """
        result = self._make()
        assert isinstance(result.E_x, np.ndarray), f"expected {np.ndarray}, got {type(result.E_x)}; test #1"
        assert isinstance(result.E_y, np.ndarray), f"expected {np.ndarray}, got {type(result.E_y)}; test #2"
        assert isinstance(result.ZPVE_x, np.ndarray), f"expected {np.ndarray}, "
        f"got {type(result.ZPVE_x)}; test #3"
        assert isinstance(result.ZPVE_y, np.ndarray), f"expected {np.ndarray}, "
        f"got {type(result.ZPVE_y)}; test #4"

    def test_two_instances_with_same_values_are_equal(self):
        """ Verify that two LevelResult instances with identical field values compare equal """
        result_1 = self._make()
        result_2 = self._make()
        value = np.array_equal(result_1.E_x, result_2.E_x)
        assert value, f"expected {True}, got {value}"

    def test_two_instances_with_different_values_are_not_equal(self):
        """ Verify that two LevelResult instances with a differing field do not compare equal """
        result_1 = self._make(e_x=np.array([1.0, 2.0]))
        result_2 = self._make(e_x=np.array([1.0, 3.0]))
        value = not np.array_equal(result_1.E_x, result_2.E_x)
        assert value, f"expected {True}, got {value}"

    def test_is_a_dataclass_instance(self):
        """ Verify that IRCDataQMT is recognized as a dataclass """
        result = self._make()
        assert dataclasses.is_dataclass(result), f"expected {True}, got {dataclasses.is_dataclass(result)}"

    def test_field_names_match_expected_set(self):
        """ Verify that the dataclass exposes exactly the four expected field names """
        field_names = {f.name for f in dataclasses.fields(IRCDataQMT)}
        expected = {"E_x", "E_y", "ZPVE_x", "ZPVE_y"}
        assert field_names == expected, f"expected {expected}, got {field_names}"

    def test_missing_required_argument_raises_type_error(self):
        """ Verify that omitting a required field raises an error. Expecting a TypeError """
        with pytest.raises(TypeError):
             IRCDataQMT(np.array([1.0]), np.array([1.0]), np.array([1.0])) # missing ZPVE field


# ===========================================================================
# --- LevelResult - normal behaviour and input validation ---
# ===========================================================================
class TestLevelResultNormal:

    def _make(self, **overrides):
        """ Should build a LevelResult instance with default or overridden field values """
        defaults = dict(
            vib_index=0,
            vib_energy=0.005,
            turning_point_left=-0.5,
            turning_point_right=0.5,
            wkb=1.2,
            transmission_probability=0.3,
            reaction_rate=1e10,
            half_s=6.93e-11,
            half_h=1.925e-14,
            half_d=8.02e-16,
            half_y=2.2e-18)
        defaults.update(overrides)
        result = LevelResult(**defaults)
        return result

    def test_fields_store_given_values(self):
        """ Verify that all LevelResult fields store the values passed to the constructor """
        result = self._make(vib_index=3, vib_energy=0.012, wkb=2.5, transmission_probability=0.42, reaction_rate=5e9)
        assert result.vib_index == 3, f"expected {3}, got {result.vib_index}; test #1"
        assert result.vib_energy == 0.012, f"expected {0.012}, got {result.vib_energy}; test #2"
        assert result.wkb == 2.5, f"expected {2.5}, got {result.wkb}; test #3"
        assert result.transmission_probability == 0.42, f"expected {0.42}, "
        f"got {result.transmission_probability}; test #4"
        assert result.reaction_rate == 5e9, f"expected {5e9}, got {result.reaction_rate}; test #5"

    def test_fields_have_expected_types(self):
        """ Verify that all LevelResult fields have the expected types """
        result = self._make()
        assert isinstance(result.vib_index, int), f"expected {int}, "
        f"got {type(result.vib_index)}; test #1"
        assert isinstance(result.vib_energy, float), f"expected {float}, "
        f"got {type(result.vib_energy)}; test #2"
        assert isinstance(result.turning_point_left, (float, type(None))), f"expected {float} "
        f"or {None}, got {type(result.turning_point_left)}; test #3"
        assert isinstance(result.turning_point_right, (float, type(None))), f"expected {float} "
        f"or {None}, got {type(result.turning_point_right)}; test #4"
        assert isinstance(result.wkb, float), f"expected {float}, "
        f"got {type(result.wkb)}; test #5"
        assert isinstance(result.transmission_probability, float), f"expected {float}, "
        f"got {type(result.transmission_probability)}; test #6"
        assert isinstance(result.reaction_rate, float), f"expected {float}, "
        f"got {type(result.reaction_rate)}; test #7"
        assert isinstance(result.half_s, float), f"expected {float}, "
        f"got {type(result.half_s)}; test #8"
        assert isinstance(result.half_h, float), f"expected {float}, "
        f"got {type(result.half_h)}; test #9"
        assert isinstance(result.half_d, float), f"expected {float}, "
        f"got {type(result.half_d)}; test #10"
        assert isinstance(result.half_y, float), f"expected {float}, "
        f"got {type(result.half_y)}; test #11"
        result = self._make(turning_point_left=None, turning_point_right=None)
        assert result.turning_point_left is None, f"expected {float} "
        f"or {None}, got {type(result.turning_point_left)}; test #12"
        assert result.turning_point_right is None, f"expected {float} "
        f"or {None}, got {type(result.turning_point_right)}; test #12"

    def test_turning_points_accept_none(self):
        """ Verify that turning_point_left and turning_point_right accept None values """
        result = self._make(turning_point_left=None, turning_point_right=None)
        assert result.turning_point_left is None, f"expected {None}, "
        f"got {result.turning_point_left}; test #1"
        assert result.turning_point_right is None, f"expected {None}, "
        f"got {result.turning_point_right}; test #2"

    def test_turning_points_accept_float_values(self):
        """ Verify that turning_point_left and turning_point_right accept float values """
        result = self._make(turning_point_left=-1.1, turning_point_right=1.1)
        assert result.turning_point_left == -1.1, f"expected {-1.1}, "
        f"got {result.turning_point_left}; test #1"
        assert result.turning_point_right == 1.1, f"expected {1.1}, "
        f"got {result.turning_point_right}; test #2"

    def test_half_life_fields_store_given_values(self):
        """ Verify that all four half-life unit fields store the values passed to the constructor """
        result = self._make(half_s=1.0, half_h=2.0, half_d=3.0, half_y=4.0)
        assert result.half_s == 1.0, f"expected {1.0}, got {result.half_s}; test #1"
        assert result.half_h == 2.0, f"expected {2.0}, got {result.half_h}; test #2"
        assert result.half_d == 3.0, f"expected {3.0}, got {result.half_d}; test #3"
        assert result.half_y == 4.0, f"expected {4.0}, got {result.half_y}; test #4"

    def test_is_a_dataclass_instance(self):
        """ Verify that LevelResult is recognized as a dataclass """
        result = self._make()
        assert dataclasses.is_dataclass(result), f"expected {True}, got {dataclasses.is_dataclass(result)}"

    def test_field_names_match_expected_set(self):
        """ Verify that the dataclass exposes exactly the eleven expected field names """
        field_names = {f.name for f in dataclasses.fields(LevelResult)}
        expected = {"vib_index", "vib_energy", "turning_point_left", "turning_point_right",
                    "wkb", "transmission_probability", "reaction_rate",
                    "half_s", "half_h", "half_d", "half_y"}
        assert field_names == expected, f"expected {expected}, got {field_names}"

    def test_two_instances_with_same_values_are_equal(self):
        """ Verify that two LevelResult instances with identical field values compare equal """
        first = self._make()
        second = self._make()
        assert first == second, f"expected {first}, got {second}"

    def test_two_instances_with_different_values_are_not_equal(self):
        """ Verify that two LevelResult instances with a differing field do not compare equal """
        first = self._make(vib_index=0)
        second = self._make(vib_index=1)
        assert first != second, f"expected instances to differ, got {first} == {second}"

    def test_missing_required_argument_raises_type_error(self):
        """ Verify that omitting a required field raises an error. Expecting a TypeError """
        with pytest.raises(TypeError):
            LevelResult(0, 0.005, -0.5, 0.5, 1.2, 0.3, 1e10, 6.93e-11, 1.925e-14, 8.02e-16)
            # missing half_y field


# ===========================================================================
# --- TemperatureAverResult - normal behaviour and input validation ---
# ===========================================================================
class TestTemperatureAverResultNormal:

    def _make(self, **overrides):
        """ Should build a TemperatureAverResult instance with default or overridden field values """
        defaults = dict(
            transmission_probability=0.25,
            reaction_rate=2e9,
            half_s=3.5e-10,
            half_h=9.7e-14,
            half_d=4.0e-15,
            half_y=1.1e-17)
        defaults.update(overrides)
        result = TemperatureAverResult(**defaults)
        return result

    def test_fields_store_given_values(self):
        """ Verify that all TemperatureAverResult fields store the values passed to the constructor """
        result = self._make(transmission_probability=0.6, reaction_rate=7e8,
                            half_s=1.0, half_h=2.0, half_d=3.0, half_y=4.0)
        assert result.transmission_probability == 0.6, f"expected {0.6}, "
        f"got {result.transmission_probability}; test #1"
        assert result.reaction_rate == 7e8, f"expected {7e8}, got {result.reaction_rate}; test #2"
        assert result.half_s == 1.0, f"expected {1.0}, got {result.half_s}; test #3"
        assert result.half_h == 2.0, f"expected {2.0}, got {result.half_h}; test #4"
        assert result.half_d == 3.0, f"expected {3.0}, got {result.half_d}; test #5"
        assert result.half_y == 4.0, f"expected {4.0}, got {result.half_y}; test #6"

    def test_fields_have_expected_types(self):
        """ Verify that all TemperatureAverResult fields have the expected types """
        result = self._make()
        assert isinstance(result.transmission_probability, float), f"expected {float}, "
        f"got {type(result.transmission_probability)}; test #1"
        assert isinstance(result.reaction_rate, float), f"expected {float}, "
        f"got {type(result.reaction_rate)}; test #2"
        assert isinstance(result.half_s, float), f"expected {float}, "
        f"got {type(result.half_s)}; test #3"
        assert isinstance(result.half_h, float), f"expected {float}, "
        f"got {type(result.half_h)}; test #4"
        assert isinstance(result.half_d, float), f"expected {float}, "
        f"got {type(result.half_d)}; test #5"
        assert isinstance(result.half_y, float), f"expected {float}, "
        f"got {type(result.half_y)}; test #6"

    def test_is_a_dataclass_instance(self):
        """ Verify that TemperatureAverResult is recognized as a dataclass """
        result = self._make()
        assert dataclasses.is_dataclass(result), f"expected {True}, got {dataclasses.is_dataclass(result)}"

    def test_field_names_match_expected_set(self):
        """ Verify that the dataclass exposes exactly the six expected field names """
        field_names = {f.name for f in dataclasses.fields(TemperatureAverResult)}
        expected = {"transmission_probability", "reaction_rate", "half_s", "half_h", "half_d", "half_y"}
        assert field_names == expected, f"expected {expected}, got {field_names}"

    def test_two_instances_with_same_values_are_equal(self):
        """ Verify that two TemperatureAverResult instances with identical field values compare equal """
        first = self._make()
        second = self._make()
        assert first == second, f"expected {first}, got {second}"

    def test_two_instances_with_different_values_are_not_equal(self):
        """ Verify that two TemperatureAverResult instances with a differing field do not compare equal """
        first = self._make(reaction_rate=1.0)
        second = self._make(reaction_rate=2.0)
        assert first != second, f"expected instances to differ, got {first} == {second}"

    def test_missing_required_argument_raises_type_error(self):
        """ Verify that omitting a required field raises an error. Expecting a TypeError """
        with pytest.raises(TypeError):
            TemperatureAverResult(0.25, 2e9, 3.5e-10, 9.7e-14, 4.0e-15)
            # missing half_y field


# ===========================================================================
# --- ArrheniusResult - normal behaviour and input validation ---
# ===========================================================================
class TestArrheniusResultNormal:

    def _make(self, **overrides):
        """ Should build an ArrheniusResult instance with default or overridden field values """
        defaults = dict(
            arrhenius_temperature=298.15,
            arrhenius_temperature_inv=1.0 / 298.15,
            log_reaction_rate=23.5)
        defaults.update(overrides)
        result = ArrheniusResult(**defaults)
        return result

    def test_fields_store_given_values(self):
        """ Verify that all ArrheniusResult fields store the values passed to the constructor """
        result = self._make(arrhenius_temperature=300.0, arrhenius_temperature_inv=1.0 / 300.0,
                            log_reaction_rate=20.1)
        assert result.arrhenius_temperature == 300.0, f"expected {300.0}, "
        f"got {result.arrhenius_temperature}; test #1"
        assert result.arrhenius_temperature_inv == 1.0 / 300.0, f"expected {1.0 / 300.0}, "
        f"got {result.arrhenius_temperature_inv}; test #2"
        assert result.log_reaction_rate == 20.1, f"expected {20.1}, "
        f"got {result.log_reaction_rate}; test #3"
        assert isinstance(result.arrhenius_temperature, float), f"expected {float}, "
        f"got {type(result.log_reaction_rate)}; test #4"
        assert isinstance(result.arrhenius_temperature_inv, float), f"expected {float}, "
        f"got {type(result.arrhenius_temperature_inv)}; test #5"
        assert isinstance(result.log_reaction_rate, float), f"expected {float}, "
        f"got {type(result.log_reaction_rate)}; test #6"

    def test_is_a_dataclass_instance(self):
        """ Verify that ArrheniusResult is recognized as a dataclass """
        result = self._make()
        assert dataclasses.is_dataclass(result), f"expected {True}, got {dataclasses.is_dataclass(result)}"

    def test_field_names_match_expected_set(self):
        """ Verify that the dataclass exposes exactly the three expected field names """
        field_names = {f.name for f in dataclasses.fields(ArrheniusResult)}
        expected = {"arrhenius_temperature", "arrhenius_temperature_inv", "log_reaction_rate"}
        assert field_names == expected, f"expected {expected}, got {field_names}"

    def test_two_instances_with_same_values_are_equal(self):
        """ Verify that two ArrheniusResult instances with identical field values compare equal """
        first = self._make()
        second = self._make()
        assert first == second, f"expected {first}, got {second}"

    def test_two_instances_with_different_values_are_not_equal(self):
        """ Verify that two ArrheniusResult instances with a differing field do not compare equal """
        first = self._make(log_reaction_rate=1.0)
        second = self._make(log_reaction_rate=2.0)
        assert first != second, f"expected instances to differ, got {first} == {second}"

    def test_missing_required_argument_raises_type_error(self):
        """ Verify that omitting a required field raises an error. Expecting a TypeError """
        with pytest.raises(TypeError):
            ArrheniusResult(298.15, 1.0 / 298.15)
            # missing log_reaction_rate field