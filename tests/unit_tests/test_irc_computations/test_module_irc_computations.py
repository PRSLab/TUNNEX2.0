# --- Test: test_module_irc_computations. Unit tests for module_irc_computations.py
# Run with: pytest test_module_irc_computations.py -v ---

# --- Modules ---
import pytest  # type: ignore
from unittest.mock import MagicMock, patch

from pathlib import Path

from tunnex_2.constants_and_dataclasses.dataclasses import FindIRCConfig  # type: ignore

# --- Module to test ---
from tunnex_2.irc_computations.module_irc_computations import irc_computations_main # type: ignore


# --- Helpers and shared data ---
TS_IN = Path("mol.gjf")
TS_OUT = Path("mol_ts.out")
IRC_OUT = Path("mol_irc.out")
REACT = Path("react.out")
PROD = Path("prod.out")
FWD = Path("tunnex_forward.txt")
BWD = Path("tunnex_backward.txt")
TUN_INP = Path("tunnex_input.txt")
FUNC_PATH = "tunnex_2.irc_computations.module_irc_computations"


def _make_irc_data(zpve_f=[(0.5, 0.03), (1.0, 0.04)], zpve_r=[(-0.5, 0.03)]):
    """ Should make the dummy IRC data """
    m = MagicMock()
    m.electronic_energies = [(-1.0, -153.10), (0.0, -153.05), (1.0, -153.09)]
    m.zpve_energies_forward = zpve_f
    m.zpve_energies_reverse = zpve_r
    return m

def _make_energy_points():
    """ Should make the dummy energy points data """
    ep = MagicMock()
    ep.transition_state_el_energy = (0.0, -153.05)
    ep.transition_state_zpve = (0.0, 0.05)
    ep.reactant_el_energy = (0.0, -153.15)
    ep.reactant_zpve = (0.0, 0.02)
    ep.product_el_energy = (0.0, -153.10)
    ep.product_zpve = (0.0, 0.02)
    return ep

def _make_attempt_freq():
    """ Should make the dummy attempt frequency data """
    af = MagicMock()
    af.att_freq_react = 1000.0
    af.att_freq_prod = 900.0
    af.att_freq_react_corr = [(1000.0, 0.9)]
    af.att_freq_prod_corr = [(900.0, 0.85)]
    return af

def _base_config(**kwargs):
    """ Should return a FindIRCConfig with minimal valid defaults """
    defaults = dict(
        prog_mode = "gaussian",
        proj_freq= True,
        calc_all = True,
        hybrid_mode = False,
        eckart = False,
        qmt_module = True,
        hess_parsing_flag = False,
        prob_aver_mode_qmt = "finite_sum",
        ts_input_file = TS_IN,
        react_input_file = None,
        prod_input_file = None,
        ts_computed_file = TS_OUT,
        irc_computed_file = IRC_OUT,
        react_computed_file = REACT,
        prod_computed_file = PROD,
        tunnex_input = TUN_INP)
    defaults.update(kwargs)
    result = FindIRCConfig(**defaults)
    return result

def _base_patches():
    """Should return a dict of patch targets """
    result = {
        f"{FUNC_PATH}.file_check":
            MagicMock(side_effect=lambda p: Path(p)),
        f"{FUNC_PATH}.ts_optimization":
            MagicMock(return_value=TS_OUT),
        f"{FUNC_PATH}.irc_computations":
            MagicMock(return_value=IRC_OUT),
        f"{FUNC_PATH}.irc_file_parsing":
            MagicMock(return_value=_make_irc_data()),
        f"{FUNC_PATH}.react_prod_optimization":
            MagicMock(return_value=(REACT, PROD)),
        f"{FUNC_PATH}.eckart_react_prod_optimization":
            MagicMock(return_value=(REACT, PROD)),
        f"{FUNC_PATH}.attempt_freq_func":
            MagicMock(return_value=_make_attempt_freq()),
        f"{FUNC_PATH}.opt_file_parsing":
            MagicMock(return_value=_make_energy_points()),
        f"{FUNC_PATH}.potential_scal_factor_comp":
            MagicMock(return_value=(1.2, 0.9)),
        f"{FUNC_PATH}.number_of_levels_func":
            MagicMock(return_value=(5, 4)),
        f"{FUNC_PATH}.tunnex_input_writing":
            MagicMock(return_value=(FWD, BWD)),
        f"{FUNC_PATH}.eckart_potential_results":
            MagicMock(return_value=_make_irc_data())}
    return result


# ===========================================================================
# --- irc_computations_main - normal behaviour ---
# ===========================================================================
class TestIRCCompMainNormal:

    def test_gaussian_mode_accepted(self):
        """ Tests that IRC computations run successfully in Gaussian mode """
        cfg = _base_config(prog_mode="gaussian")
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{
            k.split(".")[-1]: v for k, v in patches.items()}):
            result = irc_computations_main(cfg)
        assert result == (FWD, BWD), f"expected {(FWD, BWD)}, got {result}"   
    
    def test_orca_mode_accepted(self):
        """ Tests that IRC computations run successfully in ORCA mode """
        cfg = _base_config(prog_mode="orca")
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{
            k.split(".")[-1]: v for k, v in patches.items()}):
            result = irc_computations_main(cfg)
        assert result == (FWD, BWD), f"expected {(FWD, BWD)}, got {result}"   
    
    def test_ts_optimization_called_when_computed_file_none(self):
        """ Tests that TS optimization is called when no computed TS file is provided """
        cfg = _base_config(ts_computed_file=None)
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.ts_optimization"].assert_called_once()
        # it should be called once

    def test_ts_optimization_skipped_when_computed_file_given(self):
        """ Tests that TS optimization is skipped when a computed TS file is provided """
        cfg = _base_config(ts_computed_file=TS_OUT)
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.ts_optimization"].assert_not_called()
        # it should not be called

    def test_file_check_called_for_ts_computed_file(self):
        """ Tests that the computed TS file is checked when provided """
        cfg = _base_config(ts_computed_file=TS_OUT)
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        fc = patches[f"{FUNC_PATH}.file_check"]
        fc.assert_any_call(TS_OUT) # it should be called

    def test_irc_computations_called_when_irc_file_none(self):
        """ Tests that IRC computations are called when no computed IRC file is provided """
        cfg = _base_config(irc_computed_file=None)
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.irc_computations"].assert_called_once()
        # it should be called once

    def test_irc_computations_skipped_when_file_given(self):
        """ Tests that IRC computations are skipped when a computed IRC file is provided """
        cfg = _base_config(irc_computed_file=IRC_OUT)
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.irc_computations"].assert_not_called()
        # it should not be called

    def test_irc_file_parsing_always_called_for_non_eckart(self):
        """ Tests that IRC file parsing is called for non-Eckart calculations """
        cfg = _base_config(eckart=False)
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.irc_file_parsing"].assert_called_once()
        # it should be called once

    def test_irc_file_parsing_not_called_for_eckart(self):
        """ Tests that IRC file parsing is skipped for Eckart calculations """
        cfg = _base_config(eckart=True,
            react_input_file=Path("react.gjf"),
            prod_input_file=Path("prod.gjf"))
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.irc_file_parsing"].assert_not_called()
        # it should not be called

    def test_react_prod_optimization_called_when_files_none(self):
        """ Tests that reactant and product optimization is called when computed files are missing """
        cfg = _base_config(react_computed_file=None, prod_computed_file=None)
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.react_prod_optimization"].assert_called_once()
        # it should be called once

    def test_react_prod_skipped_when_files_given(self):
        """ Tests that reactant and product optimization is skipped when computed files are provided """
        cfg = _base_config(react_computed_file=REACT, prod_computed_file=PROD)
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.react_prod_optimization"].assert_not_called()
        # it should not be called

    def test_eckart_uses_eckart_react_prod_optimization(self):
        """ Tests that Eckart mode uses the dedicated reactant and product optimization """
        cfg = _base_config(eckart=True,
            react_input_file=Path("react.gjf"),
            prod_input_file=Path("prod.gjf"),
            react_computed_file=None,
            prod_computed_file=None)
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.eckart_react_prod_optimization"].assert_called_once()
        # it should be called once
        patches[f"{FUNC_PATH}.react_prod_optimization"].assert_not_called()
        # it should not be called

    def test_hybrid_non_eckart_calls_scal_factor_comp(self):
        """ Tests that hybrid non-Eckart mode calls the potential scaling factor computation """
        cfg = _base_config(hybrid_mode=True, eckart=False)
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.potential_scal_factor_comp"].assert_called_once()
        # it should be called once

    def test_non_hybrid_uses_default_scaling_factor(self):
        """ Tests that non-hybrid mode uses the default scaling factor without computing it """
        cfg = _base_config(hybrid_mode=False, eckart=False)
        patches = _base_patches()
        captured = {}
        _ = patches[f"{FUNC_PATH}.tunnex_input_writing"]
        def capture(*args, **kwargs):
            captured["args"] = args
            return (FWD, BWD)
        patches[f"{FUNC_PATH}.tunnex_input_writing"] = MagicMock(side_effect=capture)
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.potential_scal_factor_comp"].assert_not_called()
        # it should not be called

    def test_eckart_always_uses_default_scaling_factor(self):
        """ Tests that Eckart mode uses the default scaling factor without computing it """
        cfg = _base_config(eckart=True,
            hybrid_mode=True,
            react_input_file=Path("react.gjf"),
            prod_input_file=Path("prod.gjf"))
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.potential_scal_factor_comp"].assert_not_called()
        # it should not be called

    def test_eckart_potential_results_called_for_eckart(self):
        """ Tests that Eckart potential results are computed in Eckart mode """
        cfg = _base_config(eckart=True,
            react_input_file=Path("react.gjf"),
            prod_input_file=Path("prod.gjf"))
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.eckart_potential_results"].assert_called_once()
        # it should be called once

    def test_eckart_potential_results_not_called_for_non_eckart(self):
        """ Tests that Eckart potential results are not computed in non-Eckart mode """
        cfg = _base_config(eckart=False)
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.eckart_potential_results"].assert_not_called()
        # it should not be called

    def test_returns_two_paths(self):
        """ Tests that IRC computations return two paths """
        cfg = _base_config()
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            result = irc_computations_main(cfg)
        assert isinstance(result, tuple), f"expected {tuple}, got {type(result)}; test #1"
        assert len(result) == 2, f"expected {2}, got {len(result)}; test #2"

    def test_returns_forward_and_backward_paths(self):
        """ Tests that IRC computations return the expected forward and backward paths """
        cfg = _base_config()
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            fwd, bwd = irc_computations_main(cfg)
        assert fwd == FWD, f"expected {FWD}, got {fwd}; test #1"
        assert bwd == BWD, f"expected {BWD}, got {bwd}; test #2"

    def test_tunnex_input_writing_always_called(self):
        """ Tests that Tunnex input writing is always called """
        cfg = _base_config()
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.tunnex_input_writing"].assert_called_once()
        # it should be called once

    def test_number_of_levels_func_always_called(self):
        """ Tests that the number of vibrational levels is always computed """
        cfg = _base_config()
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.number_of_levels_func"].assert_called_once()
        # it should be called once

    def test_attempt_freq_func_always_called(self):
        """ Tests that the attempt frequency is always computed """
        cfg = _base_config()
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.attempt_freq_func"].assert_called_once()
        # it should be called once

    def test_opt_file_parsing_always_called(self):
        """ Tests that optimization file parsing is always called """
        cfg = _base_config()
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        patches[f"{FUNC_PATH}.opt_file_parsing"].assert_called_once()
        # it should be called once

    def test_file_check_called_for_react_and_prod_computed_files(self):
        """ Tests that file_check is called for both reactant and product computed files when provided """
        cfg = _base_config(react_computed_file=REACT, prod_computed_file=PROD)
        patches = _base_patches()
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        fc = patches[f"{FUNC_PATH}.file_check"]
        fc.assert_any_call(REACT) # it should be called
        fc.assert_any_call(PROD) # it should be called

    def test_number_of_levels_func_receives_correct_frequencies_and_scaling_factors(self):
        """
        Tests that number_of_levels_func is called with the attempt frequencies
        and scaling factors it should use
        """
        cfg = _base_config(hybrid_mode=True, eckart=False)
        patches = _base_patches()
        captured = {}
        def capture(irc_data, energy_points, freq_react, freq_prod, scal_react, scal_prod):
            captured["args"] = (freq_react, freq_prod, scal_react, scal_prod)
            return (5, 4)
        patches[f"{FUNC_PATH}.number_of_levels_func"] = MagicMock(side_effect=capture)
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        freq_react, freq_prod, scal_react, scal_prod = captured["args"]
        assert freq_react == 1000.0, f"expected {1000.0}, got {freq_react}; test #1"
        assert freq_prod == 900.0, f"expected {900.0}, got {freq_prod}; test #2"
        assert (scal_react, scal_prod) == (1.2, 0.9), f"expected {(1.2, 0.9)}, "
        f"got {(scal_react, scal_prod)}; test #3"

    def test_eckart_mode_forwards_eckart_irc_data_downstream(self):
        """
        Verifies that Eckart IRC data is correctly forwarded
        to the downstream level-counting function
        """
        eckart_irc_data = _make_irc_data(zpve_f=[(9.9, 9.9)])
        cfg = _base_config(eckart=True,
            react_input_file=Path("react.gjf"),
            prod_input_file=Path("prod.gjf"))
        patches = _base_patches()
        patches[f"{FUNC_PATH}.eckart_potential_results"] = MagicMock(return_value=eckart_irc_data)
        captured = {}
        def capture(irc_data, energy_points, freq_react, freq_prod, scal_react, scal_prod):
            captured["irc_data"] = irc_data
            return (5, 4)
        patches[f"{FUNC_PATH}.number_of_levels_func"] = MagicMock(side_effect=capture)
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            irc_computations_main(cfg)
        assert captured["irc_data"] is eckart_irc_data, f"expected {eckart_irc_data}, "
        f"got {captured['irc_data']}"


# ===========================================================================
# --- irc_computations_main - input validation ---
# ===========================================================================
class TestIRCCompMainValidation:

    def test_missing_ts_input_file_raises(self):
        """ Tests that missing TS input file raises an error. Expecting a ValueError """
        cfg = FindIRCConfig(ts_input_file=None)
        with pytest.raises(FileNotFoundError, match="TS input file is required"):
             irc_computations_main(cfg)

    def test_invalid_prog_mode_raises(self):
        """ Tests that an unsupported program mode raises an error. Expecting a ValueError """
        cfg = _base_config(prog_mode="molpro")
        with pytest.raises(ValueError, match="Expected"):
             irc_computations_main(cfg)

    def test_eckart_without_react_input_raises(self):
        """
        Tests that missing reactant input file raises
        an error in Eckart mode. Expecting a ValueError
        """
        cfg = _base_config(eckart=True,
            react_input_file=None,
            prod_input_file=Path("prod.gjf"))
        with pytest.raises(FileNotFoundError, match="Reactant input file"):
             irc_computations_main(cfg)

    def test_eckart_without_prod_input_raises(self):
        """
        Tests that missing product input file raises
        an error in Eckart mode. Expecting a ValueError
        """
        cfg = _base_config(eckart=True,
            react_input_file=Path("react.gjf"),
            prod_input_file=None)
        with pytest.raises(FileNotFoundError, match="Product input file"):
             irc_computations_main(cfg)