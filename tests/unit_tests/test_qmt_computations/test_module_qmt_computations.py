# --- Test: test_module_qmt_computations. Unit tests for module_qmt_computations.py
# Run with: pytest test_module_qmt_computations.py -v ---

# --- Modules ---
import pytest  # type: ignore
from unittest.mock import MagicMock, patch

from pathlib import Path


# --- Module to test ---
from tunnex_2.qmt_computations.module_qmt_computations import qmt_computations_main # type: ignore


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.qmt_computations.module_qmt_computations"


def _make_irc(with_zpve=True):
    """ Should make the dummy IRC data """
    m = MagicMock()
    m.E_x = [0.0, 1.0, 2.0]
    m.E_y = [-153.0, -152.9, -153.0]
    m.ZPVE_x = [0.0, 1.0, 2.0]
    if with_zpve:
        m.ZPVE_y = [0.03, 0.04, 0.03]
    else:
        m.ZPVE_y = [0.00, 0.00, 0.00]
    return m

def _make_params():
    """ Should make the dummy input parameters """
    m = MagicMock()
    m.E0 = -153.15
    m.ZPVE0 = 0.02
    m.potential_scaling_factor = 1.0
    m.T = 300.0
    return m

def _base_patches(tmp_path):
    """Should return a dict of patch targets """
    inp = tmp_path / "mol_forward.txt"
    inp.write_text("dummy", encoding="utf-8")
    result = {
        f"{FUNC_PATH}.read_input":
            MagicMock(return_value=(_make_params(), _make_irc())),
        f"{FUNC_PATH}.detect_outliers_local":
            MagicMock(),
        f"{FUNC_PATH}.Potential":
            MagicMock(return_value=MagicMock(offset=0.01)),
        f"{FUNC_PATH}.qmt_computations_pipeline":
            MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock())),
        f"{FUNC_PATH}.build_report":
            MagicMock(return_value="report text"),
        f"{FUNC_PATH}.write_output":
            MagicMock()}
    return inp, result


# ===========================================================================
# --- qmt_computations_main - normal behaviour and input validation ---
# ===========================================================================
class TestQmtComputationsMain:

    def test_string_path_accepted(self, tmp_path):
        """ Verifies that the function accepts a file path provided as a string """
        inp, patches = _base_patches(tmp_path)
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            result = qmt_computations_main(str(inp))
        assert result is None, f"expected {None}, got {result}"

    def test_path_object_accepted(self, tmp_path):
        """ Verifies that the function accepts a file path provided as a Path object """
        inp, patches = _base_patches(tmp_path)
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            result = qmt_computations_main(Path(inp))
        assert result is None, f"expected {None}, got {result}"
    
    def test_output_filename_has_tunnex_2_suffix(self, tmp_path):
        """ Verifies that the output filename has the expected suffix """
        inp, patches = _base_patches(tmp_path)
        captured = {}
        def capture_write(path, report):
            captured["path"] = path
        patches[f"{FUNC_PATH}.write_output"] = MagicMock(side_effect=capture_write)
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            qmt_computations_main(inp)
        expected = "mol_forward_tunnex_2.out"
        result = captured["path"].name
        assert result == expected, f"expected {expected}, got {result}"

    def test_output_file_in_same_directory_as_input(self, tmp_path):
        """ Verifies that the output file is created in the same directory as the input file """
        inp, patches = _base_patches(tmp_path)
        captured = {}
        def capture_write(path, report):
            captured["path"] = path
        patches[f"{FUNC_PATH}.write_output"] = MagicMock(side_effect=capture_write)
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            qmt_computations_main(inp)
        result = captured["path"].parent
        assert result == inp.parent, f"expected {inp.parent}, got {result}"

    def test_read_input_called_with_resolved_path(self, tmp_path):
        """ Verifies that read_input is called with the resolved input file path """
        inp, patches = _base_patches(tmp_path)
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            qmt_computations_main(inp)
        read_mock = patches[f"{FUNC_PATH}.read_input"]
        read_mock.assert_called_once() # it should be called once
        called_path = read_mock.call_args[0][0]
        assert called_path == inp.resolve(), f"expected {inp.resolve()}, got {called_path}"

    def test_detect_outliers_called_twice(self, tmp_path):
        """ Verifies that outlier detection is called twice """
        inp, patches = _base_patches(tmp_path)
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            qmt_computations_main(inp)
        dol = patches[f"{FUNC_PATH}.detect_outliers_local"]
        assert dol.call_count == 2, f"expected {2}, got {dol.call_count}"

    def test_detect_outliers_called_for_el_energy_and_zpve(self, tmp_path):
        """ Verifies that outlier detection is performed for electronic energy and ZPVE """
        inp, patches = _base_patches(tmp_path)
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            qmt_computations_main(inp)
        dol = patches[f"{FUNC_PATH}.detect_outliers_local"]
        modes = [c[0][2] for c in dol.call_args_list]
        expected = ("electronic energy", "ZPVE")
        assert expected[0] in modes, f"expected {expected[0]} in {dol.call_count}; test #1"
        assert expected[1] in modes, f"expected {expected[1]} in {dol.call_count}; test #2"

    def test_detect_outliers_called_with_irc_data(self, tmp_path):
        """ Verifies that outlier detection receives the IRC electronic energy data """
        inp, patches = _base_patches(tmp_path)
        irc = _make_irc()
        patches[f"{FUNC_PATH}.read_input"] = MagicMock(return_value=(_make_params(), irc))
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v 
            for k, v in patches.items()}):
            qmt_computations_main(inp)
        dol = patches[f"{FUNC_PATH}.detect_outliers_local"]
        first_call_x = dol.call_args_list[0][0][0]
        assert first_call_x is irc.E_x, f"expected {irc.E_x}, got {first_call_x}"

    def test_potential_called_with_correct_args(self, tmp_path):
        """ Verifies that Potential is initialized with the correct IRC and parameter values """
        inp, patches = _base_patches(tmp_path)
        params = _make_params()
        irc = _make_irc()
        patches[f"{FUNC_PATH}.read_input"] = MagicMock(return_value=(params, irc))
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            qmt_computations_main(inp)
        pot_cls = patches[f"{FUNC_PATH}.Potential"]
        pot_cls.assert_called_once_with(
            irc.E_x, irc.E_y,
            irc.ZPVE_x, irc.ZPVE_y,
            params.E0, params.ZPVE0,
            params.potential_scaling_factor)
        # it should be called once

    def test_pipeline_called_with_params_and_potential(self, tmp_path):
        """
        Verifies that the QMT computation pipeline is called
        with the expected parameters and potential
        """
        inp, patches = _base_patches(tmp_path)
        pot_instance = MagicMock(offset=0.01)
        patches[f"{FUNC_PATH}.Potential"] = MagicMock(return_value=pot_instance)
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            qmt_computations_main(inp)
        pipe = patches[f"{FUNC_PATH}.qmt_computations_pipeline"]
        pipe.assert_called_once() # it should be called once
        args = pipe.call_args[0]
        assert args[1] is pot_instance, f"expected {pot_instance}, got {args[1]}"

    def test_build_report_called(self, tmp_path):
        """ Verifies that build_report is called """
        inp, patches = _base_patches(tmp_path)
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            qmt_computations_main(inp)
        patches[f"{FUNC_PATH}.build_report"].assert_called_once()
        # it should be called once

    def test_write_output_called_with_report(self, tmp_path):
        """ Verifies that write_output is called with the generated report """
        inp, patches = _base_patches(tmp_path)
        patches[f"{FUNC_PATH}.build_report"] = MagicMock(return_value="my_report")
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            qmt_computations_main(inp)
        wo = patches[f"{FUNC_PATH}.write_output"]
        wo.assert_called_once() # it should be called once
        assert wo.call_args[0][1] == "my_report", f"expected {"my_report"}, "
        f"got {wo.call_args[0][1]}"

    def test_build_report_receives_offset(self, tmp_path):
        """ Verifies that build_report receives the potential energy offset """
        inp, patches = _base_patches(tmp_path)
        pot_instance = MagicMock(offset=0.099)
        patches[f"{FUNC_PATH}.Potential"] = MagicMock(return_value=pot_instance)
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            qmt_computations_main(inp)
        br = patches[f"{FUNC_PATH}.build_report"]
        assert br.call_args[0][-1] == 0.099, f"expected {0.099}, got {br.call_args[0][-1]}"

    def test_returns_none(self, tmp_path):
        """ Verifies that qmt_computations_main returns None """
        inp, patches = _base_patches(tmp_path)
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            result = qmt_computations_main(inp)
        assert result is None, f"expected {None}, got {result}"

    def test_pipeline_called_with_params_as_first_argument(self, tmp_path):
        """
        Verifies that the QMT computation pipeline receives
        the parsed input parameters as its first argument
        """
        inp, patches = _base_patches(tmp_path)
        params = _make_params()
        irc = _make_irc()
        patches[f"{FUNC_PATH}.read_input"] = MagicMock(return_value=(params, irc))
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            qmt_computations_main(inp)
        pipe = patches[f"{FUNC_PATH}.qmt_computations_pipeline"]
        args = pipe.call_args[0]
        assert args[0] is params, f"expected {params}, got {args[0]}"

    def test_build_report_receives_exact_pipeline_outputs(self, tmp_path):
        """
        Verifies that build_report is forwarded the exact objects
        returned by the pipeline, in the right positions
        """
        inp, patches = _base_patches(tmp_path)
        interpolated_irc = MagicMock(name="interpolated_irc")
        level_results = MagicMock(name="level_results")
        res_temper_aver = MagicMock(name="res_temper_aver")
        arrhenius_rate_values = MagicMock(name="arrhenius_rate_values")
        patches[f"{FUNC_PATH}.qmt_computations_pipeline"] = MagicMock(
            return_value=(interpolated_irc, level_results, res_temper_aver, arrhenius_rate_values))
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            qmt_computations_main(inp)
        br = patches[f"{FUNC_PATH}.build_report"]
        args = br.call_args[0]
        assert args[0] is level_results, f"expected {level_results}, got {args[0]}; test #1"
        assert args[1] is res_temper_aver, f"expected {res_temper_aver}, got {args[1]}; test #2"
        assert args[3] is arrhenius_rate_values, f"expected {arrhenius_rate_values}, got {args[3]}; test #3"
        assert args[4] is interpolated_irc, f"expected {interpolated_irc}, got {args[4]}; test #4"

    def test_build_report_receives_temperature_from_params(self, tmp_path):
        """ Verifies that build_report receives the temperature taken from the parsed input parameters """
        inp, patches = _base_patches(tmp_path)
        params = _make_params()
        params.T = 425.5
        irc = _make_irc()
        patches[f"{FUNC_PATH}.read_input"] = MagicMock(return_value=(params, irc))
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            qmt_computations_main(inp)
        br = patches[f"{FUNC_PATH}.build_report"]
        assert br.call_args[0][2] == 425.5, f"expected {425.5}, got {br.call_args[0][2]}"

    def test_logger_info_called_with_output_path(self, tmp_path):
        """
        Verifies that a success message is logged with
        the resolved output path after writing the report
        """
        inp, patches = _base_patches(tmp_path)
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}), \
            patch(f"{FUNC_PATH}.logger") as logger_mock:
            qmt_computations_main(inp)
        logger_mock.info.assert_called_once()
        logged_path = logger_mock.info.call_args[0][1]
        expected = inp.with_name("mol_forward_tunnex_2.out")
        assert logged_path == expected, f"expected {expected}, got {logged_path}"

    def test_output_filename_only_strips_the_last_suffix(self, tmp_path):
        """
        Verifies that an input filename with mutliple dots
        only has its final suffix replaced, and not the whole stem
        """
        multi_dot_inp = tmp_path / "mol.v2.forward.txt"
        multi_dot_inp.write_text("dummy", encoding="utf-8")
        _, patches = _base_patches(tmp_path)
        captured = {}
        def capture_write(path, report):
            captured["path"] = path
        patches[f"{FUNC_PATH}.write_output"] = MagicMock(side_effect=capture_write)
        with patch.multiple(FUNC_PATH, **{k.split(".")[-1]: v
            for k, v in patches.items()}):
            qmt_computations_main(multi_dot_inp)
        expected = "mol.v2.forward_tunnex_2.out"
        result = captured["path"].name
        assert result == expected, f"expected {expected}, got {result}"

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """ Verifies that a missing input file raises an error. Expecting a FileNotFoundError """
        with pytest.raises(FileNotFoundError, match="File not found"):
            qmt_computations_main(tmp_path / "nonexistent.txt")