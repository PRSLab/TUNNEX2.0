# --- Test: test_main. Unit tests for main.py
# Run with: pytest test_main.py -v ---

# --- Modules ---
from unittest.mock import MagicMock, patch, call

from pathlib import Path

from tunnex_2.constants_and_dataclasses.dataclasses import FindIRCConfig  # type: ignore


# --- Module to test ---
from tunnex_2.main import main # type: ignore


# --- Helpers and shared data ---
FWD = Path("tunnex_2_forward.txt")
BWD = Path("tunnex_2_backward.txt")
TS_IN = Path("mol.gjf")
TUN_IN = Path("tunnex_2_input.txt")
FUNC_PATH = "tunnex_2.main"


def _cfg(**kwargs) -> FindIRCConfig:
    """ Should return FindIRCConfig for Tunnex 2.0 """
    defaults = dict(
        qmt_module = True,
        ts_input_file = TS_IN,
        tunnex_input = None)
    defaults.update(kwargs)
    return FindIRCConfig(**defaults)


# ===========================================================================
# --- main - branch 1 (tunnex_input is None, run IRC pipeline) ---
# ===========================================================================
class TestMainWithoutTunnexInput:

    def test_irc_pipeline_called_when_tunnex_input_none(self):
        """ Runs the IRC pipeline when no Tunnex input is provided """
        cfg = _cfg(tunnex_input=None)
        with patch(FUNC_PATH+".irc_computations_main",
             return_value=(FWD, BWD)) as mock_pipe, \
             patch(FUNC_PATH+".qmt_computations_main"), \
             patch(FUNC_PATH+".log_run_start"), \
             patch(FUNC_PATH+".setup_logger", return_value=MagicMock()):
             main(cfg)
        mock_pipe.assert_called_once_with(cfg) # it should be called once

    def test_qmt_called_twice_when_qmt_module_true(self):
        """ Calls the QMT computation twice when the QMT module is enabled """
        cfg = _cfg(tunnex_input=None, qmt_module=True)
        with patch(FUNC_PATH+".irc_computations_main",
             return_value=(FWD, BWD)), \
             patch(FUNC_PATH+".qmt_computations_main") as mock_qmt, \
             patch(FUNC_PATH+".log_run_start"), \
             patch(FUNC_PATH+".setup_logger", return_value=MagicMock()):
             main(cfg)
        assert mock_qmt.call_count == 2, f"Expected {2}, got {mock_qmt.call_count}"

    def test_qmt_called_with_forward_and_backward(self):
        """ Calls QMT with the forward and backward IRC data """
        cfg = _cfg(tunnex_input=None, qmt_module=True,
        prob_aver_mode_qmt="finite_sum")
        with patch(FUNC_PATH+".irc_computations_main",
             return_value=(FWD, BWD)), \
             patch(FUNC_PATH+".qmt_computations_main") as mock_qmt, \
             patch(FUNC_PATH+".log_run_start"), \
             patch(FUNC_PATH+".setup_logger", return_value=MagicMock()):
             main(cfg)
        calls = mock_qmt.call_args_list
        expected = [call(FWD), call(BWD)]
        for i in range(len(calls)):
            assert calls[i] == expected[i], f"Expected {expected[i]}, got {calls[i]}; test #{i+1}"

    def test_qmt_not_called_when_module_false(self):
        """ Does not call QMT when the QMT module is disabled """
        cfg = _cfg(tunnex_input=None, qmt_module=False)
        with patch(FUNC_PATH+".irc_computations_main",
             return_value=(FWD, BWD)), \
             patch(FUNC_PATH+".qmt_computations_main") as mock_qmt, \
             patch(FUNC_PATH+".log_run_start"), \
             patch(FUNC_PATH+".setup_logger", return_value=MagicMock()):
             result = main(cfg)
        assert result is None, f"Expected {None}, got {result}"
        mock_qmt.assert_not_called() # it should not be called

    def test_log_run_start_is_called(self):
        """ Logs the run is called """
        cfg = _cfg(tunnex_input=None)
        with patch(FUNC_PATH+".irc_computations_main",
             return_value=(FWD, BWD)), \
             patch(FUNC_PATH+".qmt_computations_main"), \
             patch(FUNC_PATH+".log_run_start") as mock_log, \
             patch(FUNC_PATH+".setup_logger", return_value=MagicMock()):
             main(cfg)
        mock_log.assert_called_once() # it should be called once

    def test_log_run_start_called_with_ts_input(self):
        """ Logs the run using the transition state input file """
        cfg = _cfg(tunnex_input=None)
        with patch(FUNC_PATH+".irc_computations_main",
             return_value=(FWD, BWD)), \
             patch(FUNC_PATH+".qmt_computations_main"), \
             patch(FUNC_PATH+".log_run_start") as mock_log, \
             patch(FUNC_PATH+".setup_logger", return_value=MagicMock()):
             main(cfg)
        assert mock_log.call_args[0][1] == TS_IN, f"Expected {TS_IN}, got {mock_log.call_args[0][1]}"


# ===========================================================================
# --- main - branch 2 (tunnex_input is not None, run QMT only) ---
# ===========================================================================
class TestMainWithTunnexInput:

    def test_pipeline_not_called_when_tunnex_input_given(self):
        """ Skips the IRC pipeline when Tunnex input is provided """
        cfg = _cfg(tunnex_input=TUN_IN)
        with patch(FUNC_PATH+".irc_computations_main") as mock_pipe, \
             patch(FUNC_PATH+".qmt_computations_main"), \
             patch(FUNC_PATH+".log_run_start"), \
             patch(FUNC_PATH+".setup_logger", return_value=MagicMock()):
             main(cfg)
        mock_pipe.assert_not_called() # it should not be called

    def test_qmt_called_once_with_tunnex_input(self):
        """ Calls QMT once with the provided Tunnex input """
        cfg = _cfg(tunnex_input=TUN_IN, prob_aver_mode_qmt="finite_sum")
        with patch(FUNC_PATH+".irc_computations_main"), \
             patch(FUNC_PATH+".qmt_computations_main") as mock_qmt, \
             patch(FUNC_PATH+".log_run_start"), \
             patch(FUNC_PATH+".setup_logger", return_value=MagicMock()):
             result = main(cfg)
        assert result is None, f"Expected {None}, got {result}"
        mock_qmt.assert_called_once_with(TUN_IN)
        # it should be called once

    def test_log_run_start_called_with_tunnex_input(self):
        """
        Logs the run using the provided Tunnex input and
        calls QMT regardless of the QMT module flag when Tunnex input is provided
        """
        cfg = _cfg(tunnex_input=TUN_IN)
        with patch(FUNC_PATH+".irc_computations_main"), \
             patch(FUNC_PATH+".qmt_computations_main") as mock_qmt, \
             patch(FUNC_PATH+".log_run_start") as mock_log, \
             patch(FUNC_PATH+".setup_logger", return_value=MagicMock()):
             main(cfg)
        assert mock_log.call_args[0][1] == TUN_IN, f"Expected {TUN_IN}, got {mock_log.call_args[0][1]}"
        mock_log.assert_called_once() # it should be called once
        mock_qmt.assert_called_once() # it should be called once


# ===========================================================================
# --- main - runpy (runs main) ---
# ===========================================================================
class TestMainRun:

    def test_main_entrypoint_2(self):
        """ Calls the main function """
        import runpy
        runpy.run_path("tunnex_2/__main__.py", run_name="__main__")
        # it should be called
