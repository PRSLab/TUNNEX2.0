# --- Test: test_interface. Unit tests for interface.py
# Run with: pytest test_interface.py -v ---

# --- Modules ---
import pytest # type: ignore
from unittest.mock import MagicMock, patch

import re
import logging
import subprocess
from pathlib import Path


# --- Module to test ---
from tunnex_2.irc_computations.ancillary_functions.interface import ( # type: ignore
    MaxLevelFilter,
    setup_logger,
    log_run_start,
    file_check,
    _insert_command,
    _command_preparation,
    _submit_slurm_job,
    _wait_for_slurm_job,
    _check_slurm_job,
    run_software,
    gauss_out_filename,
    orca_out_filename,
    gauss_error_check,
    _orca_normal_term_check,
    orca_error_check)


# --- Helpers and shared data ---
DUMMY_COMP = "tunnex_2.irc_computations.ancillary_functions.interface.subprocess.run"
DUMMY_INT = "tunnex_2.irc_computations.ancillary_functions.interface"
ENV = {"PATH": "/usr/bin"}


# ===========================================================================
# --- MaxLevelFilter - normal behaviour and input validation ---
# ===========================================================================
class TestMaxLevelFilter:

    def test_passes_record_at_max_level(self):
        """ Should pass a record at the maximum level """
        f = MaxLevelFilter(logging.INFO)
        record = logging.LogRecord("", logging.INFO, "", 0, "", (), None)
        assert f.filter(record) is True, f"expecting {True}, got {f.filter(record)}"

    def test_passes_record_below_max_level(self):
        """ Should pass a record below the maximum level """
        f = MaxLevelFilter(logging.INFO)
        record = logging.LogRecord("", logging.DEBUG, "", 0, "", (), None)
        assert f.filter(record) is True, f"expecting {True}, got {f.filter(record)}"

    def test_blocks_record_above_max_level(self):
        """ Should block a record above the maximum level """
        f = MaxLevelFilter(logging.INFO)
        record = logging.LogRecord("", logging.WARNING, "", 0, "", (), None)
        assert f.filter(record) is False, f"expecting {False}, got {f.filter(record)}"

    def test_blocks_error_level(self):
        """ Should block an error-level record """
        f = MaxLevelFilter(logging.INFO)
        record = logging.LogRecord("", logging.ERROR, "", 0, "", (), None)
        assert f.filter(record) is False, f"expecting {False}, got {f.filter(record)}"


# ===========================================================================
# --- setup_logger - normal behaviour and input validation ---
# ===========================================================================
class TestSetupLogger:

    def test_returns_logger_instance(self, tmp_path):
        """ Should return a Logger instance """
        log_file = tmp_path / "test.log"
        logger = setup_logger(str(log_file))
        assert isinstance(logger, logging.Logger), f"expected {logging.Logger}, got {type(logger)}"

    def test_logger_has_two_handlers(self, tmp_path):
        """ Should create two handlers """
        log_file = tmp_path / "test.log"
        logger = setup_logger(str(log_file))
        set_len = 2
        assert len(logger.handlers) == set_len, f"expected len_handlers={set_len}, got {len(logger.handlers)}"

    def test_logger_level_is_debug(self, tmp_path):
        """ Should set the logger level to DEBUG """
        log_file = tmp_path / "test.log"
        logger = setup_logger(str(log_file))
        assert logger.level == logging.DEBUG, f"expected {logging.DEBUG}, got {logger.level}"

    def test_calling_twice_does_not_duplicate_handlers(self, tmp_path):
        """ Should not duplicate handlers when called twice """
        log_file = tmp_path / "test.log"
        setup_logger(str(log_file))
        logger = setup_logger(str(log_file))
        set_len = 2
        assert len(logger.handlers) == set_len, f"expected len_handlers={set_len}, got {len(logger.handlers)}"

    def test_writes_to_log_file(self, tmp_path):
        """ Should write messages to the log file """
        log_file = tmp_path / "test.log"
        logger = setup_logger(str(log_file))
        line = "hello from test"
        logger.debug(line)
        for h in logger.handlers: # it flushes handlers
            h.flush()
        assert line in log_file.read_text(encoding="utf-8"), \
        f"expected {line} in {log_file}"

    def test_propagate_is_false(self, tmp_path):
        """ Should disable log propagation """
        log_file = tmp_path / "test.log"
        logger = setup_logger(str(log_file))
        assert logger.propagate is False, f"expected {False}, got {logger.propagate}"


# ===========================================================================
# --- log_run_start - normal behaviour and input validation ---
# ===========================================================================
class TestLogRunStart:

    def test_logs_calculation_path(self, tmp_path):
        """ Should log the calculation file path """
        log_file = tmp_path / "test.log"
        logger = setup_logger(str(log_file))
        file = "mol.gjf"
        log_run_start(logger, Path(file))
        for h in logger.handlers: # it flushes handlers
            h.flush()
        content = log_file.read_text(encoding="utf-8")
        assert file in content, f"expected {file} in {content}"

    def test_logs_separator_lines(self, tmp_path):
        """ Should log separator lines """
        log_file = tmp_path / "test.log"
        logger = setup_logger(str(log_file))
        log_run_start(logger, Path("mol.gjf"))
        for h in logger.handlers: # it flushes handlers
            h.flush()
        content = log_file.read_text(encoding="utf-8")
        sample = "=" * 10
        assert sample in content, f"expected {sample} in {content}"


# ===========================================================================
# --- file_check - normal behaviour and input validation ---
# ===========================================================================
class TestFileCheck:

    def test_existing_file_returns_path(self, tmp_path):
        """ Should return a Path object for Path input """
        file = tmp_path / "test.log" # type(file) is Path
        file.write_text("content") # file is deleted automatically after the test
        result = file_check(file)
        assert result == file, f"expected {file}, got {result}; test #1"
        assert isinstance(result, Path), f"expected {Path}, got {type(result)}; test #2"

    def test_string_input_accepted(self, tmp_path):
        """ Should return a Path object for str input """
        file = tmp_path / "test.log"
        file.write_text("something")
        result = file_check(str(file))
        assert isinstance(result, Path), f"expected {Path}, got {type(result)}"

    def test_missing_file_raises(self, tmp_path):
        """ Should check the file existence. Expecting a FileNotFoundError """
        with pytest.raises(FileNotFoundError, match="was not found"):
            file_check(tmp_path / "nonexistent.log")

    def test_directory_raises(self, tmp_path):
        """ A directory is not a file. Expecting a ValueError """
        with pytest.raises(ValueError, match="Expected a file"):
            file_check(tmp_path)


# ===========================================================================
# --- _insert_command - normal behaviour and input validation ---
# ===========================================================================
class TestInsertCommand:

    def test_reads_existing_non_empty_file(self, tmp_path):
        """ The function should read the existing non-empty command file """
        cmd_file = tmp_path / "command_settings.txt"
        test_line = "g16"
        cmd_file.write_text(test_line, encoding="utf-8")
        result = _insert_command(cmd_file)
        assert result == test_line, f"expected {test_line}, got {result}"

    def test_strips_whitespace_from_file(self, tmp_path):
        """ The function should remove the whitespace in the existing non-empty command file """
        cmd_file = tmp_path / "command_settings.txt"
        test_line = "  g16  \n"
        cmd_file.write_text(test_line, encoding="utf-8")
        result = _insert_command(cmd_file)
        assert result == test_line.strip(), f"expected {test_line.strip()}, got {result}"

    def test_missing_file_prompts_and_writes(self, tmp_path):
        """ The function should write the command line from a terminal if the file is missing """
        cmd_file = tmp_path / "command_settings.txt"
        test_line = "orca"
        with patch("builtins.input", return_value=test_line):
            result = _insert_command(cmd_file)
        assert result == test_line, f"expected {test_line}, got {result}; test #1"
        file_read = cmd_file.read_text(encoding="utf-8")
        assert file_read == test_line, f"expected {test_line}, got {file_read}; test #2"

    def test_empty_file_prompts_and_writes(self, tmp_path):
        """ The function should write the command line from a terminal if the file is empty """
        cmd_file = tmp_path / "command_settings.txt"
        cmd_file.write_text("", encoding="utf-8")
        test_line = "g16"
        with patch("builtins.input", return_value=test_line):
            result = _insert_command(cmd_file)
        file_read = cmd_file.read_text(encoding="utf-8")
        assert result == test_line, f"expected {test_line}, got {result}; test #1"
        assert file_read == test_line, f"expected {test_line}, got {file_read}; test #2"

    def test_input_whitespace_is_stripped_and_written(self, tmp_path):
        """ The function should strip whitespace from terminal input """
        cmd_file = tmp_path / "command_settings.txt"
        test_line = "  g16  \n"
        with patch("builtins.input", return_value=test_line):
            result_1 = _insert_command(cmd_file)
        assert result_1 == test_line.strip(), f"expected {test_line.strip()}, got {result_1}; test #1"
        result_2 = cmd_file.read_text(encoding="utf-8")
        assert result_2 == test_line.strip(), f"expected {test_line.strip()}, got {result_2}; test #2"

    def test_directory_path_prompts_and_writes_file(self, tmp_path):
        """ The function should prompt and write when the command path is not a file """
        cmd_file = tmp_path / "command_settings.txt"
        cmd_file.mkdir()
        test_line = "orca"
        with patch("builtins.input", return_value=test_line):
            result_1 = _insert_command(cmd_file)
        assert result_1 == test_line, f"expected {test_line}, got {result_1}; test #1"
        result_2 = (cmd_file / "command_settings.txt").read_text(encoding="utf-8")
        assert result_2 == test_line, f"expected {test_line}, got {result_2}; test #2"


# ===========================================================================
# --- _command_preparation - normal behaviour and input validation ---
# ===========================================================================
class TestCommandPreparation:

    @patch(DUMMY_INT+"._insert_command", return_value="g16")
    @patch(DUMMY_INT+".file_check", return_value=Path("mol.gjf"))
    def test_single_word_command(self, fc, ic):
        """ Should prepare a single-word command """
        cmd_file = Path("command_settings.txt")
        expected = ["g16", "mol.gjf"]
        result = _command_preparation("mol.gjf", cmd_file)
        assert result == expected, f"expected {expected}, got {result}"

    @patch(DUMMY_INT+"._insert_command", return_value="mpirun -np 4 g16")
    @patch(DUMMY_INT+".file_check", return_value=Path("mol.gjf"))
    def test_multi_word_command_split_by_shlex(self, fc, ic):
        """ Should split a multi-word command with shlex """
        cmd_file = Path("command_settings.txt")
        result = _command_preparation("mol.gjf", cmd_file)
        expected = ["mpirun", "-np", "4", "g16", "mol.gjf"]
        assert result == expected, f"expected {expected}, got {result}"

    @patch(DUMMY_INT+"._insert_command", return_value="orca")
    @patch(DUMMY_INT+".file_check", side_effect=[Path("mol.inp"), Path("mol.out")])
    def test_output_file_appended_when_provided(self, fc, ic):
        """ Should append the output file when provided """
        cmd_file = Path("command_settings.txt")
        result = _command_preparation("mol.inp", cmd_file, "mol.out")
        expected_1, expected_2 = "mol.out", "mol.inp"
        assert result[-1] == "mol.out", f"expected {expected_1}, got {result[-1]}; test #1"
        assert result[-2] == "mol.inp", f"expected {expected_2}, got {result[-2]}; test #2"

    @patch(DUMMY_INT+"._insert_command", return_value="g16")
    @patch(DUMMY_INT+".file_check", return_value=Path("mol.gjf"))
    def test_no_output_file_when_none(self, fc, ic):
        """ Should omit the output file when not provided """
        cmd_file = Path("command_settings.txt")
        result = _command_preparation("mol.gjf", cmd_file, None)
        expected = 2
        assert len(result) == expected, f"expected {expected}, got {len(result)}" # command + input only

    @patch(DUMMY_INT+"._insert_command", return_value="g16")
    @patch(DUMMY_INT+".file_check", side_effect=FileNotFoundError("not found"))
    def test_missing_input_file_raises(self, fc, ic):
        """ Should check for a missing input file. Expecting a FileNotFoundError """
        cmd_file = Path("command_settings.txt")
        with pytest.raises(FileNotFoundError): # A file is missing
            _command_preparation("missing.gjf", cmd_file)


# ===========================================================================
# --- _submit_slurm_job - normal behaviour and input validation ---
# ===========================================================================
class TestSubmitSlurmJob:

    def _make_proc(self, stdout: str):
        """ Imitating the subprocess.run() """
        proc = MagicMock()
        proc.stdout = stdout
        return proc

    @patch(DUMMY_INT+".subprocess.run")
    def test_returns_job_id(self, mock_run):
        """ Should return the submitted SLURM job ID """
        mock_run.return_value = self._make_proc("Submitted batch job 12345\n")
        job_id = _submit_slurm_job("mol.gjf", ["sbatch", "mol.gjf"], ENV)
        expected = "12345"
        assert job_id == expected, f"expected {expected}, got {job_id}"

    @patch(DUMMY_INT+".subprocess.run")
    def test_logs_job_id(self, mock_run):
        """ Should log the submitted SLURM job ID """
        expected = "99"
        mock_run.return_value = self._make_proc(f"Submitted batch job {expected}\n")
        with patch(DUMMY_INT+".logger") as mock_logger:
            _submit_slurm_job("mol.gjf", ["sbatch", "mol.gjf"], ENV)
        mock_logger.info.assert_called_once()
        result = str(mock_logger.info.call_args)
        assert expected in result, f"expected {expected} in {result}"

    @patch(DUMMY_INT+".subprocess.run")
    def test_passes_env_to_subprocess(self, mock_run):
        """ Should pass the environment to subprocess """
        mock_run.return_value = self._make_proc("Submitted batch job 1\n")
        custom_env = {"MY_VAR": "value"}
        _submit_slurm_job("mol.gjf", ["sbatch"], custom_env)
        result = mock_run.call_args[1]["env"]
        assert mock_run.call_args[1]["env"] == custom_env, f"expected {custom_env}, got {result}"

    @patch(DUMMY_INT+".subprocess.run", side_effect=subprocess.CalledProcessError(1, "sbatch", stderr="error"))
    def test_subprocess_error_raises_runtime(self, mock_run):
        """ Should convert subprocess errors to the function errors. Expecting a RuntimeError """
        with pytest.raises(RuntimeError, match="Failed to submit"):
            _submit_slurm_job("mol.gjf", ["sbatch", "mol.gjf"], ENV)

    @patch(DUMMY_INT+".subprocess.run")
    def test_unexpected_sbatch_output_raises(self, mock_run):
        """ Should raise an error for unexpected sbatch output. Expecting a RuntimeError """
        mock_run.return_value = self._make_proc("Something unexpected\n")
        with pytest.raises(RuntimeError, match="Unexpected sbatch output"):
            _submit_slurm_job("mol.gjf", ["sbatch", "mol.gjf"], ENV)

    @patch(DUMMY_INT+".subprocess.run")
    def test_empty_sbatch_output_raises(self, mock_run):
        """ Should raise an error for empty sbatch output. Expecting a RuntimeError """
        mock_run.return_value = self._make_proc("")
        with pytest.raises(RuntimeError):
            _submit_slurm_job("mol.gjf", ["sbatch", "mol.gjf"], ENV)


# ===========================================================================
# --- _wait_for_slurm_job - normal behaviour and input validation ---
# ===========================================================================
class TestWaitForSlurmJob:

    def _make_proc(self, stdout: str):
        """ Imitating the subprocess.run() """
        proc = MagicMock()
        proc.stdout = stdout
        return proc

    @patch(DUMMY_INT+".time.sleep")
    @patch(DUMMY_INT+".subprocess.run")
    def test_exits_when_queue_empty(self, mock_run, mock_sleep):
        """ Should exit when the SLURM queue is empty """
        mock_run.return_value = self._make_proc("")
        _wait_for_slurm_job("42", ENV, time_wait_seconds=0)
        mock_sleep.assert_not_called() # exit when the SLURM queue is empty

    @patch(DUMMY_INT+".time.sleep")
    @patch(DUMMY_INT+".subprocess.run")
    def test_loops_until_queue_empty(self, mock_run, mock_sleep):
        """ Should wait until the SLURM job leaves the queue """
        mock_run.side_effect = [self._make_proc("42 RUNNING"), self._make_proc("42 RUNNING"), self._make_proc("")]
        _wait_for_slurm_job("42", ENV, time_wait_seconds=0)
        expected = 2
        assert mock_sleep.call_count == expected, f"expected {expected}, got {mock_sleep.call_count}"

    @patch(DUMMY_INT+".time.sleep")
    @patch(DUMMY_INT+".subprocess.run")
    def test_logs_at_correct_interval(self, mock_run, mock_sleep):
        """ Should log at the specified interval """
        mock_run.side_effect = [self._make_proc("running")] * 4 + [self._make_proc("")]
        with patch(DUMMY_INT+".logger") as mock_logger:
            _wait_for_slurm_job("42", ENV, time_wait_seconds=0, log_interval=3)
        expected = 2 # Iterations 0 and 3 trigger logging
        assert mock_logger.info.call_count == expected, f"expected {expected}, got {mock_logger.info.call_count}"

    @patch(DUMMY_INT+".time.sleep")
    @patch(DUMMY_INT+".subprocess.run")
    def test_passes_env_to_squeue(self, mock_run, mock_sleep):
        """ Should pass the environment to squeue """
        mock_run.return_value = self._make_proc("")
        custom_env = {"MY_KEY": "val"}
        _wait_for_slurm_job("42", custom_env, time_wait_seconds=0)
        result = mock_run.call_args[1]["env"]
        assert result == custom_env, f"expected {custom_env}, got {result}"

    @patch(DUMMY_INT+".time.sleep")
    @patch(DUMMY_INT+".subprocess.run")
    def test_returns_none(self, mock_run, mock_sleep):
        """ Should return None when the job finishes """
        mock_run.return_value = self._make_proc("")
        result = _wait_for_slurm_job("42", ENV, time_wait_seconds=0)
        assert result is None, f"expected {None}, got {result}"

    @patch(DUMMY_INT+".time.sleep")
    @patch(DUMMY_INT+".subprocess.run", side_effect=subprocess.CalledProcessError(1, "squeue", stderr="err"))
    def test_squeue_error_raises_runtime(self, mock_run, mock_sleep):
        """ Should convert squeue errors to an erroe. Expecting a RuntimeError """
        with pytest.raises(RuntimeError, match="Failed squeue"):
            _wait_for_slurm_job("42", ENV, time_wait_seconds=0)


# ===========================================================================
# _check_slurm_job - normal behaviour and input validation ---
# ===========================================================================
class TestCheckSlurmJob:

    def _make_proc(self, stdout: str):
        """ Imitating the subprocess.run() """
        proc = MagicMock()
        proc.stdout = stdout
        return proc

    @patch(DUMMY_INT+".subprocess.run")
    def test_completed_job_passes(self, mock_run):
        """ Should accept a successfully completed job """
        mock_run.return_value = self._make_proc("COMPLETED|0:0\n")
        _check_slurm_job("42", ENV) # should not raise

    @patch(DUMMY_INT+".subprocess.run")
    def test_logs_success(self, mock_run):
        """ Should log successful job completion """
        mock_run.return_value = self._make_proc("COMPLETED|0:0\n")
        expected = "42"
        with patch(DUMMY_INT+".logger") as mock_logger:
            _check_slurm_job(expected, ENV)
        mock_logger.info.assert_called_once()
        result = str(mock_logger.info.call_args)
        assert expected in result, f"expected {expected} in {result}"

    @patch(DUMMY_INT+".subprocess.run")
    def test_failed_state_raises(self, mock_run):
        """ Should raise an error for a failed job state. Expecting a RuntimeError """
        mock_run.return_value = self._make_proc("FAILED|1:0\n")
        with pytest.raises(RuntimeError, match="failed"):
            _check_slurm_job("42", ENV)

    @patch(DUMMY_INT+".subprocess.run")
    def test_non_zero_exit_code_raises(self, mock_run):
        """ Should raise an error for a non-zero exit code. Expecting a RuntimeError """
        mock_run.return_value = self._make_proc("COMPLETED|1:0\n")
        with pytest.raises(RuntimeError, match="failed"):
            _check_slurm_job("42", ENV)

    @patch(DUMMY_INT+".subprocess.run")
    def test_cancelled_state_raises(self, mock_run):
        """ Should raise an error for a cancelled job. Expecting a RuntimeError """
        mock_run.return_value = self._make_proc("CANCELLED|0:0\n")
        with pytest.raises(RuntimeError, match="failed"):
            _check_slurm_job("42", ENV)

    @patch(DUMMY_INT+".subprocess.run")
    def test_empty_sacct_output_raises(self, mock_run):
        """ Should raise an error when sacct returns no status. Expecting a RuntimeError """
        mock_run.return_value = self._make_proc("")
        with pytest.raises(RuntimeError, match="no status"):
            _check_slurm_job("42", ENV)

    @patch(DUMMY_INT+".subprocess.run")
    def test_malformed_sacct_line_raises(self, mock_run):
        """ Should raise an error for malformed sacct output. Expecting a RuntimeError """
        mock_run.return_value = self._make_proc("COMPLETED\n")
        with pytest.raises(RuntimeError, match="Unexpected sacct"):
            _check_slurm_job("42", ENV)

    @patch(DUMMY_INT+".subprocess.run",
           side_effect=subprocess.CalledProcessError(1, "sacct", stderr="err"))
    def test_sacct_subprocess_error_raises(self, mock_run):
        """ Should convert sacct errors to the function errors. Expecting a RuntimeError """
        with pytest.raises(RuntimeError, match="sacct failed"):
            _check_slurm_job("42", ENV)


# ===========================================================================
# --- run_software - normal behaviour and input validation ---
# ===========================================================================
class TestRunSoftware:

    def test_direct_run_with_correct_args(self, tmp_path):
        """ Runs the function with the correct arguments without the launching of an external program """
        input_file = tmp_path / "mol.gjf"
        input_file.write_text("input")
        command_file = tmp_path / "command.txt"
        command_file.write_text("g16")
        with patch(DUMMY_COMP) as mock_run: # Running the dummy computations
            run_software(input_file, command_file, envr=ENV)
        mock_run.assert_called_once_with(["g16", str(input_file)], check=True, env=ENV)

    def test_direct_run_with_output(self, tmp_path):
        """ Runs the function with the correct arguments (and an output file) without the launching of an external program """
        input_file = tmp_path / "mol.gjf"
        input_file.write_text("input")
        output_file = tmp_path / "mol.out"
        output_file.write_text("output")
        command_file = tmp_path / "command.txt"
        command_file.write_text("g16")
        with patch(DUMMY_COMP) as mock_run: # Running the dummy computations
            run_software(input_file, command_file, filename_out_key=True, envr=ENV)
        mock_run.assert_called_once_with(["g16", str(input_file), str(output_file)], check=True, env=ENV)

    def test_multi_word_command_split_correctly(self, tmp_path):
        """ Runs the function with the multiword command without the launching of an external program """
        input_file = tmp_path / "mol.gjf"
        input_file.write_text("%mem=1GB")
        command_file = tmp_path / "cmd.txt"
        command_file.write_text("g16 -n 4")
        with patch(DUMMY_COMP) as mock_run: # Running the dummy computations
            run_software(input_file, command_file, envr=ENV)
        mock_run.assert_called_once_with(["g16", "-n", "4", str(input_file)], check=True, env=ENV)

    def test_missing_input_file_raises(self, tmp_path):
        """ Should check the file existence. Expecting a FileNotFoundError """
        cmd_file = tmp_path / "cmd.txt"
        cmd_file.write_text("g16")
        with pytest.raises(FileNotFoundError, match="was not found"):
            run_software(tmp_path / "nonexistent.gjf", cmd_file)

    def test_subprocess_error_propagates(self, tmp_path):
        """ Subprocess errors should propagate unchanged """
        input_file = tmp_path / "mol.gjf"
        input_file.write_text("data")
        cmd_file = tmp_path / "cmd.txt"
        cmd_file.write_text("g16")
        with patch(DUMMY_COMP, side_effect=subprocess.CalledProcessError(1, "g16")): # Running the dummy computations
            with pytest.raises(subprocess.CalledProcessError):
                run_software(input_file, cmd_file)

    def test_slurm_run(self, tmp_path):
        """ Runs the function with the correct arguments in a SLURM mode without the launching of an external program """
        input_file = tmp_path / "mol.gjf"
        input_file.write_text("input")
        command_file = tmp_path / "command.txt"
        command_file.write_text("sbatch job.sh")
        with patch(DUMMY_INT+"._submit_slurm_job", return_value="123456") as mock_submit, patch(
        DUMMY_INT+"._wait_for_slurm_job") as mock_wait, patch(
        DUMMY_INT+"._check_slurm_job") as mock_check: # Running the dummy computations
            run_software(input_file, command_file, slurm_key=True, envr=ENV, time_wait_seconds=5, log_interval=2)
        mock_submit.assert_called_once_with(input_file, ["sbatch", "job.sh", str(input_file)], ENV)
        mock_wait.assert_called_once_with("123456", ENV, 5, 2)
        mock_check.assert_called_once_with("123456", ENV)

    def test_default_environment(self, tmp_path):
        """ Runs the function with the default environment without the launching of an external program """
        input_file = tmp_path / "mol.gjf"
        input_file.write_text("input")
        command_file = tmp_path / "command.txt"
        command_file.write_text("g16")
        with patch(DUMMY_INT+".os.environ.copy", return_value={"PATH": "test"}), \
            patch(DUMMY_COMP) as mock_run:
            run_software(input_file, command_file)
        mock_run.assert_called_once_with(["g16", str(input_file)], check=True, env={"PATH": "test"})

    @patch(DUMMY_INT+".subprocess.run")
    @patch(DUMMY_INT+"._command_preparation", return_value=["g16", "mol.gjf"])
    def test_direct_submission_calls_subprocess(self, rsp, mock_run):
        """ Should submit the command directly """
        cmd_file = Path("command_settings.txt")
        run_software("mol.gjf", cmd_file, slurm_key=False) # Expected a direct submission of a command
        mock_run.assert_called_once_with(["g16", "mol.gjf"], check=True, env=mock_run.call_args[1]["env"])

    @patch(DUMMY_INT+".subprocess.run")
    @patch(DUMMY_INT+"._command_preparation", return_value=["g16", "mol.gjf"])
    def test_direct_submission_returns_none(self, rsp, mock_run):
        """ Should return None after direct submission """
        cmd_file = Path("command_settings.txt")
        result = run_software("mol.gjf", cmd_file, slurm_key=False)
        assert result is None, f"expected {None}, got {result}"

    @patch(DUMMY_INT+"._check_slurm_job")
    @patch(DUMMY_INT+"._wait_for_slurm_job")
    @patch(DUMMY_INT+"._submit_slurm_job", return_value="99")
    @patch(DUMMY_INT+"._command_preparation", return_value=["sbatch", "mol.gjf"])
    def test_slurm_calls_all_three_helpers(self, rsp, ssj, wsj, csj):
        """ Should call all SLURM helper functions """
        cmd_file = Path("command_settings.txt")
        run_software("mol.gjf", cmd_file, slurm_key=True, envr=ENV)
        ssj.assert_called_once()
        wsj.assert_called_once_with("99", ENV, 10, 6)
        csj.assert_called_once_with("99", ENV) # Expected a call of all SLURM helpers

    @patch(DUMMY_INT+"._check_slurm_job")
    @patch(DUMMY_INT+"._wait_for_slurm_job")
    @patch(DUMMY_INT+"._submit_slurm_job", return_value="77")
    @patch(DUMMY_INT+"._command_preparation", return_value=["sbatch", "mol.gjf"])
    def test_slurm_passes_custom_wait_and_interval(self, rsp, ssj, wsj, csj):
        """ Should pass custom wait time and log interval """
        cmd_file = Path("command_settings.txt")
        run_software("mol.gjf", cmd_file, slurm_key=True, envr=ENV, time_wait_seconds=30, log_interval=2)
        wsj.assert_called_once_with("77", ENV, 30, 2) # Expected a passing of the custom wait time and log interval

    @patch(DUMMY_INT+".subprocess.run")
    @patch(DUMMY_INT+"._command_preparation", return_value=["g16", "mol.gjf"])
    def test_default_env_is_os_environ_copy(self, rsp, mock_run):
        """ Should use a copy of the default environment """
        import os
        cmd_file = Path("command_settings.txt")
        run_software("mol.gjf", cmd_file, slurm_key=False)
        actual_env = mock_run.call_args[1]["env"] # Should be a copy of os.environ, not the same object
        assert actual_env == dict(os.environ), f"expected {dict(os.environ)}, got {actual_env}"

    @patch(DUMMY_INT+".subprocess.run")
    @patch(DUMMY_INT+"._command_preparation", return_value=["g16", "mol.gjf"])
    def test_logs_direct_submission(self, rsp, mock_run):
        """ Should log direct job submission """
        cmd_file = Path("command_settings.txt")
        with patch(DUMMY_INT+".logger") as mock_logger:
            run_software("mol.gjf", cmd_file, slurm_key=False)
        mock_logger.info.assert_called_once() # Expected a logging of a direct job submission


# ===========================================================================
# --- gauss_out_filename - normal behaviour and input validation ---
# --- orca_out_filename - normal behaviour and input validation ---
# ===========================================================================
class TestOutFilenames:

    # gauss_out_filename
    def test_gauss_replaces_suffix_with_out(self):
        """ The function should replace any .abc suffix with .out """
        result = gauss_out_filename("mol.gjf")
        expected = Path("mol.out")
        assert result == expected, f"expected {expected}, got {result}"

    def test_gauss_already_out_unchanged(self):
        """ The function should not change .out suffix """
        result = gauss_out_filename("mol.out")
        expected = Path("mol.out")
        assert result == expected, f"expected {expected}, got {result}"

    def test_gauss_accepts_path_object(self):
        """ The function should accept a Path object """
        result = gauss_out_filename(Path("dir/mol.gjf"))
        expected = Path("dir/mol.out")
        assert result == expected, f"expected {expected}, got {result}"

    # orca_out_filename
    def test_orca_out_suffix(self):
        """ The function should put the pre-defined suffix (.out) """
        suffix = ".out"
        result = orca_out_filename("mol.inp", suffix)
        expected = Path("mol.out")
        assert result == expected, f"expected {expected}, got {result}"

    def test_orca_hess_suffix(self):
        """ The function should put the pre-defined suffix (.hess) """
        suffix = ".hess"
        result = orca_out_filename("mol.inp", suffix)
        expected = Path("mol.hess")
        assert result == expected, f"expected {expected}, got {result}"

    def test_orca_xyz_suffix_adds_irc_stem(self):
        """ The function should add '_IRC_Full_trj' to the stem and put the pre-defined suffix (if it is .xyz) """
        suffix = ".xyz"
        result = orca_out_filename("mol.inp", suffix)
        expected = Path("mol_IRC_Full_trj.xyz")
        assert result == expected, f"expected {expected}, got {result}"

    def test_orca_xyz_suffix_case_insensitive(self):
        """ The function should not be sensitive to the suffix register (if it is .xyz) """
        suffix = ".XYZ"
        result = orca_out_filename("mol.inp", suffix)
        expected = Path("mol_IRC_Full_trj.xyz")
        assert result == expected, f"expected {expected}, got {result}"

    def test_orca_accepts_path_object(self):
        """ The function should accept a Path object """
        result = orca_out_filename(Path("dir/mol.inp"), ".out")
        expected = Path("dir/mol.out")
        assert result == expected, f"expected {expected}, got {result}"


# ===========================================================================
# --- gauss_error_check - normal behaviour and input validation ---
# --- _orca_normal_term_check - normal behaviour and input validation ---
# --- orca_error_check - normal behaviour and input validation ---
# ===========================================================================
class TestErrorCheck:

    # gauss_error_check
    def test_no_error_marker_returns_none(self, tmp_path):
        """ The function should return None normally """
        file = tmp_path / "calc.out"
        file.write_text("Normal termination of Gaussian 16 at Fri Jul 13 10:00:00 2018.")
        result = gauss_error_check(file)
        assert result is None, f"expected {None}, got {result}"

    def test_error_marker_raises_runtime(self, tmp_path):
        """ The function should notice the failed computations. Expecting a RuntimeError """
        file = tmp_path / "calc.out"
        file.write_text("Some output\nError termination via Lnk1e in l101.exe\nmore text")
        # Failed computations
        with pytest.raises(RuntimeError, match="Error termination via") as exc_info_1:
            gauss_error_check(file)
        print("Caught (test #1):", exc_info_1.value)
        # Computations with unclear status
        file = tmp_path / "calc.out"
        file.write_text("Some incomplete Gaussian output")
        with pytest.raises(RuntimeError, match="status could not be determined") as exc_info_2:
            gauss_error_check(file)
        print("Caught (test #2):", exc_info_2.value)

    def test_error_message_contains_filename(self, tmp_path):
        """ The function should print the filename and error message. Expecting a RuntimeError """
        file = tmp_path / "calc.out"
        file.write_text("Error termination via Lnk1e in l101.exe")
        with pytest.raises(RuntimeError, match=re.escape(str(file))):
            gauss_error_check(file)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """ Should check the file existence. Expecting a FileNotFoundError """
        with pytest.raises(FileNotFoundError, match="was not found"):
            gauss_error_check(tmp_path / "missing.out")

    # _orca_normal_term_check
    """ The function should return None normally """
    def test_normal_termination_passes(self, tmp_path):
        file = tmp_path / "calc.out"
        file.write_text("****ORCA TERMINATED NORMALLY****")
        result = _orca_normal_term_check(file)
        assert result is None, f"expected {None}, got {result}"

    def test_abnormal_termination_raises(self, tmp_path):
        """ The function should notice the computations which were not finished normally. Expecting a RuntimeError """
        file = tmp_path / "calc.out"
        file.write_text("Something went wrong.")
        with pytest.raises(RuntimeError, match="error"):
            _orca_normal_term_check(file)

    # orca_error_check
    def test_xyz_delegates_to_out_file(self, tmp_path):
        """ For .xyz input, the function checks the corresponding .out file """
        xyz = tmp_path / "mol_IRC_Full_trj.xyz"
        xyz.write_text("coords")
        out = tmp_path / "mol.out"
        out.write_text("****ORCA TERMINATED NORMALLY****")
        result = orca_error_check(xyz)
        assert result is None, f"expected {None}, got {result}"

    def test_xyz_out_file_missing_raises(self, tmp_path):
        """ For .xyz input, checking .out file if it does not exist. Expecting a FileNotFoundError """
        xyz = tmp_path / "mol_IRC_Full_trj.xyz"
        xyz.write_text("coords")
        with pytest.raises(FileNotFoundError): # corresponding .out file does not exist
            orca_error_check(xyz)

    def test_out_file_normal_termination(self, tmp_path):
        """ For .out input, the function checks the .out file """
        file = tmp_path / "calc.out"
        file.write_text("****ORCA TERMINATED NORMALLY****")
        result = orca_error_check(file)
        assert result is None, f"expected {None}, got {result}"

    def test_out_file_error_termination_raises(self, tmp_path):
        """ For .out input, checking the abnormal termination of the program. Expecting a RuntimeError """
        file = tmp_path / "calc.out"
        file.write_text("Something went wrong.")
        with pytest.raises(RuntimeError): # the computations failed at some point
            orca_error_check(file)

    def test_hess_file_always_passes(self, tmp_path):
        """ For .hess input, the function just return None """
        file = tmp_path / "calc.hess"
        file.write_text("hessian data")
        result = orca_error_check(file)
        assert result is None, f"expected {None}, got {result}"

    def test_unknown_suffix_raises_value_error(self, tmp_path):
        """ The unknown suffix shall not pass! Expecting a ValueError """
        file = tmp_path / "calc.log"
        file.write_text("data")
        with pytest.raises(ValueError, match="undocumented suffix"):
            orca_error_check(file)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """ The function checks if the file exists. Expecting a FileNotFoundError """
        with pytest.raises(FileNotFoundError): # corresponding file does not exist
            orca_error_check(tmp_path / "nonexistent.out")