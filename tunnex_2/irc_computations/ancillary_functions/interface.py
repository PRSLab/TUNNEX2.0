# --- Modules ---
from pathlib import Path
import os
import shlex
import subprocess
import logging
import time


# --- Defining the class for the logging function ---
class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level

    def filter(self, record):
        return record.levelno <= self.max_level


# --- Defining the logging function ---
def setup_logger(log_file="calculations_log.log"):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False # Should prevent log messages from propagating to parent loggers
    logger.handlers.clear() # Should prevent duplicate messages if setup_logger() is called more than once

    # --- File: DEBUG and above ---
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # --- Console: INFO only ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.addFilter(MaxLevelFilter(logging.INFO))

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# --- Defining the logging function ---
def log_run_start(logger, calculation_path):
    logger.info("=" * 80)
    logger.info("Starting calculation: %s", calculation_path)
    logger.info("=" * 80)


# --- Defining the logging function ---
logger = setup_logger()


# --- Defining the function which checks if the file exist ---
def file_check(filename: str | Path) -> Path:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' was not found. Please check '{path}'.")
    if not path.is_file():
        raise ValueError(f"Expected a file, got {path}.")
    return path


# --- Defining the function to read the command to execute the quantum chemistry program (Gaussian, ORCA etc.) ---
def _insert_command(command_file: str | Path) -> str:
    command_file_path = Path(command_file)

    if command_file_path.is_dir():
        command_file_path = command_file_path / "command_settings.txt"

    if command_file_path.is_file() and command_file_path.stat().st_size > 0:
        return command_file_path.read_text(encoding="utf-8").strip()

    command_line = input("Please specify the command used to run Gaussian or ORCA:\n").strip()
    command_file_path.write_text(command_line, encoding="utf-8")
    return command_line


# --- Defining the function to prepare the Gaussian, ORCA or another quantum chemistry program computations ---
def _command_preparation(filename_in: str | Path, command_file: str | Path,
    filename_out_key: bool = False) -> list[str]:

    file_in = file_check(filename_in)

    command_line = _insert_command(command_file)
    command = [*shlex.split(command_line), str(file_in)]

    if filename_out_key:
        file_out = file_in.with_suffix(".out")
        command.append(str(file_out))
    
    return command


# --- Defining the function to submit the job to SLURM ---
def _submit_slurm_job(filename_in: str | Path, command: list[str], envr: dict[str, str]) -> str:

    # --- Submitting the job to SLURM ---
    path = Path(filename_in)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, env=envr)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to submit job '{path}':\n"
                           f"  Return code: {e.returncode}\n"
                           f"  stderr: {e.stderr.strip()}") from e

    # --- Extracting the Slurm job ID ---
    parts = result.stdout.strip().split()
    if len(parts) < 4 or parts[-2] != "job":
        raise RuntimeError(f"Unexpected sbatch output: {result.stdout.strip()!r}")
    job_id = parts[-1]
    logger.info("Submitted batch job %s", job_id)
    return job_id


# --- Defining the function to wait for the SLURM job ---
def _wait_for_slurm_job(job_id: str, envr: dict[str, str], time_wait_seconds: float = 10, log_interval: int = 6) -> None:

    # --- How often a user should be notified (for time_wait_seconds=10 and log_interval=6,
    # a message would appear once per minute) ---
    i = 0
    while True:
        try:
            result = subprocess.run(["squeue", "-h", "-j", job_id],
                                    capture_output=True, text=True,
                                    check=True, env=envr)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed squeue for job {job_id}:\n"
                               f"  Return code: {e.returncode}\n"
                               f"  stderr: {e.stderr.strip()}") from e

        if i % log_interval == 0:
            logger.info('Waiting for a job %s to be finished...', job_id)

        # --- Job is no longer present in the queue ---
        if not result.stdout.strip():
            break
        i += 1
        time.sleep(time_wait_seconds)


# --- Defining the function to check the SLURM job final status ---
def _check_slurm_job(job_id: str, envr: dict[str, str]) -> None:
    try:
        result = subprocess.run(["sacct", "-j", job_id, "--format=State,ExitCode", "--noheader", "--parsable2"],
                                capture_output=True, text=True, check=True, env=envr)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"sacct failed for job {job_id}:\n"
            f"  Return code: {e.returncode}\n"
            f"  stderr: {e.stderr.strip()}") from e

    lines = result.stdout.strip().splitlines()

    if not lines:
        raise RuntimeError(f"sacct returned no status for job {job_id}.")

    job_status = lines[0].split("|")

    if len(job_status) != 2:
        raise RuntimeError(f"Unexpected sacct output for job {job_id}: {result.stdout.strip()!r}")

    state, exit_code = job_status
    
    if state != "COMPLETED" or exit_code != "0:0":
        raise RuntimeError(f"Slurm job {job_id} failed:\n"
                           f"  State: {state}\n"
                           f"  ExitCode: {exit_code}")

    logger.info("Job %s completed successfully", job_id)


# --- Defining the function to run the Gaussian, ORCA or another quantum chemistry program ---
def run_software(filename_in: str | Path, command_file: str | Path, filename_out_key: bool = False,
    slurm_key: bool = False, envr: dict[str, str] | None = None,
    time_wait_seconds: float = 10, log_interval: int = 6) -> None:

    # --- Setting the input parameters ---
    path = Path(filename_in)
    if envr is None:
        envr = os.environ.copy()
    command = _command_preparation(path, command_file, filename_out_key)

    # --- Submitting the job directly ---
    if not slurm_key:
        logger.info("Submitted job %s", path)
        subprocess.run(command, check=True, env=envr)
        return # None
    
    # --- Submitting the job to SLURM ---
    job_id = _submit_slurm_job(path, command, envr)

    # --- Waiting for the job to finish ---
    _wait_for_slurm_job(job_id, envr, time_wait_seconds, log_interval)

    # --- Checking the final job status ---
    _check_slurm_job(job_id, envr)


# --- Defining the function to read Gaussian output files ---
def gauss_out_filename(filename: str | Path) -> Path:
    return Path(filename).with_suffix(".out")


# --- Defining the function to read Orca output files ---
def orca_out_filename(filename: str | Path, suffix: str) -> Path:
    path = Path(filename)
    if suffix.lower() == ".xyz":
        return path.with_stem(path.stem + "_IRC_Full_trj").with_suffix(suffix)
    else:
        return path.with_suffix(suffix)


# --- Checking Gaussian output for abnormal termination ---
def gauss_error_check(filename: str | Path) -> None:

    path_filename = file_check(filename)
    text = path_filename.read_text(encoding="utf-8")

    marker_error = "Error termination via"
    pos_error = text.find(marker_error)
    if pos_error == -1:
        marker_norm = "Normal termination of Gaussian"
        if marker_norm not in text:
            raise RuntimeError(f"Gaussian termination status could not be determined: {path_filename}")
        return # None
    
    raise RuntimeError(f"Gaussian terminated with an error in '{path_filename}':\n\n"f"{text[pos_error:]}")


# --- Checking ORCA output for normal termination ---
def _orca_normal_term_check(file: str | Path) -> None:
    path = Path(file)
    text = path.read_text(encoding="utf-8")
    marker = "****ORCA TERMINATED NORMALLY****"
    if marker not in text:
        raise RuntimeError(f"ORCA terminated with an error in '{path}'. Please check '{path}'.")


# --- Checking ORCA output for existing and normal termination ---
def orca_error_check(filename: str | Path) -> None:

    path_filename = file_check(filename)

    suffix = path_filename.suffix.lower()
    suffix_list = (".xyz", ".out", ".hess")
    if suffix not in suffix_list:
        raise ValueError(f"Found the undocumented suffix '{suffix}' in '{filename}'. Expected {suffix_list}.")

    if suffix == suffix_list[0]:
        new_filename = path_filename.with_stem(path_filename.stem.removesuffix("_IRC_Full_trj")).with_suffix(".out")
        new_filename_path = file_check(new_filename)
        _orca_normal_term_check(new_filename_path)
    elif suffix == suffix_list[1]:
        _orca_normal_term_check(path_filename)
    else: # suffix = suffix_list[2], ".hess", doing nothing
        return # None


# --- It was needed to test the PRIRODA run ---
# env = os.environ.copy()
# env["PATH"] = r"C:\cygwin64\bin;" + env["PATH"]
# filename_in = r"cp_1\min.in"
# command_file = r"cp_1\sett.in"
# filename_out = r"cp_1\min1.out"
# run_software(filename_in, command_file, filename_out, slurm_key=False, envr=env)
# --- Reference (PRIRODA): Laikov, D. N., & Ustynyuk, Y. A., Russ. Chem. Bull., 54(3), 820-826 (2005). ---


# --- It was needed to test the SLURM computations ---
# filename_in = r"cp_1\min.in"
# command_file = r"cp_1\sett.in"
# run_software(filename_in, command_file, filename_out=None, slurm_key=True, envr=env, time_wait_seconds=10, log_interval=6)
# --- Reference (SLURM): Yoo, A. B., Jette, M. A., & Grondona, M.
# In Workshop on job scheduling strategies for parallel processing (pp. 44-60).
# Berlin, Heidelberg: Springer Berlin Heidelberg (2003, June). ---