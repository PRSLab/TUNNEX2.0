# --- Test: test_irc_computations_pipeline_int. Integration tests for irc_computations_pipeline.py
# Run with: pytest test_irc_computations_pipeline_int.py -v ---

# --- Modules ---
import pytest  # type: ignore
pytestmark = pytest.mark.integration  # type: ignore

import numpy as np
import shutil
import hashlib
import logging
import copy
import re
from pathlib import Path

from tunnex_2.constants_and_dataclasses.dataclasses import ( # type: ignore
    IRCData,
    EnergyPointsData,
    AttemptFreq)


# --- Module to test ---
import tunnex_2.irc_computations.pipeline.irc_computations_pipeline as pipeline  # type: ignore
from tunnex_2.irc_computations.pipeline.irc_computations_pipeline import (  # type: ignore
    ts_optimization,
    irc_computations,
    irc_file_parsing,
    react_prod_optimization,
    opt_file_parsing,
    attempt_freq_func,
    number_of_levels_func,
    potential_scal_factor_comp,
    tunnex_input_writing,
    eckart_react_prod_optimization,
    eckart_potential_results)


# --- Helpers and shared data ---
DATA = Path(__file__).parent

TS_OPT_CASES = {"gaussian": dict(guess="ch3coh.gjf", ok="ch3coh_ts.out",
                failed="ch3coh_ts_failed.out", no_status="ch3coh_ts_no_status.out"),
                "orca": dict(guess="ch3coh.inp", ok="ch3coh_ts.out", failed="ch3coh_ts_failed.out")}

PRECOMPUTED_TS_OPT_OUTPUT_NAME = {"gaussian": "ch3coh_ts.out", "orca": "ch3coh_ts.out"}

IRC_COMP_CASES = {"gaussian": dict(ts_out="ch3coh_ts.out", ok="ch3coh_irc.out",
             failed="ch3coh_irc_failed.out", no_status="ch3coh_irc_no_status.out"),
             "orca": dict(ts_out="ch3coh_ts.out", ok="ch3coh_irc_IRC_Full_trj.xyz", ok_log="ch3coh_irc.out",
             failed="ch3coh_irc_IRC_Full_trj_failed.xyz", failed_log="ch3coh_irc_failed.out")}

PARSE_IRC_OUTLIER_FILE = {"gaussian": "ch3coh_irc_outlier.out", "orca": "ch3coh_irc_IRC_Full_trj_outlier.xyz"}

RP_OPT_CASES = {"gaussian": dict(react_ok="ch3coh_react.out", prod_ok="ch3coh_prod.out",
               react_failed="ch3coh_react_failed.out", prod_failed="ch3coh_prod_failed.out"),
               "orca": dict(react_ok="ch3coh_react.out", prod_ok="ch3coh_prod.out",
               react_failed="ch3coh_react_failed.out", prod_failed="ch3coh_prod_failed.out")}

ATT_FREQ_REACT, ATT_FREQ_PROD = 2083.49, 1387.33  # cm-1, CH3CHO and CH3COH, ORCA
AF_FIELDS = ("att_freq_react", "att_freq_react_corr", "att_freq_prod", "att_freq_prod_corr")

SCALING_FACTOR_REACT, SCALING_FACTOR_PROD = 1.00, 0.95  # Test values, should be different

EP_EL_FIELDS = ("transition_state_el_energy", "reactant_el_energy", "product_el_energy")
EP_ZPVE_FIELDS = ("transition_state_zpve", "reactant_zpve", "product_zpve")
EP_ROLE_FIELDS = {"ts": ("transition_state_el_energy", "transition_state_zpve"),
                  "react": ("reactant_el_energy", "reactant_zpve"),
                  "prod": ("product_el_energy", "product_zpve")}

PARSE_EXTRA_FILES = {"gaussian": ["gaussian_atom_masses.out"], "orca": ["orca_atom_masses.out",
                     "ch3coh_irc_IRC_B.xyz", "ch3coh_irc_IRC_F.xyz",
                    "ch3coh_ts.hess", "ch3coh_react.hess", "ch3coh_prod.hess"]}

TI_NUM_LEV_REACT, TI_NUM_LEV_PROD = 17, 23  # these values were made up to detect swapped forward and backward files
TI_SCALING_REACT, TI_SCALING_PROD = 1.37, 0.83  # these values were made up to detect swapped forward and backward files

NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
# it is the pattern to check the Tunnex input files for the QMT computations


@pytest.fixture
def workdir(tmp_path):
    " It creates the temporary work directory "
    dst = tmp_path / "work"
    dst.mkdir()
    return dst

@pytest.fixture(params=["gaussian", "orca"])
def mode(request):
    " It runs the tests with Gaussian and ORCA "
    return request.param
    # All of the test receiving this parameter will run both Gaussian and ORCA tests

@pytest.fixture
def cmd_file(workdir):
    " It creates the command file in temporary folder "
    cmd = workdir / "run.sh"
    cmd.write_text("#!/bin/bash\n")
    return cmd

@pytest.fixture
def guess_file(workdir, mode):
    " It creates the copy of the TS guess file in temporary folder "
    src = DATA / mode / TS_OPT_CASES[mode]["guess"]
    dst = workdir / src.name
    shutil.copy(src, dst)
    return dst

@pytest.fixture
def ts_output_file(workdir, mode):
    " It creates the copy of the TS output file in temporary folder "
    src = DATA / mode / TS_OPT_CASES[mode]["ok"]
    dst = workdir / src.name
    shutil.copy(src, dst)
    for name in PARSE_EXTRA_FILES[mode]:
        shutil.copy(DATA / mode / name, workdir / name)
    return dst

@pytest.fixture
def irc_output_file(workdir, mode):
    " It creates the copy of the IRC output files in temporary folder "
    src = DATA / mode / IRC_COMP_CASES[mode]["ok"]
    dst = workdir / src.name
    shutil.copy(src, dst)
    log = IRC_COMP_CASES[mode].get("ok_log")
    if log is not None:
        shutil.copy(DATA / mode / log, dst.with_suffix(".out"))
    for name in PARSE_EXTRA_FILES[mode]:
        shutil.copy(DATA / mode / name, workdir / name)
    return dst

@pytest.fixture
def fake_ts_opt_software(monkeypatch, mode):
    """
    It mocks the external software execution by copying
    a prepared output file and recording the call arguments
    """
    calls = []
    sample = {"name": "ok"}
    def fake_run(ts_input_file, command_file, filename_out_key, slurm_key, envr):
        calls.append((ts_input_file, command_file, filename_out_key, slurm_key, envr))
        src = DATA / mode / TS_OPT_CASES[mode][sample["name"]]
        shutil.copy(src, _expected_output(mode, Path(ts_input_file)))
    monkeypatch.setattr(pipeline, "run_software", fake_run)
    fake_run.calls = calls
    fake_run.sample = sample
    return fake_run

@pytest.fixture
def fake_irc_software(monkeypatch, mode):
    """
    It mocks the external software execution by copying
    a prepared IRC output file and recording the call arguments
    """
    calls = []
    sample = {"name": "ok"}
    def fake_run(irc_input_file, command_file, filename_out_key, slurm_key, envr):
        calls.append((irc_input_file, command_file, filename_out_key, slurm_key, envr))
        _place_irc_output(mode, sample["name"], irc_input_file)
    monkeypatch.setattr(pipeline, "run_software", fake_run)
    fake_run.calls = calls
    fake_run.sample = sample
    return fake_run

@pytest.fixture(params=[False, True], ids=["without_proj_freq", "with_proj_freq"])
def proj_freq(request, mode):
    " It runs the tests with and without proj_freq "
    if mode == "orca":
        return False
        # set the False option for ORCA always,
        # since we do not have the computed Hessians for each IRC point
    return request.param

@pytest.fixture(params=[False, True], ids=["no_hess", "hess"])
def hess_parsing(request):
    " It runs the tests with and without hess_parsing "
    return request.param

@pytest.fixture
def outlier_spy(monkeypatch):
    " It records the labels passed to detect_outliers_local and keeps the real check "
    labels = []
    real = pipeline.detect_outliers_local
    def spy(irc, values, label, *args, **kwargs):
        labels.append(label)
        return real(irc, values, label, *args, **kwargs)
    monkeypatch.setattr(pipeline, "detect_outliers_local", spy)
    return labels

@pytest.fixture
def fake_rp_software(monkeypatch, mode):
    """
    It mocks the external software execution by copying a prepared output file
    and recording the call arguments. The first call is the reactant, the second is the product
    """
    calls = []
    sample = {"react": "ok", "prod": "ok"}
    def fake_run(input_file, command_file, filename_out, slurm_key, envr):
        role = "react" if len(calls) == 0 else "prod"
        calls.append((input_file, command_file, filename_out, slurm_key, envr))
        _place_rp_output(mode, role, sample[role], input_file)
    monkeypatch.setattr(pipeline, "run_software", fake_run)
    fake_run.calls = calls
    fake_run.sample = sample
    return fake_run

@pytest.fixture
def react_output_file(workdir, mode):
    " It creates the copy of the reactant output file in temporary folder "
    src = DATA / mode / RP_OPT_CASES[mode]["react_ok"]
    dst = workdir / f"{src.name}"
    shutil.copy(src, dst)
    return dst

@pytest.fixture
def prod_output_file(workdir, mode):
    " It creates the copy of the product output file in temporary folder "
    src = DATA / mode / RP_OPT_CASES[mode]["prod_ok"]
    dst = workdir / f"{src.name}"
    shutil.copy(src, dst)
    return dst

@pytest.fixture
def attempt_freq(af_react=ATT_FREQ_REACT, af_prod=ATT_FREQ_PROD):
    " It creates the attempt frequencies "
    result = AttemptFreq(att_freq_react=af_react,
                         att_freq_react_corr = [(af_react, 1.0)],
                         att_freq_prod=af_prod,
                         att_freq_prod_corr = [(af_prod, 1.0)])
    return result

@pytest.fixture
def _num_of_vib_levels_chain(irc_output_file, ts_output_file, guess_file, cmd_file,
    react_output_file, prod_output_file, mode, proj_freq, hess_parsing):
    " It runs several steps of the pipeline on the real files and collects their results "
    irc_data = _irc_parse(irc_output_file, ts_output_file, guess_file, cmd_file, mode, proj_freq, hess_parsing)
    att_freq = _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing)
    energy_points = opt_file_parsing(ts_output_file, react_output_file, prod_output_file, att_freq, mode,
    hess_parsing)
    return irc_data, att_freq, energy_points

@pytest.fixture
def reader_spy(monkeypatch, mode):
    " It records the arguments passed to the structure file reader and keeps the real reader "
    calls = []
    name = _reader_name(mode)
    real = getattr(pipeline, name)
    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)
    monkeypatch.setattr(pipeline, name, spy)
    return calls

@pytest.fixture
def att_freq_calc_spy(monkeypatch):
    " It records the arguments and the results of attempt_freq_calc and keeps the real function "
    calls = []
    real = pipeline.attempt_freq_calc
    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append((args, result))
        return result
    monkeypatch.setattr(pipeline, "attempt_freq_calc", spy)
    return calls

@pytest.fixture
def write_corr_spy(monkeypatch):
    " It records the arguments of write_freq_corr and keeps the real function "
    calls = []
    real = pipeline.write_freq_corr
    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)
    monkeypatch.setattr(pipeline, "write_freq_corr", spy)
    return calls

@pytest.fixture
def num_of_vib_levels_spy(monkeypatch):
    " It records the arguments of compute_number_of_vib_levels and keeps the real function "
    calls = []
    real = pipeline.compute_number_of_vib_levels
    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)
    monkeypatch.setattr(pipeline, "compute_number_of_vib_levels", spy)
    return calls

@pytest.fixture
def react_guess_file(workdir, mode):
    " It creates the copy of the reactant guess file in temporary folder "
    src = DATA / mode / TS_OPT_CASES[mode]["guess"]
    dst = workdir / f"react_guess_{src.name}"
    shutil.copy(src, dst)
    return dst

@pytest.fixture
def prod_guess_file(workdir, mode):
    " It creates the copy of the product guess file in temporary folder "
    src = DATA / mode / TS_OPT_CASES[mode]["guess"]
    dst = workdir / f"prod_guess_{src.name}"
    shutil.copy(src, dst)
    return dst

@pytest.fixture
def mode_reader_eckart_spy(monkeypatch, mode):
    " It records the arguments and the results of the imaginary frequency reader "
    mode_reader_name = "gauss_mode_coordinate_reader" if mode == "gaussian" else "orca_mode_coordinate_reader"
    return _make_eckart_spy(monkeypatch, mode_reader_name)

@pytest.fixture
def params_eckart_spy(monkeypatch):
    " It records the arguments and the results of eckart_potential_parameters "
    return _make_eckart_spy(monkeypatch, "eckart_potential_parameters")

@pytest.fixture
def data_eckart_spy(monkeypatch):
    " It records the arguments and the results of eckart_potential_data "
    return _make_eckart_spy(monkeypatch, "eckart_potential_data")


def _irc_output_path(mode, irc_input_file):
    """ Returns the expected IRC output path using the project's filename generation logic """
    if mode == "gaussian":
        result = pipeline.gauss_out_filename(irc_input_file)
    else:
        result = pipeline.orca_out_filename(irc_input_file, ".xyz")
    return result

def _place_irc_output(mode, name, irc_input_file):
    """ Copies the prepared IRC output and log to the expected locations """
    dst = _irc_output_path(mode, Path(irc_input_file))
    case = IRC_COMP_CASES[mode]
    shutil.copy(DATA / mode / case[name], dst)
    log = case.get(f"{name}_log")
    if log is not None:
        log_dst = dst.parent / "ch3coh_irc.out"
        shutil.copy(DATA / mode / log, log_dst)
    return dst

def _irc_parse(irc_output_file, ts_output_file, guess_file, cmd_file, mode, proj_freq, hess_parsing=False, **kwargs):
    """ Calls irc_file_parsing without running the external software """
    result = irc_file_parsing(irc_output_file, ts_output_file, guess_file, cmd_file, mode, proj_freq=proj_freq,
    hess_parsing=hess_parsing, run_software_key=False, **kwargs)
    return result

def _place_rp_output(mode: str, role: str, status: str, input_file: Path) -> Path:
    """ Copies the prepared reactant or product output to where the project expects it """
    dst = _expected_output(mode, Path(input_file))
    shutil.copy(DATA / mode / RP_OPT_CASES[mode][f"{role}_{status}"], dst)
    return dst

def _snapshot_workdir(folder: Path) -> list:
    """ Returns the sorted names of the files in a folder """
    result = sorted(path.name for path in folder.iterdir())
    return result

def _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing=False,
    corr_analysis=False, **kwargs):
    """ Calls attempt_freq_func with the common arguments """
    result = attempt_freq_func(ts_output_file, react_output_file, prod_output_file, corr_analysis, mode,
    hess_parsing, **kwargs)
    return result

def _number_of_levels(_num_of_vib_levels_chain, **overrides):
    """
    Calls number_of_levels_func with the _num_of_vib_levels_chain data
    and the default frequencies and scaling factors
    """
    args = dict(attempt_freq_react=_num_of_vib_levels_chain[1].att_freq_react,
    attempt_freq_prod=_num_of_vib_levels_chain[1].att_freq_prod,
    potential_scaling_factor_react=SCALING_FACTOR_REACT, potential_scaling_factor_prod=SCALING_FACTOR_PROD)
    args.update(overrides)
    result = number_of_levels_func(_num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2], **args)
    return result

def _eckart_tail(mode: str) -> str:
    """ Returns the block which the project writes into the reactant and product inputs """
    if mode == "gaussian":
        result = pipeline.gaussian_patterns.METHOD_TAIL_REACT_PROD
    else:
        result = pipeline.orca_patterns.LINE_TAIL_REACT_PROD
    return result

def _eckart_rp_opt(react_guess, prod_guess, cmd_file, mode, **kwargs):
    """ Calls eckart_react_prod_optimization with the common arguments """
    result = eckart_react_prod_optimization(react_guess, prod_guess, cmd_file, mode, **kwargs)
    return result


def _sha(path):
    """ Returns the SHA-256 hash of a file """
    result = hashlib.sha256(path.read_bytes()).hexdigest()
    return result

def _expected_output(mode: str, ts_input_file: Path) -> Path:
    """ Returns the expected output file path using the project's filename generation logic """
    if mode == "gaussian":
        result = pipeline.gauss_out_filename(ts_input_file)
    else:
        result = pipeline.orca_out_filename(ts_input_file, ".out")
    return result

def _expected_heat_message(energies):
    """ Returns the expected heat effect message for the _heat_check_func function """
    react = min(energies, key=lambda x: x[0])[1]
    prod = max(energies, key=lambda x: x[0])[1]
    diff = -(prod - react) / pipeline.KJ_MOL_M1_TO_HARTREE
    if np.isclose(diff, 0.0, atol=1e-1):
        word = "thermoneutral"
        result = f"the reaction is {word} ({abs(diff):.1f} kJ mol-1)", f"forbidden message"
        # Since there is no > 0 and < 0 symmetry, put "forbidden message" as
        # a forbidden message
    else:
        word = "exothermic" if diff > 0 else "endothermic"
        other = "endothermic" if diff > 0 else "exothermic"
        result = f"forward reaction is {word} by {abs(diff):.1f} kJ mol-1", f"forward reaction is {other}"
    return result

def _reader_name(mode: str) -> str:
    """ Returns the name of the structure file reader used in the given mode """
    result = "reading_gauss_struct_file" if mode == "gaussian" else "reading_orca_struct_file"
    return result

def _raw_values(mode: str, paths: dict, hess_parsing: bool) -> dict:
    """ Returns the energies read directly by the project's reader, without the attempt frequency correction """
    reader = getattr(pipeline, _reader_name(mode))
    result = {role: reader(path, role, hess_parsing) for role, path in paths.items()}
    return result

def _expected_potential_scal_factor(irc_data, energy_points):
    """ Returns scaling factors by sorting along the IRC """
    points = sorted(irc_data.electronic_energies, key=lambda point: point[0])
    react_el, prod_el = points[0][1], points[-1][1]
    ts_el = sorted(points, key=lambda point: abs(point[0]))[0][1]
    react_zpve = sorted(irc_data.zpve_energies_reverse)[0][1] if irc_data.zpve_energies_reverse is not None else 0.0
    prod_zpve = sorted(irc_data.zpve_energies_forward)[-1][1] if irc_data.zpve_energies_forward is not None else 0.0
    ep = energy_points
    ts_zpve_irc = ep.transition_state_zpve[1] if irc_data.zpve_energies_reverse is not None else 0.0
    ts_zpve = ep.transition_state_zpve[1]
    ep_react = ep.transition_state_el_energy[1] + ts_zpve - ep.reactant_el_energy[1] - ep.reactant_zpve[1]
    ep_prod = ep.transition_state_el_energy[1] + ts_zpve - ep.product_el_energy[1] - ep.product_zpve[1]
    irc_react = ts_el + ts_zpve_irc - react_el - react_zpve
    irc_prod = ts_el + ts_zpve_irc - prod_el - prod_zpve
    return abs(ep_react / irc_react), abs(ep_prod / irc_prod)

def _numbers_findall(path: Path) -> list:
    """ Returns all the numbers found in the text of a file, in the order of appearance """
    result = [float(item) for item in NUMBER_PATTERN.findall(path.read_text())]
    return result

def _file_has_number(numbers: list, value: float, tol=1e-5) -> bool:
    """ Checks whether the value is among the numbers """
    result = any(np.isclose(number, value, rtol=tol, atol=tol) for number in numbers)
    return result

def _first_position_number(numbers: list, value: float, tol=1e-5) -> int:
    """ Returns the position of the first occurrence of the value, -1 if there is None """
    for position, number in enumerate(numbers):
        if np.isclose(number, value, rtol=tol, atol=tol):
            return position
    return -1

def _write_tunn_inp(_num_of_vib_levels_chain, guess_file, **overrides):
    """ Calls tunnex_input_writing with the chain data and the default levels and scaling factors """
    args = dict(num_lev_react=TI_NUM_LEV_REACT, num_lev_prod=TI_NUM_LEV_PROD,
    potential_scal_factor_react=TI_SCALING_REACT, potential_scal_factor_prod=TI_SCALING_PROD)
    args.update(overrides)
    irc_data = args.pop("irc_data", _num_of_vib_levels_chain[0])
    result = tunnex_input_writing(guess_file, irc_data,
        _num_of_vib_levels_chain[2], _num_of_vib_levels_chain[1], **args)
    return result

def _pairs_irc_data(irc_data) -> np.ndarray:
    """ Returns the (IRC, energy) pairs of the IRC data as an array """
    result = np.array(irc_data.electronic_energies, dtype=float)
    return result

def _eckart_results(ts_file, energy_points, mode, hess_parsing=False, **kwargs):
    """ Calls eckart_potential_results with the common arguments """
    result = eckart_potential_results(ts_file, energy_points, mode, hess_parsing, **kwargs)
    return result

def _make_eckart_spy(monkeypatch, name):
    """ Replaces pipeline.<name> with a spy which keeps the real function and records """
    calls = []
    real = getattr(pipeline, name)
    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append((args, result))
        return result
    monkeypatch.setattr(pipeline, name, spy)
    return calls


# ===========================================================================
# --- ts_optimization - normal behaviour and input validation ---
# ===========================================================================
class TestTSOptimization:

    def test_full_flow_returns_output_file(self, guess_file, workdir, mode, fake_ts_opt_software):
        """ Verifies that the full workflow produces the expected output file """
        cmd = workdir / "run.sh"
        cmd.write_text("#!/bin/bash\n")
        out = ts_optimization(guess_file, cmd, mode)
        expected = (DATA / mode / TS_OPT_CASES[mode]["ok"]).read_text()
        assert out.exists(), f"expected {True}, got {out.exists()}, test #1"
        assert out.parent == workdir, f"expected {workdir}, got {out.parent}, test #2"
        assert out.read_text() == expected, f"expected {expected}, got {out.read_text()}, test #3"
        # The last test cannot be failed with fake_ts_opt_software

    def test_ts_input_file_is_created_with_ts_route(self, guess_file, mode, fake_ts_opt_software, workdir):
        """ Verifies that a TS input file is created with the correct TS route """
        ts_optimization(guess_file, workdir / "run.sh", mode)
        ts_input = Path(fake_ts_opt_software.calls[0][0])
        assert ts_input.exists(), f"expected {True}, got {ts_input.exists()}, test #1"
        assert ts_input != guess_file, f"expected {ts_input} != {guess_file}, test #2"          

    def test_ts_input_file_is_created_with_stem_and_ts(self, guess_file, mode, fake_ts_opt_software, workdir):
        """ Verifies that a TS input file is created with stem + '_ts' """
        ts_optimization(guess_file, workdir / "run.sh", mode)
        ts_input = Path(fake_ts_opt_software.calls[0][0])        
        result = ts_input.stem.endswith("_ts")
        assert result, f"expected {True}, got {result}"

    def test_ts_input_file_is_created_with_correct_lines(self, guess_file, mode, fake_ts_opt_software, workdir):
        """ Verifies that a TS input file is created with the correct lines in it """
        ts_optimization(guess_file, workdir / "run.sh", mode)
        ts_input = Path(fake_ts_opt_software.calls[0][0])
        text = ts_input.read_text()
        marker = "ts" if mode  == "gaussian" else "optts"
        assert marker in text.lower(), f"expected {marker} in {text.lower()}"

    def test_run_software_receives_expected_arguments(self, guess_file, mode, fake_ts_opt_software, workdir):
        """ Verifies that run_software receives the expected arguments """
        cmd = workdir / "run.sh"
        expected = ("custom_out", pipeline.SLURM_MODE_KEY, pipeline.ENVR)
        ts_optimization(guess_file, cmd, mode, filename_out_key=expected[0])
        (_, command_file, filename_out_key, slurm_key, envr) = fake_ts_opt_software.calls[0]
        assert Path(command_file) == cmd, f"expected {cmd}, got {Path(command_file)}, test #1"
        assert filename_out_key == expected[0], f"expected {expected[0]}, got {filename_out_key}, test #2"
        assert slurm_key == expected[1], f"expected {expected[1]}, got {slurm_key}, test #3"
        assert envr == expected[2], f"expected {expected[2]}, got {envr}, test #4"
        assert len(fake_ts_opt_software.calls) == 1, \
        f"expected {1} call, got {len(fake_ts_opt_software.calls)}; test #5"

    def test_original_guess_file_is_not_modified(self, guess_file, mode, fake_ts_opt_software, workdir):
        """ Verifies that the original guess file remains unchanged """
        before = _sha(guess_file)
        ts_optimization(guess_file, workdir / "run.sh", mode)
        assert _sha(guess_file) == before, f"expected {before}, got {_sha(guess_file)}"

    def test_accepts_str_and_path(self, guess_file, mode, fake_ts_opt_software, workdir):
        """ Verifies that the workflow accepts both string and Path arguments """
        out_path = ts_optimization(guess_file, workdir / "run.sh", mode)
        out_str = ts_optimization(str(guess_file), str(workdir / "run.sh"), mode)
        assert out_path == out_str, f"expected {out_str}, got {out_path}"

    def test_without_running_software_uses_precomputed_output(self, guess_file, mode, workdir, monkeypatch):
        """ Verifies that a precomputed output is used without running external software """
        monkeypatch.setattr(pipeline, "run_software",
        lambda *a, **kw: pytest.fail("run_software should not be called"))
        # it should use the pre-computed output
        precomputed = workdir / PRECOMPUTED_TS_OPT_OUTPUT_NAME[mode]
        shutil.copy(DATA / mode / TS_OPT_CASES[mode]["ok"], precomputed)
        cmd = workdir / "run.sh"
        cmd.write_text("#!/bin/bash\n")
        out = ts_optimization(guess_file, cmd, mode, run_software_key=False)
        expected = (DATA / mode / TS_OPT_CASES[mode]["ok"]).read_text()
        assert out == precomputed, f"expected {precomputed}, got {out}; test #1"
        assert out.read_text() == expected, f"expected {expected}, got {out.read_text()}; test #2"

    def test_without_running_software_missing_output_raises(self, guess_file, mode, workdir, monkeypatch):
        """ Verifies that a missing precomputed output raises an error. Expecting a FileNotFoundError """
        monkeypatch.setattr(pipeline, "run_software", lambda *a, **kw: pytest.fail("should not be called"))
        # it should use the pre-computed output
        with pytest.raises(FileNotFoundError, match="was not found"):
            ts_optimization(guess_file, workdir / "run.sh", mode, run_software_key=False)
            # file_check in gauss_error_check and orca_error_check

    def test_invalid_mode_raises_and_creates_nothing(self, tmp_path):
        """
        Verifies that an invalid program mode raises an error
        and creates no files. Expecting a ValueError
        """
        guess = tmp_path / "guess.xyz"
        guess.write_text("dummy")
        with pytest.raises(ValueError, match="program mode"):
            ts_optimization(guess, tmp_path / "run.sh", "psi4")
        assert list(tmp_path.iterdir()) == [guess], f"expected {[guess]}, got {list(tmp_path.iterdir())}"

    def test_failed_calculation_is_detected(self, guess_file, mode, fake_ts_opt_software, workdir):
        """ Verifies that a failed calculation is detected and raises a error. Expecting a RuntimeError """
        fake_ts_opt_software.sample["name"] = "failed"
        with pytest.raises(RuntimeError, match="terminated with an error"):
            ts_optimization(guess_file, workdir / "run.sh", mode)
   
    @pytest.mark.parametrize("mode", ["gaussian"])  # overrides the two-mode fixture to "gaussian" only
    def test_no_status_calculation_is_detected_gaussian(self, guess_file, mode, fake_ts_opt_software, workdir):
        """
        Verifies that a Gaussian calculation with no status
        is detected and raises a error. Expecting a RuntimeError
        """
        fake_ts_opt_software.sample["name"] = "no_status"
        with pytest.raises(RuntimeError, match="could not be determined"):
            ts_optimization(guess_file, workdir / "run.sh", mode)


# ===========================================================================
# --- irc_computations - normal behaviour and input validation ---
# ===========================================================================
class TestIRCComputations:

    def test_full_flow_returns_output_file(self, guess_file, ts_output_file, cmd_file, mode, fake_irc_software):
        """ Verifies that the full workflow returns the expected IRC output file """
        out = irc_computations(guess_file, ts_output_file, cmd_file, mode)
        irc_input = Path(fake_irc_software.calls[0][0])
        expected = _irc_output_path(mode, irc_input)
        assert out.exists(), f"expected {True}, got {out.exists()}; test #1"
        assert out == expected, f"expected {expected}, got {out}; test #2"
        assert out.parent == guess_file.parent, f"expected {guess_file.parent}, got {out.parent}; test #3"
        # The last test cannot be failed with fake_irc_software

    def test_output_suffix_depends_on_mode(self, guess_file, ts_output_file, cmd_file, mode, fake_irc_software):
        """ Verifies that the returned file is the IRC trajectory """
        out = irc_computations(guess_file, ts_output_file, cmd_file, mode)
        expected = ".xyz" if mode == "orca" else Path(pipeline.gauss_out_filename(Path("a.gjf"))).suffix
        assert out.suffix == expected, f"expected {expected}, got {out.suffix}"

    def test_irc_input_file_is_created_with_irc_route(self, guess_file, ts_output_file, cmd_file, mode,
        fake_irc_software):
        """ Verifies that an IRC input file is created and contains the IRC keyword """
        irc_computations(guess_file, ts_output_file, cmd_file, mode)
        irc_input = Path(fake_irc_software.calls[0][0])
        assert irc_input.exists(), f"expected {True}, got {irc_input.exists()}; test #1"
        assert irc_input != guess_file, f"expected {irc_input} != {guess_file}; test #2"
        assert irc_input != ts_output_file, f"expected {irc_input} != {ts_output_file}; test #3"
        marker = "irc=" if mode  == "gaussian" else "inithess calc_anfreq"
        assert marker in irc_input.read_text().lower(), \
        f"expected {marker} in {irc_input.read_text().lower()}; test #4"

    def test_run_software_receives_expected_arguments(self, guess_file, ts_output_file, cmd_file, mode,
        fake_irc_software):
        """ Verifies that run_software receives the expected arguments """
        expected = ("custom_out", pipeline.SLURM_MODE_KEY, pipeline.ENVR)
        irc_computations(guess_file, ts_output_file, cmd_file, mode, filename_out_key=expected[0])
        (irc_input, command_file, filename_out_key, slurm_key, envr) = fake_irc_software.calls[0]
        assert Path(command_file) == cmd_file, f"expected {cmd_file}, got {Path(command_file)}; test #1"
        assert filename_out_key == expected[0], f"expected {expected[0]}, got {filename_out_key}; test #2"
        assert slurm_key == expected[1], f"expected {expected[1]}, got {slurm_key}; test #3"
        assert envr == expected[2], f"expected {expected[2]}, got {envr}; test #4"
        assert len(fake_irc_software.calls) == 1, \
        f"expected {1} call, got {len(fake_irc_software.calls)}; test #5"

    def test_original_files_are_not_modified(self, guess_file, ts_output_file, cmd_file, mode, fake_irc_software):
        """ Verifies that the TS guess and the TS output files remain unchanged """
        before = (_sha(guess_file), _sha(ts_output_file))
        irc_computations(guess_file, ts_output_file, cmd_file, mode)
        after = (_sha(guess_file), _sha(ts_output_file))
        assert after == before, f"expected {before}, got {after}"

    def test_accepts_str_and_path(self, guess_file, ts_output_file, cmd_file, mode, fake_irc_software):
        """ Verifies that the workflow accepts both string and Path arguments """
        out_path = irc_computations(guess_file, ts_output_file, cmd_file, mode)
        out_str = irc_computations(str(guess_file), ts_output_file, str(cmd_file), mode)
        assert out_path == out_str, f"expected {out_str}, got {out_path}"

    def test_without_running_software_uses_existing_output(self, guess_file, ts_output_file, cmd_file, mode,
        fake_irc_software, monkeypatch):
        """ Verifies that an existing output is used without running external software """
        first = irc_computations(guess_file, ts_output_file, cmd_file, mode)
        text_before = first.read_text()
        monkeypatch.setattr(pipeline, "run_software",
        lambda *a, **kw: pytest.fail("run_software should not be called"))
        # it should use the pre-computed output
        second = irc_computations(guess_file, ts_output_file, cmd_file, mode, run_software_key=False)
        assert len(fake_irc_software.calls) == 1, f"expected 1 call, got {len(fake_irc_software.calls)}; test #1"
        assert second == first, f"expected {first}, got {second}; test #2"
        assert second.read_text() == text_before, f"expected {text_before}, got {second.read_text()}; test #3"

    @pytest.mark.parametrize("proj_freq", [True, False])
    @pytest.mark.parametrize("calc_all", [True, False])
    @pytest.mark.parametrize("mode", ["gaussian"])  # overrides the two-mode fixture to "gaussian" only
    def test_gaussian_options_are_forwarded(self, guess_file, ts_output_file, cmd_file, 
        mode, workdir, monkeypatch, fake_irc_software, proj_freq, calc_all):
        """ Verifies that Gaussian options are forwarded and reflected in the input """
        real = pipeline.create_gauss_irc_input
        spy = []
        created_input = []
        def spy_create(*args, **kwargs):
            result = real(*args, **kwargs)
            spy.append((args, kwargs))
            created_input.append(result)
            return result
        monkeypatch.setattr(pipeline, "create_gauss_irc_input", spy_create)
        irc_computations(guess_file, ts_output_file, cmd_file, mode,
        proj_freq=proj_freq, calc_all=calc_all)
        (args, _), = spy
        assert args == (guess_file, ts_output_file, proj_freq, calc_all), \
        f"expected {(guess_file, ts_output_file, proj_freq, calc_all)}, got {args}; test #1"
        text = created_input[0].read_text()
        expected_pf = "iop(1/73=2)" if proj_freq else "irc="
        assert expected_pf in text, \
        f"expected {expected_pf} in {text}; test #2"
        expected_calc = "calcall" if calc_all else "calcfc"
        assert expected_calc in text, \
        f"expected {expected_calc} in {text}; test #3"

    @pytest.mark.parametrize("mode", ["orca"])  # overrides the two-mode fixture to "orca" only
    def test_orca_ignores_gaussian_only_options(self, guess_file, ts_output_file, mode,
        workdir, cmd_file, monkeypatch, fake_irc_software):
        """ Verifies that proj_freq and calc_all do not change the ORCA IRC input """
        real = pipeline.create_orca_irc_input
        created_input = []
        def spy_create(*args, **kwargs):
            result = real(*args, **kwargs)
            created_input.append(result)
            return result
        monkeypatch.setattr(pipeline, "create_orca_irc_input", spy_create)
        irc_computations(guess_file, ts_output_file, cmd_file, mode, proj_freq=True, calc_all=True)
        irc_computations(guess_file, ts_output_file, cmd_file, mode, proj_freq=False, calc_all=False)
        texts = [path.read_text() for path in created_input]
        assert texts[0] == texts[1], f"expected equal inputs, got {texts[0]} and {texts[1]}"

    def test_without_running_software_missing_output_raises(self, guess_file, ts_output_file, cmd_file, mode,
        monkeypatch):
        """ Verifies that a missing precomputed output raises an error. Expecting a FileNotFoundError """
        monkeypatch.setattr(pipeline, "run_software", lambda *a, **kw: pytest.fail("should not be called"))
        # it should use the pre-computed output
        with pytest.raises(FileNotFoundError, match="was not found"):
            irc_computations(guess_file, ts_output_file, cmd_file, mode, run_software_key=False)
            # file_check in gauss_error_check and orca_error_check

    def test_invalid_mode_raises_and_creates_nothing(self, tmp_path):
        """
        Verifies that an invalid program mode raises an error
        and creates no files. Expecting a ValueError
        """
        guess = tmp_path / "guess.xyz"
        ts_out = tmp_path / "ts.out"
        guess.write_text("dummy")
        ts_out.write_text("dummy")
        with pytest.raises(ValueError, match="program mode"):
            irc_computations(guess, ts_out, tmp_path / "run.sh", "psi4")
        result = sorted(tmp_path.iterdir())
        assert result == sorted([guess, ts_out]), f"expected {sorted([guess, ts_out])}, got {result}"

    def test_failed_calculation_is_detected(self, guess_file, ts_output_file, cmd_file, mode, fake_irc_software):
        """ Verifies that a failed IRC calculation is detected and raises an error. Expecting a RuntimeError """
        fake_irc_software.sample["name"] = "failed"
        with pytest.raises(RuntimeError, match="terminated with an error"):
            irc_computations(guess_file, ts_output_file, cmd_file, mode)

    @pytest.mark.parametrize("mode", ["gaussian"])  # overrides the two-mode fixture to "gaussian" only
    def test_no_status_calculation_is_detected_gaussian(self, guess_file, ts_output_file, cmd_file,
        mode, fake_irc_software):
        """
        Verifies that a Gaussian calculation with no status
        is detected and raises a error. Expecting a RuntimeError
        """
        fake_irc_software.sample["name"] = "no_status"
        with pytest.raises(RuntimeError, match="could not be determined"):
            irc_computations(guess_file, ts_output_file, cmd_file, mode)

# ===========================================================================
# --- irc_file_parsing - normal behaviour and input validation ---
#  --- _heat_check_func - tests are included ---
# ===========================================================================
class TestIRCFileParsing:

    def test_returns_irc_data_with_energies(self, irc_output_file, ts_output_file, guess_file, cmd_file, mode,
        proj_freq, hess_parsing):
        """ Verifies that the parsed real file gives an IRCData with a valid list of (IRC, energy) pairs """
        data = _irc_parse(irc_output_file, ts_output_file, guess_file, cmd_file, mode, proj_freq, hess_parsing)
        energies = data.electronic_energies
        assert isinstance(data, IRCData), f"expected {IRCData}, got {type(data)}; test #1"
        assert len(energies) > 0, f"expected at least one point, got {len(energies)}; test #2"
        assert all(len(point) == 2 for point in energies), f"expected pairs, got {energies[:3]}; test #3"
        assert np.isfinite(np.array(energies)).all(), f"expected finite values, got {energies}; test #4"

    def test_outlier_check_covers_energy_and_zpve(self, irc_output_file, ts_output_file, guess_file, cmd_file,
        mode, proj_freq, hess_parsing, outlier_spy):
        """ Verifies that the outlier check runs for the electronic energy and for ZPVE only if it is parsed """
        data = _irc_parse(irc_output_file, ts_output_file, guess_file, cmd_file, mode, proj_freq, hess_parsing)
        has_zpve = data.zpve_energies_forward is not None and data.zpve_energies_reverse is not None
        expected = ["electronic energy", "ZPVE"] if has_zpve else ["electronic energy"]
        assert outlier_spy == expected, f"expected {expected}, got {outlier_spy}"

    def test_outlier_check_can_be_disabled(self, irc_output_file, ts_output_file, guess_file, cmd_file, mode,
        proj_freq, hess_parsing, outlier_spy):
        """ Verifies that outlier_check=False skips detect_outliers_local completely """
        _irc_parse(irc_output_file, ts_output_file, guess_file, cmd_file, mode, proj_freq, hess_parsing, outlier_check=False)
        assert outlier_spy == [], f"expected no calls, got {outlier_spy}"

    def test_heat_effect_is_logged(self, irc_output_file, ts_output_file, guess_file, cmd_file, mode,
        proj_freq, hess_parsing, caplog):
        """ Verifies that the logged heat effect corresponds to the energies of the returned IRC data """
        with caplog.at_level(logging.INFO):
            data = _irc_parse(irc_output_file, ts_output_file, guess_file, cmd_file, mode, proj_freq, hess_parsing)
        expected, forbidden = _expected_heat_message(data.electronic_energies)
        assert expected in caplog.text, f"expected '{expected}' in the log, got {caplog.text}; test #1"
        assert forbidden not in caplog.text, f"expected no '{forbidden}' in {caplog.text}; test #2"

    def test_original_files_are_not_modified(self, irc_output_file, ts_output_file, guess_file, cmd_file, mode,
        proj_freq, hess_parsing):
        """ Verifies that parsing does not change the IRC and TS output files """
        before = (_sha(irc_output_file), _sha(ts_output_file))
        _irc_parse(irc_output_file, ts_output_file, guess_file, cmd_file, mode, proj_freq, hess_parsing)
        after = (_sha(irc_output_file), _sha(ts_output_file))
        assert after == before, f"expected {before}, got {after}"

    def test_accepts_str_and_path(self, irc_output_file, ts_output_file, guess_file, cmd_file, mode,
        proj_freq, hess_parsing):
        """ Verifies that the function accepts both string and Path arguments """
        from_path = _irc_parse(irc_output_file, ts_output_file, guess_file, cmd_file, mode, proj_freq, hess_parsing)
        from_str = _irc_parse(str(irc_output_file), str(ts_output_file), str(guess_file), str(cmd_file), mode,
        proj_freq, hess_parsing)
        assert from_path == from_str, f"expected {from_str}, got {from_path}"

    def test_outlier_file_is_irc_parsed_when_check_disabled(self, workdir, ts_output_file, guess_file, cmd_file,
        proj_freq, mode, outlier_spy):
        """ Verifies that a file with an outlier is parsed without errors when outlier_check=False """
        irc_out = workdir / PARSE_IRC_OUTLIER_FILE[mode]
        shutil.copy(DATA / mode / PARSE_IRC_OUTLIER_FILE[mode], irc_out)
        data = _irc_parse(irc_out, ts_output_file, guess_file, cmd_file, mode, proj_freq, outlier_check=False)
        assert isinstance(data, IRCData), f"expected {IRCData}, got {type(data)}; test #1"
        assert outlier_spy == [], f"expected no calls, got {outlier_spy}; test #2"
    
    @pytest.mark.parametrize("mode", ["gaussian"])  # overrides the two-mode fixture to "gaussian" only
    def test_hess_parsing_selects_reader(self, irc_output_file, ts_output_file, guess_file, cmd_file, mode,
        proj_freq, hess_parsing, monkeypatch):
        """ Verifies that hess_parsing selects the reader and that proj_freq is forwarded to it """
        calls = []
        for name in ("reading_gauss_irc", "reading_gauss_irc_tunnex"):
            def make_spy(name=name, real=getattr(pipeline, name)):
                def spy(*args, **kwargs):
                    calls.append((name, args))
                    return real(*args, **kwargs)
                return spy
            monkeypatch.setattr(pipeline, name, make_spy())
        _irc_parse(irc_output_file, ts_output_file, guess_file, cmd_file, mode, proj_freq, hess_parsing)
        expected = "reading_gauss_irc_tunnex" if hess_parsing else "reading_gauss_irc"
        assert [name for name, _ in calls] == [expected], f"expected {[expected]}, got {calls}; test #1"
        assert calls[0][1][-1] == proj_freq, f"expected {proj_freq}, got {calls[0][1][-1]}; test #2"

    def test_missing_irc_output_raises(self, workdir, ts_output_file, guess_file, cmd_file, mode, hess_parsing):
        """ Verifies that a missing IRC output file raises an error. Expecting a FileNotFoundError """
        with pytest.raises(FileNotFoundError, match="was not found"):
            _irc_parse(workdir / "nope.out", ts_output_file, guess_file, cmd_file, mode, hess_parsing)
            # file_check in gauss_error_check and orca_error_check

    def test_invalid_mode_raises_and_creates_nothing(self, tmp_path):
        """
        Verifies that an invalid program mode raises an error
        and creates no files. Expecting a ValueError
        """
        irc_out = tmp_path / "irc.out"
        irc_out.write_text("dummy")
        with pytest.raises(ValueError, match="program mode"):
            irc_file_parsing(irc_out, tmp_path / "ts.out", tmp_path / "guess.xyz", tmp_path / "run.sh", "psi4")
        result = list(tmp_path.iterdir())
        assert result == [irc_out], f"expected {[irc_out]}, got {result}"


# ===========================================================================
# --- react_prod_optimization - normal behaviour and input validation ---
# ===========================================================================
class TestReactProdOptimization:

    def test_full_flow_returns_two_output_files(self, guess_file, irc_output_file, cmd_file, mode,
        fake_rp_software):
        """ Verifies that the full workflow returns two different existing output files in the work directory """
        react_out, prod_out = react_prod_optimization(guess_file, irc_output_file, cmd_file, mode)
        assert react_out.exists(), f"expected {True}, got {react_out.exists()}; test #1"
        assert prod_out.exists(), f"expected {True}, got {prod_out.exists()}; test #2"
        assert react_out != prod_out, f"expected different files, got {react_out} twice; test #3"
        assert react_out.parent == guess_file.parent, f"expected {guess_file.parent}, got {react_out.parent}; test #4"
        assert prod_out.parent == guess_file.parent, f"expected {guess_file.parent}, got {prod_out.parent}; test #5"
        # The last test cannot be failed with fake_ts_opt_software

    def test_outputs_are_not_swapped(self, guess_file, irc_output_file, cmd_file, mode, fake_rp_software):
        """ Verifies that the first returned file is the reactant output and the second one is the product output """
        react_out, prod_out = react_prod_optimization(guess_file, irc_output_file, cmd_file, mode)
        expected_react = (DATA / mode / RP_OPT_CASES[mode]["react_ok"]).read_text()
        expected_prod = (DATA / mode / RP_OPT_CASES[mode]["prod_ok"]).read_text()
        assert react_out.read_text() == expected_react, f"expected {expected_react}, got {react_out.read_text()}; test #1"
        assert prod_out.read_text() == expected_prod, f"expected {expected_prod}, got {prod_out.read_text()}; test #2"

    def test_input_files_are_created(self, guess_file, irc_output_file, cmd_file, mode, fake_rp_software):
        """ Verifies that two new different input files are created and the source files are not reused """
        react_prod_optimization(guess_file, irc_output_file, cmd_file, mode)
        react_input, prod_input = (Path(call[0]) for call in fake_rp_software.calls)
        assert react_input.exists(), f"expected {True}, got {react_input.exists()}; test #1"
        assert prod_input.exists(), f"expected {True}, got {prod_input.exists()}; test #2"
        assert react_input != prod_input, f"expected different inputs, got {react_input} twice; test #3"
        assert {react_input, prod_input}.isdisjoint({guess_file, irc_output_file}), \
        f"expected new files, got {react_input} and {prod_input}; test #4"
        assert react_input.read_text() != prod_input.read_text(), "expected different input contents; test #5"
        marker = "opt freq=noraman" if mode  == "gaussian" else "opt freq"
        assert marker in react_input.read_text().lower(), \
        f"expected {marker} in {react_input.read_text()}; test #6"
        assert marker in prod_input.read_text().lower(), \
        f"expected {marker} in {prod_input.read_text().lower()}; test #7"

    def test_run_software_is_called_for_reactant_and_product(self, guess_file, irc_output_file, cmd_file, mode,
        fake_rp_software):
        """ Verifies that run_software is called exactly twice, the reactant first and the product second """
        react_out, prod_out = react_prod_optimization(guess_file, irc_output_file, cmd_file, mode)
        calls = fake_rp_software.calls
        assert len(calls) == 2, f"expected {2} calls, got {len(calls)}; test #1"
        result = (_expected_output(mode, Path(calls[0][0])), _expected_output(mode, Path(calls[1][0])))
        assert result == (react_out, prod_out), f"expected {(react_out, prod_out)}, got {result}; test #2"

    def test_run_software_receives_expected_arguments(self, guess_file, irc_output_file, cmd_file, mode,
        fake_rp_software):
        """ Verifies that both run_software calls receive the expected arguments """
        expected = ("custom_out", pipeline.SLURM_MODE_KEY, pipeline.ENVR)
        react_prod_optimization(guess_file, irc_output_file, cmd_file, mode, filename_out_key=expected[0])
        for number, (_, command_file, filename_out, slurm_key, envr) in enumerate(fake_rp_software.calls, 1):
            assert Path(command_file) == cmd_file, \
            f"expected {cmd_file}, got {Path(command_file)}; call #{number}, test #1"
            assert filename_out == expected[0], \
            f"expected {expected[0]}, got {filename_out}; call #{number}, test #2"
            assert slurm_key == expected[1], f"expected {expected[1]}, got {slurm_key}; call #{number}, test #3"
            assert envr == expected[2], f"expected {expected[2]}, got {envr}; call #{number}, test #4"

    def test_original_files_are_not_modified(self, guess_file, irc_output_file, cmd_file, mode, fake_rp_software):
        """ Verifies that the TS guess and the IRC output files remain unchanged """
        before = (_sha(guess_file), _sha(irc_output_file))
        react_prod_optimization(guess_file, irc_output_file, cmd_file, mode)
        after = (_sha(guess_file), _sha(irc_output_file))
        assert after == before, f"expected {before}, got {after}"

    def test_accepts_str_and_path(self, guess_file, irc_output_file, cmd_file, mode, fake_rp_software):
        """ Verifies that the workflow accepts both string and Path arguments """
        from_path = react_prod_optimization(guess_file, irc_output_file, cmd_file, mode)
        from_str = react_prod_optimization(str(guess_file), str(irc_output_file), str(cmd_file), mode)
        assert from_path == from_str, f"expected {from_str}, got {from_path}"

    def test_without_running_software_uses_existing_outputs(self, guess_file, irc_output_file, cmd_file, mode,
        fake_rp_software, monkeypatch):
        """ Verifies that existing outputs are used without running external software """
        first = react_prod_optimization(guess_file, irc_output_file, cmd_file, mode)
        texts_before = tuple(path.read_text() for path in first)
        monkeypatch.setattr(pipeline, "run_software",
        lambda *a, **kw: pytest.fail("run_software should not be called"))
        # it should use the pre-computed output
        second = react_prod_optimization(guess_file, irc_output_file, cmd_file, mode, run_software_key=False)
        texts_after = tuple(path.read_text() for path in second)
        assert len(fake_rp_software.calls) == 2, f"expected 2 calls, got {len(fake_rp_software.calls)}; test #1"
        assert second == first, f"expected {first}, got {second}; test #2"
        assert texts_after == texts_before, f"expected {texts_before}, got {texts_after}; test #3"

    def test_without_running_software_missing_outputs_raises(self, guess_file, irc_output_file, cmd_file, mode,
        monkeypatch):
        """ Verifies that missing precomputed outputs raise an error. Expecting a FileNotFoundError """
        monkeypatch.setattr(pipeline, "run_software", lambda *a, **kw: pytest.fail("should not be called"))
        # it should use the pre-computed output
        with pytest.raises(FileNotFoundError, match="was not found"):
            react_prod_optimization(guess_file, irc_output_file, cmd_file, mode, run_software_key=False)
            # file_check in gauss_error_check and orca_error_check

    def test_without_running_software_missing_product_output_raises(self, guess_file, irc_output_file, cmd_file,
        mode, fake_rp_software, monkeypatch):
        """
        Verifies that a missing product output raises an error even if the reactant output exists.
        Expecting a FileNotFoundError
        """
        _, prod_out = react_prod_optimization(guess_file, irc_output_file, cmd_file, mode)
        prod_out.unlink()
        monkeypatch.setattr(pipeline, "run_software", lambda *a, **kw: pytest.fail("should not be called"))
        # it should use the pre-computed output
        with pytest.raises(FileNotFoundError, match="was not found"):
            react_prod_optimization(guess_file, irc_output_file, cmd_file, mode, run_software_key=False)
            # file_check in gauss_error_check and orca_error_check

    def test_invalid_mode_raises_and_creates_nothing(self, tmp_path):
        """
        Verifies that an invalid program mode raises an error
        and creates no files. Expecting a ValueError
        """
        guess = tmp_path / "guess.xyz"
        irc_out = tmp_path / "irc.out"
        guess.write_text("dummy")
        irc_out.write_text("dummy")
        with pytest.raises(ValueError, match="program mode"):
            react_prod_optimization(guess, irc_out, tmp_path / "run.sh", "psi4")
        result = sorted(tmp_path.iterdir())
        assert result == sorted([guess, irc_out]), f"expected {sorted([guess, irc_out])}, got {result}"

    @pytest.mark.parametrize("role", ["react", "prod"])
    def test_failed_calculation_is_detected(self, guess_file, irc_output_file, cmd_file, mode, fake_rp_software,
        role):
        """
        Verifies that a failed reactant or product calculation is detected
        and raises an error. Expecting a RuntimeError
        """
        fake_rp_software.sample[role] = "failed"
        with pytest.raises(RuntimeError, match="terminated with an error"):
            react_prod_optimization(guess_file, irc_output_file, cmd_file, mode)

# ===========================================================================
# --- opt_file_parsing - normal behaviour and input validation ---
# ===========================================================================
class TestOptFileParsing:

    def test_returns_energy_points_data(self, ts_output_file, react_output_file, prod_output_file, attempt_freq,
        mode, hess_parsing):
        """ Verifies that the parsed real files give an EnergyPointsData with finite energies in all the fields """
        data = opt_file_parsing(ts_output_file, react_output_file, prod_output_file, attempt_freq, mode,
        hess_parsing)
        assert isinstance(data, EnergyPointsData), f"expected {EnergyPointsData}, got {type(data)}; test #1"
        for field in EP_EL_FIELDS:
            value = getattr(data, field)
            assert len(value) == 2, f"expected a pair, got {value}; field {field}, test #2"
            assert np.isfinite(value[1]), \
            f"expected finite values, got {value[1]}; field {field}, test #3"
            # irc can be infinite
        for field in EP_ZPVE_FIELDS:
            value = getattr(data, field)
            assert len(value) == 2, f"expected a pair, got {value}; field {field}, test #4"
            assert np.isfinite(value[1]), \
            f"expected finite values, got {value[1]}; field {field}, test #5"
            # irc can be infinite

    def test_energies_come_from_the_right_files(self, ts_output_file, react_output_file, prod_output_file,
        attempt_freq, mode, hess_parsing):
        """ Verifies that the electronic energies and the TS ZPVE are those read from the corresponding files """
        paths = {"ts": ts_output_file, "react": react_output_file, "prod": prod_output_file}
        raw = _raw_values(mode, paths, hess_parsing)
        data = opt_file_parsing(ts_output_file, react_output_file, prod_output_file, attempt_freq, mode,
        hess_parsing)
        for role, (el_field, _) in EP_ROLE_FIELDS.items():
            assert getattr(data, el_field) == raw[role][0], \
            f"expected {raw[role][0]}, got {getattr(data, el_field)}; role {role}, test #1"
        assert data.transition_state_zpve == raw["ts"][1], \
        f"expected {raw['ts'][1]}, got {data.transition_state_zpve}; test #2"
        # we cannot compare ZPVEs for the reactant and product in the same manner, since
        # they are corrected by the attempt frequency inside the function

    def test_attempt_freq_is_removed_from_reactant_and_product_zpve(self, ts_output_file, react_output_file,
        prod_output_file, attempt_freq, mode, hess_parsing):
        """ Verifies that half of the attempt frequency is subtracted from the ZPVE of the reactant and product only """
        paths = {"ts": ts_output_file, "react": react_output_file, "prod": prod_output_file}
        raw = _raw_values(mode, paths, hess_parsing)
        data = opt_file_parsing(ts_output_file, react_output_file, prod_output_file, attempt_freq, mode,
        hess_parsing)
        expected_react = raw["react"][1][1] - 0.5 * ATT_FREQ_REACT * pipeline.CM_M1_TO_HARTREE
        expected_prod = raw["prod"][1][1] - 0.5 * ATT_FREQ_PROD * pipeline.CM_M1_TO_HARTREE
        assert np.isclose(data.reactant_zpve[1], expected_react, rtol=1e-12), \
        f"expected {expected_react}, got {data.reactant_zpve[1]}; test #1"
        assert np.isclose(data.product_zpve[1], expected_prod, rtol=1e-12), \
        f"expected {expected_prod}, got {data.product_zpve[1]}; test #2"
        assert data.reactant_zpve[0] == raw["react"][1][0], \
        f"expected {raw['react'][1][0]}, got {data.reactant_zpve[0]}; test #3"
        assert data.product_zpve[0] == raw["prod"][1][0], \
        f"expected {raw['prod'][1][0]}, got {data.product_zpve[0]}; test #4"
        assert raw["react"] != raw["prod"], \
        f"expected {raw["react"]} != {raw["prod"]}; test #5"

    def test_zero_attempt_freq_keeps_zpve(self, ts_output_file, react_output_file, prod_output_file, mode,
        hess_parsing, attempt_freq):
        """ Verifies that zero attempt frequencies leave the reactant and product ZPVE as they were read """
        paths = {"ts": ts_output_file, "react": react_output_file, "prod": prod_output_file}
        raw = _raw_values(mode, paths, hess_parsing)
        attempt_freq.att_freq_react = 0.0
        attempt_freq.att_freq_prod = 0.0
        data = opt_file_parsing(ts_output_file, react_output_file, prod_output_file, attempt_freq, mode, hess_parsing)
        assert data.reactant_zpve == raw["react"][1], f"expected {raw['react'][1]}, got {data.reactant_zpve}; test #1"
        assert data.product_zpve == raw["prod"][1], f"expected {raw['prod'][1]}, got {data.product_zpve}; test #2"

    def test_readers_receive_roles_files_and_hess_flag(self, ts_output_file, react_output_file, prod_output_file,
        attempt_freq, mode, hess_parsing, reader_spy):
        """ Verifies that each file is read with its own role and that hess_parsing is forwarded """
        opt_file_parsing(ts_output_file, react_output_file, prod_output_file, attempt_freq, mode, hess_parsing)
        expected = [(ts_output_file, "ts", hess_parsing), (react_output_file, "react", hess_parsing),
        (prod_output_file, "prod", hess_parsing)]
        assert reader_spy == expected, f"expected {expected}, got {reader_spy}"

    def test_original_files_and_attempt_freq_are_not_modified(self, ts_output_file, react_output_file,
        prod_output_file, attempt_freq, mode, hess_parsing):
        """ Verifies that parsing does not change the output files and the attempt frequencies """
        files = (ts_output_file, react_output_file, prod_output_file)
        before = (tuple(_sha(path) for path in files), copy.deepcopy(attempt_freq))
        opt_file_parsing(*files, attempt_freq, mode, hess_parsing)
        after = (tuple(_sha(path) for path in files), attempt_freq)
        assert after == before, f"expected {before}, got {after}"

    def test_accepts_str_and_path(self, ts_output_file, react_output_file, prod_output_file, attempt_freq, mode,
        hess_parsing):
        """ Verifies that the function accepts both string and Path arguments """
        from_path = opt_file_parsing(ts_output_file, react_output_file, prod_output_file, attempt_freq, mode,
        hess_parsing)
        from_str = opt_file_parsing(str(ts_output_file), str(react_output_file), str(prod_output_file),
        attempt_freq, mode, hess_parsing)
        assert from_path == from_str, f"expected {from_str}, got {from_path}"

    @pytest.mark.parametrize("role", ["ts", "react", "prod"])
    def test_missing_output_raises(self, workdir, ts_output_file, react_output_file, prod_output_file,
        attempt_freq, mode, hess_parsing, role):
        """
        Verifies that a missing TS, reactant or product output file raises an error.
        Expecting a FileNotFoundError
        """
        paths = {"ts": ts_output_file, "react": react_output_file, "prod": prod_output_file}
        paths[role] = workdir / "nope.out"
        with pytest.raises(FileNotFoundError, match="was not found"):
            opt_file_parsing(paths["ts"], paths["react"], paths["prod"], attempt_freq, mode, hess_parsing)

    def test_invalid_mode_raises_and_creates_nothing(self, tmp_path, attempt_freq):
        """
        Verifies that an invalid program mode raises an error
        and creates no files. Expecting a ValueError
        """
        ts_out = tmp_path / "ts.out"
        ts_out.write_text("dummy")
        with pytest.raises(ValueError, match="program mode"):
            opt_file_parsing(ts_out, tmp_path / "react.out", tmp_path / "prod.out", attempt_freq, "psi4")
        result = list(tmp_path.iterdir())
        assert result == [ts_out], f"expected {[ts_out]}, got {result}"


# ===========================================================================
# --- attempt_freq_func - normal behaviour and input validation ---
# ===========================================================================
class TestAttemptFreqFunc:

    def test_returns_attempt_freq_with_finite_values(self, ts_output_file, react_output_file, prod_output_file,
        mode, hess_parsing):
        """ Verifies that the real files give an AttemptFreq with finite attempt frequencies """
        data = _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing)
        assert isinstance(data, AttemptFreq), f"expected {AttemptFreq}, got {type(data)}; test #1"
        for field in ("att_freq_react", "att_freq_prod"):
            value = getattr(data, field)
            assert np.isfinite(value), f"expected a finite value, got {value}; field {field}, test #2"
            assert value > 0, f"expected a positive frequency, got {value}; field {field}, test #3"

    def test_calc_receives_roles_files_and_flags(self, ts_output_file, react_output_file, prod_output_file,
        mode, hess_parsing, att_freq_calc_spy):
        """ Verifies that the reactant and product are processed in this order with their own files and the flags """
        _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing)
        result = [args for args, _ in att_freq_calc_spy]
        expected = [(ts_output_file, react_output_file, "reactant", mode, hess_parsing),
        (ts_output_file, prod_output_file, "product", mode, hess_parsing)]
        assert result == expected, f"expected {expected}, got {result}"

    def test_default_picks_weighted_average(self, ts_output_file, react_output_file, prod_output_file, mode,
        hess_parsing, att_freq_calc_spy):
        """ Verifies that the correlation-weighted average is returned by default """
        data = _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing)
        react_aver, prod_aver = att_freq_calc_spy[0][1][0], att_freq_calc_spy[1][1][0]
        react_max, prod_max = att_freq_calc_spy[0][1][2][0], att_freq_calc_spy[1][1][2][0]
        assert data.att_freq_react == react_aver, f"expected {react_aver}, got {data.att_freq_react}; test #1"
        assert data.att_freq_prod == prod_aver, f"expected {prod_aver}, got {data.att_freq_prod}; test #2"
        assert react_aver != react_max, f"expected {react_aver} != {react_max}; test #3"
        assert prod_aver != prod_max, f"expected {react_aver} != {react_max}; test #4"

    def test_pick_freq_max_corr_picks_the_most_correlated(self, ts_output_file, react_output_file,
        prod_output_file, mode, hess_parsing, att_freq_calc_spy):
        """ Verifies that pick_freq_max_corr=True returns the frequencies with the highest correlation """
        data = _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing,
        pick_freq_max_corr=True)
        react_aver, prod_aver = att_freq_calc_spy[0][1][0], att_freq_calc_spy[1][1][0]
        react_max, prod_max = att_freq_calc_spy[0][1][2][0], att_freq_calc_spy[1][1][2][0]
        assert data.att_freq_react == react_max, f"expected {react_max}, got {data.att_freq_react}; test #1"
        assert data.att_freq_prod == prod_max, f"expected {prod_max}, got {data.att_freq_prod}; test #2"
        assert react_aver != react_max, f"expected {react_aver} != {react_max}; test #3"
        assert prod_aver != prod_max, f"expected {react_aver} != {react_max}; test #4"

    def test_corr_values_do_not_depend_on_pick_flag(self, ts_output_file, react_output_file, prod_output_file,
        mode, hess_parsing):
        """ Verifies that the stored correlations are the same for both values of pick_freq_max_corr """
        default = _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing)
        picked = _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing,
        pick_freq_max_corr=True)
        assert default.att_freq_react_corr == picked.att_freq_react_corr, \
        f"expected {default.att_freq_react_corr}, got {picked.att_freq_react_corr}; test #1"
        assert default.att_freq_prod_corr == picked.att_freq_prod_corr, \
        f"expected {default.att_freq_prod_corr}, got {picked.att_freq_prod_corr}; test #2"

    def test_corr_values_come_from_the_right_calculation(self, ts_output_file, react_output_file,
        prod_output_file, mode, hess_parsing, att_freq_calc_spy):
        """ Verifies that the reactant and product correlations are not swapped """
        data = _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing)
        react_corr, prod_corr = att_freq_calc_spy[0][1][1], att_freq_calc_spy[1][1][1]
        assert data.att_freq_react_corr == react_corr, f"expected {react_corr}, got {data.att_freq_react_corr}; test #1"
        assert data.att_freq_prod_corr == prod_corr, f"expected {prod_corr}, got {data.att_freq_prod_corr}; test #2"

    def test_accepts_str_and_path(self, ts_output_file, react_output_file, prod_output_file, mode, hess_parsing):
        """ Verifies that the function accepts both string and Path arguments """
        from_path = _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing)
        from_str = _att_freq_call(str(ts_output_file), str(react_output_file), str(prod_output_file), mode, hess_parsing)
        assert from_path == from_str, f"expected {from_str}, got {from_path}"

    def test_is_repeatable(self, ts_output_file, react_output_file, prod_output_file, mode, hess_parsing):
        """ Verifies that two runs on the same files give the same result """
        first = _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing)
        second = _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing)
        assert first == second, f"expected {first}, got {second}"

    def test_no_files_without_corr_analysis(self, workdir, ts_output_file, react_output_file, prod_output_file,
        mode, hess_parsing, write_corr_spy):
        """ Verifies that corr_analysis=False neither calls write_freq_corr nor creates files """
        before = _snapshot_workdir(workdir)
        _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing, corr_analysis=False)
        after = _snapshot_workdir(workdir)
        assert write_corr_spy == [], f"expected no calls, got {write_corr_spy}; test #1"
        assert after == before, f"expected {before}, got {after}; test #2"
        # in fact, a function may create a new file with the atom masses, but here we assume that
        # it exists already

    def test_corr_files_are_created(self, workdir, ts_output_file, react_output_file, prod_output_file, mode,
        hess_parsing):
        """ Verifies that corr_analysis=True creates new non-empty files in the work directory """
        before = _snapshot_workdir(workdir)
        _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing, corr_analysis=True)
        new = sorted(set(_snapshot_workdir(workdir)) - set(before))
        assert len(new) == 2, f"expected 2 new files (reactant and product), got {new}; test #1"
        assert all((workdir / name).stat().st_size > 0 for name in new), f"expected non-empty files, got {new}; test #2"
        # in fact, a function may create a new file with the atom masses, but here we assume that
        # it exists already

    def test_write_freq_corr_receives_own_data(self, ts_output_file, react_output_file, prod_output_file, mode,
        hess_parsing, att_freq_calc_spy, write_corr_spy):
        """ Verifies that write_freq_corr is called for the reactant first and the product second with their data """
        _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing, corr_analysis=True)
        expected = [(react_output_file, *att_freq_calc_spy[0][1]), (prod_output_file, *att_freq_calc_spy[1][1])]
        assert write_corr_spy == expected, f"expected {expected}, got {write_corr_spy}"

    def test_corr_analysis_does_not_change_the_result(self, ts_output_file, react_output_file, prod_output_file,
        mode, hess_parsing):
        """ Verifies that corr_analysis only adds files and does not change the returned data """
        without = _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing, corr_analysis=False)
        with_corr = _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing,
        corr_analysis=True)
        assert with_corr == without, f"expected {without}, got {with_corr}"

    @pytest.mark.parametrize("corr_analysis", [False, True])
    def test_original_files_are_not_modified(self, ts_output_file, react_output_file, prod_output_file, mode,
        hess_parsing, corr_analysis):
        """ Verifies that the TS, reactant and product output files remain unchanged """
        files = (ts_output_file, react_output_file, prod_output_file)
        before = tuple(_sha(path) for path in files)
        _att_freq_call(*files, mode, hess_parsing, corr_analysis=corr_analysis)
        after = tuple(_sha(path) for path in files)
        assert after == before, f"expected {before}, got {after}"

    def test_repeated_corr_analysis_does_not_break(self, workdir, ts_output_file, react_output_file,
        prod_output_file, mode, hess_parsing):
        """ Verifies that a second run with corr_analysis=True works and does not create extra files """
        _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing, corr_analysis=True)
        before = _snapshot_workdir(workdir)
        _att_freq_call(ts_output_file, react_output_file, prod_output_file, mode, hess_parsing, corr_analysis=True)
        after = _snapshot_workdir(workdir)
        assert after == before, f"expected {before}, got {after}"
        # in fact, a function may create a new file with the atom masses, but here we assume that
        # it exists already

    @pytest.mark.parametrize("role", ["ts", "react", "prod"])
    def test_missing_output_raises(self, workdir, ts_output_file, react_output_file, prod_output_file, mode,
        hess_parsing, role):
        """
        Verifies that a missing TS, reactant or product output file
        raises an error. Expecting a FileNotFoundError
        """
        paths = {"ts": ts_output_file, "react": react_output_file, "prod": prod_output_file}
        paths[role] = workdir / "nope.out"
        with pytest.raises(FileNotFoundError, match="was not found"):
            _att_freq_call(paths["ts"], paths["react"], paths["prod"], mode, hess_parsing)

    def test_missing_output_creates_nothing(self, workdir, ts_output_file, react_output_file, prod_output_file,
        mode, hess_parsing):
        """ Verifies that a failed run with corr_analysis=True does not leave new files """
        before = _snapshot_workdir(workdir)
        with pytest.raises(FileNotFoundError, match="was not found"):
            _att_freq_call(ts_output_file, react_output_file, workdir / "nope.out", mode, hess_parsing, corr_analysis=True)
        after = _snapshot_workdir(workdir)
        assert after == before, f"expected {before}, got {after}"
        # in fact, a function may create a new file with the atom masses, but here we assume that
        # it exists already
    
    @pytest.mark.parametrize("bad_mode", ["psi4", "Gaussian", "ORCA", ""])
    def test_invalid_mode_raises_and_creates_nothing(self, tmp_path, bad_mode, write_corr_spy):
        """
        Verifies that an invalid program mode raises an error before any file is read or written,
        even with corr_analysis=True. Expecting a ValueError
        """
        ts_out = tmp_path / "ts.out"
        react_out = tmp_path / "react.out"
        prod_out = tmp_path / "prod.out"
        for path in (ts_out, react_out, prod_out):
            path.write_text("dummy")
        before = _snapshot_workdir(tmp_path)
        with pytest.raises(ValueError, match="Unsupported program key"):
            _att_freq_call(ts_out, react_out, prod_out, bad_mode, corr_analysis=True)
        after = _snapshot_workdir(tmp_path)
        assert write_corr_spy == [], f"expected no calls, got {write_corr_spy}; test #1"
        assert after == before, f"expected {before}, got {after}; test #2"


# ===========================================================================
# --- number_of_levels_func - normal behaviour and input validation ---
# ===========================================================================
class TestNumberOfLevelsFunc:

    def test_returns_two_finite_positive_numbers(self, _num_of_vib_levels_chain):
        """ Verifies that the data from the previous steps give finite non-negative numbers of levels """
        result = _number_of_levels(_num_of_vib_levels_chain)
        assert len(result) == 2, f"expected {2} values, got {len(result)}; test #1"
        for name, value in zip(("reactant", "product"), result):
            assert np.isfinite(value), f"expected a finite value, got {value}; {name}, test #2"
            assert value >= 0, f"expected a non-negative number, got {value}; {name}, test #3"

    def test_compute_receives_own_arguments_in_order(self, _num_of_vib_levels_chain, num_of_vib_levels_spy):
        """ Verifies that the reactant is processed first and the product second, each with its own arguments """
        _number_of_levels(_num_of_vib_levels_chain)
        expected = [(_num_of_vib_levels_chain[1].att_freq_react,
        _num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2], SCALING_FACTOR_REACT, "reactant"), \
        (_num_of_vib_levels_chain[1].att_freq_prod, _num_of_vib_levels_chain[0],
        _num_of_vib_levels_chain[2], SCALING_FACTOR_PROD, "product")]
        assert num_of_vib_levels_spy == expected, f"expected {expected}, got {num_of_vib_levels_spy}"

    def test_results_are_not_swapped(self, _num_of_vib_levels_chain):
        """
        Verifies that the returned values are the results
        of the reactant and product calculations in this order
        """
        real = pipeline.compute_number_of_vib_levels
        expected_react = real(_num_of_vib_levels_chain[1].att_freq_react, _num_of_vib_levels_chain[0],
        _num_of_vib_levels_chain[2], SCALING_FACTOR_REACT, "reactant")
        expected_prod = real(_num_of_vib_levels_chain[1].att_freq_prod, _num_of_vib_levels_chain[0],
        _num_of_vib_levels_chain[2], SCALING_FACTOR_PROD, "product")
        result = _number_of_levels(_num_of_vib_levels_chain)
        assert result == (expected_react, expected_prod), f"expected {(expected_react, expected_prod)}, got {result}"

    def test_reactant_does_not_depend_on_product_arguments(self, _num_of_vib_levels_chain):
        """
        Verifies that changing the product frequency and scaling factor
        does not change the reactant result
        """
        base = _number_of_levels(_num_of_vib_levels_chain)
        changed = _number_of_levels(_num_of_vib_levels_chain,
        attempt_freq_prod=1.5 * _num_of_vib_levels_chain[1].att_freq_prod,
        potential_scaling_factor_prod=0.9)
        assert changed[0] == base[0], f"expected {base[0]}, got {changed[0]}; test #1"
        assert changed[1] != base[1], f"expected a changed product result, got {changed[1]}; test #2"

    def test_product_does_not_depend_on_reactant_arguments(self, _num_of_vib_levels_chain):
        """
        Verifies that changing the reactant frequency and scaling factor
        does not change the product result
        """
        base = _number_of_levels(_num_of_vib_levels_chain)
        changed = _number_of_levels(_num_of_vib_levels_chain,
        attempt_freq_react=1.5 * _num_of_vib_levels_chain[1].att_freq_react,
        potential_scaling_factor_react=0.9)
        assert changed[1] == base[1], f"expected {base[1]}, got {changed[1]}; test #1"
        assert changed[0] != base[0], f"expected a changed reactant result, got {changed[0]}; test #2"

    def test_inputs_are_not_modified(self, _num_of_vib_levels_chain):
        """ Verifies that the IRC data, attempt frequencies, and energy points remain unchanged """
        before = (copy.deepcopy(_num_of_vib_levels_chain[0]),
        copy.deepcopy(_num_of_vib_levels_chain[1]), copy.deepcopy(_num_of_vib_levels_chain[2]))
        _number_of_levels(_num_of_vib_levels_chain)
        after = (_num_of_vib_levels_chain[0], _num_of_vib_levels_chain[1], _num_of_vib_levels_chain[2])
        assert after == before, f"expected {before}, got {after}"

    def test_depends_on_frequency_and_potential_scaling_factor(self, _num_of_vib_levels_chain):
        """
        Verifies that increasing the attempt frequency decreases
        and potential scaling factor increases the number of vibrational levels
        """
        first = _number_of_levels(_num_of_vib_levels_chain)
        second = _number_of_levels(_num_of_vib_levels_chain,
        attempt_freq_react=1.5 * _num_of_vib_levels_chain[1].att_freq_react)
        third = _number_of_levels(_num_of_vib_levels_chain,
        potential_scaling_factor_react=SCALING_FACTOR_REACT*1.5)
        assert first[0] > second[0], f"expected {first[0]} > {second[0]}; test #1"
        assert third[0] > first[0], f"expected {third[0]} > {first[0]}; test #2"
        assert first[1] == second[1], f"expected {first[1]} == {second[1]}; test #3"
        assert first[1] == third[1], f"expected {first[1]} == {third[1]}; test #4"
    
    @pytest.mark.parametrize("hess_parsing", [False]) # overrides the two-mode fixture to False only
    @pytest.mark.parametrize("mode", ["gaussian"])  # overrides the two-mode fixture to "gaussian" only
    def test_result_matches_manually_computed_data(self, _num_of_vib_levels_chain, proj_freq):
        """ Verifies that the result matches the manually computed data """
        result = _number_of_levels(_num_of_vib_levels_chain)
        expected = (11, 6) if proj_freq else (12, 6)
        assert result == expected, f"expected {expected}, got {result}"

    def test_does_not_create_files(self, workdir, _num_of_vib_levels_chain):
        """ Verifies that the computation does not create or remove files in the work directory """
        before = _snapshot_workdir(workdir)
        _number_of_levels(_num_of_vib_levels_chain)
        after = _snapshot_workdir(workdir)
        assert after == before, f"expected {before}, got {after}"

    def test_non_positive_frequency_react_raises(self, _num_of_vib_levels_chain):
        """ Verifies that non-positive reactant frequency an error. Expecting a ValueError """
        irc_data, att_freq, energy_points = _num_of_vib_levels_chain
        att_freq.att_freq_react = 0.0
        with pytest.raises(ValueError, match="must be positive"):
            number_of_levels_func(irc_data, energy_points, att_freq.att_freq_react, att_freq.att_freq_prod,
            SCALING_FACTOR_REACT, SCALING_FACTOR_PROD)

    def test_non_positive_frequency_prod_raises(self, _num_of_vib_levels_chain):
        """ Verifies that non-positive product frequency an error. Expecting a ValueError """
        irc_data, att_freq, energy_points = _num_of_vib_levels_chain
        att_freq.att_freq_prod = 0.0
        with pytest.raises(ValueError, match="must be positive"):
            number_of_levels_func(irc_data, energy_points, att_freq.att_freq_react, att_freq.att_freq_prod,
            SCALING_FACTOR_REACT, SCALING_FACTOR_PROD)
    
    def test_missing_electronic_energies_raise(self, _num_of_vib_levels_chain):
        """ Verifies that IRCData without electronic energies raises an error. Expecting a ValueError """
        irc_data, att_freq, energy_points = _num_of_vib_levels_chain
        irc_data.electronic_energies = None
        with pytest.raises(ValueError, match="array of electronic energies"):
            number_of_levels_func(irc_data, energy_points, att_freq.att_freq_react, att_freq.att_freq_prod,
            SCALING_FACTOR_REACT, SCALING_FACTOR_PROD)


# ===========================================================================
# --- potential_scal_factor_comp - normal behaviour and input validation ---
# ===========================================================================
class TestPotentialScalFactorComp:

    def test_returns_two_finite_positive_numbers(self, _num_of_vib_levels_chain):
        """ Verifies that the data from the previous steps give two finite positive scaling factors """
        result = potential_scal_factor_comp(_num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2])
        assert len(result) == 2, f"expected {2} values, got {len(result)}; test #1"
        for name, value in zip(("reactant", "product"), result):
            assert np.isfinite(value), f"expected a finite value, got {value}; {name}, test #2"
            assert value > 0, f"expected a positive number, got {value}; {name}, test #3"

    def test_result_matches_independent_calculation(self, _num_of_vib_levels_chain):
        """ Verifies the factors against the formula applied to the independently found energies """
        result = potential_scal_factor_comp(_num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2])
        expected = _expected_potential_scal_factor(_num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2])
        assert np.allclose(result, expected, rtol=1e-12), f"expected {expected}, got {result}"

    def test_without_zpve_in_irc_data(self, _num_of_vib_levels_chain):
        """ Verifies that the function works and matches the independent calculation without the ZPVE """
        _num_of_vib_levels_chain[0].zpve_energies_forward = None
        _num_of_vib_levels_chain[0].zpve_energies_reverse = None
        result = potential_scal_factor_comp(_num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2])
        expected = _expected_potential_scal_factor(_num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2])
        assert np.allclose(result, expected, rtol=1e-12), f"expected {expected}, got {result}"

    def test_result_is_close_to_one(self, _num_of_vib_levels_chain):
        """
        Verifies that the factors are of a reasonable size:
        the IRC and the optimized barriers must be comparable
        """
        result = potential_scal_factor_comp(_num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2])
        borders = (0.25, 4.00)
        assert all(borders[0] < value < borders[1] for value in result), \
        f"expected values between {borders[0]} and {borders[1]}, got {result}"
        # The range is a sanity check
    
    def test_no_warning_for_a_normal_barrier(self, _num_of_vib_levels_chain, caplog):
        """ Verifies that no flat barrier warning is logged for the real data """
        with caplog.at_level(logging.WARNING):
            potential_scal_factor_comp(_num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2])
        assert "Very small barrier" not in caplog.text, f"expected no warning, got {caplog.text}"

    def test_result_is_usable_by_number_of_levels_func(self, _num_of_vib_levels_chain):
        """ Verifies that the factors can be passed to number_of_levels_func """
        react, prod = potential_scal_factor_comp(_num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2])
        result = _number_of_levels(_num_of_vib_levels_chain,
        potential_scaling_factor_react=react, potential_scaling_factor_prod=prod)
        assert all(isinstance(value, int) and value >= 0 for value in result), \
        f"expected non-negative integers, got {result}"

    def test_inputs_are_not_modified(self, _num_of_vib_levels_chain):
        """ Verifies that the IRC data and the energy points remain unchanged """
        before = (copy.deepcopy(_num_of_vib_levels_chain[0]), copy.deepcopy(_num_of_vib_levels_chain[2]))
        potential_scal_factor_comp(_num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2])
        after = (_num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2])
        assert after == before, f"expected {before}, got {after}"

    def test_is_repeatable(self, _num_of_vib_levels_chain):
        """ Verifies that two runs on the same data give the same result """
        first = potential_scal_factor_comp(_num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2])
        second = potential_scal_factor_comp(_num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2])
        assert first == second, f"expected {first}, got {second}"

    def test_does_not_create_files(self, workdir, _num_of_vib_levels_chain):
        """ Verifies that the computation does not create or remove files in the work directory """
        before = _snapshot_workdir(workdir)
        potential_scal_factor_comp(_num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2])
        after = _snapshot_workdir(workdir)
        assert after == before, f"expected {before}, got {after}"

    @pytest.mark.parametrize("hess_parsing", [False]) # overrides the two-mode fixture to False only
    @pytest.mark.parametrize("mode", ["gaussian"])  # overrides the two-mode fixture to "gaussian" only
    def test_result_matches_manually_computed_data(self, _num_of_vib_levels_chain, proj_freq):
        """ Verifies that the result matches the manually computed data """
        result = potential_scal_factor_comp(_num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2])
        expected = (1.105, 1.080) if proj_freq else (1.060, 1.085)
        assert np.allclose(result, expected, rtol=1e-3), f"expected {expected}, got {result}"


# ===========================================================================
# --- tunnex_input_writing - normal behaviour and input validation ---
# ===========================================================================
class TestTunnexInputWriting:

    def test_returns_two_new_non_empty_files(self, workdir, _num_of_vib_levels_chain, guess_file):
        """ Verifies that two different new non-empty files are created in the work directory """
        before = set(_snapshot_workdir(workdir))
        forward, backward = _write_tunn_inp(_num_of_vib_levels_chain, guess_file)
        new = set(_snapshot_workdir(workdir)) - before
        assert forward.exists() and backward.exists(), f"expected existing files, got {forward}, {backward}; test #1"
        assert forward != backward, f"expected different files, got {forward} and {backward}; test #2"
        assert {forward.name, backward.name} <= new, f"expected {forward.name}, {backward.name} in {new}; test #3"
        assert forward.parent == guess_file.parent, f"expected {guess_file.parent}, got {forward.parent}; test #4"
        assert backward.parent == guess_file.parent, f"expected {guess_file.parent}, got {backward.parent}; test #5"
        assert forward.stat().st_size > 0 and backward.stat().st_size > 0, "expected non-empty files; test #6"

    def test_file_names_show_the_direction(self, _num_of_vib_levels_chain, guess_file):
        """ Verifies that the first returned file is the forward one and the second one is the backward one """
        forward, backward = _write_tunn_inp(_num_of_vib_levels_chain, guess_file)
        assert "_forward" in forward.stem, f"expected _forward in {forward.name}; test #1"
        assert "_backward" in backward.stem, f"expected _backward in {backward.name}; test #2"

    def test_forward_file_has_reactant_settings(self, _num_of_vib_levels_chain, guess_file):
        """ Verifies that the forward file contains the reactant frequency, energies, levels and scaling factor """
        forward, _ = _write_tunn_inp(_num_of_vib_levels_chain, guess_file)
        numbers = _numbers_findall(forward)
        ep, af = _num_of_vib_levels_chain[2], _num_of_vib_levels_chain[1]
        expected = {"frequency": af.att_freq_react, "E0": ep.reactant_el_energy[1], "ZPVE0": ep.reactant_zpve[1],
        "levels": TI_NUM_LEV_REACT, "scaling factor": TI_SCALING_REACT}
        for name, value in expected.items():
            assert _file_has_number(numbers, value), f"expected {value} in the forward file; {name}, test #1"
        for name, value in {"levels": TI_NUM_LEV_PROD, "scaling factor": TI_SCALING_PROD}.items():
            assert not _file_has_number(numbers, value), \
            f"expected no product {value} in the forward file; {name}, test #2"

    def test_backward_file_has_product_settings(self, _num_of_vib_levels_chain, guess_file):
        """ Verifies that the backward file contains the product frequency, energies, levels and scaling factor """
        _, backward = _write_tunn_inp(_num_of_vib_levels_chain, guess_file)
        numbers = _numbers_findall(backward)
        ep, af = _num_of_vib_levels_chain[2], _num_of_vib_levels_chain[1]
        expected = {"frequency": af.att_freq_prod, "E0": ep.product_el_energy[1], "ZPVE0": ep.product_zpve[1],
        "levels": TI_NUM_LEV_PROD, "scaling factor": TI_SCALING_PROD}
        for name, value in expected.items():
            assert _file_has_number(numbers, value), f"expected {value} in the backward file; {name}, test #1"
        for name, value in {"levels": TI_NUM_LEV_REACT, "scaling factor": TI_SCALING_REACT}.items():
            assert not _file_has_number(numbers, value), \
            f"expected no reactant {value} in the backward file; {name}, test #2"

    def test_irc_energies_are_written_in_opposite_orders(self, _num_of_vib_levels_chain, guess_file):
        """
        Verifies that all the IRC electronic energies are written,
        forward in the IRC order, backward reversed
        """
        forward, backward = _write_tunn_inp(_num_of_vib_levels_chain, guess_file)
        energies = [energy for _, energy in _num_of_vib_levels_chain[0].electronic_energies]
        for path, name in ((forward, "forward"), (backward, "backward")):
            numbers = _numbers_findall(path)
            assert all(_file_has_number(numbers, energy) for energy in energies), \
            f"expected all the energies; {name}, test #1"
        first, last = energies[0], energies[-1]
        assert _first_position_number(_numbers_findall(forward), first) < _first_position_number(
        _numbers_findall(forward), last), \
        "expected the first IRC point before the last one in the forward file; test #2"
        assert _first_position_number(_numbers_findall(backward), first) > _first_position_number(
        _numbers_findall(backward), last), \
        "expected the last IRC point before the first one in the backward file; test #3"

    def test_scaling_factors_change_only_their_own_file(self, _num_of_vib_levels_chain, guess_file):
        """ Verifies that a changed reactant scaling factor changes the forward file only """
        base = [path.read_text() for path in _write_tunn_inp(_num_of_vib_levels_chain, guess_file)]
        changed = [path.read_text() for path in _write_tunn_inp(
        _num_of_vib_levels_chain, guess_file, potential_scal_factor_react=2.5)]
        assert changed[0] != base[0], "expected a changed forward file; test #1"
        assert changed[1] == base[1], "expected an unchanged backward file; test #2"

    def test_default_scaling_factors_are_one(self, _num_of_vib_levels_chain, guess_file):
        """ Verifies that the omitted scaling factors give the same files as the explicit 1.0 """
        explicit = [path.read_text() for path in _write_tunn_inp(
        _num_of_vib_levels_chain, guess_file, potential_scal_factor_react=1.0,
        potential_scal_factor_prod=1.0)]
        omitted = [path.read_text() for path in tunnex_input_writing(
        guess_file, _num_of_vib_levels_chain[0], _num_of_vib_levels_chain[2],
        _num_of_vib_levels_chain[1], TI_NUM_LEV_REACT, TI_NUM_LEV_PROD)]
        assert omitted == explicit, f"expected {explicit}, got {omitted}"

    def test_original_file_and_data_are_not_modified(self, _num_of_vib_levels_chain, guess_file):
        """ Verifies that the TS guess file, the IRC data, the energy points and the frequencies remain unchanged """
        before = (_sha(guess_file), copy.deepcopy(_num_of_vib_levels_chain[0]),
        copy.deepcopy(_num_of_vib_levels_chain[1]), copy.deepcopy(_num_of_vib_levels_chain[2]))
        _write_tunn_inp(_num_of_vib_levels_chain, guess_file)
        after = (_sha(guess_file), _num_of_vib_levels_chain[0], _num_of_vib_levels_chain[1],
        _num_of_vib_levels_chain[2])
        assert after == before, f"expected {before}, got {after}"

    def test_accepts_str_and_path(self, _num_of_vib_levels_chain, guess_file):
        """ Verifies that the function accepts both string and Path arguments """
        from_path = _write_tunn_inp(_num_of_vib_levels_chain, guess_file)
        texts_path = [path.read_text() for path in from_path]
        from_str = tunnex_input_writing(str(guess_file), _num_of_vib_levels_chain[0],
        _num_of_vib_levels_chain[2], _num_of_vib_levels_chain[1],
        TI_NUM_LEV_REACT, TI_NUM_LEV_PROD, TI_SCALING_REACT, TI_SCALING_PROD)
        assert from_str == from_path, f"expected {from_path}, got {from_str}; test #1"
        assert [path.read_text() for path in from_str] == texts_path, "expected the same texts; test #2"

    def test_repeated_run_overwrites_and_does_not_append(self, workdir, _num_of_vib_levels_chain, guess_file):
        """ Verifies that a second run gives the same paths and texts and does not append to the old files """
        first = _write_tunn_inp(_num_of_vib_levels_chain, guess_file)
        texts = [path.read_text() for path in first]
        before = _snapshot_workdir(workdir)
        second = _write_tunn_inp(_num_of_vib_levels_chain, guess_file)
        assert second == first, f"expected {first}, got {second}; test #1"
        assert [path.read_text() for path in second] == texts, "expected the same texts; test #2"
        assert _snapshot_workdir(workdir) == before, f"expected {before}, got {_snapshot_workdir(workdir)}; test #3"

    def test_without_zpve_in_irc_data(self, _num_of_vib_levels_chain, guess_file):
        """ Verifies that the files are written if the IRC data has no ZPVE """
        _num_of_vib_levels_chain[0].zpve_energies_forward=None
        _num_of_vib_levels_chain[0].zpve_energies_reverse=None
        forward, backward = _write_tunn_inp(_num_of_vib_levels_chain, guess_file, irc_data=_num_of_vib_levels_chain[0])
        energies = [energy for _, energy in _num_of_vib_levels_chain[0].electronic_energies]
        for path in (forward, backward):
            assert all(_file_has_number(_numbers_findall(path), energy)
            for energy in energies), f"expected all the energies; {path.name}"

    def test_missing_guess_file_raises_and_creates_nothing(self, workdir, _num_of_vib_levels_chain):
        """
        Verifies that a missing TS guess file raises
        an error and creates no files. Expecting a FileNotFoundError
        """
        before = _snapshot_workdir(workdir)
        with pytest.raises(FileNotFoundError, match="was not found"):
            tunnex_input_writing(workdir / "nope.gjf", _num_of_vib_levels_chain[0],
            _num_of_vib_levels_chain[2], _num_of_vib_levels_chain[1], TI_NUM_LEV_REACT, TI_NUM_LEV_PROD)
        assert _snapshot_workdir(workdir) == before, f"expected {before}, got {_snapshot_workdir(workdir)}"


# ==============================================================================
# --- eckart_react_prod_optimization - normal behaviour and input validation ---
# ==============================================================================
class TestEckartReactProdOptimization:

    def test_full_flow_returns_two_output_files(self, react_guess_file, prod_guess_file, cmd_file, mode,
        fake_rp_software):
        """ Verifies that the full workflow returns two different existing output files in the work directory """
        react_out, prod_out = _eckart_rp_opt(react_guess_file, prod_guess_file, cmd_file, mode, run_software_key=True)
        assert react_out.exists(), f"expected {True}, got {react_out.exists()}; test #1"
        assert prod_out.exists(), f"expected {True}, got {prod_out.exists()}; test #2"
        assert react_out != prod_out, f"expected different files, got {react_out} and {prod_out}; test #3"
        assert react_out.parent == react_guess_file.parent, \
        f"expected {react_guess_file.parent}, got {react_out.parent}; test #4"
        assert prod_out.parent == prod_guess_file.parent, \
        f"expected {prod_guess_file.parent}, got {prod_out.parent}; test #5"

    def test_outputs_are_not_swapped(self, react_guess_file, prod_guess_file, cmd_file, mode, fake_rp_software):
        """
        Verifies that the first returned file is the reactant output
        and the second one is the product output
        """
        react_out, prod_out = _eckart_rp_opt(react_guess_file, prod_guess_file, cmd_file, mode, run_software_key=True)
        expected_react = (DATA / mode / RP_OPT_CASES[mode]["react_ok"]).read_text()
        expected_prod = (DATA / mode / RP_OPT_CASES[mode]["prod_ok"]).read_text()
        assert react_out.read_text() == expected_react, \
        f"expected {expected_react}, got {react_out.read_text()}; test #1"
        assert prod_out.read_text() == expected_prod, \
        f"expected {expected_prod}, got {prod_out.read_text()}; test #2"

    def test_input_files_are_created(self, react_guess_file, prod_guess_file, cmd_file, mode, fake_rp_software):
        """ Verifies that two new different input files with the optimization keywords are created """
        _eckart_rp_opt(react_guess_file, prod_guess_file, cmd_file, mode, run_software_key=True)
        react_input, prod_input = (Path(call[0]) for call in fake_rp_software.calls)
        tail = _eckart_tail(mode).strip()
        assert react_input.exists(), f"expected {True}, got {react_input.exists()}; test #1"
        assert prod_input.exists(), f"expected {True}, got {prod_input.exists()}; test #2"
        assert react_input != prod_input, f"expected different inputs, got {react_input} and {prod_input}; test #3"
        assert {react_input, prod_input}.isdisjoint({react_guess_file, prod_guess_file}), \
        f"expected new files, got {react_input} and {prod_input}; test #4"
        assert react_input.stem.endswith("_react"), f"expected the {"_react"} suffix, got {react_input.name}; test #5"
        assert prod_input.stem.endswith("_prod"), f"expected the {"_prod"} suffix, got {prod_input.name}; test #6"
        assert tail in react_input.read_text(), f"expected the keywords in {react_input.name}; test #7"
        assert tail in prod_input.read_text(), f"expected the keywords in {prod_input.name}; test #8"

    def test_input_files_come_from_own_guess(self, react_guess_file, prod_guess_file, cmd_file, mode,
        fake_rp_software):
        """
        Verifies that the reactant input is made from the reactant guess
        and the product input from the product guess
        """
        _eckart_rp_opt(react_guess_file, prod_guess_file, cmd_file, mode, run_software_key=True)
        react_input, prod_input = (Path(call[0]) for call in fake_rp_software.calls)
        assert react_guess_file.stem in react_input.name, \
        f"expected {react_guess_file.stem} in {react_input.name}; test #1"
        assert prod_guess_file.stem in prod_input.name, \
        f"expected {prod_guess_file.stem} in {prod_input.name}; test #2"

    def test_run_software_receives_expected_arguments(self, react_guess_file, prod_guess_file, cmd_file, mode,
        fake_rp_software):
        """ Verifies that run_software is called twice, the reactant first, with the expected arguments """
        expected = ("custom_out", pipeline.SLURM_MODE_KEY, pipeline.ENVR)
        react_out, prod_out = _eckart_rp_opt(react_guess_file, prod_guess_file, cmd_file, mode,
        run_software_key=True, filename_out_key=expected[0])
        calls = fake_rp_software.calls
        assert len(calls) == 2, f"expected {2} calls, got {len(calls)}; test #1"
        result = (_expected_output(mode, Path(calls[0][0])), _expected_output(mode, Path(calls[1][0])))
        assert result == (react_out, prod_out), f"expected {(react_out, prod_out)}, got {result}; test #2"
        for number, (_, command_file, filename_out, slurm_key, envr) in enumerate(calls, 1):
            assert Path(command_file) == cmd_file, f"expected {cmd_file}, got {Path(command_file)}; call #{number}, test #3"
            assert filename_out == expected[0], f"expected {expected[0]}, got {filename_out}; call #{number}, test #4"
            assert slurm_key == expected[1], f"expected {expected[1]}, got {slurm_key}; call #{number}, test #5"
            assert envr == expected[2], f"expected {expected[2]}, got {envr}; call #{number}, test #6"

    def test_missing_outputs_raise(self, react_guess_file, prod_guess_file, cmd_file, mode, monkeypatch):
        """ Verifies that missing precomputed outputs raise an error. Expecting a FileNotFoundError """
        monkeypatch.setattr(pipeline, "run_software", lambda *a, **kw: pytest.fail("should not be called"))
        with pytest.raises(FileNotFoundError, match="was not found"):
            _eckart_rp_opt(react_guess_file, prod_guess_file, cmd_file, mode, run_software_key=False)

    def test_missing_product_output_raises(self, react_guess_file, prod_guess_file, cmd_file, mode,
        fake_rp_software, monkeypatch):
        """
        Verifies that a missing product output raises an error even if the reactant output exists.
        Expecting a FileNotFoundError
        """
        _, prod_out = _eckart_rp_opt(react_guess_file, prod_guess_file, cmd_file, mode, run_software_key=True)
        prod_out.unlink()
        monkeypatch.setattr(pipeline, "run_software", lambda *a, **kw: pytest.fail("should not be called"))
        with pytest.raises(FileNotFoundError, match="was not found"):
            _eckart_rp_opt(react_guess_file, prod_guess_file, cmd_file, mode, run_software_key=False)

    def test_original_guess_files_are_not_modified(self, react_guess_file, prod_guess_file, cmd_file, mode,
        fake_rp_software):
        """ Verifies that the reactant and product guess files remain unchanged """
        before = (_sha(react_guess_file), _sha(prod_guess_file))
        _eckart_rp_opt(react_guess_file, prod_guess_file, cmd_file, mode, run_software_key=True)
        after = (_sha(react_guess_file), _sha(prod_guess_file))
        assert after == before, f"expected {before}, got {after}"

    def test_accepts_str_and_path(self, react_guess_file, prod_guess_file, cmd_file, mode, fake_rp_software):
        """ Verifies that the workflow accepts both string and Path arguments """
        from_path = _eckart_rp_opt(react_guess_file, prod_guess_file, cmd_file, mode, run_software_key=True)
        from_str = _eckart_rp_opt(str(react_guess_file), str(prod_guess_file), str(cmd_file),
            mode, run_software_key=True)
        assert from_path == from_str, f"expected {from_str}, got {from_path}"

    def test_invalid_mode_raises_and_creates_nothing(self, tmp_path):
        """
        Verifies that an invalid program mode raises an error
        and creates no files. Expecting a ValueError
        """
        react = tmp_path / "react.xyz"
        prod = tmp_path / "prod.xyz"
        react.write_text("dummy")
        prod.write_text("dummy")
        with pytest.raises(ValueError, match="program mode"):
            _eckart_rp_opt(react, prod, tmp_path / "run.sh", "psi4")
        result = sorted(tmp_path.iterdir())
        assert result == sorted([react, prod]), f"expected {sorted([react, prod])}, got {result}"

    @pytest.mark.parametrize("role", ["react", "prod"])
    def test_failed_calculation_is_detected(self, react_guess_file, prod_guess_file, cmd_file, mode,
        fake_rp_software, role):
        """
        Verifies that a failed reactant or product calculation is detected
        and raises an error. Expecting a RuntimeError
        """
        fake_rp_software.sample[role] = "failed"
        with pytest.raises(RuntimeError, match="terminated with an error"):
            _eckart_rp_opt(react_guess_file, prod_guess_file, cmd_file, mode, run_software_key=True)


# ===========================================================================
# --- eckart_potential_results - normal behaviour and input validation ---
# ===========================================================================
class TestEckartPotentialResults:

    def test_returns_irc_data_with_finite_energies(self, ts_output_file, _num_of_vib_levels_chain, mode, hess_parsing):
        """
        Verifies that the real TS output and the energy points give
        an IRCData with a valid Eckart potential
        """
        data = _eckart_results(ts_output_file, _num_of_vib_levels_chain[2], mode, hess_parsing)
        pairs = _pairs_irc_data(data)
        assert isinstance(data, IRCData), f"expected {IRCData}, got {type(data)}; test #1"
        assert pairs.ndim == 2 and pairs.shape[1] == 2, \
        f"expected (IRC, energy) pairs, got shape {pairs.shape}; test #2"
        assert len(pairs) > 1, f"expected several points, got {len(pairs)}; test #3"
        assert np.isfinite(pairs).all(), f"expected finite values, got {pairs[:3]}; test #4"

    def test_reader_receives_file_and_hess_flag(self, ts_output_file, _num_of_vib_levels_chain,
        mode, hess_parsing, mode_reader_eckart_spy):
        """ Verifies that the frequency reader gets the TS output file and the hess_parsing flag """
        _eckart_results(ts_output_file, _num_of_vib_levels_chain[2], mode, hess_parsing)
        result = [args for args, _ in mode_reader_eckart_spy]
        expected = [(ts_output_file, hess_parsing)]
        assert result == expected, f"expected {expected}, got {result}"

    def test_most_negative_frequency_is_used(self, ts_output_file, _num_of_vib_levels_chain,
        mode, hess_parsing, mode_reader_eckart_spy, params_eckart_spy):
        """
        Verifies that the transition state imaginary frequency
        is passed to eckart_potential_parameters
        """
        _eckart_results(ts_output_file, _num_of_vib_levels_chain[2], mode, hess_parsing)
        freq_and_modes = mode_reader_eckart_spy[0][1][0]
        expected = min(item[0] for item in freq_and_modes)
        result = params_eckart_spy[0][0][0]
        assert result == expected, f"expected {expected}, got {result}; test #1"
        assert all(result <= item[0] for item in freq_and_modes), \
        f"expected the lowest of {freq_and_modes}; test #2"
        assert result < 0, f"expected an imaginary (negative) frequency, got {result}; test #3"

    def test_barriers_are_computed_from_energy_points(self, ts_output_file, _num_of_vib_levels_chain,
        mode, hess_parsing, params_eckart_spy):
        """ Verifies that dV1 and dV2 are the TS - reactant and TS - product energies including ZPVE """
        _eckart_results(ts_output_file, _num_of_vib_levels_chain[2], mode, hess_parsing)
        ep = _num_of_vib_levels_chain[2]
        ts_total = ep.transition_state_el_energy[1] + ep.transition_state_zpve[1]
        expected = (ts_total - (ep.reactant_el_energy[1] + ep.reactant_zpve[1]),
        ts_total - (ep.product_el_energy[1] + ep.product_zpve[1]))
        result = params_eckart_spy[0][0][1:]
        assert np.allclose(result, expected, rtol=1e-12), f"expected {expected}, got {result}"

    def test_barriers_for_hand_calculated_energy_points(self, ts_output_file, mode, hess_parsing, params_eckart_spy):
        """ Verifies dV1 and dV2 for the small data set """
        energy_points_syn = EnergyPointsData(transition_state_el_energy=(0.0, 0.1),
                                         transition_state_zpve = (0.0, 0.0),
                                         reactant_el_energy = (-1.0, 0.0),
                                         reactant_zpve = (-1.0, 0.0),
                                         product_el_energy = (1.0, -0.1),
                                         product_zpve = (1.0, 0.0))
        _eckart_results(ts_output_file, energy_points_syn, mode, hess_parsing)
        result = params_eckart_spy[0][0][1:]
        expected = (0.1, 0.2)
        assert np.allclose(result, expected, atol=1e-12), f"expected {expected}, got {result}"

    def test_potential_is_built_from_the_computed_parameters(self, ts_output_file, _num_of_vib_levels_chain,
        mode, hess_parsing, params_eckart_spy, data_eckart_spy):
        """ Verifies that eckart_potential_data gets exactly the (A, B, L) returned by eckart_potential_parameters """
        data = _eckart_results(ts_output_file, _num_of_vib_levels_chain[2], mode, hess_parsing)
        params = params_eckart_spy[0][1]
        result = data_eckart_spy[0][0]
        assert tuple(result) == tuple(params), f"expected {tuple(params)}, got {tuple(result)}; test #1"
        assert data is data_eckart_spy[0][1], \
        "expected the result of eckart_potential_data to be returned; test #2"

    def test_swapped_reactant_and_product_give_a_different_potential(self, ts_output_file, _num_of_vib_levels_chain,
        mode, hess_parsing):
        """ Verifies that the reactant and product are not interchangeable (the barriers are different) """
        ep = _num_of_vib_levels_chain[2]
        ep_swapped = EnergyPointsData(transition_state_el_energy=ep.transition_state_el_energy,
                                                transition_state_zpve = ep.transition_state_zpve,
                                                reactant_el_energy = ep.product_el_energy,
                                                reactant_zpve = ep.product_zpve,
                                                product_el_energy = ep.reactant_el_energy,
                                                product_zpve = ep.reactant_zpve)
        base = _pairs_irc_data(_eckart_results(ts_output_file, ep, mode, hess_parsing))
        other = _pairs_irc_data(_eckart_results(ts_output_file, ep_swapped, mode, hess_parsing))
        assert base.shape != other.shape or not np.allclose(base, other), \
        "expected a different potential for the swapped reactant and product"

    def test_symmetric_barrier_has_the_barrier_height_at_maximum(self, ts_output_file, _num_of_vib_levels_chain,
        mode, hess_parsing):
        """ Verifies that for equal reactant and product energies the potential maximum equals the barrier height """
        ep = _num_of_vib_levels_chain[2]
        ep_sym = EnergyPointsData(transition_state_el_energy=ep.transition_state_el_energy,
                                                        transition_state_zpve = ep.transition_state_zpve,
                                                        reactant_el_energy = ep.reactant_el_energy,
                                                        reactant_zpve = ep.reactant_zpve,
                                                        product_el_energy = ep.reactant_el_energy,
                                                        product_zpve = ep.reactant_zpve)
        barrier = (ep.transition_state_el_energy[1] + ep.transition_state_zpve[1]
        - ep.reactant_el_energy[1] - ep.reactant_zpve[1])
        pairs = _pairs_irc_data(_eckart_results(ts_output_file, ep_sym, mode, hess_parsing))
        result = pairs[:, 1].max()
        assert np.allclose(result, barrier, rtol=1e-2), f"expected {barrier}, got {result}"

    def test_heat_effect_is_logged(self, ts_output_file, _num_of_vib_levels_chain, mode, hess_parsing, caplog):
        """ Verifies that the logged heat effect corresponds to the energies of the returned Eckart potential """
        with caplog.at_level(logging.INFO):
            data = _eckart_results(ts_output_file, _num_of_vib_levels_chain[2], mode, hess_parsing)
        expected, forbidden = _expected_heat_message(data.electronic_energies)
        assert expected in caplog.text, f"expected '{expected}' in the log, got {caplog.text}; test #1"
        assert forbidden not in caplog.text, f"expected no '{forbidden}' in the log; test #2"

    def test_outlier_check_covers_the_potential(self, ts_output_file, _num_of_vib_levels_chain,
        mode, hess_parsing, outlier_spy):
        """ Verifies that the outlier check runs for the electronic energy and for ZPVE only if it is present """
        data = _eckart_results(ts_output_file, _num_of_vib_levels_chain[2], mode, hess_parsing)
        has_zpve = data.zpve_energies_forward is not None and data.zpve_energies_reverse is not None
        expected = ["electronic energy", "ZPVE"] if has_zpve else ["electronic energy"]
        assert outlier_spy == expected, f"expected {expected}, got {outlier_spy}"

    def test_outlier_check_can_be_disabled(self, ts_output_file, _num_of_vib_levels_chain,
        mode, hess_parsing, outlier_spy):
        """ Verifies that outlier_check=False skips detect_outliers_local completely """
        _eckart_results(ts_output_file, _num_of_vib_levels_chain[2], mode, hess_parsing, outlier_check=False)
        assert outlier_spy == [], f"expected no calls, got {outlier_spy}"

    def test_smooth_potential_is_not_reported_as_outlier(self, ts_output_file, _num_of_vib_levels_chain,
        mode, hess_parsing):
        """ Verifies that the analytic potential passes the outlier check, i.e. both values of the flag work """
        with_check = _pairs_irc_data(_eckart_results(ts_output_file, _num_of_vib_levels_chain[2], mode, hess_parsing))
        without = _pairs_irc_data(_eckart_results(ts_output_file, _num_of_vib_levels_chain[2], mode, hess_parsing,
        outlier_check=False))
        assert np.array_equal(with_check, without), "expected the same potential for both values of the flag"

    def test_inputs_are_not_modified(self, ts_output_file, _num_of_vib_levels_chain, mode, hess_parsing):
        """ Verifies that the TS output file and the energy points remain unchanged """
        before = (_sha(ts_output_file), copy.deepcopy(_num_of_vib_levels_chain[2]))
        _eckart_results(ts_output_file, _num_of_vib_levels_chain[2], mode, hess_parsing)
        after = (_sha(ts_output_file), _num_of_vib_levels_chain[2])
        assert after == before, f"expected {before}, got {after}"

    def test_does_not_create_files(self, workdir, ts_output_file, _num_of_vib_levels_chain, mode, hess_parsing):
        """ Verifies that the computation does not create or remove files in the work directory """
        before = _snapshot_workdir(workdir)
        _eckart_results(ts_output_file, _num_of_vib_levels_chain[2], mode, hess_parsing)
        after = _snapshot_workdir(workdir)
        assert after == before, f"expected {before}, got {after}"

    def test_accepts_str_and_path(self, ts_output_file, _num_of_vib_levels_chain, mode, hess_parsing):
        """ Verifies that the function accepts both string and Path arguments """
        from_path = _pairs_irc_data(_eckart_results(ts_output_file,
            _num_of_vib_levels_chain[2], mode, hess_parsing))
        from_str = _pairs_irc_data(_eckart_results(str(ts_output_file),
            _num_of_vib_levels_chain[2], mode, hess_parsing))
        assert np.array_equal(from_path, from_str), f"expected {from_str[:3]}, got {from_path[:3]}"

    def test_is_repeatable(self, ts_output_file, _num_of_vib_levels_chain, mode, hess_parsing):
        """ Verifies that two runs on the same data give the same potential """
        first = _pairs_irc_data(_eckart_results(ts_output_file,
            _num_of_vib_levels_chain[2], mode, hess_parsing))
        second = _pairs_irc_data(_eckart_results(ts_output_file,
            _num_of_vib_levels_chain[2], mode, hess_parsing))
        assert np.array_equal(first, second), f"expected {first[:3]}, got {second[:3]}"

    def test_missing_ts_output_raises(self, workdir, _num_of_vib_levels_chain, mode, hess_parsing):
        """ Verifies that a missing TS output file raises an error. Expecting a FileNotFoundError """
        with pytest.raises(FileNotFoundError, match="was not found"):
            _eckart_results(workdir / "nope.out", _num_of_vib_levels_chain[2], mode, hess_parsing)

    def test_invalid_mode_raises_and_creates_nothing(self, tmp_path):
        """
        Verifies that an invalid program mode raises an error
        and creates no files. Expecting a ValueError
        """
        ts_out = tmp_path / "ts.out"
        ts_out.write_text("dummy")
        energy_points_syn = EnergyPointsData(transition_state_el_energy=(0.0, 0.1),
                                                 transition_state_zpve = (0.0, 0.0),
                                                 reactant_el_energy = (-1.0, 0.0),
                                                 reactant_zpve = (-1.0, 0.0),
                                                 product_el_energy = (1.0, -0.1),
                                                 product_zpve = (1.0, 0.0))
        with pytest.raises(ValueError, match="program mode"):
            _eckart_results(ts_out, energy_points_syn, "psi4")
        result = list(tmp_path.iterdir())
        assert result == [ts_out], f"expected {[ts_out]}, got {result}"