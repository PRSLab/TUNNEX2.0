# --- Modules ---
from pathlib import Path

from tunnex_2.irc_computations.ancillary_functions.attempt_freq_calc import detect_outliers_local  # type: ignore
from tunnex_2.irc_computations.ancillary_functions.interface import setup_logger  # type: ignore

from tunnex_2.qmt_computations.core.potential import Potential  # type: ignore
from tunnex_2.qmt_computations.data_reader_writer.reader import read_input  # type: ignore
from tunnex_2.qmt_computations.data_reader_writer.writer import write_output, build_report  # type: ignore
from tunnex_2.qmt_computations.pipeline.qmt_computations_pipeline import qmt_computations_pipeline  # type: ignore


# --- Defining the logging function ---
logger = setup_logger()


# --- Defining the main function to compute the QMT half-lives ---
def qmt_computations_main(input_path: str | Path) -> None:

    # --- Checking the input file and creating the output file ---
    input_path_read = Path(input_path).resolve()

    if not input_path_read.is_file():
        raise FileNotFoundError(f"File not found: {input_path_read}.")

    output_path = input_path_read.with_name(input_path_read.stem + "_tunnex_2.out")

    # --- Reading the parameters and IRC data from the input file ---
    params, irc = read_input(input_path_read)

    detect_outliers_local(irc.E_x, irc.E_y, 'electronic energy')
    detect_outliers_local(irc.ZPVE_x, irc.ZPVE_y, 'ZPVE')

    # --- Building the potential function based on the input IRC datapoints ---
    potential = Potential(
        irc.E_x,
        irc.E_y,
        irc.ZPVE_x,
        irc.ZPVE_y,
        params.E0,
        params.ZPVE0,
        params.potential_scaling_factor)

    # --- Computing the interpolated IRC, vibrational levels, tunneling data, temeprature averaged rates,
    # and Arrhenius plot values ---
    interpolated_irc, level_results, res_temper_aver, arrhenius_rate_values = qmt_computations_pipeline(params,
    potential)

    # --- Forming the output file, writing it, and reporting the succesful creating of the output file ---
    general_report = build_report(
        level_results,
        res_temper_aver,
        params.T,
        arrhenius_rate_values,
        interpolated_irc,
        potential.offset)
    
    write_output(output_path, general_report)
    logger.info("Results were saved to: %s.", output_path)