# --- Modules ---
from tunnex_2.constants_and_dataclasses.dataclasses import FindIRCConfig # type: ignore

from tunnex_2.irc_computations.ancillary_functions.interface import log_run_start, setup_logger # type: ignore

from tunnex_2.irc_computations.module_irc_computations import irc_computations_main # type: ignore

from tunnex_2.qmt_computations.module_qmt_computations import qmt_computations_main # type: ignore


# --- Defining the logging function ---
logger = setup_logger()


# --- Defining the main function to compute IRC and run TUNNEX 2.0 ---
def main(configs: FindIRCConfig) -> None:

    if configs.tunnex_input is None: 
        log_run_start(setup_logger(), configs.ts_input_file)
        tunnex_forward, tunnex_backward = irc_computations_main(configs)

        if configs.qmt_module:
            qmt_computations_main(tunnex_forward)
            qmt_computations_main(tunnex_backward)

    else:
        log_run_start(setup_logger(), configs.tunnex_input)
        qmt_computations_main(configs.tunnex_input)


# How to install (requires-python = ">=3.10"):
# 1. In terminal type: conda create -n tunnex_2 python=3.13                                          # Creating a conda environment
# 2. In terminal type: conda activate tunnex_2                                                       # Activating a conda environment
# 3. Go to a project folder (where pyproject.toml is located) and in terminal type: pip install -e . # Installing TUNNEX 2.0
# 4. In terminal type: 'tunnex_2 --help'                                                             # Checking the installed files
# 5. Use TUNNEX 2.0 as 'tunnex_2 your_filename.txt'.                                                 # Done. You're breathtaking!


# An alternative way to run the code for debugging etc.
#if __name__ == "__main__":

    #folder = Path("floder_name")
    #file = folder / "filename.inp"
    #program = 'orca'
    #configs = FindIRCConfig(ts_input_file=file, prog_mode=program)
    
    #main(configs)