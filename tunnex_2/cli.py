# --- Modules ---
import argparse
from typing import NoReturn

from tunnex_2.constants_and_dataclasses.dataclasses import FindIRCConfig # type: ignore

from tunnex_2.main import main # type: ignore


# --- Dealing with key errors ---
def _parser_error(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(f"{message}\n"
    "Use --help for more information.")


# --- Parsing command line arguments ---
def cli() -> None:

    # --- Defining the program name ---
    parser = argparse.ArgumentParser(prog="tunnex_2")

    # --- Defining the files which will be transferred to the program ---
    parser.add_argument("files", nargs="*",
                        help="input files for tunnex_2 computations")

    # --- Defining the bool arguments for the program ---
    parser.add_argument("-o", "--orca", action="store_true",
                    help="use ORCA to compute the IRC data (default is Gaussian)")
    parser.add_argument("-z", "--no-zpve", "--no_zpve", action="store_true",
                    help="do not calculate ZPVE")  
    parser.add_argument("-f", "--calcfc", action="store_true",
                    help="faster but less accurate method for projected frequencies (only for Gaussian)")
    parser.add_argument("-y", "--hybrid", action="store_true",
                    help="scale IRC curve using stationary points (cannot be used with Eckart potential)")
    parser.add_argument("-e", "--eckart", action="store_true",
                    help="use Eckart potential to compute the IRC data")
    parser.add_argument("-i", "--iso", action="store_true",
                    help="use computed Hessians to calculate frequencies (isotopic substitution analysis)")
    parser.add_argument("-q", "--qmt-only", "--qmt_only", action="store_true",
                    help="run the QMT computations for the already computed IRC path")
    parser.add_argument("-p", "--no-qmt", "--no_qmt", action="store_true",
                    help="compute only IRC data without running the QMT computations")
    parser.add_argument("-c", "--comp", action="store_true",
                    help="use already computed files instead of running calculations")

    args = parser.parse_args()
    
    # --- Tunnex 2.0 supports only Gaussian and ORCA [01.10.2026] ---
    prog_mode_key = 'orca' if args.orca else 'gaussian'

    # --- Building the input FindIRCConfig dataclass object ---
    configs_data = FindIRCConfig(
    prog_mode = prog_mode_key,
    proj_freq = not args.no_zpve,
    calc_all = not args.calcfc,
    hybrid_mode = args.hybrid,
    eckart = args.eckart,
    hess_parsing_flag = args.iso,
    qmt_module = not args.no_qmt)

    # --- Running only the QMT module: it supportns only one file (Tunnex input file)
    # and cannot be run with IRC calculation options or the --no_qmt flag ---
    if args.qmt_only:
        # --- Defining the forbidden combinations ---
        qmt_only_forbidden_flags = (args.orca or args.no_zpve
        or args.calcfc or args.hybrid or args.eckart or args.iso or args.comp)

        if qmt_only_forbidden_flags:
            _parser_error(parser, "--qmt_only cannot be combined with IRC calculation options")

        if args.no_qmt:
            _parser_error(parser, "--qmt_only cannot be combined with --no_qmt")
        
        # --- Defining the required number of files (QMT module) ---
        if len(args.files) != 1:
            _parser_error(parser,
                f"--qmt_only requires exactly 1 input file, got {len(args.files)}.\n"
                "Expected: Tunnex input file")
        configs_data.tunnex_input = args.files[0]

    else:
        # --- Defining the forbidden combinations ---
        if args.orca and args.calcfc:
            _parser_error(parser, "--calcfc cannot be used with --orca (Gaussian only)")
            # --calcfc works only for Gaussian
            # did not forbid --calcfc and --no_zpve combination,
            # because --calcfc is expected to compute IRC faster but less accurate
            # (even without the computation of projected frequencies)
      
        if args.eckart and args.no_zpve:
            _parser_error(parser, "--no_zpve cannot be used with --eckart")
            # --no_zpve has no effect when used with --eckart, so this combination is forbidden to avoid confusion

        if args.eckart and args.calcfc:
            _parser_error(parser, "--calcfc cannot be used with --eckart")
            # --calcfc has no effect when used with --eckart, so this combination is forbidden to avoid confusion

        if args.eckart and args.hybrid:
            _parser_error(parser, "--hybrid cannot be used with --eckart")
            # --hybrid has no effect when used with --eckart, so this combination is forbidden to avoid confusion
        
        # --- Defining the required number of files (--eckart and --comp) ---
        if args.eckart and args.comp:
            if len(args.files) != 4:
                _parser_error(parser,
                    f"--eckart with --comp requires exactly 4 input files, got {len(args.files)}.\n"
                    "Expected order: TS input, TS computed, reactant computed, product computed")
            configs_data.ts_input_file = args.files[0]
            configs_data.ts_computed_file = args.files[1]
            configs_data.react_input_file = args.files[2]
            configs_data.react_computed_file = args.files[2] # intentionally assigned the same file
            configs_data.prod_input_file = args.files[3]
            configs_data.prod_computed_file = args.files[3] # intentionally assigned the same file
            # The guessed transition state, reactant, and product input files are kept for later use
        
        # --- Defining the required number of files (default mode and --comp) ---
        elif args.comp:
            allowed_number_of_files = (2, 3, 5)
            if len(args.files) not in allowed_number_of_files:
                _parser_error(parser,
                    f"--comp requires 2, 3 or 5 input files, got {len(args.files)}.\n"
                    "Expected order:\n"
                    "  2 files: TS input, TS computed\n"
                    "  3 files: TS input, TS computed, IRC computed\n"
                    "  5 files: TS input, TS computed, IRC computed, reactant computed, product computed\n"
                    "Reactant and product must be given together and only after the IRC file")
            configs_data.ts_input_file, configs_data.ts_computed_file = args.files[0], args.files[1]
            configs_data.irc_computed_file = args.files[2] if len(args.files) > 2 else None
            configs_data.react_computed_file = args.files[3] if len(args.files) > 3 else None
            configs_data.prod_computed_file = args.files[4] if len(args.files) > 4 else None
            # The guessed transition state input file is kept for later use
        
        # --- Defining the required number of files (--eckart) ---
        elif args.eckart:
            if len(args.files) != 3:
                _parser_error(parser,
                    f"--eckart requires exactly 3 input files, got {len(args.files)}.\n"
                    "Expected order: TS input, reactant input, product input")
            configs_data.ts_input_file, configs_data.react_input_file, configs_data.prod_input_file = args.files
        
        # --- Defining the required number of files (default mode) ---
        else:
            if len(args.files) != 1:
                _parser_error(parser,
                    f"Default mode requires exactly 1 input file, got {len(args.files)}.\n"
                    "Expected: TS input")
            configs_data.ts_input_file = args.files[0]

    main(configs=configs_data)