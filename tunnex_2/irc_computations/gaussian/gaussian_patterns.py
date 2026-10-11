# --- Modules ---
import re


# =============================================================================================
# --- Gaussian file parcing patterns (Cartesian Gaussian geometry is expected in all cases) ---
# --- String pattenrs ---
# =============================================================================================
METHOD_TAIL_TS = r"opt=(ts,calcfc,noeigen) freq=NoRaman"
# A line to add to the input-file for the optimization of the transition state geometry
# METHOD_TAIL_IRC is constructed inside the create_gauss_irc_input function
METHOD_TAIL_REACT_PROD = r"Opt freq=NoRaman"
# This pattern is used for the optimization of the reactant and product geometries
PATTERN_COMMENT = r"Comment"
# A line defining the comment section
IRC_SPLIT_MARKER_FORWARD = r"Calculation of FORWARD path complete."
# It is a marker to split the IRC file to the forward (before the marker) and reverse (after the marker) IRC fragments
IRC_SPLIT_MARKER_REVERSE = r"Calculation of REVERSE path complete."
# It is a marker to identify the end the IRC computations (not used now)
# Since Gaussian computes the Hessian and prints it at each IRC point,
# no special pattern for Hessian computations are needed (LINE_HESS_COMP)


# ============================================================================
# --- Input file parsing ---
# ============================================================================
PATTERN_COMPUTATIONAL_METHOD_GAUSS = re.compile(r'^(#\S+)\s+(\S+).*$')
# It searches for a method line in a gaussian input file. The input line should be '#p METHOD/BASIS ...'
PATTERN_GEOMETRY_LINE_INPUT_GAUSS = re.compile(
    r"^\s*([A-Z][a-z]?|\d+)\s+"                       # Atom [captured]
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"           # X [captured]
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"           # Y [captured]
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*$",         # Z [captured]
    re.MULTILINE)
# It searches the pattern of type 'C 0.000000 1.234567 -0.123456'
# or '6 0.000000 1.234567 -0.123456' in the input files
# (no block patterns are used for the input file because these files are simple)
# it is identical to one in ORCA


# ============================================================================
# --- Optimization output file parsing ---
# ============================================================================
PATTERN_OPTIMIZED_SPECIES_GEOMETRY_BLOCK = re.compile(r"Standard orientation:\s*-+\s*Center.*?\s*-+\s*"
    r"(.*?)\s*-+\s*Rotational constants", re.DOTALL)
# It searches the blocks containing the geometry in the optimization output files
PATTERN_GEOMETRY_LINE_OUTPUT_GAUSS = re.compile(
    r'^\s*\d+\s+'                    # Center number
    r'(\d+)\s+'                      # Atomic number [captured]
    r'(?:\d+\s+)?'                   # Atomic type (optional)
    r'([-+]?\d*\.\d+)\s+'            # X [captured]
    r'([-+]?\d*\.\d+)\s+'            # Y [captured]
    r'([-+]?\d*\.\d+)\s*$',          # Z [captured]
    re.MULTILINE)
# It searches the line patterns of type '1 6 0.000000 0.635864 0.000000'
# and '1 6 0 0.000000 0.635864 0.000000' in the optimization output file (it also works for the IRC path)
# it is identical to one in ORCA
PATTERN_EL_ENERGY_SINGLE_POINT = re.compile(r"SCF Done:\s+E\([^)]+\)\s*=\s*(-?\d+\.\d+)")
# It searches for the electronic energy value in the optimization output file
PATTERN_MASS_LINES_AND_VALUES = re.compile(r"(Atom\s+\d+\s+has atomic number\s+\d+\s+and mass\s+"r"([0-9]+(?:\.[0-9]+)?))")
# It searches for atom masses and lines (it works only for the optimization output file)


# ============================================================================
# --- IRC output file parsing ---
# ============================================================================
PATTERN_IRC_POINT_REACT_PROD_GEOMETRY = re.compile(r"\*{4}\s+End of Projected Frequency Analysis\s+\*{4}")
# This pattern opens the geometry table in the IRC output file.
# It was designed to find the last geometry at each IRC end (reactant and product) but is not used anywhere now
PATTERN_IRC_INPUT_ORIENTATION_BLOCK = re.compile(r'Input orientation:\s*.*?\n\s*-{5,}\s*\n\s*Distance matrix', re.DOTALL)
# It searches for the transition state geometry in the IRC file (it is the first 'Input orientation' block)
PATTERN_IRC_CURRENT_STRUCTURE_BLOCK = re.compile(
    r'CURRENT STRUCTURE\s+Cartesian Coordinates \(Ang\):.*?NET REACTION COORDINATE UP TO THIS POINT\s*=\s*[-+]?\d+(?:\.\d+)?',
    re.DOTALL)
# It searches for the IRC point geometries in the IRC file (the 'Current structure' blocks)
PATTERN_TS_ENERGY_IRC = re.compile(r"Energies reported relative to the TS energy of\s+([-+]?\d+\.\d+)")
# It searches the energy of the transition state in the irc output file
PATTERN_EL_ENERGY_IRC = re.compile(r"^\s*\d+\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)", re.MULTILINE)
# It searches the IRC lines in the electronic irc table at the end of the IRC output file
PATTERN_NET_REACTION_COORDINATE = re.compile(r"NET REACTION COORDINATE UP TO THIS POINT\s*=\s*(-?\d+\.\d+)")
# It searches for the IRC reaction coordinate in the structure block of IRC file
# it should be something like a 'NET REACTION COORDINATE UP TO THIS POINT = 0.42130' line


# ============================================================================
# --- Frequencies, modes, and Hessians parsing ---
# ============================================================================
PATTERN_HESSIAN_IN_IRC_BLOCK = re.compile(r"Force constants in Cartesian coordinates:\s*\n(.*?)Leave Link", re.DOTALL)
# It searches for a Hessian in the IRC file
PATTERN_HESSIAN_IN_OPT_BLOCK = re.compile(r"Force constants in Cartesian coordinates:\s*\n(.*?)FormGI", re.DOTALL)
PATTERN_FREQ_MODES_BLOCK = re.compile(r"and normal coordinates:(.*?)-+\s*\n\s*- Thermochemistry -", re.DOTALL)
# It searches for the block containing frequencies and vector coordinates.
# It is needed to compute the attempt frequency
# Frequencies, normal modes, and Hessian values are stored in Gaussian in simple lines,
# so no PATTERN_VIBR_FREQUENCIES, PATTERN_NORMAL_MODES_BLOCK, and PATTERN_HESSIAN_LINE are needed
PATTERN_FREQ_MODES_LINES = re.compile(r"-?\d+\.\d+")
# It is a pattern to search the frequencies in the 'Frequencies --  -2091.8757    36.5420    306.8203' line.
# It is also used for values of the frequency vectors,
# e.g., " 1   6     0.00   0.00  -0.08    -0.06   0.17   0.00     0.00   0.00  -0.12"
# In fact, it can find a number of the 1.23 or -1.23 type in any line