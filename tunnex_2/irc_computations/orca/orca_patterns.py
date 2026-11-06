# --- Modules ---
import re


# =============================================================================================
# --- ORCA file parcing patterns (Cartesian ORCA geometry is expected in all cases) ---
# --- String pattenrs ---
# =============================================================================================
BLOCK_TAIL_TS = r"""OptTS Freq

%geom
Calc_Hess true
NumHess true
end
"""
# A block to add to the input-file for the optimization of the geometry of the transition state
BLOCK_TAIL_IRC = r"""IRC

%irc
    MaxIter 100
    PrintLevel 1
    Direction both
    InitHess calc_anfreq
    hessMode 0
    Init_Displ length
    Scale_Init_Displ 0.10
    Scale_Displ_SD   0.10
    Adapt_Scale_Displ false
end
"""
# It is needed to be put inside the IRC input-file for ORCA.
# To keep the step size exactly 0.1 bohr (ORCA cannot follow mass-weighted coordinates now)
# the following keywords can be added:
#----------------------#
# SD_ParabolicFit false
# Do_SD_Corr false
#----------------------#
# However, it may require a lot more iterations to compute the IRC, MaxIter 100 is often exceeded.
# Thus, it is not a default option.
LINE_TAIL_REACT_PROD = r"""Opt Freq
"""
# A line to add to the input-file for the optimization of the geometry of the transition state
# No PATTERN_COMMENT, IRC_SPLIT_MARKER_FORWARD, and IRC_SPLIT_MARKER_REVERSE are needed for ORCA program
LINE_HESS_COMP = r"""Freq # E {}
"""
# It is needed for a Hessian computation input file for ORCA


# ============================================================================
# --- Input file parsing ---
# ============================================================================
PATTERN_COMPUTATIONAL_METHOD_ORCA = re.compile(r'^\s*!\s+(\S+)\s+(\S+)')
# It searches for a method line in an orca input file. The input line should be '! METHOD BASIS ...'
# ORCA stores the coordinates uniformly, so the unified PATTERN_GEOMETRY_LINE_ORCA is
# used instead PATTERN_GEOMETRY_LINE_INPUT_ORCA


# ============================================================================
# --- Optimization output file parsing ---
# ============================================================================
PATTERN_OPTIMIZED_SPECIES_GEOM_BLOCK = re.compile(r'CARTESIAN COORDINATES \(ANGSTROEM\)\n-+\n'
    r'((?:[ \t]*(?:[A-Za-z]{1,2}|\d+)(?:[ \t]+-?\d+\.\d+){3}[ \t]*\n?)+)')
# It searches the blocks containing the geometry in the optimization output files
PATTERN_GEOMETRY_LINE_ORCA = re.compile(
    r"^\s*([A-Z][a-z]?|\d+)\s+"                       # Atom [captured]
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"           # X [captured]
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"           # Y [captured]
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*$",         # Z [captured]
    re.MULTILINE)
# It searches the line patterns of type '1 6 0.000000 0.635864 0.000000'
# and '1 6 0 0.000000 0.635864 0.000000' in the optimization output file (it also works for the IRC path)
# it is identical to one in Gaussian
PATTERN_EL_ENERGY_ZPVE_SINGLE_POINT = re.compile(r'Electronic energy\s+\.\.\.\s+([-+]?\d+\.\d+)\s+Eh\s+'
    r'Zero point energy\s+\.\.\.\s+([-+]?\d+\.\d+)\s+Eh')
# It searches for the electronic energy and ZPVE in the optimization output file
PATTERN_MASS_BLOCK = re.compile(
    r'CARTESIAN COORDINATES \(A\.U\.\)\n'                                        # CARTESIAN COORDINATES (A.U.)
    r'-+\n'                                                                      # ----------------------------
    r'[ \t]*NO[ \t]+LB[ \t]+ZA[ \t]+FRAG[ \t]+MASS[ \t]+X[ \t]+Y[ \t]+Z[ \t]*\n' # NO LB ZA FRAG MASS X Y Z
    r'((?:'                                                                      # Start of atom data
    r'[ \t]*\d+[ \t]+'                                                           # NO
    r'(?:[A-Za-z]{1,2}|\d+)[ \t]+'                                               # LB
    r'[-+]?\d+(?:\.\d+)?[ \t]+'                                                  # ZA
    r'\d+[ \t]+'                                                                 # FRAG
    r'[-+]?\d+(?:\.\d+)?[ \t]+'                                                  # MASS
    r'[-+]?\d+(?:\.\d+)?[ \t]+'                                                  # X
    r'[-+]?\d+(?:\.\d+)?[ \t]+'                                                  # Y
    r'[-+]?\d+(?:\.\d+)?[ \t]*'                                                  # Z
    r'\n?'                                                                       # [the whole block is captured]
    r')+)')
# It searches for atom masses and lines


# ============================================================================
# --- IRC output file parsing ---
# ============================================================================
PATTERN_IRC_COORDINATES_BLOCK = re.compile(
    r"E\s+([-+]?\d+(?:\.\d+)?)\s*\n"             # E + energy; energy is [captured]
    r"("                                         # Start of coordinates block
    r"(?:"
    r"[ \t]*(?:[A-Za-z]{1,2}|\d+)"               # Atom [captured]
    r"[ \t]+[-+]?\d+(?:\.\d+)?"                  # X [captured]
    r"[ \t]+[-+]?\d+(?:\.\d+)?"                  # Y [captured]
    r"[ \t]+[-+]?\d+(?:\.\d+)?"                  # Z [captured]
    r"[ \t]*\n?"
    r")+"
    r")")
# It searches the coordinates block in the IRC output file
# No PATTERN_IRC_INPUT_ORIENTATION_BLOCK, PATTERN_IRC_CURRENT_STRUCTURE_BLOCK,
# PATTERN_TS_ENERGY_IRC, PATTERN_EL_ENERGY_IRC,
# and PATTERN_NET_REACTION_COORDINATE are needed for ORCA program


# ============================================================================
# --- Frequencies, modes, and Hessians parsing ---
# ============================================================================
PATTERN_HESSIAN_BLOCK = re.compile(r"\$hessian\s+(\d+)(.*?)\$vibrational_frequencies", re.DOTALL)
# It is needed to search for a hessian block in a Hessian file
# ORCA stores a uniform Hessian file, so splitting to PATTERN_HESSIAN_IN_IRC_BLOCK
# PATTERN_HESSIAN_IN_OPT_BLOCK and is not needed
PATTERN_VIBR_BLOCK = re.compile(r'-+\s*VIBRATIONAL FREQUENCIES\s*-+(.*?)\-+\s*NORMAL MODES\s*-+', re.DOTALL)
# It searches for the vibrational frequencies block in output files
PATTERN_VIBR_FREQUENCIES = re.compile(r'^\s*\d+:\s+([-+]?\d+(?:\.\d+)?)\s+cm\*\*-1', re.MULTILINE)
# It searches for the vibrational frequencies in output files
PATTERN_NORMAL_MODES_BLOCK = re.compile(
    r'------------\s*'                                                # Separator
    r'NORMAL MODES\s*'                                                # Normal modes header
    r'------------\s*'                                                # Separator
    r'.*?'                                                            # Text before normalized vectors
    r'Thus, these vectors are normalized but \*not\* orthogonal\s*'   # Vector description
    r'(.*?)'                                                          # Normal modes data
    r'\s*-----------\s*'                                              # Separator
    r'IR SPECTRUM', re.DOTALL)                                        # End of normal modes block 
                                                                      # [the whole block is captured]
# It searches for the vibrational normal modes in output files
PATTERN_HESSIAN_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+)(?:[Ee][-+]?\d+)?"
# It is a pattern to search the Hessian values in an ORCA Hessian file
# In fact, it can find a number of the 1.23, -1.23, 1.23E-05 etc. type in any line
PATTERN_HESSIAN_LINE = re.compile(rf"^\s*(\d+)\s+((?:{PATTERN_HESSIAN_FLOAT}\s+){{4}}{PATTERN_HESSIAN_FLOAT})\s*$",re.MULTILINE)
# It is needed to extract a Hessian line from an ORCA Hessian file