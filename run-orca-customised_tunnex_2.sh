#!/bin/bash
#SBATCH --job-name=orca_computations
#SBATCH --ntasks=8
#SBATCH --time=00-06:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --hint=nomultithread
#SBATCH --nodes=1-1
#SBATCH -p main
#SBATCH --qos=standard

# Load modules
module purge
module load ORCA/6.0.1-gompi-2023b-avx2
module load NBO/7.0

# Input file passed as the first argument
INPUT="$1"

if [[ -z "$INPUT" ]]; then
    echo "Usage: sbatch script.sh path/to/file.inp"
    exit 1
fi

if [[ ! -f "$INPUT" ]]; then
    echo "Error: input file not found: $INPUT"
    exit 1
fi

# Resolve the input file and its actual directory
INPUT_PATH=$(realpath "$INPUT")
INPUT_DIR=$(dirname "$INPUT_PATH")
INPUT_FILE=$(basename "$INPUT_PATH")
JOB_NAME="${INPUT_FILE%.inp}"

echo "Input file: $INPUT_PATH"
echo "Output directory: $INPUT_DIR"
echo "Job name: $JOB_NAME"

# Scratch directory
SCRATCH=/localscratch
TEMPDIR="$SCRATCH/$USER/orca/$SLURM_JOB_ID.$SLURM_JOB_NAME"

echo "Creating scratch directory: $TEMPDIR"

export NBOEXE=/trinity/shared/easybuild/software/NBO/7.0/bin/nbo7.i4.exe
export orcabin=$(which orca)

if [[ -z "$orcabin" || ! -x "$orcabin" ]]; then
    echo "Error: ORCA executable not found."
    exit 1
fi

if [[ -d "$TEMPDIR" ]]; then
    echo "Error: scratch directory already exists: $TEMPDIR"
    exit 1
fi

mkdir -p "$TEMPDIR" || exit 1

# Copy input file to scratch, keeping the .inp extension
cp -p "$INPUT_PATH" "$TEMPDIR/$INPUT_FILE" || exit 1

# Link scratch directory in the input file's directory
ln -s "$TEMPDIR" "$INPUT_DIR/scratch_$SLURM_JOB_ID"

echo "ORCA executable: $orcabin"
echo "Execution begun: $(date)"

# Run ORCA in scratch; write the main log directly to the input directory
cd "$TEMPDIR" || exit 1

"$orcabin" "$INPUT_FILE" > "$INPUT_DIR/$JOB_NAME.out" 2>&1
ORCA_EXIT=$?

cd "$INPUT_DIR" || exit 1

if [[ "$ORCA_EXIT" -eq 0 ]]; then
    echo "ORCA calculation finished successfully."
else
    echo "ERROR: ORCA exited with code $ORCA_EXIT"
fi

# Move calculation results into the directory containing the input file.
# Keep the original .inp file unchanged.
find "$TEMPDIR" -maxdepth 1 -type f ! -name "$INPUT_FILE" \
    ! -name "$JOB_NAME.out" \
    -exec mv -t "$INPUT_DIR" -- {} +

# Remove temporary files left in scratch
rm -f "$TEMPDIR"/*tmp* "$TEMPDIR"/*proc*

# Remove scratch directory and symbolic link
rm -rf "$TEMPDIR"
rm -f "$INPUT_DIR/scratch_$SLURM_JOB_ID"

echo
echo "Calculation statistics:"
seff "$SLURM_JOB_ID"

echo
echo "Output log: $INPUT_DIR/$JOB_NAME.out"
echo "Execution finished: $(date)"
echo "==========================================="

exit "$ORCA_EXIT"