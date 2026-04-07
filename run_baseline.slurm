#!/bin/bash
#SBATCH --exclusive

# Usage:
#   sbatch --job-name=pgame  run_baseline.slurm pgame.py
#   sbatch --job-name=qdpg   run_baseline.slurm qdpg.py

SCRIPT=$1

echo "Running baseline script: $SCRIPT"

cd "$(dirname "$0")"
uv run python "$SCRIPT"