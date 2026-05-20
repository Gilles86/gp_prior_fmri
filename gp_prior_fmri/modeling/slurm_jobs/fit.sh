#!/bin/bash
#SBATCH --job-name=gp_prior
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=360:00
#SBATCH --output=/dev/null

# Usage:
#   sbatch --array=1-10,12-22,24-41%10 fit.sh neural_priors NPCr "--joint_hyperparams --prior_params mu --tag joint_mu"
#   sbatch --array=1-10%10              fit.sh tms_risk      NPCr "--joint_hyperparams --prior_params mu --tag joint_mu_tms"

PARTICIPANT=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
ADAPTER=${1:-neural_priors}
ROI=${2:-NPCr}
EXTRA_FLAGS="${3:-}"

case "$ADAPTER" in
    neural_priors)  BIDS=/shares/zne.uzh/gdehol/ds-neuralpriors ;;
    tms_risk)       BIDS=/shares/zne.uzh/gdehol/ds-tmsrisk      ;;
    *) echo "Unknown adapter $ADAPTER" >&2; exit 1 ;;
esac

# Tag log file by smoothed + tag for easy grep.
SMOOTH=""
[[ "$EXTRA_FLAGS" == *"--smoothed"* ]] && SMOOTH=".smoothed"
TAG="default"
[[ "$EXTRA_FLAGS" =~ --tag[[:space:]]+([^[:space:]]+) ]] && TAG="${BASH_REMATCH[1]}"
LOGFILE="$HOME/logs/gp_${ADAPTER}_${ROI}${SMOOTH}.${TAG}_s${PARTICIPANT}.txt"
mkdir -p "$(dirname "$LOGFILE")"
scontrol update JobId=$SLURM_JOB_ID JobName="gp_${ADAPTER}.${ROI}${SMOOTH}.${TAG}_s${PARTICIPANT}"
exec > "$LOGFILE" 2>&1

export PYTHONUNBUFFERED=1
PYTHON=$HOME/data/conda/envs/gp_prior_fmri_cuda/bin/python

$PYTHON -m gp_prior_fmri.modeling.fit \
    $PARTICIPANT \
    --adapter $ADAPTER \
    --bids_folder $BIDS \
    --roi $ROI \
    $EXTRA_FLAGS
