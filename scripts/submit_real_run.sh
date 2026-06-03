#!/bin/bash
# Submit a real-data pipeline run via SLURM.
#
# Required env vars (pass via --export=ALL,...):
#   DATA_CFG   - Hydra data config name  (e.g. mnist_subset, fashionmnist_subset, cifar10_subset)
#
# Optional env vars:
#   SEED            - random seed (default: 7)
#   DATA_ROOT       - path to dataset root (default: ./outputs/datasets)
#   OUTPUT_ROOT     - path to output root  (default: ./outputs)
#   PYTHON          - python executable    (default: python)
#   DATA_DOWNLOAD_OVERRIDE - set to empty string to skip data.download=false override
#                            (needed for datasets that have no torchvision download key,
#                            e.g. Galaxy10 loaded from a local .h5 file)
#   EXTRA_OVERRIDES - extra Hydra overrides (e.g. "model.bottleneck.enabled=true train.bottleneck_warmup_epochs=2")

#SBATCH --job-name=mcwola_real
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.out

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO}/outputs/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/outputs}"

mkdir -p "$OUTPUT_ROOT"
cd "$REPO"
MPLCONFIGDIR=/tmp/mpl_$$ PYTHONPATH=src "$PYTHON" scripts/run_real.py \
    experiment=realdata_hidden_priors \
    data="${DATA_CFG:-mnist_subset}" \
    seed="${SEED:-7}" \
    data.data_root="$DATA_ROOT" \
    ${DATA_DOWNLOAD_OVERRIDE-data.download=false} \
    output_root="$OUTPUT_ROOT" \
    ${EXTRA_OVERRIDES:-}
