# Multiclass Classification without Labels via Posterior Simplex Geometry

Anonymous code release accompanying the paper of the same title.

This repository implements a prior-free recovery procedure for multiclass
classification from multiple unlabeled mixtures, exploiting the simplex
geometry of the Bayes-optimal mixture posterior. Both post-hoc simplex
fitting and an architectural bottleneck variant are provided, alongside the
baselines reported in the paper (supervised oracle, OvR, Wei CCM, oracle
simplex with known mixing matrix, KSBS-Demix).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Tested on Python 3.10-3.12 with PyTorch >= 2.2 and a CUDA GPU. Pure-CPU runs
are possible on the small-image benchmarks.

## Data

| Dataset | How it is obtained |
| --- | --- |
| MNIST, Fashion-MNIST, CIFAR-10 | Downloaded automatically by `torchvision` into `./outputs/datasets` on first run. |
| Galaxy10 DECaLS | Manual download required. Place `Galaxy10_DECals.h5` under `./outputs/datasets/` (see `configs/data/galaxy10.yaml`). The dataset is publicly available from the official `astroNN` distribution. |

## Reproducing the paper

All commands below assume the project root and write under `outputs/`.

### Posterior-simplex geometry figure

Run the CIFAR-10 `K=3, M=6` experiment used for the simplex-geometry figure:

```bash
python scripts/run_real.py experiment=realdata_hidden_priors \
  data=cifar10_subset model=cnn_large seed=7 \
  data.K=3 data.M=6 data.class_subset='[0,1,2]'
```

Each run writes diagnostic plots directly under `outputs/<experiment>/<run>/plots/`:
- `mspace_posterior.png` -- M-space source posterior geometry (paper figure): raw source
  posteriors `g(x)` or reconstructed bottleneck posteriors `alpha(x) @ V_hat`, with
  recovered simplex vertices and oracle simplex overlay.
- `kspace_posterior.png` -- K-space latent posterior `alpha(x)` (supplementary diagnostic).

The run also writes `simplex_fit_arrays.npz` and `simplex_fit_arrays.json` containing
the raw geometry arrays: `fitted_vertices_M` (shape `KxM`), `decoded_alpha_test`, and
evaluation-only fields `class_alignment`, `oracle_vertices_M_eval`.

### Multi-seed image benchmarks

```bash
# Run a single seed; repeat for seeds 7, 42, 123, ...
python scripts/run_real.py experiment=realdata_hidden_priors data=mnist_subset seed=7
python scripts/run_real.py experiment=realdata_hidden_priors data=fashionmnist_subset seed=7
python scripts/run_real.py experiment=realdata_hidden_priors data=cifar10_subset seed=7

# Aggregate results across seeds
python scripts/plot_method_comparison.py \
  --input-root outputs/realdata_hidden_priors \
  --output-dir outputs/figures
```

### Galaxy10 DECaLS

Standard comparison (K=10, M=10):

```bash
python scripts/run_galaxy10.py seed=7
```

M-scaling ablation (vary number of sources):

```bash
python scripts/run_galaxy10.py data=galaxy10_m12 seed=7
python scripts/run_galaxy10.py data=galaxy10_m20 seed=7
# ... see configs/data/galaxy10_m*.yaml for all variants
```

Mixture-purity ablation (vary source overlap):

```bash
python scripts/run_galaxy10.py data=galaxy10_purity_p00 seed=7
python scripts/run_galaxy10.py data=galaxy10_purity_p50 seed=7
# ... see configs/data/galaxy10_purity_*.yaml for all variants
```

### Bottleneck variant

Add `model.bottleneck.enabled=true` to any `run_real.py` or `run_galaxy10.py` call:

```bash
python scripts/run_real.py experiment=realdata_hidden_priors \
  data=cifar10_subset seed=7 \
  model.bottleneck.enabled=true train.bottleneck_warmup_epochs=2
```

### Synthetic experiments

```bash
python scripts/run_synth.py seed=7
```

### SLURM (HPC)

`scripts/submit_real_run.sh` is a SLURM batch script. Configure paths via
environment variables before submitting:

```bash
export PYTHON=/path/to/your/venv/bin/python
export DATA_ROOT=/path/to/datasets
export OUTPUT_ROOT=/path/to/outputs
sbatch --export=ALL,DATA_CFG=cifar10_subset,SEED=7 scripts/submit_real_run.sh
```

## Output locations

Per-run results live under `outputs/<experiment>/<run>/` and contain:

- `resolved_config.yaml`
- `mixture_metadata.json`
- `metrics.json`, `metrics.csv`
- `simplex_details.json`
- `simplex_fit_arrays.npz`, `simplex_fit_arrays.json`
- `test_true_labels.npy` -- true class labels for the test split (for plot regeneration)
- `plots/` -- per-run diagnostic plots (`mspace_posterior.png`, `kspace_posterior.png`,
  `pi_hat_heatmap.png`, `pi_oracle_heatmap.png`, `source_training_curve.png`)

## Repository structure

```
configs/    Hydra configs (data, model, train, simplex fitter, experiment)
src/        library code (method, baselines, data, simplex fitters, training)
scripts/    runnable entry points
```

## License

See `LICENSE` (MIT).
