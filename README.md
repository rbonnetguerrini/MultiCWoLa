# MultiCWoLa

MultiCWoLa implements multiclass classification without class labels from
multiple unlabeled mixtures. The method trains on source labels only, then
recovers class posteriors by exploiting the simplex geometry of the
Bayes-optimal source posterior.

This repository provides two implementations:

- Post-hoc simplex fitting: train a source classifier, calibrate its source
  posterior, fit the posterior simplex, and decode class posteriors.
- Bottleneck variant: constrain the classifier through a low-dimensional
  bottleneck so the latent posterior directly matches the recovered simplex
  structure.

The repository also includes the baselines and experiment configurations used
for the accompanying paper: supervised oracle, one-vs-rest source baseline,
Wei CCM, known-prior oracle simplex, and KSBS-Demix.

## How to use MultiCWoLa

Use MultiCWoLa when you have examples drawn from several unlabeled mixtures and
you know which mixture or source each example came from, but you do not know
the underlying class label of each example. The intended setting is:

- `K` latent classes.
- `M` observed mixtures or sources.
- Training labels identify only the source index, not the class.
- The mixture proportions are unknown.

The method estimates class posteriors up to a permutation of the latent classes.
For benchmark datasets, true labels are used only for evaluation and alignment.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The code has been tested with Python 3.10-3.12 and PyTorch >= 2.2. A CUDA GPU
is recommended for image benchmarks. Small synthetic and small-image runs can
also be executed on CPU.

## Quickstart

Run a small synthetic experiment:

```bash
python scripts/run_synth.py seed=7
```

Run a real-data benchmark with hidden class priors:

```bash
python scripts/run_real.py experiment=realdata_hidden_priors data=mnist_subset seed=7
```

Run on user-provided arrays where source labels are observed and class labels
are not:

```bash
python scripts/run_real.py experiment=realdata_hidden_priors \
  data=source_labeled_arrays data.path=/path/to/source_labeled_data.npz data.K=3 \
  run_baselines=false run_wei_baselines=false
```

Enable the bottleneck variant on the same workflow:

```bash
python scripts/run_real.py experiment=realdata_hidden_priors \
  data=mnist_subset seed=7 \
  model.bottleneck.enabled=true train.bottleneck_warmup_epochs=2
```

All commands use Hydra configuration files under `configs/` and write run
outputs under `outputs/` by default.

## Python API (bring your own model)

The library never trains a model — **you plug in your own**. `MultiCWoLa` provides the
structure around the two recovery methods (post-hoc simplex fitting and the bottleneck),
plus alignment, trust diagnostics, and mixture-vs-latent plotting. The only supervision
required is the **mixture identity** of each example; true class labels are optional and
used only to resolve the latent-class permutation and to evaluate.

### Post-hoc (R1) — fit a simplex to your classifier's posteriors

Train any M-way classifier (any framework, any data loader), then hand over its source
posteriors `g(x) = P(m | x)`:

```python
from multiclass_cwola import MultiCWoLa

g_train = my_model.predict_proba(X_train)        # (N, M) = P(mixture | x), your model
model = MultiCWoLa(K=3).fit_posteriors(g_train, source, y=y_optional)

model.class_posteriors_     # decoded latent posteriors alpha(x), (N, K)
model.pi_                   # recovered mixing matrix Pi_hat (M, K) — often the science target
alpha_eval = model.predict_proba(g_eval)         # decode held-out posteriors

print(model.report())                            # A1/A2/A3 trust diagnostics (below)
model.plot("out/", latent_labels=y, true_pi=pi)  # comparison figures (below)
```

Prefer plain functions? The same steps without the class:

```python
from multiclass_cwola import fit_simplex, decode_posteriors, align_latent_classes

fit = fit_simplex(g_train, num_vertices=3, config={"method": "archetypal"})
print(fit.diagnostics)                           # per-fit A1/A2/A3 checks
alpha_eval = decode_posteriors(g_eval, fit)
aligned, mapping = align_latent_classes(alpha_eval, y_small, labeled_index=idx)  # resolves K=2 flip
```

**Calibration (optional but recommended).** Overconfident logits distort the posterior
cloud. Fit a temperature/vector scaler on a held-out split of the source logits first:

```python
from multiclass_cwola import fit_calibrator, calibrate_logits

calib = fit_calibrator(val_logits, val_source, method="temperature")
g_train = calibrate_logits(train_logits, calib)
```

### Bottleneck (R2) — a drop-in head you train yourself

Replace your model's final linear layer with `BottleneckSimplexHead`, so the network
factorises as `g(x) = Pi @ alpha(x)`. Train it with ordinary cross-entropy / NLL on the
**mixture labels** — no custom loss — then extract the recovered structure and hand it over:

```python
from multiclass_cwola import BottleneckSimplexHead, MultiCWoLa

head = BottleneckSimplexHead(input_dim=embed_dim, num_sources=M, num_classes=K)
# attach to your backbone, train on mixture labels (a few warm-up epochs with Pi frozen helps) ...

alpha = head.predict_latent(features).cpu().numpy()   # (N, K) latent posteriors
pi    = head.pi().detach().cpu().numpy()              # (M, K) column-stochastic mixing matrix

model = MultiCWoLa(K=K).fit_bottleneck(alpha, pi, source, y=y_optional)
print(model.report());  model.plot("out/", latent_labels=y, true_pi=pi)
```

### Plots — mixture results vs. latent classes

`model.plot(outdir, latent_labels=None, true_pi=None)` writes the comparison figures
into `outdir`: the decoded latent-posterior scatter, the M-space source-posterior
geometry with the recovered simplex (and the oracle simplex overlaid when `true_pi` is
given), the recovered `Pi_hat` heatmap (plus the oracle when available), and the MxM
mixture-recovery confusion. Points are coloured by latent class when `latent_labels` is
provided, otherwise by mixture id.

### Trust diagnostics

On your own data there are usually no labels to validate against, so the method's
assumptions are surfaced as runnable checks via `model.report()`:

- **A2 (rank):** conditioning / volume of the recovered simplex — flags collapse.
- **A3 (separability):** how close the cloud gets to each vertex — weak anchors
  shrink the simplex and degrade identification.
- **A1 (shared class-conditionals):** an MxM mixture-recovery check comparing the
  classifier's empirical mixture separability against the composition limit;
  excess separability signals a per-mixture artifact rather than real class
  structure.

Each check returns an `ok` / `warn` / `fail` status. As the paper notes, recovered
classes should still be validated on held-out labels before high-stakes use.

## Method Variants

### Post-hoc Simplex Fitting

The default pipeline trains a classifier to predict the source index. Its
calibrated source posterior is then treated as a point cloud in source-posterior
space. MultiCWoLa fits a simplex to that point cloud and uses barycentric
coordinates inside the fitted simplex as recovered class posteriors.

Example:

```bash
python scripts/run_real.py experiment=realdata_hidden_priors \
  data=cifar10_subset model=cnn_large seed=7
```

Simplex fitting behavior is controlled by configs in `configs/simplex/`, for
example:

```bash
python scripts/run_real.py experiment=realdata_hidden_priors \
  data=cifar10_subset simplex=archetypal_corner seed=7
```

### Bottleneck Variant

The bottleneck variant adds a constrained class-posterior layer before decoding
back to source posteriors. This can be enabled with Hydra overrides:

```bash
python scripts/run_real.py experiment=realdata_hidden_priors \
  data=cifar10_subset seed=7 \
  model.bottleneck.enabled=true train.bottleneck_warmup_epochs=2
```

The same flag can be used with Galaxy10 runs:

```bash
python scripts/run_galaxy10.py seed=7 \
  model.bottleneck.enabled=true train.bottleneck_warmup_epochs=2
```

## Data

| Dataset | How it is obtained |
| --- | --- |
| MNIST, Fashion-MNIST, CIFAR-10 | Downloaded automatically by `torchvision` into `./outputs/datasets` on first run. |
| Galaxy10 DECaLS | Manual download required. Place `Galaxy10_DECals.h5` under `./outputs/datasets/` or set `data.data_root`. See `configs/data/galaxy10.yaml`. |
| Synthetic Gaussian data | Generated by the repository from `configs/data/synthetic_gaussian.yaml`. |
| Source-labeled arrays | Load a NumPy NPZ file with `data=source_labeled_arrays`. |

For custom data, create an NPZ file containing:

```text
train_x, train_source
val_x, val_source
test_x, test_source
```

The `*_x` arrays contain features or images. The `*_source` arrays contain the
observed source index for each example. True latent class labels are not
required.

Optional arrays are:

```text
train_y, val_y, test_y
train_true_posteriors, val_true_posteriors, test_true_posteriors
train_true_source_posteriors, val_true_source_posteriors, test_true_source_posteriors
pi
source_given_class
```

If the NPZ file does not include `pi`, `source_given_class`, or true posterior
arrays, set `data.K` explicitly. If `data.M` is omitted, it is inferred from
`train_source`.

Custom dataset builders can also return the dataset bundle interface in
`src/multiclass_cwola/data/common.py`. A dataset builder should return train,
validation, and test splits with feature arrays and source labels. True class
labels and true mixing matrices are optional and are used only for evaluation.

## Outputs

Per-run results are written under:

```text
outputs/<experiment>/<run>/
```

Typical files include:

- `resolved_config.yaml`: fully resolved Hydra configuration.
- `mixture_metadata.json`: dataset and mixture metadata.
- `metrics.json` and `metrics.csv`: scalar results.
- `simplex_details.json`: simplex fit diagnostics.
- `simplex_fit_arrays.npz` and `simplex_fit_arrays.json`: fitted vertices,
  decoded posteriors, and geometry arrays.
- `alpha_test.npy`: recovered test-set class posteriors in the fitted class
  order.
- `test_true_labels.npy`: true labels for evaluation and plot regeneration when
  available.
- `plots/`: diagnostic figures such as posterior geometry, heatmaps, and
  training curves.

In unlabeled-class runs, latent class accuracy, class alignment, oracle simplex
distances, and oracle heatmaps are unavailable and are saved as `null` or
omitted. The fitted class index order is still consistent across
`decoded_alpha_*`, `fitted_vertices_M`, and `pi_hat_MK`.

## Reproducing Paper Experiments

All commands below assume the project root and write under `outputs/`.

### Posterior Simplex Geometry Figure

Run the CIFAR-10 `K=3, M=6` experiment used for the simplex-geometry figure:

```bash
python scripts/run_real.py experiment=realdata_hidden_priors \
  data=cifar10_subset model=cnn_large seed=7 \
  data.K=3 data.M=6 data.class_subset='[0,1,2]'
```

Each run writes diagnostic plots under `outputs/<experiment>/<run>/plots/`:

- `mspace_posterior.png`: M-space source posterior geometry, with recovered
  simplex vertices and oracle simplex overlay when evaluation labels are
  available.
- `kspace_posterior.png`: K-space latent posterior diagnostic.

The run also writes `simplex_fit_arrays.npz` and `simplex_fit_arrays.json`.
Important arrays include `fitted_vertices_M`, `decoded_alpha_test`,
`class_alignment`, and `oracle_vertices_M_eval`.

### Multi-seed Image Benchmarks

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

Standard comparison:

```bash
python scripts/run_galaxy10.py seed=7
```

M-scaling ablation:

```bash
python scripts/run_galaxy10.py data=galaxy10_m12 seed=7
python scripts/run_galaxy10.py data=galaxy10_m20 seed=7
```

Mixture-purity ablation:

```bash
python scripts/run_galaxy10.py data=galaxy10_purity_p00 seed=7
python scripts/run_galaxy10.py data=galaxy10_purity_p50 seed=7
```

See `configs/data/galaxy10_m*.yaml` and
`configs/data/galaxy10_purity_*.yaml` for all variants.

### Synthetic Experiments

```bash
python scripts/run_synth.py seed=7
```

### SLURM

`scripts/submit_real_run.sh` is a SLURM batch script. Configure paths with
environment variables before submitting:

```bash
export PYTHON=/path/to/your/venv/bin/python
export DATA_ROOT=/path/to/datasets
export OUTPUT_ROOT=/path/to/outputs
sbatch --export=ALL,DATA_CFG=cifar10_subset,SEED=7 scripts/submit_real_run.sh
```

## Repository Structure

```text
configs/    Hydra configs for data, models, training, simplex fitting, and experiments
scripts/    Runnable experiment and plotting entry points
src/        Python package implementation
```

Key package areas:

- `multiclass_cwola.data`: synthetic, image, and Galaxy10 dataset builders.
- `multiclass_cwola.training`: source-classifier training and inference.
- `multiclass_cwola.simplex`: simplex projection, fitting, and selection.
- `multiclass_cwola.calibration`: post-hoc source-posterior calibration.
- `multiclass_cwola.baselines`: comparison methods.
- `multiclass_cwola.experiments`: end-to-end experiment orchestration.

## License

This project is released under the MIT License. See `LICENSE`.
