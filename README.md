<p align="center">
  <img src="docs/_static/MultiCWoLa_logo.png" width="450", alt="logo">
</p>

<h2 align="center">MultiCWoLa: multiclass classification without class labels</h2>

<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://pytorch.org"><img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.0-DC583A.svg?style=flat&logo=pytorch"></a>
</p>

MultiCWoLa recovers latent class posteriors from multiple unlabeled mixtures,
no class labels required. The only supervision is the **mixture identity** of each
example (`source`). Class posteriors are recovered up to a permutation by exploiting
the simplex geometry of the Bayes-optimal source posterior.

## When to use MultiCWoLa

Use MultiCWoLa when you have examples drawn from several unlabeled mixtures and you
know which mixture each example came from, but not its underlying class label:

- `K` latent classes (unknown at training time).
- `M` observed mixtures or sources.
- Labels identify only the source index, not the class.
- The mixture proportions are unknown.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires Python 3.10-3.12 and PyTorch >= 2.2.

## Python API (bring your own model)

The library never trains a model, **you plug in your own**. `MultiCWoLa` provides the
structure around two recovery methods (post-hoc simplex fitting and the bottleneck),
plus alignment, trust diagnostics, and mixture-vs-latent plotting.

### Post-hoc (R1): fit a simplex to your classifier's posteriors

Train any M-way classifier (any framework, any data loader), then hand over its source
posteriors `g(x) = P(m | x)`:

```python
from multiclass_cwola import MultiCWoLa

g_train = my_model.predict_proba(X_train)        # (N, M) = P(mixture | x), your model
model = MultiCWoLa(K=3).fit_posteriors(g_train, source, y=y_optional)

model.class_posteriors_     # decoded latent posteriors alpha(x), (N, K)
model.pi_                   # recovered mixing matrix Pi_hat (M, K), often the science target
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

### Bottleneck (R2): a drop-in head you train yourself

Replace your model's final linear layer with `BottleneckSimplexHead`, so the network
factorises as `g(x) = Pi @ alpha(x)`. Train it with ordinary cross-entropy / NLL on the
**mixture labels**, no custom loss, then extract the recovered structure and hand it over:

```python
from multiclass_cwola import BottleneckSimplexHead, MultiCWoLa

head = BottleneckSimplexHead(input_dim=embed_dim, num_sources=M, num_classes=K)
# attach to your backbone, train on mixture labels (a few warm-up epochs with Pi frozen helps) ...

alpha = head.predict_latent(features).cpu().numpy()   # (N, K) latent posteriors
pi    = head.pi().detach().cpu().numpy()              # (M, K) column-stochastic mixing matrix

model = MultiCWoLa(K=K).fit_bottleneck(alpha, pi, source, y=y_optional)
print(model.report());  model.plot("out/", latent_labels=y, true_pi=pi)
```

### Plots: mixture results vs. latent classes

`model.plot(outdir, latent_labels=None, true_pi=None)` writes the comparison figures
into `outdir`: the decoded latent-posterior scatter, the M-space source-posterior
geometry with the recovered simplex (and the oracle simplex overlaid when `true_pi` is
given), the recovered `Pi_hat` heatmap (plus the oracle when available), and the MxM
mixture-recovery confusion. Points are coloured by latent class when `latent_labels` is
provided, otherwise by mixture id.

### Trust diagnostics

On your own data there are usually no labels to validate against, so the method's
assumptions are surfaced as runnable checks via `model.report()`:

- **A2 (rank):** conditioning / volume of the recovered simplex, flags collapse.
- **A3 (separability):** how close the cloud gets to each vertex, weak anchors
  shrink the simplex and degrade identification.
- **A1 (shared class-conditionals):** an MxM mixture-recovery check comparing the
  classifier's empirical mixture separability against the composition limit;
  excess separability signals a per-mixture artifact rather than real class
  structure.

Each check returns an `ok` / `warn` / `fail` status. Recovered classes should still be
validated on held-out labels before high-stakes use.

## License

This project is released under the MIT License. See `LICENSE`.
