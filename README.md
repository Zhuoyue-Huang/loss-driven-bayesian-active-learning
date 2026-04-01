# Loss-Driven Bayesian Active Learning

This repository accompanies the paper *From Decision to Acquisition: Loss-driven Bayesian Active Learning*. It contains the source code, experiment entry points, and dataset-staging utilities needed to reproduce the repository's experiments without committing generated figures, cached datasets, or intermediate result files.

## Environment

- Python 3.10 or newer
- CPU-only execution is sufficient for the provided scripts
- Real-data experiments require network access for OpenML and UCI downloads unless the datasets are staged in advance

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If you need a GPU-enabled PyTorch build, install it before `pip install -r requirements.txt`.

## Repository Layout

- `acquisition/`: models, acquisition strategies, metric helpers, and plotting utilities
- `data/`: synthetic problems plus tabular dataset loaders and preprocessing
- `experiments/`: public experiment entry points
- `config/datasets/defaults.json`: named dataset groups for staging

Generated outputs are written under `results/` and `checkpoint/` on demand. Those artefacts are intentionally excluded from version control.

## Dataset Staging

To prefetch the default datasets used by the experiment scripts:

```bash
python scripts/data/stage_default_datasets.py --group regression --group classification
```

To inspect the available dataset groups:

```bash
python scripts/data/stage_default_datasets.py --list
```

To stage a specific dataset manually:

```bash
python scripts/data/stage_datasets.py --dataset yacht_hydrodynamics@openml
```

## Experiment Recipes

Run all modules from the repository root.

### Synthetic Regression Figure

```bash
python -m experiments.run_reg
```

This generates a synthetic 1D regression acquisition figure under `results/regression/pm_bump/sklearn_gp/custom/`.

### Synthetic Regression Comparison

```bash
python -m experiments.run_reg_compare --n-runs 5 --weight exp
```

### Real-Data Regression Comparison

```bash
python -m experiments.run_reg_data_compare --dataset yacht --p-init 10 --n-runs 50 --weight exp_neg --n-steps 20 --n-samples 1000 --scale-target
```

By default the metric plots use `log10` loss; pass `--transform none` for raw loss.

### Real-Data LINEX Comparison

```bash
python -m experiments.run_reg_data_compare_linex --dataset estate --n-runs 50 --stat-type median
```

This compares squared-loss (variance reduction) and LINEX acquisition on the real-estate benchmark and writes LINEX comparison plots under `results/regression/<dataset>/.../`. By default the primary figure uses `log10` loss and the companion figure uses raw loss; pass `--transform none` to swap that order.

### Real-Data Classification Comparison

```bash
python -m experiments.run_clf_data_compare --dataset vowel --n-runs 50 --weight 1,1,1,1,1,1,50,50,1,1,1
```

This runs random, entropy, and weighted-entropy acquisition on the Vowel dataset, saves evaluation files, and by default produces NLL metric and class-proportion plots. Add `--plot-accuracy` to also generate accuracy plots.

### Synthetic Classification Visualisation

```bash
python -m experiments.run_clf_compare --n-runs 1 --viz-runs 1 --weight 1,50,50
```

This writes an acquisition visualisation for the synthetic ternary angular problem, showing how acquired points change the entropy and weighted-entropy acquisition values across the 2D region. The main output is `results/classification/ternary_angular/<model>/.../eval/clf_entropy_w_1_50_50_iter0.svg`. Add `--plot-metrics` to also generate aggregate NLL plots, and combine it with `--plot-accuracy` if you also want accuracy plots.
