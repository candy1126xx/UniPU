[English](README.md) | [中文](README.zh-CN.md)

# UniPU

A federated positive-unlabeled (PU) learning framework for label-side selection bias and unlabeled-side composition heterogeneity. In UniPU, a classifier trained with a weighted PU risk is calibrated at the client level to the posterior of the unlabeled set, so its outputs can directly serve as bias-correction weights, merging the estimator and the classifier, and estimation and training, into a single model without an additional propensity-score estimator.

This repository contains only the code needed for the paper's main experiments (end-to-end main results); data and result packages are not distributed with the repository.

## Directory structure

```
.
├─ common\
│  ├─ acs_data.py            # ACSIncome data loader
│  ├─ cxr14_data.py          # ChestX-ray14 Effusion data loader
│  ├─ isic_data.py           # ISIC 2019 data loader
│  ├─ mnist_data.py          # MNIST data loader
│  ├─ federated_pu_proto.py  # PU protocol generation (clients, labeled/unlabeled set construction)
│  ├─ metrics_v2.py          # metric computation
│  └─ model_v2.py            # model construction
├─ methods\
│  └─ federated_pu_v2.py     # core method implementation (get_method entry point)
├─ experiments\
│  └─ run_proto_7x5.py       # main experiment entry point: 6 methods × 5 seeds
└─ EXPERIMENT_ENV.md         # experiment environment configuration
```

## Requirements

- Python 3.10 (3.10.11 in the original experiments)
- PyTorch (2.6.0+cu124 with CUDA 12.4 in the original experiments)
- Other dependencies: numpy, pandas, scikit-learn, pyarrow (for reading parquet)

We recommend installing the dependencies in a virtual environment; without a GPU the code falls back to CPU automatically (`DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'`).

## Data preparation and path configuration

The four data loaders no longer contain any local absolute paths. Data directories are specified entirely through environment variables, so the code runs on other machines without source changes. The following environment variables are required:

| Dataset | Environment variable (required) |
|---|---|
| ACSIncome | `UNIPU_ACS_DATA_DIR` |
| MNIST | `UNIPU_MNIST_DATA_ROOT` |
| ISIC 2019 | `UNIPU_ISIC_DATA_DIR` |
| ChestX-ray14 | `UNIPU_CXR_DATA_DIR` |

If an environment variable is unset or empty when a loader is imported, the loader immediately raises a RuntimeError naming the variable that must be set, instead of falling back to any default path.

PowerShell example for the current session:

```powershell
$env:UNIPU_ACS_DATA_DIR = '<your-data-directory>'
```

Expected directory contents:

- ACSIncome: folktables ACS 2018 single-year slice parquet files;
- MNIST: raw IDX files (`train-images-idx3-ubyte`, etc.);
- ISIC 2019: offline-extracted frozen ResNet-18 feature npz files and metadata json;
- ChestX-ray14: offline-extracted frozen ResNet-18 feature npz files and metadata json.

## Running the main experiment

Run from the repository root (replace `<python>` with a Python interpreter that has the dependencies configured):

```powershell
# ACSIncome (default dataset, 20 rounds)
< python > experiments\run_proto_7x5.py --dataset acs

# ISIC 2019 (20 rounds, U+ budget 450)
< python > experiments\run_proto_7x5.py --dataset isic

# ChestX-ray14 Effusion (20 rounds, U budget 400)
< python > experiments\run_proto_7x5.py --dataset cxr14eff

# MNIST (50 rounds)
< python > experiments\run_proto_7x5.py --dataset mnist
```

Optional arguments:

- `--seeds 42,123`: run only the specified seeds (default 42, 123, 456, 789, 1024);
- `--methods nnPU,UniPU-Cluster`: run only the specified methods (default nnPU, UniPU-true, UniPU-Cluster, PUe-LocalEst, NTC, FedPU);
- `--rounds N`, `--model mlp2`: override the default number of rounds and model;
- `--tag _suffix`: suffix for the result directory.

The script supports resuming: existing result json files are skipped rather than recomputed.

## Output

Results for each dataset are written to `experiments\<out>\` (see the runner's `DATASETS` configuration for the default directory names), one json per method × seed containing `final_metrics`, the per-round `comm_curve`, the runtime, and so on; after all runs finish, a `summary.json` is generated to aggregate the means and standard deviations.
