[English](README.md) | [中文](README.zh-CN.md)

# UniPU

A federated positive-unlabeled (PU) learning framework for label-side selection bias and unlabeled-side composition heterogeneity. In UniPU, a classifier trained with a weighted PU risk is calibrated at the client level to the posterior of the unlabeled set, so its outputs can directly serve as bias-correction weights, merging the estimator and the classifier, and estimation and training, into a single model without an additional propensity-score estimator.

This repository contains the code needed for the paper's main experiments (end-to-end main results). The two small datasets (MNIST, ISIC 2019) are committed under `data/`; the two large datasets (ACSIncome, ChestX-ray14) are git-ignored and must be placed locally as described below. Result packages are not included.

## Directory structure

```
.
├─ data\                     # datasets (see "Data preparation")
│  ├─ acs\                   # [git-ignored] ACSIncome parquet files
│  ├─ chestxray14\           # [git-ignored] ChestX-ray14 feature npz + metadata json
│  ├─ isic2019\              # [committed] ISIC 2019 feature npz + metadata json
│  └─ mnist\                 # [committed] MNIST raw IDX files
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

Each loader resolves its data directory in two steps:

1. if the corresponding environment variable is set, it is used;
2. otherwise the loader falls back to a repository-local relative path under `data/`, resolved from the loader file itself, so it does not depend on the current working directory.

No environment variable is required and no source change is needed: once the files are placed in the expected `data/` subdirectories, the loaders work out of the box. The environment variables remain available to point at a data directory elsewhere.

| Dataset | Default relative path | Optional override variable |
|---|---|---|
| ACSIncome | `data\acs\` | `UNIPU_ACS_DATA_DIR` |
| MNIST | `data\mnist\` | `UNIPU_MNIST_DATA_ROOT` |
| ISIC 2019 | `data\isic2019\` | `UNIPU_ISIC_DATA_DIR` |
| ChestX-ray14 | `data\chestxray14\` | `UNIPU_CXR_DATA_DIR` |

Expected files in each `data/` subdirectory:

- `data\acs\`: folktables ACS 2018 single-year slice parquet files;
- `data\mnist\`: raw IDX files (`train-images-idx3-ubyte`, etc.);
- `data\isic2019\`: offline-extracted frozen ResNet-18 feature npz and metadata json;
- `data\chestxray14\`: offline-extracted frozen ResNet-18 feature npz and metadata json.

### Which datasets are committed

GitHub blocks plain-Git files larger than 100 MiB and warns on files larger than 50 MiB, and recommends keeping a repository below about 1 GB.

- **Committed (small, 95.7 MiB in total):** MNIST and ISIC 2019. Every file is below 50 MiB (the largest is 44.9 MiB), so they are tracked with plain Git, need no Git LFS, and produce no warnings.
- **Git-ignored (large):** ACSIncome and ChestX-ray14. The ChestX-ray14 npz is about 129.8 MiB (over the 100 MiB hard limit, so a push containing it would be rejected), and the two ACS training parquet files are about 70.6 MiB each (over the 50 MiB warning threshold). These two directories are listed in `.gitignore`.

To run the ACS/CXR experiments, obtain the corresponding files from their public sources (see the paper's Data Availability statement) and place them in `data\acs\` and `data\chestxray14\` respectively, or point the optional override variable at another directory for the current PowerShell session:

```powershell
$env:UNIPU_ACS_DATA_DIR = '<your-data-directory>'
```

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
