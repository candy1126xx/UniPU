[English](EXPERIMENT_ENV.md) | [中文](EXPERIMENT_ENV.zh-CN.md)

# Experiment Environment Configuration

## Python interpreter

| Item | Value |
|----|-----|
| Runtime environment | Python interpreter in the project virtual environment |
| Python version | 3.10.11 (MSC v.1929, 64-bit AMD64) |

## Deep learning environment

| Item | Value |
|----|-----|
| torch | 2.6.0+cu124 |
| CUDA | 12.4, `cuda_available=True` |
| cuDNN | 9.1.0 (90100) |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |

## Usage

All experiment commands use the virtual-environment interpreter with the dependencies installed:

```powershell
# Example: run the ACS main experiment
<python> experiments\run_proto_7x5.py --dataset acs
```

Replace `<python>` with the current virtual environment's Python interpreter; make sure this interpreter has the dependencies installed, rather than using a system-default Python that does not have torch.
