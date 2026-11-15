[English](README.md) | [中文](README.zh-CN.md)

# UniPU

面向标注侧选择偏置与无标注侧构成异质性的联邦正例-无标注（PU）学习框架。UniPU 以加权 PU 风险训练的分类器在客户端级校准到无标注集后验，使分类器输出可直接担任偏置校正权重，将估计器与分类器、估计与训练合并进单一模型，无需额外的倾向得分估计器。

本仓库仅包含论文主实验（端到端主结果）所需代码，数据与结果包不随仓库分发。

## 目录结构

```
.
├─ common\
│  ├─ acs_data.py            # ACSIncome 数据加载
│  ├─ cxr14_data.py          # ChestX-ray14 Effusion 数据加载
│  ├─ isic_data.py           # ISIC 2019 数据加载
│  ├─ mnist_data.py          # MNIST 数据加载
│  ├─ federated_pu_proto.py  # PU 协议生成（客户端、标注与无标注集构造）
│  ├─ metrics_v2.py          # 指标计算
│  └─ model_v2.py            # 模型构造
├─ methods\
│  └─ federated_pu_v2.py     # 核心方法实现（get_method 入口）
├─ experiments\
│  └─ run_proto_7x5.py       # 主实验入口：6 方法 × 5 种子
└─ EXPERIMENT_ENV.md         # 原始实验环境配置
```

## 环境要求

- Python 3.10（原始实验为 3.10.11）
- PyTorch（原始实验为 2.6.0+cu124，CUDA 12.4）
- 其他依赖：numpy、pandas、scikit-learn、pyarrow（读取 parquet）

建议在虚拟环境中安装依赖；无 GPU 时代码会自动回退到 CPU（`DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'`）。

## 数据准备与路径配置

四个数据加载器不再内置任何本机绝对路径，数据目录完全由环境变量指定，无需修改源码即可在其他机器上运行。以下环境变量为必填：

| 数据集 | 环境变量（必填） |
|---|---|
| ACSIncome | `UNIPU_ACS_DATA_DIR` |
| MNIST | `UNIPU_MNIST_DATA_ROOT` |
| ISIC 2019 | `UNIPU_ISIC_DATA_DIR` |
| ChestX-ray14 | `UNIPU_CXR_DATA_DIR` |

导入加载器时若对应环境变量未设置或为空，会立即抛出 RuntimeError 提示需要设置的变量名，而不是回退到任何默认路径。

PowerShell 中为当前会话设置环境变量示例：

```powershell
$env:UNIPU_ACS_DATA_DIR = '<你的数据目录>'
```

各数据集期望的目录内容：

- ACSIncome：folktables ACS 2018 单年切片 parquet 文件；
- MNIST：原始 IDX 文件（`train-images-idx3-ubyte` 等）；
- ISIC 2019：离线提取的冻结 ResNet-18 特征 npz 及元数据 json；
- ChestX-ray14：离线提取的冻结 ResNet-18 特征 npz 及元数据 json。

## 运行主实验

在仓库根目录运行（将 `<python>` 替换为已配置依赖的 Python 解释器）：

```powershell
# ACSIncome（默认数据集，20 轮）
< python > experiments\run_proto_7x5.py --dataset acs

# ISIC 2019（20 轮，U+ 预算 450）
< python > experiments\run_proto_7x5.py --dataset isic

# ChestX-ray14 Effusion（20 轮，U 预算 400）
< python > experiments\run_proto_7x5.py --dataset cxr14eff

# MNIST（50 轮）
< python > experiments\run_proto_7x5.py --dataset mnist
```

可选参数：

- `--seeds 42,123`：只跑指定种子（默认 42、123、456、789、1024）；
- `--methods nnPU,UniPU-Cluster`：只跑指定方法（默认 nnPU、UniPU-true、UniPU-Cluster、PUe-LocalEst、NTC、FedPU）；
- `--rounds N`、`--model mlp2`：覆盖默认轮数与模型；
- `--tag _suffix`：结果目录后缀。

脚本支持断点续跑：已存在的结果 json 会被跳过，不会重复计算。

## 输出

每个数据集的结果写入 `experiments\<out>\`（默认目录名见 runner 的 `DATASETS` 配置），每个方法 × 种子一个 json，含 `final_metrics`、逐轮 `comm_curve`、运行耗时等；全部跑完后另生成 `summary.json` 汇总均值与标准差。
