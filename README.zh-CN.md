[English](README.md) | [中文](README.zh-CN.md)

# UniPU

面向标注侧选择偏置与无标注侧构成异质性的联邦正例-无标注（PU）学习框架。UniPU 以加权 PU 风险训练的分类器在客户端级校准到无标注集后验，使分类器输出可直接担任偏置校正权重，将估计器与分类器、估计与训练合并进单一模型，无需额外的倾向得分估计器。

本仓库包含论文主实验（端到端主结果）所需代码。两个小数据集（MNIST、ISIC 2019）已提交在 `data/` 下；两个大数据集（ACSIncome、ChestX-ray14）被 Git 忽略，需按下方说明在本地放置。结果包不包含在内。

## 目录结构

```
.
├─ data\                     # 数据（见“数据准备”）
│  ├─ acs\                   # [Git 忽略] ACSIncome parquet 文件
│  ├─ chestxray14\           # [Git 忽略] ChestX-ray14 特征 npz + 元数据 json
│  ├─ isic2019\              # [已提交] ISIC 2019 特征 npz + 元数据 json
│  └─ mnist\                 # [已提交] MNIST 原始 IDX 文件
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

每个加载器按两步定位数据目录：

1. 若设置了对应环境变量，则使用该变量；
2. 否则回退到仓库内 `data/` 下的相对路径，该路径基于加载器文件自身解析，不依赖当前工作目录。

无需设置环境变量、也无需修改源码：只要把文件放入 `data/` 下对应子目录，加载器即可直接运行。环境变量仍可用于指向仓库外的其他数据目录。

| 数据集 | 默认相对路径 | 可选覆盖变量 |
|---|---|---|
| ACSIncome | `data\acs\` | `UNIPU_ACS_DATA_DIR` |
| MNIST | `data\mnist\` | `UNIPU_MNIST_DATA_ROOT` |
| ISIC 2019 | `data\isic2019\` | `UNIPU_ISIC_DATA_DIR` |
| ChestX-ray14 | `data\chestxray14\` | `UNIPU_CXR_DATA_DIR` |

各 `data/` 子目录期望的文件：

- `data\acs\`：folktables ACS 2018 单年切片 parquet 文件；
- `data\mnist\`：原始 IDX 文件（`train-images-idx3-ubyte` 等）；
- `data\isic2019\`：离线提取的冻结 ResNet-18 特征 npz 及元数据 json；
- `data\chestxray14\`：离线提取的冻结 ResNet-18 特征 npz 及元数据 json。

### 哪些数据集纳入版本控制

GitHub 对大于 100 MiB 的普通 Git 文件会直接阻止，对大于 50 MiB 的文件会发出警告，并建议仓库总体保持在约 1 GB 以内。

- **已提交（小数据，合计 95.7 MiB）：** MNIST 与 ISIC 2019。每个文件都小于 50 MiB（最大 44.9 MiB），以普通 Git 跟踪，无需 Git LFS，也不会产生任何警告。
- **Git 忽略（大数据）：** ACSIncome 与 ChestX-ray14。ChestX-ray14 的 npz 约 129.8 MiB（超过 100 MiB 硬限制，包含它的 push 会被拒绝），ACS 的两个训练 parquet 各约 70.6 MiB（超过 50 MiB 警告阈值）。这两个目录已写入 `.gitignore`。

若要运行 ACS/CXR 实验，请从公开来源获取对应文件（见论文的数据可用性声明），分别放入 `data\acs\` 与 `data\chestxray14\`；或在当前 PowerShell 会话中用可选覆盖变量指向其他目录：

```powershell
$env:UNIPU_ACS_DATA_DIR = '<你的数据目录>'
```

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
