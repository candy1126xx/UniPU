[English](EXPERIMENT_ENV.md) | [中文](EXPERIMENT_ENV.zh-CN.md)

# 实验环境配置

## Python 解释器

| 项 | 值 |
|----|-----|
| 运行环境 | 项目虚拟环境中的 Python 解释器 |
| Python 版本 | 3.10.11（MSC v.1929，64-bit AMD64） |

## 深度学习环境

| 项 | 值 |
|----|-----|
| torch | 2.6.0+cu124 |
| CUDA | 12.4，`cuda_available=True` |
| cuDNN | 9.1.0（90100） |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |

## 用法

所有实验命令都使用已安装依赖的虚拟环境解释器调用：

```powershell
# 示例：跑 ACS 主实验
<python> experiments\run_proto_7x5.py --dataset acs
```

其中 `<python>` 替换为当前虚拟环境的 Python 解释器；请确认该解释器已安装下列依赖，而不是使用未安装 torch 的系统默认 Python。

## 数据位置

加载器默认从仓库内的 `data/` 目录读取数据，文件就位后无需设置环境变量。小数据集（MNIST、ISIC 2019）已提交在 `data/` 下，大数据集（ACSIncome、ChestX-ray14）被 Git 忽略、需在本地自行添加。各子目录的期望布局、按大小划分的依据与可选覆盖变量见 README.zh-CN.md 的“数据准备与路径配置”章节。
