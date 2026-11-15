# 实验环境配置

## Python 解释器

| 项 | 值 |
|----|-----|
| 可执行文件 | `D:\PUDP\.venv\Scripts\python.exe` |
| venv 前缀 | `D:\PUDP\.venv` |
| Python 版本 | 3.10.11（MSC v.1929，64-bit AMD64） |

## 深度学习环境

| 项 | 值 |
|----|-----|
| torch | 2.6.0+cu124 |
| CUDA | 12.4，`cuda_available=True` |
| cuDNN | 9.1.0（90100） |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |

## 用法

所有实验命令都使用绝对路径调用：

```powershell
# 示例：跑 ACS 主实验
& "D:\PUDP\.venv\Scripts\python.exe" experiments\run_proto_7x5.py --dataset acs
```

**不要**直接用 `python` 命令，默认 Python 3.13.16 没有 torch。
