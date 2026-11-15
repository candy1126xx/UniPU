"""
MNIST 数据加载器（任务书v10.22新增）。

正例：偶数数字（0,2,4,6,8），负例：奇数数字（1,3,5,7,9），π≈0.5
簇：K=5，每个偶数数字独立成簇，奇数数字按数字顺序分配到5簇
特征：28×28展平=784维，标准化到[0,1]
"""
import numpy as np
import os

DATA_ROOT = os.environ.get('UNIPU_MNIST_DATA_ROOT')
if not DATA_ROOT:
    raise RuntimeError('环境变量 UNIPU_MNIST_DATA_ROOT 未设置，无法定位 MNIST 数据目录')

# 簇映射：偶数数字→簇号（0→0, 2→1, 4→2, 6→3, 8→4）
# 奇数数字按顺序分配：1→簇0, 3→簇1, 5→簇2, 7→簇3, 9→簇4
EVEN_DIGITS = [0, 2, 4, 6, 8]
ODD_DIGITS = [1, 3, 5, 7, 9]
DIGIT_TO_CLUSTER = {}
for i, d in enumerate(EVEN_DIGITS):
    DIGIT_TO_CLUSTER[d] = i
for i, d in enumerate(ODD_DIGITS):
    DIGIT_TO_CLUSTER[d] = i

K_MNIST = 5


def _load_idx_images(path):
    """加载IDX格式图像文件。"""
    with open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        n = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def _load_idx_labels(path):
    """加载IDX格式标签文件。"""
    with open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        n = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.astype(np.int64)


def load_mnist(data_root=None):
    """加载MNIST，返回 (train_X, train_y, train_cluster, test_X, test_y, test_cluster)。

    train_y: 1=偶数(正例), 0=奇数(负例)
    train_cluster: 0-4，每个偶数数字一簇，奇数按序分配
    """
    if data_root is None:
        data_root = DATA_ROOT

    train_X = _load_idx_images(os.path.join(data_root, 'train-images-idx3-ubyte'))
    train_labels = _load_idx_labels(os.path.join(data_root, 'train-labels-idx1-ubyte'))
    test_X = _load_idx_images(os.path.join(data_root, 't10k-images-idx3-ubyte'))
    test_labels = _load_idx_labels(os.path.join(data_root, 't10k-labels-idx1-ubyte'))

    # 二分类标签：偶数=1(正例), 奇数=0(负例)
    train_y = (train_labels % 2 == 0).astype(np.int64)
    test_y = (test_labels % 2 == 0).astype(np.int64)

    # 簇标签
    train_cluster = np.array([DIGIT_TO_CLUSTER[d] for d in train_labels], dtype=np.int64)
    test_cluster = np.array([DIGIT_TO_CLUSTER[d] for d in test_labels], dtype=np.int64)

    return train_X, train_y, train_cluster, test_X, test_y, test_cluster


if __name__ == '__main__':
    train_X, train_y, train_cluster, test_X, test_y, test_cluster = load_mnist()
    print(f"train: {train_X.shape}, pi={train_y.mean():.3f}")
    print(f"test: {test_X.shape}, pi={test_y.mean():.3f}")
    print(f"train cluster dist: {np.bincount(train_cluster)}")
    print(f"test cluster dist: {np.bincount(test_cluster)}")
