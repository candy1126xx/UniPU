"""ChestX-ray14 Effusion PU 加载器（ResNet18 ImageNet 冻结特征已离线提取）。

簇 = age3 (<40, 40-60, 60+) x gender (M/F) = 6 簇。
正类 = 报告含 Effusion，负类 = No Finding。

特征 512 维（ResNet18 avgpool），按官方 train_val_list / test_list 切分。
"""
import json
import os

import numpy as np

_DEFAULT_DATA = r'D:\Deepseek\data\chestxray\built'
DATA = os.environ.get('UNIPU_CXR_DATA_DIR', _DEFAULT_DATA)

VARIANTS = {
    'effusion6': dict(npz='cxr14_effusion6.npz', meta='cxr14_effusion6_meta.json',
                      K=6, names=['<40|M', '<40|F', '40-60|M', '40-60|F',
                                  '60+|M', '60+|F']),
}

K_CXR = 6


def load_cxr14(variant='effusion6', verbose=True):
    cfg = VARIANTS[variant]
    z = np.load(os.path.join(DATA, cfg['npz']), allow_pickle=True)
    f_tr = z['feat_train'].astype(np.float32)
    f_te = z['feat_test'].astype(np.float32)
    mu, sd = f_tr.mean(0, keepdims=True), f_tr.std(0, keepdims=True) + 1e-6
    Xtr, Xte = (f_tr - mu) / sd, (f_te - mu) / sd

    d = dict(Xtr=Xtr, ytr=z['y_train'].astype(np.int64),
             ctr=z['cluster_train'].astype(np.int64),
             Xte=Xte, yte=z['y_test'].astype(np.int64),
             cte=z['cluster_test'].astype(np.int64),
             layout={'n_cont': Xtr.shape[1], 'blocks': []},
             meta_tr=z['meta_train'].astype(np.float32),
             meta_te=z['meta_test'].astype(np.float32),
             # meta 9 维全 one-hot：age5(0:5) gender2(5:7) view2(7:9)
             meta_layout={'n_cont': 0,
                          'blocks': [(0, 5), (5, 7), (7, 9)]},
             K=cfg['K'], cluster_names=cfg['names'], variant=variant)

    if verbose:
        pos = np.bincount(d['ctr'], minlength=cfg['K']).tolist()
        cpos = np.bincount(d['ctr'][d['ytr'] == 1], minlength=cfg['K']).tolist()
        print(f'[CXR14:{variant}] train={Xtr.shape} test={Xte.shape} '
              f'pos={d["ytr"].mean():.3f} K={cfg["K"]}')
        print(f'  簇={cfg["names"]}')
        print(f'  簇分布={pos}  正例按簇={cpos}')
    return d


if __name__ == '__main__':
    for v in VARIANTS:
        load_cxr14(v)
        print()
