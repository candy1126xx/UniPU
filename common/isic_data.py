"""ISIC 2019 加载器（ResNet-18 冻结特征已离线提取，此处只读 npz）。

npz 由 build_isic2019.py 生成（已离线完成，本文件不依赖 ResNet）。
簇 = Age3 × Sex（6 簇），正类 = 恶性(MEL+BCC+SCC)。

特征 512 维（ResNet-18 全局池化），训练/测试按 lesion_id 分组切分，无泄漏。
"""
import json
import os

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATA = os.path.join(_REPO_ROOT, 'data', 'isic2019')
# 环境变量优先；未设置时回退到仓库内 data/isic2019（基于本文件定位，不依赖当前工作目录）
DATA = os.environ.get('UNIPU_ISIC_DATA_DIR') or _DEFAULT_DATA

VARIANTS = {
    'meta6': dict(npz='isic2019_meta6.npz', meta='isic2019_meta6_meta.json',
                  K=6, names=['<45|M', '<45|F', '45-65|M', '45-65|F',
                              '65+|M', '65+|F']),
}

K_ISIC = 6


def load_isic(variant='meta6', verbose=True):
    """读取 ISIC npz，返回 dict（字段与其它 loader 一致）。

    返回键：Xtr, ytr, ctr, Xte, yte, cte, layout, meta_tr, meta_te,
            meta_layout, K, cluster_names, variant
    """
    cfg = VARIANTS[variant]
    z = np.load(os.path.join(DATA, cfg['npz']), allow_pickle=True)
    f_tr = z['feat_train'].astype(np.float32)
    f_te = z['feat_test'].astype(np.float32)
    mu, sd = f_tr.mean(0, keepdims=True), f_tr.std(0, keepdims=True) + 1e-6
    Xtr, Xte = (f_tr - mu) / sd, (f_te - mu) / sd

    meta_layout = None
    meta_path = os.path.join(DATA, cfg['meta'])
    if os.path.exists(meta_path):
        info = json.load(open(meta_path, encoding='utf-8'))
        ml = info.get('meta_layout') or {}
        if ml.get('blocks'):
            meta_layout = {'n_cont': int(ml.get('n_cont', 0)),
                           'blocks': [tuple(b) for b in ml['blocks']]}

    d = dict(Xtr=Xtr, ytr=z['y_train'].astype(np.int64),
             ctr=z['cluster_train'].astype(np.int64),
             Xte=Xte, yte=z['y_test'].astype(np.int64),
             cte=z['cluster_test'].astype(np.int64),
             layout={'n_cont': Xtr.shape[1], 'blocks': []},
             meta_tr=z['meta_train'].astype(np.float32),
             meta_te=z['meta_test'].astype(np.float32),
             meta_layout=meta_layout,
             K=cfg['K'], cluster_names=cfg['names'], variant=variant)

    if verbose:
        pos = np.bincount(d['ctr'], minlength=cfg['K']).tolist()
        cpos = np.bincount(d['ctr'][d['ytr'] == 1], minlength=cfg['K']).tolist()
        print(f'[ISIC:{variant}] train={Xtr.shape} test={Xte.shape} '
              f'pos={d["ytr"].mean():.3f} K={cfg["K"]}')
        print(f'  簇={cfg["names"]}')
        print(f'  簇分布={pos}  正例按簇={cpos}')
        print(f'  meta_layout={meta_layout}')
    return d


if __name__ == '__main__':
    for v in VARIANTS:
        load_isic(v)
        print()
