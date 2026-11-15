"""ACSIncome (folktables) loader for the PU protocol.

Task: positive = PINCP == 1 (personal income > $50k), 2018 single-year slice.

Clusters (10) = OCCP major groups (Census occupation code // 1000). OCCP is a
visible feature, so the upstream task is to identify it as the bias carrier.

Feature layout (native-categorical gain path):
    [0:2]   AGEP, WKHP (standardized)
    then one one-hot block per categorical, in CATS order, with rare
    categories (< MIN_FREQ) merged into 'other'.
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_DEFAULT_DATA_DIR = r'D:\Deepseek\data\acs\folktables_acs_income'
DATA_DIR = os.environ.get('UNIPU_ACS_DATA_DIR', _DEFAULT_DATA_DIR)

CONT = ['AGEP', 'WKHP']
CATS = ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']
MIN_FREQ = 0.005
N_ROWS = 600000
K_ACS = 10


def _read_2018():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=CONT + CATS + ['STATE', 'YEAR',
                                                       'PINCP'])
        parts.append(df[df['YEAR'] == 2018])
        del df
    return pd.concat(parts, ignore_index=True)


def _consolidate(s: pd.Series) -> pd.Series:
    vc = s.value_counts(normalize=True)
    keep = set(vc[vc >= MIN_FREQ].index)
    return s.where(s.isin(keep), other='other').astype(str)


def load_acs(n_rows=N_ROWS, test_size=0.2, seed=42, verbose=True):
    df = _read_2018()
    if n_rows and len(df) > n_rows:
        df = df.sample(n=n_rows, random_state=seed).reset_index(drop=True)

    y = (df['PINCP'] > 0.5).astype(np.int64).values
    cluster = np.clip(df['OCCP'].astype(int) // 1000, 0, 9).values.astype(np.int64)
    states = df['STATE'].astype(str).values

    X_num = StandardScaler().fit_transform(
        df[CONT].values.astype(np.float64)).astype(np.float32)

    blocks, mats = [], [X_num]
    for c in CATS:
        if c == 'OCCP':
            # 数据生成时簇就是 OCCP 十大类，因此训练者看到的也应是十大类
            cons = pd.Series([f'g{v}' for v in cluster], index=df.index)
        else:
            cons = _consolidate(df[c])
        d = pd.get_dummies(cons, prefix=c, drop_first=False)
        blocks.append((c, d.shape[1]))
        mats.append(d.values.astype(np.float32))
    X = np.hstack(mats)

    layout, cur = {'n_cont': len(CONT)}, len(CONT)
    for c, w in blocks:
        layout.setdefault('blocks', []).append((cur, cur + w))
        cur += w
    assert cur == X.shape[1], (cur, X.shape[1])

    rs = np.random.RandomState(seed)
    idx = rs.permutation(len(X))
    n_te = int(len(X) * test_size)
    te, tr = idx[:n_te], idx[n_te:]

    if verbose:
        print(f'[ACS] 样本 {len(X):,}（2018 切片抽样）  维数 {X.shape[1]}  '
              f'正例率 {y[tr].mean():.3f}')
        print(f'  类别块: {[(c, w) for c, w in blocks]}')
        print(f'  簇分布(train): {np.bincount(cluster[tr], minlength=K_ACS).tolist()}')
        print(f'  正例簇分布(train): '
              f'{np.bincount(cluster[tr][y[tr] == 1], minlength=K_ACS).tolist()}')
        print(f'  训练集州数={len(np.unique(states[tr]))}')
    return dict(Xtr=X[tr], ytr=y[tr], ctr=cluster[tr], states_tr=states[tr],
                Xte=X[te], yte=y[te], cte=cluster[te], layout=layout)


if __name__ == '__main__':
    d = load_acs()
    print({k: (v.shape if hasattr(v, 'shape') else v) for k, v in d.items()})
