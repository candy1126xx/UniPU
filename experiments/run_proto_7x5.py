"""新协议数据：6 方法 × 5 seeds（可断点续跑），支持多数据集。

用法:
    python run_proto_7x5.py                  # 默认 acs
    python run_proto_7x5.py --dataset mnist

各数据集的配置沿用旧 runner：
  acs     K=10   semantic_v2 k=20  mlp2       20 轮  按州取客户端（10 州）
  mnist   K=5    spectral    k=10  mlp2       50 轮  按协议均分
  isic    K=6    semantic_v2 k=20  mlp2       20 轮  按协议均分（U+=450）
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(REPO, 'experiments')
for p in (REPO, EXP):
    if p not in sys.path:
        sys.path.insert(0, p)

import common.acs_data as acs  # noqa: E402
import common.cxr14_data as cxr14  # noqa: E402
import common.isic_data as isic  # noqa: E402
import common.mnist_data as mn  # noqa: E402
from common.federated_pu_proto import generate_pu_proto  # noqa: E402
from methods.federated_pu_v2 import get_method  # noqa: E402

SEEDS = [42, 123, 456, 789, 1024]
SCALARS = ['AUC', 'AP', 'ECE', 'MCE', 'Brier', 'F1', 'Recall', 'Precision',
           'ACC']
METHODS = ['nnPU', 'UniPU-true', 'UniPU-Cluster', 'PUe-LocalEst', 'NTC',
           'FedPU']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

N_P, N_PLUS, N_MINUS = 400, 600, 600
CLIENT_STATES = ['CA', 'TX', 'NY', 'FL', 'PA', 'IL', 'OH', 'NC', 'MI', 'GA']

DATASETS = {
    'acs': dict(loader=acs.load_acs, K=acs.K_ACS, K_CLUSTER=20,
                cluster_method='semantic_v2', rounds=20, model='mlp2',
                clients='state', n_p=N_P, n_plus=N_PLUS, n_minus=N_MINUS,
                out='E_acs_proto_p400_7x5'),
    # MNIST：MLP 20 轮未收敛（92% 水平），50 轮完全平台化（r45→r50 变化 <0.1）
    'mnist': dict(loader=mn.load_mnist, K=mn.K_MNIST, K_CLUSTER=10,
                  cluster_method='spectral', rounds=50, model='mlp2',
                  clients='proto', n_p=N_P, n_plus=N_PLUS, n_minus=N_MINUS,
                  out='E_mnist_mlp_p400_7x50'),
    # ISIC：训练集正例仅 6,621，10 客户端下 U+ 用 450（600 会触发容量断言）；
    # 用元数据分簇、图像训练（cluster_method='semantic_v2'）
    'isic': dict(loader=isic.load_isic, K=isic.K_ISIC, K_CLUSTER=20,
                 cluster_method='semantic_v2', rounds=20, model='mlp2',
                 clients='proto', n_p=N_P, n_plus=450, n_minus=450,
                 out='E_isic_proto_p400_7x5'),
    # CXR14 Effusion：训练正例 8,659，n_plus=400（5 seed 容量全过）；
    # 年龄梯度 π 0.108→0.208（1.9×），贴近 400:400:400
    'cxr14eff': dict(loader=lambda: cxr14.load_cxr14(variant='effusion6'),
                     K=cxr14.K_CXR, K_CLUSTER=20,
                     cluster_method='semantic_v2', rounds=20, model='mlp2',
                     clients='proto', n_p=N_P, n_plus=400, n_minus=400,
                     out='E_cxr14eff_proto_p400_7x5'),
}


def compute_cluster_quality_true_gt(clients_data, method_clients, K):
    """簇纯度（ground truth=数据生成协议真实簇标签，pred=方法重分簇结果）。

    - clients_data: generate_pu_proto 输出，'P_cluster'/'U_cluster' 为协议真实簇
      （numpy，方法 setup_clients 不修改 clients_data，故保持不变）。
    - method_clients: 方法训练用 self.clients，'P_cluster'/'U_cluster' 为重分簇结果（torch）。
    返回 (agg, per_client) 或 (None, None)。"""
    from sklearn.metrics import (
        homogeneity_score, v_measure_score,
        normalized_mutual_info_score, adjusted_rand_score,
    )

    def _purity(labels_true, labels_pred):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels_true, labels_pred)
        return float(cm.max(axis=0).sum() / cm.sum())

    per_client = []
    for c, mc in zip(clients_data, method_clients):
        q = {}
        for split in ('P', 'U'):
            true_np = c[f'{split}_cluster']
            pred_np = mc[f'{split}_cluster'].cpu().numpy()
            if len(true_np) != len(pred_np):
                # U 被规则丢弃（-1），长度不一致，跳过该 split 的纯度评估
                continue
            if len(np.unique(true_np)) < 2 or len(np.unique(pred_np)) < 2:
                continue
            q[split] = {
                'purity': _purity(true_np, pred_np),
                'homogeneity': float(homogeneity_score(true_np, pred_np)),
                'v_measure': float(v_measure_score(true_np, pred_np)),
                'nmi': float(normalized_mutual_info_score(true_np, pred_np)),
                'ari': float(adjusted_rand_score(true_np, pred_np)),
            }
        per_client.append(q)
    if not per_client:
        return None, None
    splits = [k for k in per_client[0].keys()]
    agg = {}
    for split in splits:
        agg[split] = {}
        for metric in ('purity', 'homogeneity', 'v_measure', 'nmi', 'ari'):
            vals = [q[split][metric] for q in per_client if split in q]
            agg[split][metric] = float(np.mean(vals))
    return agg, per_client


def load_dataset(ds):
    """统一返回 (Xtr, ytr, ctr, Xte, yte, layout, states)。"""
    cfg = DATASETS[ds]
    raw = cfg['loader']()
    if isinstance(raw, dict):
        return dict(Xtr=raw['Xtr'], ytr=raw['ytr'], ctr=raw['ctr'],
                    Xte=raw['Xte'], yte=raw['yte'], layout=raw.get('layout'),
                    states=raw.get('states_tr'),
                    meta=raw.get('meta_tr'),
                    meta_layout=raw.get('meta_layout'))
    Xtr, ytr, ctr, Xte, yte, _cte = raw
    return dict(Xtr=Xtr, ytr=ytr, ctr=ctr, Xte=Xte, yte=yte, layout=None,
                states=None, meta=None, meta_layout=None)


def attach_meta(arrays, target_len_p, target_len_u, Z):
    """从 [X|M] 拼合矩阵 Z 拆出元数据部分，挂到 P_X_cluster/U_X_cluster。"""
    n_x = arrays['Xtr'].shape[1]
    return Z[:, n_x:]


def build_clients(ds, arrays, seed):
    """按数据集的客户端构造方式生成 10 个客户端。

    若数据集带独立元数据（arrays['meta'] 非空），把 [X|M] 拼起来喂给协议
    （采样只依赖索引，结果与只传 X 完全一致），再把返回的 P_X/U_X 拆回
    图像部分（用于训练）与元数据部分（挂到 P_X_cluster/U_X_cluster，
    供 _cluster_client 分簇用）——实现“元数据分簇、图像训练”。
    """
    cfg = DATASETS[ds]
    Xtr, ytr, ctr = arrays['Xtr'], arrays['ytr'], arrays['ctr']
    states = arrays['states']
    meta = arrays['meta']
    if meta is not None:
        Ztr = np.hstack([Xtr, meta]).astype(np.float32)
        n_x = Xtr.shape[1]
    else:
        Ztr, n_x = Xtr, None

    def _split(cl):
        if n_x is None:
            return cl
        for c in cl:
            pz, uz = c['P_X'], c['U_X']
            c['P_X'], c['P_X_cluster'] = pz[:, :n_x], pz[:, n_x:]
            c['U_X'], c['U_X_cluster'] = uz[:, :n_x], uz[:, n_x:]
            c['P_X'] = np.ascontiguousarray(c['P_X'])
            c['U_X'] = np.ascontiguousarray(c['U_X'])
        return cl

    if cfg['clients'] == 'state':
        clients = []
        for st in CLIENT_STATES:
            rows = np.where(states == st)[0]
            cl, _ = generate_pu_proto(Ztr[rows], ytr[rows], ctr[rows],
                                      K=cfg['K'], n_clients=1,
                                      n_p=cfg['n_p'], n_plus=cfg['n_plus'],
                                      n_minus=cfg['n_minus'],
                                      seed=seed, verbose=False)
            clients.append(cl[0])
        return _split(clients)
    clients, _ = generate_pu_proto(Ztr, ytr, ctr, K=cfg['K'],
                                   n_p=cfg['n_p'], n_plus=cfg['n_plus'],
                                   n_minus=cfg['n_minus'],
                                   seed=seed, verbose=False)
    return _split(clients)


def build(ds, name, dim, alpha, seed):
    cfg = DATASETS[ds]
    kw = dict(input_dim=dim, output_dim=1, model_type=cfg['model'],
              n_clients=10, total_rounds=cfg['rounds'], local_epochs=1,
              batch_size=128, lr=1e-3, weight_decay=0.0, pi=alpha,
              device=DEVICE, seed=seed)
    if name == 'UniPU-true':
        kw.update(K=cfg['K'], cluster_method='true', cluster_k=None,
                  dynamic_local_pi=True)
        return get_method('UniPU-Cluster', **kw)
    if name == 'UniPU-Cluster':
        # 有独立元数据时用元数据分簇（图像仍用于训练），否则用 X 分簇
        kw.update(K=cfg['K_CLUSTER'], cluster_method=cfg['cluster_method'],
                  cluster_k=cfg['K_CLUSTER'],
                  dynamic_local_pi=True,
                  native_layout=(DATASETS[ds].get('meta_layout')
                                 or DATASETS[ds].get('layout')))
        return get_method('UniPU-Cluster', **kw)
    return get_method(name, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='acs', choices=list(DATASETS))
    ap.add_argument('--rounds', type=int, default=None,
                    help='覆盖默认轮数（默认按数据集配置）')
    ap.add_argument('--model', default=None, help='覆盖模型，如 mlp2 / small_cnn')
    ap.add_argument('--seeds', default=None,
                    help='逗号分隔的种子子集，如 42')
    ap.add_argument('--methods', default=None,
                    help='逗号分隔的方法子集，如 nnPU,UniPU-Cluster')
    ap.add_argument('--tag', default='', help='结果目录后缀，如 _r100')
    args = ap.parse_args()
    ds = args.dataset
    cfg = DATASETS[ds]
    seeds = list(SEEDS)
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    methods = list(METHODS)
    if args.methods:
        methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    if args.rounds is not None:
        cfg['rounds'] = args.rounds
    if args.model is not None:
        cfg['model'] = args.model
    out = os.path.join(EXP, cfg['out'] + args.tag)

    arrays = load_dataset(ds)
    Xtr, ytr, ctr = arrays['Xtr'], arrays['ytr'], arrays['ctr']
    Xte, yte = arrays['Xte'], arrays['yte']
    DATASETS[ds]['layout'] = arrays['layout']
    DATASETS[ds]['meta_layout'] = arrays['meta_layout']
    if arrays['meta'] is not None:
        print(f'[meta] 分簇用元数据 {arrays["meta"].shape[1]} 维，'
              f'训练用特征 {Xtr.shape[1]} 维')

    os.makedirs(out, exist_ok=True)
    rows = []
    for seed in seeds:
        clients = build_clients(ds, arrays, seed)
        alpha = float(np.mean(np.concatenate([c['U_y_true']
                                              for c in clients])))
        print(f'\n[{ds} seed={seed}]  P={len(clients[0]["P_X"])}  '
              f'U={len(clients[0]["U_X"])}  α={alpha:.3f}')
        for name in methods:
            path = os.path.join(out, f'{name}_seed{seed}.json')
            if os.path.exists(path):
                with open(path, encoding='utf-8') as fh:
                    rec = json.load(fh)
                fm = rec['final_metrics']
                print(f'  {name:<15} [已存在] AUC={fm["AUC"]:.2f}')
                rows.append(rec)
                continue
            t0 = time.time()
            m = build(ds, name, Xtr.shape[1], alpha, seed)
            m.setup_clients(clients)
            m.fit(test_X=Xte, test_y=yte, eval_interval=1,
                  use_cosine_annealing=True)
            raw = m.evaluate(Xte, yte)
            fm = {k: float(raw[k]) for k in SCALARS if k in raw}
            rec = {'seed': seed, 'method': name, 'dataset': ds,
                   'protocol': 'pu_proto',
                   'alpha_u': alpha, 'final_metrics': fm,
                   'comm_curve': m.per_round_results,
                   'wall_seconds': round(time.time() - t0, 1)}
            if name in ('UniPU-true', 'UniPU-Cluster'):
                cq, _ = compute_cluster_quality_true_gt(
                    clients, m.clients,
                    cfg['K'] if name == 'UniPU-true' else cfg['K_CLUSTER'])
                rec['cluster_quality'] = cq
                rec['cluster_audit'] = m.cluster_audit
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(rec, fh, indent=2, ensure_ascii=False)
            rows.append(rec)
            print(f'  {name:<15} AUC={fm["AUC"]:.2f} AP={fm["AP"]:.2f} '
                  f'F1={fm["F1"]:.2f} ECE={fm["ECE"]:.2f} '
                  f'({rec["wall_seconds"]}s)')

    print('\n' + '=' * 96)
    print(f'{ds}: 5 seeds 均值 ± 标准差（P={cfg["n_p"]} / '
          f'U+={cfg["n_plus"]} / U-={cfg["n_minus"]}）')
    print(f'{"method":<16}{"AUC":>16}{"AP":>14}{"F1":>14}{"ECE":>14}')
    print('-' * 96)
    summary = {}
    for name in methods:
        ds_rows = [r['final_metrics'] for r in rows if r['method'] == name]
        if not ds_rows:
            continue
        agg = {k: (np.mean([d[k] for d in ds_rows]),
                   np.std([d[k] for d in ds_rows]))
               for k in ('AUC', 'AP', 'F1', 'ECE')}
        summary[name] = {k: [float(v[0]), float(v[1])]
                         for k, v in agg.items()}
        print(f'{name:<16}'
              + ''.join(f'{agg[k][0]:>10.2f}±{agg[k][1]:<5.2f}'
                        for k in ('AUC', 'AP', 'F1', 'ECE')))
    with open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump({'dataset': ds, 'summary': summary}, fh, indent=2,
                  ensure_ascii=False)
    print(f'\n结果目录: {out}')


if __name__ == '__main__':
    main()
