"""联邦 PU 采样协议（双侧偏倚版）——按《联邦PU采样协议伪代码》实现。

与旧协议的关键差异：
  · π_i 来自"均匀 Dirichlet + 降序排序"，所有客户端【同向】偏向大簇
  · η_i,k ∝ 1/(π_i,k + ε)^β，再用 q_i = normalize(η_i ⊙ γ) 得到 P 的簇分布
  · U+ / U− 全局无放回分配（客户端间不重叠）
  · P 库独立从 D 正例均匀采（≈γ），与 U+ 可重叠（case-control）

默认超参（文档给出的典型值）：
  N_clients=10, N_plus=600, N_minus=600, N_P=400,
  alpha_D=1.0 (0.5–1.0), beta=1.0 (1.0–2.0), epsilon=1e-6,
  rho=1.0 (工作容量比例), N_pool=None (=D 的全部正例)
"""
import numpy as np

DEFAULTS = {
    'n_clients': 10,
    'n_plus': 600,        # 每客户端 U+
    'n_minus': 600,       # 每客户端 U-
    'n_p': 400,           # 每客户端 P
    'n_pool': None,       # 共享 P 库；None = D 的全部正例
    'alpha_D': 1.0,       # 均匀 Dirichlet 浓度
    'beta': 1.0,          # η 反向强度
    'epsilon': 1e-6,      # Dirichlet 平滑
    'rho': 1.0,           # 工作容量比例（D = subsample(D*, rho)）
    'seed': 42,
}


def _geometric_series(K, ratio, descending=True):
    """最大/最小 = ratio 的几何级数，归一化为概率。「8:1」「16:1」即 ratio。"""
    if K == 1:
        return np.array([1.0])
    r = float(ratio) ** (1.0 / (K - 1))
    if descending:
        w = np.array([r ** (K - 1 - i) for i in range(K)])
    else:
        w = np.array([r ** i for i in range(K)])
    return w / w.sum()


def _temperature_scale(base, T):
    """逐客户端温度缩放（T~LogNormal(0,σ²)），保持簇间单调序。"""
    lb = np.log(np.clip(base, 1e-10, None)) / max(T, 1e-6)
    lb = lb - lb.max()
    e = np.exp(lb)
    return e / e.sum()


def _cluster_independence(X, y, cluster, K):
    """簇独立度 = 该簇正例质心到其他簇正例质心的平均距离（v12 口径）。"""
    Xf = X.reshape(len(X), -1).astype(np.float64)
    cents = np.zeros((K, Xf.shape[1]))
    for k in range(K):
        m = (y == 1) & (cluster == k)
        if m.sum():
            cents[k] = Xf[m].mean(0)
    out = np.zeros(K)
    for k in range(K):
        d = [np.linalg.norm(cents[k] - cents[j])
             for j in range(K) if j != k]
        out[k] = float(np.mean(d)) if d else 0.0
    return out


def generate_pu_proto(X, y, cluster, K=None, n_clients=None, n_plus=None,
                      n_minus=None, n_p=None, n_pool=None, alpha_D=None,
                      beta=None, epsilon=None, rho=None, seed=None,
                      pi_mode='dirichlet', eta_mode='inv_pi',
                      pi_ratio=8.0, eta_ratio=16.0,
                      temp_sigma=0.4, eta_descending=False, verbose=True):
    """返回 (clients, info)。

    clients[i] = {'P_X','P_y','P_cluster','U_X','U_y','U_y_true',
                  'U_cluster','n_samples'}

    因子分解实验用的两个开关（默认值 = 主协议行为，向后兼容）：
      pi_mode='dirichlet' ：π_i ~ sort(Dirichlet(α_D))[::-1]，对齐簇大小序（主协议）
      pi_mode='gamma'     ：π_i = γ（U+ 按全局正例簇分布，客户端间 IID）
      pi_mode='uniform'   ：π_i = 1/K
      pi_mode='geo'       ：π_i ∝ 几何级数（最大/最小 = pi_ratio，逐客户端温度缩放）
      eta_mode='inv_pi'   ：η_i = 1/(π_i+ε)^β（主协议，与 π 反向）
      eta_mode='uniform'  ：q_i = γ（从共享库均匀抽取）
      eta_mode='pi'       ：q_i ∝ π_i（与 π 同向）
      eta_mode='reversed_ref'：q_i 用参考 Dirichlet 的反序
      eta_mode='geo'      ：η_i ∝ 几何级数（最大/最小 = eta_ratio；方向由
                            eta_descending 控制：True=与 π 同向，False=反向）
      所有模式统一 NIPW：q_i = normalize(η_i · γ)，反倾向性权重 w ∝ 1/η
      使 q_i w ∝ γ。geo 仅决定 η 的几何形状，两通道仍可独立精确控制。
    """
    p = dict(DEFAULTS)
    for k, v in (('n_clients', n_clients), ('n_plus', n_plus),
                 ('n_minus', n_minus), ('n_p', n_p), ('n_pool', n_pool),
                 ('alpha_D', alpha_D), ('beta', beta), ('epsilon', epsilon),
                 ('rho', rho), ('seed', seed)):
        if v is not None:
            p[k] = v
    K = int(K if K is not None else int(cluster.max()) + 1)
    rng = np.random.RandomState(p['seed'])
    NC = int(p['n_clients'])

    # ---------- Step 0: D = subsample(D*, rho) + 按簇建正例池 ----------
    n_all = len(y)
    if p['rho'] < 1.0:
        keep = rng.choice(n_all, size=int(n_all * p['rho']), replace=False)
    else:
        keep = np.arange(n_all)
    yD, cD = y[keep], cluster[keep]
    posD = keep[yD == 1]
    negD = keep[yD == 0]
    posD_cl = cluster[posD]

    pool_by_k = {k: posD[posD_cl == k] for k in range(K)}
    sizes = np.array([len(pool_by_k[k]) for k in range(K)])

    # ---------- Step 1: 簇按池大小降序 + 同向 Dirichlet ----------
    order = np.argsort(-sizes)                 # order[0] = 最大簇
    gamma_sorted = sizes[order] / sizes.sum()  # 全局正例簇分布（按大小序）

    pi_sorted = np.zeros((NC, K))
    for i in range(NC):
        raw = rng.dirichlet(np.full(K, p['alpha_D']))
        pi_sorted[i] = np.sort(raw)[::-1]      # 降序，与簇大小序对齐

    # 映射回原始簇编号
    pi = np.zeros((NC, K))
    for rank, k in enumerate(order):
        pi[:, k] = pi_sorted[:, rank]
    gamma = np.zeros(K)
    for rank, k in enumerate(order):
        gamma[k] = gamma_sorted[rank]

    # π 通道
    #   'dirichlet'（默认）：每客户端独立 Dirichlet，标签偏斜
    #   'gamma'            ：π = γ（U+ 跟随全局正例分布，无 U 侧偏倚；F00/F10 用）
    #   'uniform'          ：π = 1/K（均匀；仅作对照，注意容量更紧）
    #   'geo'              ：几何级数（最大/最小 = pi_ratio），按簇独立度排序
    geo_order = None
    if pi_mode == 'geo' or eta_mode == 'geo':
        # 按簇正例池大小降序：最大 π 分给最大池，保证容量可行
        geo_order = np.argsort(-sizes)
    if pi_mode == 'geo':
        base = _geometric_series(K, pi_ratio, descending=True)
        pi = np.zeros((NC, K))
        for i in range(NC):
            T = float(rng.lognormal(0.0, temp_sigma))
            v = _temperature_scale(base, T)
            for rank, k in enumerate(geo_order):
                pi[i, k] = v[rank]
    elif pi_mode == 'gamma':
        pi = np.tile(gamma, (NC, 1))
    elif pi_mode == 'uniform':
        pi = np.full((NC, K), 1.0 / K)

    # η 通道（逐客户端），供 Step 5 的 q_i 使用
    eta = np.zeros((NC, K))
    for i in range(NC):
        if eta_mode == 'geo':
            base = _geometric_series(K, eta_ratio,
                                     descending=bool(eta_descending))
            T = float(rng.lognormal(0.0, temp_sigma))
            v = _temperature_scale(base, T)
            for rank, k in enumerate(geo_order):
                eta[i, k] = v[rank]
        elif eta_mode == 'uniform':
            eta[i] = np.ones(K)
        elif eta_mode == 'pi':
            # 与 π 同向：直接用该客户端 π（映射回簇序）
            eta[i] = pi[i]
        elif eta_mode == 'reversed_ref':
            # 用同一参照 Dirichlet 的反序（与 π 反向），π 本身可以是均匀的
            rev = pi_sorted[i][::-1]
            for rank, k in enumerate(order):
                eta[i, k] = rev[rank]
        else:  # 'inv_pi'（主协议）
            eta[i] = 1.0 / (pi[i] + p['epsilon']) ** p['beta']

    # ---------- Step 2: U+ 全局无放回分配 ----------
    n_ik = np.zeros((NC, K), dtype=int)
    for i in range(NC):
        for k in range(K):
            n_ik[i, k] = int(round(p['n_plus'] * pi[i, k]))
    # 容量检查
    demand = n_ik.sum(axis=0)
    if verbose:
        print('  各簇 U+ 需求 vs 池:')
        for k in range(K):
            flag = '  <-- 超出' if demand[k] > sizes[k] else ''
            print(f'    簇{k}: 池 {sizes[k]:>7,}  需求 {demand[k]:>7,}{flag}')
    if np.any(demand > sizes):
        raise AssertionError(
            f'正例不够分: 需求 {demand.tolist()} 池 {sizes.tolist()}')

    U_plus = [[] for _ in range(NC)]
    for k in range(K):
        pool = pool_by_k[k].copy()
        rng.shuffle(pool)
        ptr = 0
        for i in range(NC):
            take = int(n_ik[i, k])
            U_plus[i].extend(pool[ptr:ptr + take].tolist())
            ptr += take

    # ---------- Step 3: U- 全局无放回分配 ----------
    need_neg = NC * p['n_minus']
    if need_neg > len(negD):
        raise AssertionError(f'负例不够分: 需要 {need_neg}, 池 {len(negD)}')
    negpool = negD.copy()
    rng.shuffle(negpool)
    U_minus = [negpool[i * p['n_minus']:(i + 1) * p['n_minus']]
               for i in range(NC)]

    # ---------- Step 4: 共享 P 库（从 D 正例均匀采，≈γ） ----------
    n_pool = int(p['n_pool']) if p['n_pool'] is not None else len(posD)
    n_pool = min(n_pool, len(posD))
    P_pool = rng.choice(posD, size=n_pool, replace=False)
    ppool_by_k = {k: P_pool[cluster[P_pool] == k] for k in range(K)}

    # ---------- Step 5: 各客户端按 NIPW 从共享库采 P（所有模式统一） ----------
    # 共享 P 库的簇构成是 γ（Step 4 从全体正例均匀采）；客户端以偏倚 η_i 从 γ
    # 采样，故实际 P 构成 q_i = η_i ⊙ γ / Σ_j η_{ij} γ_j。
    # 反倾向性权重 1/η_i 只抵消偏倚、把 q_i 还原到 γ（见 UniPU-EtaInv 变体）。
    P_idx, q_all, dropped = [], np.zeros((NC, K)), 0
    for i in range(NC):
        q = eta[i] * gamma
        q = q / q.sum()
        q_all[i] = q
        parts = []
        for k in range(K):
            want = int(round(p['n_p'] * q[k]))
            pool = ppool_by_k[k]
            take = min(want, len(pool))
            dropped += want - take
            if take > 0:
                parts.append(rng.choice(pool, size=take, replace=False))
        P_idx.append(np.concatenate(parts) if parts else
                     np.array([], dtype=np.int64))

    # ---------- 组装 ----------
    clients = []
    for i in range(NC):
        pidx = P_idx[i]
        uidx = np.concatenate([np.array(U_plus[i], dtype=np.int64),
                               np.asarray(U_minus[i], dtype=np.int64)])
        rng.shuffle(uidx)
        clients.append({
            'P_X': X[pidx].astype(np.float32),
            'P_y': np.ones(len(pidx), dtype=np.int64),
            'P_cluster': cluster[pidx].astype(np.int64),
            'U_X': X[uidx].astype(np.float32),
            'U_y': np.zeros(len(uidx), dtype=np.int64),
            'U_y_true': y[uidx].astype(np.int64),
            'U_cluster': cluster[uidx].astype(np.int64),
            'n_samples': len(pidx) + len(uidx)})

    info = {'params': p, 'K': K, 'cluster_order': order.tolist(),
            'gamma': gamma.tolist(), 'pi': pi.tolist(), 'eta': eta.tolist(),
            'q': q_all.tolist(),
            'pool_sizes': sizes.tolist(), 'u_plus_demand': demand.tolist(),
            'P_mean': float(np.mean([len(x) for x in P_idx])),
            'P_dropped_by_capacity': int(dropped)}
    return clients, info
