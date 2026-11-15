"""
Federated PU Learning Methods v2: Unified framework with 7 methods.
"""
import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.model_v2 import make_model_v2
from common.metrics_v2 import compute_metrics, balance_test_set

def pu_loss(logits, labels, pi, risk_type='nnPU'):
    """无加权 PU risk。nnPU: pi*R_P^+ + max(0, R_U^- - pi*R_U^+)
    加权变体由各算法自己在 _local_train 中实现。
    """
    if logits.dim() > 1:
        logits = logits.squeeze(-1)
    loss_pos = F.softplus(-logits)
    loss_neg = F.softplus(logits)
    P_mask = (labels == 1).float()
    U_mask = (labels == 0).float()
    n_P = max(P_mask.sum().item(), 1)
    n_U = max(U_mask.sum().item(), 1)
    R_P_plus = (loss_pos * P_mask).sum() / n_P
    R_P_minus = (loss_neg * P_mask).sum() / n_P
    R_U_minus = (loss_neg * U_mask).sum() / n_U
    if risk_type == 'uPU':
        return pi * R_P_plus + R_U_minus - pi * R_P_minus
    elif risk_type == 'nnPU':
        nnPU_neg = torch.clamp(R_U_minus - pi * R_P_minus, min=0)
        return pi * R_P_plus + nnPU_neg
    else:
        raise ValueError(f"Unknown risk_type: {risk_type}")

class FederatedPUBase:
    def __init__(self, input_dim=784, output_dim=1, model_type='mlp6',
                 n_clients=10, total_rounds=100, local_epochs=1, batch_size=256,
                 lr=5e-3, weight_decay=5e-3, pi=0.5, device='cuda', seed=42,
                 checkpoint_dir=None, checkpoint_rounds=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_type = model_type
        self.n_clients = n_clients
        self.total_rounds = total_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.pi = pi
        self.device = device
        self.seed = seed
        self.global_model = None
        self.clients = []
        self.per_round_results = []
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_rounds = checkpoint_rounds or []
        self.saved_checkpoints = {}

    def setup_clients(self, clients_data):
        self.clients = []
        for data in clients_data:
            # v3数据不输出簇标签，默认填0（仅UniPUFed会重写setup_clients做k-means）
            p_cluster = data.get('P_cluster', np.zeros(len(data['P_y']), dtype=int))
            u_cluster = data.get('U_cluster', np.zeros(len(data['U_y']), dtype=int))
            p_class = data.get('P_class', np.zeros(len(data['P_y']), dtype=int))
            u_class = data.get('U_class', np.zeros(len(data['U_y']), dtype=int))
            u_y_true = data.get('U_y_true', np.zeros(len(data['U_y']), dtype=int))
            self.clients.append({
                'P_X': torch.tensor(data['P_X'], dtype=torch.float32, device=self.device),
                'P_y': torch.ones(len(data['P_y']), dtype=torch.long, device=self.device),  # P集标签全为1
                'P_cluster': torch.tensor(p_cluster, dtype=torch.long, device=self.device),
                'P_class': torch.tensor(p_class, dtype=torch.long, device=self.device),
                'U_X': torch.tensor(data['U_X'], dtype=torch.float32, device=self.device),
                'U_y': torch.zeros(len(data['U_y']), dtype=torch.long, device=self.device),  # U集标签全为0（未标注）
                'U_y_true': torch.tensor(u_y_true, dtype=torch.long, device=self.device),  # U集真实标签（用于验证集）
                'U_cluster': torch.tensor(u_cluster, dtype=torch.long, device=self.device),
                'U_class': torch.tensor(u_class, dtype=torch.long, device=self.device),
                'n_samples': data['n_samples'],
            })

    def _make_model(self):
        return make_model_v2(self.model_type, self.input_dim, self.output_dim, self.device)

    def _local_train(self, client, global_state, round_idx):
        raise NotImplementedError

    def _fedavg_aggregate(self, local_states, n_samples_list):
        total = sum(n_samples_list)
        avg_state = copy.deepcopy(local_states[0])
        for key in avg_state:
            if avg_state[key].dtype in (torch.float32, torch.float16, torch.float64):
                avg_state[key] = torch.zeros_like(avg_state[key])
        for state, n in zip(local_states, n_samples_list):
            w = n / total
            for key in avg_state:
                if avg_state[key].dtype in (torch.float32, torch.float16, torch.float64):
                    avg_state[key] += w * state[key]
        return avg_state

    def fit(self, test_X=None, test_y=None, eval_interval=10, use_cosine_annealing=False):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        print(f"[{self.__class__.__name__}] start: {self.n_clients} clients, {self.total_rounds} rounds, cosine_annealing={use_cosine_annealing}")
        self.global_model = self._make_model()
        global_state = self.global_model.state_dict()
        _initial_lr = self.lr
        for round_idx in range(self.total_rounds):
            if use_cosine_annealing:
                half = self.total_rounds // 2
                if round_idx < half:
                    self.lr = _initial_lr * (round_idx + 1) / max(half, 1)
                else:
                    t = (round_idx - half) / max(self.total_rounds - half - 1, 1)
                    self.lr = _initial_lr * 0.5 * (1.0 + np.cos(np.pi * t))
            local_states, n_samples_list = [], []
            for c in range(self.n_clients):
                ls, n = self._local_train(self.clients[c], global_state, round_idx)
                local_states.append(ls); n_samples_list.append(n)
            global_state = self._fedavg_aggregate(local_states, n_samples_list)
            self.global_model.load_state_dict(global_state)
            # 检查点保存
            if (round_idx + 1) in self.checkpoint_rounds and self.checkpoint_dir:
                ckpt_path = os.path.join(self.checkpoint_dir, f"ckpt_round{round_idx + 1}.pt")
                torch.save({
                    'model_state': self.global_model.state_dict(),
                    'round': round_idx + 1,
                }, ckpt_path)
                self.saved_checkpoints[round_idx + 1] = ckpt_path
            if (round_idx + 1) % eval_interval == 0 or round_idx == self.total_rounds - 1:
                if test_X is not None:
                    self._pre_evaluate_hook()
                    metrics = self.evaluate(test_X, test_y)
                    self.per_round_results.append({'round': round_idx + 1, 'metrics': metrics})
                    print(f"  round {round_idx+1}/{self.total_rounds}: lr={self.lr:.6f} AUC={metrics['AUC']:.2f}, F1={metrics['F1']:.2f}")
        self.lr = _initial_lr
        print(f"[{self.__class__.__name__}] done")
        return self

    def _pre_evaluate_hook(self):
        """评估前钩子：子类可重写以在评估前计算阈值等。默认空实现。"""
        pass

    def predict_proba(self, X):
        self.global_model.eval()
        with torch.no_grad():
            X_dev = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.global_model(X_dev)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            return torch.sigmoid(logits).cpu().numpy()

    def evaluate(self, test_X, test_y):
        # 原始测试集评估（不下采样平衡）
        return compute_metrics(test_y, self.predict_proba(test_X))

class UPUFed(FederatedPUBase):
    def _local_train(self, client, global_state, round_idx):
        model = self._make_model()
        model.load_state_dict(global_state)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        X = torch.cat([client['P_X'], client['U_X']], dim=0)
        y = torch.cat([client['P_y'], client['U_y']], dim=0)
        n = len(X)
        for _ in range(self.local_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = perm[start:end]
                optimizer.zero_grad()
                pu_loss(model(X[idx]), y[idx], self.pi, risk_type='uPU').backward()
                optimizer.step()
        return model.state_dict(), n

class NNPUFed(FederatedPUBase):
    def _local_train(self, client, global_state, round_idx):
        model = self._make_model()
        model.load_state_dict(global_state)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        P_X, U_X = client['P_X'], client['U_X']
        n_P, n_U = len(P_X), len(U_X)
        half_bs = self.batch_size // 2
        n_batches = max(1, max(n_P, n_U) // max(half_bs, 1))
        for _ in range(self.local_epochs):
            perm_p = torch.randperm(n_P, device=self.device)
            perm_u = torch.randperm(n_U, device=self.device)
            for b in range(n_batches):
                p_start = (b * half_bs) % n_P
                p_end = min(p_start + half_bs, n_P)
                u_start = (b * half_bs) % n_U
                u_end = min(u_start + half_bs, n_U)
                p_idx = perm_p[p_start:p_end]
                u_idx = perm_u[u_start:u_end]
                if len(p_idx) == 0 or len(u_idx) == 0:
                    continue
                X = torch.cat([P_X[p_idx], U_X[u_idx]], dim=0)
                y = torch.cat([torch.ones(len(p_idx), device=self.device, dtype=torch.long),
                               torch.zeros(len(u_idx), device=self.device, dtype=torch.long)], dim=0)
                optimizer.zero_grad()
                pu_loss(model(X), y, self.pi, risk_type='nnPU').backward()
                optimizer.step()
        return model.state_dict(), n_P + n_U

class FedPUOfficialFed(FederatedPUBase):
    """官方 FedPU-torch 的二分类移植（最小改动）。

    逐行对应 refs/FedPU-torch-main/modules/loss.py::MPULoss_V2：
        pu3 = (-log(1-pU + eps) * mask).sum() / n_U / |C_P|
        pu1 = (-log(1-pP + eps) * mask).sum() * prior[mask[0]]
              / n_P / (C - |C_P|)
        pu2 = -(log_res * prior).sum() / n_P,   log_res = -log(1 - pP*onehot + eps)
        objective = (pu3 + pu1 + pu2) * puW + CE(pP, labelsP)

    只做的改动：
      1. C = 2，正类索引 = 1，U 哨兵标签 = 2（官方为 num_classes）
      2. prior 由 pi 推导（prior_mode='pi'）或从 U_y_true 统计（'u_true'）；
         官方 priorlist 来自数据划分时的类别占比
      3. eps 默认 0.01（与官方一致）
    """
    NC = 2
    POS_INDEX = 1
    U_SENTINEL = 2

    def __init__(self, puW=1.0, eps=1e-6, prior_mode='pi', client_priors=None,
                 **kwargs):
        kwargs['output_dim'] = self.NC
        super().__init__(**kwargs)
        self.puW = puW
        self.eps = eps
        self.prior_mode = prior_mode
        self.prior = None
        # 【A 方案】逐客户端的 pool 先验（总体正例率），而不是 U 内正例比例
        self.client_priors = client_priors
        self.priors = None
        self._cur_prior = None
        self.positive_classes = [self.POS_INDEX]

    def setup_clients(self, clients_data):
        super().setup_clients(clients_data)
        if self.prior_mode == 'u_true':
            counts = np.zeros(self.NC)
            for c in self.clients:
                uy = c['U_y_true'].cpu().numpy()
                counts[0] += int((uy == 0).sum())
                counts[self.POS_INDEX] += int((uy == 1).sum())
            prior = counts / max(counts.sum(), 1)
        else:
            pi = float(self.pi)
            prior = np.array([1.0 - pi, pi])
        self.prior = torch.tensor(prior, dtype=torch.float32,
                                  device=self.device)
        # 逐客户端先验：client_priors 为长度 n_clients 的正例率列表
        idx0 = self.POS_INDEX
        if self.client_priors is not None:
            rows = []
            for p_i in self.client_priors:
                r = torch.zeros(self.NC, device=self.device)
                r[0] = 1.0 - float(p_i)
                r[idx0] = float(p_i)
                rows.append(r)
            self.priors = torch.stack(rows)
        else:
            self.priors = self.prior.unsqueeze(0).repeat(self.n_clients, 1)
        for i, c in enumerate(self.clients):
            c['_fed_idx'] = i

    def _local_train(self, client, global_state, round_idx):
        model = self._make_model()
        model.load_state_dict(global_state)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        self._cur_prior = self.priors[int(client.get('_fed_idx', 0))]
        P_labels = torch.full((len(client['P_y']),), self.POS_INDEX,
                              dtype=torch.long, device=self.device)
        U_labels = torch.full((len(client['U_y']),), self.U_SENTINEL,
                              dtype=torch.long, device=self.device)
        X = torch.cat([client['P_X'], client['U_X']], dim=0)
        y = torch.cat([P_labels, U_labels], dim=0)
        n = len(X)
        # 官方 indexlist：长度 C 的 one-hot 掩码，正类位置为 1
        mask = torch.zeros(self.NC, dtype=torch.float32, device=self.device)
        mask[self.POS_INDEX] = 1.0
        for _ in range(self.local_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = perm[start:end]
                optimizer.zero_grad()
                self._mpu_loss(model(X[idx]), y[idx], mask).backward()
                optimizer.step()
        return model.state_dict(), n

    def _mpu_loss(self, outputs, labels, mask):
        """逐行对应官方 MPULoss_V2.forward（含其 len(indexlist) 语义）。"""
        outputs_soft = F.softmax(outputs, dim=1)
        eps = self.eps
        nc = self.NC
        prior = self._cur_prior if self._cur_prior is not None else self.prior
        P_mask = (labels <= nc - 1).nonzero(as_tuple=False).view(-1)
        labels_P = labels[P_mask]
        outputs_P = outputs[P_mask]
        outputs_P_soft = outputs_soft[P_mask]
        U_mask = (labels > nc - 1).nonzero(as_tuple=False).view(-1)
        outputs_U_soft = outputs_soft[U_mask]

        # 官方的 len(indexlist) 是"正类个数"，不是类别总数。
        # 单正类（C=2, positive_classes=[1]）时 len=1，
        # 因此 num_neg = C-1 = 1，pu1 参与计算。
        n_index = len(self.positive_classes)

        pu_loss = torch.zeros(1, device=outputs.device)

        # pu3
        if outputs_U_soft.size(0) > 0:
            pu3 = (-torch.log(1 - outputs_U_soft + eps) * mask).sum() \
                / max(1, outputs_U_soft.size(0)) / max(1, n_index)
            pu_loss = pu_loss + pu3

        # pu1：官方条件 if numClass > len(indexlist)
        num_neg = nc - n_index
        if num_neg > 0:
            idx0 = int(self.positive_classes[0])   # 官方 priorlist[indexlist[0]]
            pu1 = (-torch.log(1 - outputs_P_soft + eps) * mask).sum() \
                * prior[idx0] \
                / max(1, outputs_P_soft.size(0)) / num_neg
            pu_loss = pu_loss + pu1

        # pu2
        if outputs_P_soft.size(0) > 0:
            label_onehot = torch.zeros(labels_P.size(0), nc,
                                       device=outputs.device)
            label_onehot.scatter_(1, labels_P.unsqueeze(1), 1)
            log_res = -torch.log(1 - outputs_P_soft * label_onehot + eps)
            pu2 = -(log_res * prior).sum() \
                / max(1, outputs_P_soft.size(0))
            pu_loss = pu_loss + pu2

        cross_loss = F.cross_entropy(outputs_P, labels_P) \
            if outputs_P_soft.size(0) > 0 else torch.zeros(
                1, device=outputs.device)

        return pu_loss * self.puW + cross_loss

    def predict_proba(self, X):
        self.global_model.eval()
        with torch.no_grad():
            X_dev = torch.tensor(X, dtype=torch.float32, device=self.device)
            probs = F.softmax(self.global_model(X_dev), dim=1)
            return probs[:, self.positive_classes].sum(dim=1).cpu().numpy()

class PUEFed(FederatedPUBase):
    """PUe-Fed: two-stage (propensity + NIPW weighted nnPU)."""

    def __init__(self, est_rounds=None, alpha_e=15.0, lambda_reg=15.0,
                 checkpoint_dir=None, checkpoint_rounds=None, **kwargs):
        super().__init__(**kwargs)
        # 默认估计-训练轮数比例为1:1（est_rounds = total_rounds // 2）
        self.est_rounds = est_rounds if est_rounds is not None else max(1, self.total_rounds // 2)
        self.train_rounds = self.total_rounds - self.est_rounds
        self.alpha_e = alpha_e
        self.lambda_reg = lambda_reg  # 倾向性估计模型的L2正则化（PUe原文默认λ=15）
        self.propensity_model = None
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_rounds = checkpoint_rounds or []
        self.saved_checkpoints = {}

    def _train_propensity(self, client, global_state):
        model = self._make_model()
        model.load_state_dict(global_state)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.lambda_reg)
        P_X, U_X = client['P_X'], client['U_X']
        n_P, n_U = len(P_X), len(U_X)
        half_bs = self.batch_size // 2
        steps = max(n_P, n_U) // max(half_bs, 1)
        for _ in range(self.local_epochs):
            for _ in range(max(steps, 1)):
                P_idx = torch.randperm(n_P, device=self.device)[:min(half_bs, n_P)]
                U_idx = torch.randperm(n_U, device=self.device)[:min(half_bs, n_U)]
                optimizer.zero_grad()
                P_pred = torch.sigmoid(model(P_X[P_idx]).squeeze(-1))
                U_pred = torch.sigmoid(model(U_X[U_idx]).squeeze(-1))
                # 论文公式14：π1/(n_P) Σ L(P,+1) + (1-π1)/(n_U) Σ L(U,-1) + α_e|Σ e - n_P|
                # batch内mean等价于1/|batch|Σ，再乘全局π1/(1-π1)
                pi1 = n_P / (n_P + n_U)
                loss_P = pi1 * F.binary_cross_entropy(P_pred, torch.ones_like(P_pred), reduction='none').mean()
                loss_U = (1 - pi1) * F.binary_cross_entropy(U_pred, torch.zeros_like(U_pred), reduction='none').mean()
                # 正则化：约束E[h]≈P(S=1)=n_P/(n_P+n_U)（论文公式13）
                global_sum_est = n_P * P_pred.mean() + n_U * U_pred.mean()
                reg = self.alpha_e * torch.abs(global_sum_est - n_P) / (n_P + n_U)
                (loss_P + loss_U + reg).backward()
                optimizer.step()
        return model.state_dict(), n_P + n_U

    def _compute_nipw_weights(self, client):
        self.propensity_model.eval()
        with torch.no_grad():
            e_hat = torch.sigmoid(self.propensity_model(client['P_X']).squeeze(-1)).clamp(min=1e-6, max=1-1e-6)
            raw_weights = 1.0 / e_hat
            # 论文归一化：Σ ω_i = 1
            return raw_weights / raw_weights.sum()

    def _local_train_classifier(self, client, global_state, weights):
        """PUe 第二阶段：样本级 NIPW 加权 nnPU。"""
        model = self._make_model()
        model.load_state_dict(global_state)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        P_X, U_X = client['P_X'], client['U_X']
        n_P, n_U = len(P_X), len(U_X)
        half_bs = self.batch_size // 2
        n_batches = max(1, max(n_P, n_U) // max(half_bs, 1))
        for _ in range(self.local_epochs):
            perm_p = torch.randperm(n_P, device=self.device)
            perm_u = torch.randperm(n_U, device=self.device)
            for b in range(n_batches):
                p_start = (b * half_bs) % n_P
                p_end = min(p_start + half_bs, n_P)
                u_start = (b * half_bs) % n_U
                u_end = min(u_start + half_bs, n_U)
                p_idx = perm_p[p_start:p_end]
                u_idx = perm_u[u_start:u_end]
                if len(p_idx) == 0 or len(u_idx) == 0:
                    continue
                logits_p = model(P_X[p_idx]).squeeze(-1)
                logits_u = model(U_X[u_idx]).squeeze(-1)
                w_p = weights[p_idx]
                # 样本级 NIPW 加权 nnPU loss（论文公式6-7、9：正损失和截断项负损失都用NIPW权重）
                l_plus_p = F.softplus(-logits_p)
                l_minus_p = F.softplus(logits_p)
                l_minus_u = F.softplus(logits_u)
                w_sum = max(w_p.sum().item(), 1e-8)
                R_P_plus = (l_plus_p * w_p).sum() / w_sum
                R_P_minus = (l_minus_p * w_p).sum() / w_sum  # 截断项也用NIPW权重
                R_U_minus = l_minus_u.mean()
                nnPU_neg = torch.clamp(R_U_minus - self.pi * R_P_minus, min=0)
                loss = self.pi * R_P_plus + nnPU_neg
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict(), n_P + n_U

    def fit(self, test_X=None, test_y=None, eval_interval=10, use_cosine_annealing=False):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        train_rounds = self.train_rounds  # 默认1:1分配（est_rounds = total_rounds // 2）
        _initial_lr = self.lr
        print(f"[PUe-Fed] two-stage: est {self.est_rounds} + train {train_rounds}, cosine={use_cosine_annealing}")
        if self.est_rounds > 0:
            print("  stage 1: propensity")
            self.propensity_model = self._make_model()
            prop_state = self.propensity_model.state_dict()
            for r in range(self.est_rounds):
                if use_cosine_annealing:
                    half = self.est_rounds // 2
                    if r < half:
                        self.lr = _initial_lr * (r + 1) / max(half, 1)
                    else:
                        t = (r - half) / max(self.est_rounds - half - 1, 1)
                        self.lr = _initial_lr * 0.5 * (1.0 + np.cos(np.pi * t))
                ls_list, ns_list = [], []
                for c in range(self.n_clients):
                    ls, ns = self._train_propensity(self.clients[c], prop_state)
                    ls_list.append(ls); ns_list.append(ns)
                prop_state = self._fedavg_aggregate(ls_list, ns_list)
                self.propensity_model.load_state_dict(prop_state)
                if (r + 1) % eval_interval == 0:
                    print(f"    est {r+1}/{self.est_rounds}")
            client_weights = [self._compute_nipw_weights(c) for c in self.clients]
            print("  NIPW weights done")
        else:
            print("  est=0: skip propensity, use uniform weights (degenerate to nnPU)")
            client_weights = [torch.ones(len(c['P_y']), device=self.device) for c in self.clients]
        print("  stage 2: NIPW nnPU")
        self.global_model = self._make_model()
        global_state = self.global_model.state_dict()
        for r in range(train_rounds):
            if use_cosine_annealing:
                half = train_rounds // 2
                if r < half:
                    self.lr = _initial_lr * (r + 1) / max(half, 1)
                else:
                    t = (r - half) / max(train_rounds - half - 1, 1)
                    self.lr = _initial_lr * 0.5 * (1.0 + np.cos(np.pi * t))
            ls_list, ns_list = [], []
            for c in range(self.n_clients):
                ls, ns = self._local_train_classifier(self.clients[c], global_state, client_weights[c])
                ls_list.append(ls); ns_list.append(ns)
            global_state = self._fedavg_aggregate(ls_list, ns_list)
            self.global_model.load_state_dict(global_state)
            total_round = self.est_rounds + r + 1
            # 检查点保存
            if total_round in self.checkpoint_rounds and self.checkpoint_dir:
                ckpt_path = os.path.join(self.checkpoint_dir, f"ckpt_round{total_round}.pt")
                torch.save({
                    'model_state': self.global_model.state_dict(),
                    'propensity_state': self.propensity_model.state_dict() if self.propensity_model else None,
                    'round': total_round,
                }, ckpt_path)
                self.saved_checkpoints[total_round] = ckpt_path
            if total_round % eval_interval == 0 or r == train_rounds - 1:
                if test_X is not None:
                    metrics = self.evaluate(test_X, test_y)
                    self.per_round_results.append({'round': self.est_rounds + r + 1, 'metrics': metrics})
                    print(f"    train {r+1}/{train_rounds}: AUC={metrics['AUC']:.2f}, F1={metrics['F1']:.2f}")
        self.lr = _initial_lr
        print("[PUe-Fed] done")
        return self

class PUELocalEstFed(PUEFed):
    """PUe-Fed 变体：估计阶段纯本地训练（无 FedAvg 聚合），训练阶段 FedAvg。

    用于 PUe 预算分配消融（E3_pue_budget_local_est）：
    - 估计阶段：每客户端独立 propensity 模型，本地训练 est_rounds 轮，**不聚合**；
    - 训练阶段：与 PUEFed 完全相同（样本级 NIPW 加权 nnPU + FedAvg 聚合），train_rounds 轮。
    - 调用方显式传 est_rounds=total_rounds、train_rounds 恒等于 total_rounds（各=当前总轮数）。
    """

    def __init__(self, est_rounds=None, alpha_e=15.0, lambda_reg=15.0,
                 checkpoint_dir=None, checkpoint_rounds=None,
                 budget_split=False, **kwargs):
        super().__init__(est_rounds=est_rounds, alpha_e=alpha_e, lambda_reg=lambda_reg,
                         checkpoint_dir=checkpoint_dir, checkpoint_rounds=checkpoint_rounds,
                         **kwargs)
        # 预算语义固定为“各阶段独立使用 total_rounds”（原 budget_split=False）：
        #   est_rounds=None -> 估计阶段 = total_rounds；训练阶段 = total_rounds。
        # budget_split 参数仅为向后兼容而保留，取值一律忽略。
        if budget_split:
            print('[PUe-LocalEst] 警告: budget_split 已废弃，固定按 False 处理')
        self.budget_split = False
        if est_rounds is None:
            self.est_rounds = self.total_rounds
        self.train_rounds = self.total_rounds
        self.prop_models = None  # 每客户端独立 propensity 模型

    def _compute_nipw_weights(self, client, prop_model=None):
        model = prop_model if prop_model is not None else self.propensity_model
        model.eval()
        with torch.no_grad():
            e_hat = torch.sigmoid(model(client['P_X']).squeeze(-1)).clamp(min=1e-6, max=1 - 1e-6)
            raw_weights = 1.0 / e_hat
            return raw_weights / raw_weights.sum()

    def fit(self, test_X=None, test_y=None, eval_interval=10, use_cosine_annealing=False):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        train_rounds = self.train_rounds
        _initial_lr = self.lr
        print(f"[PUe-LocalEst] two-stage: est {self.est_rounds}(LOCAL, no FedAvg) "
              f"+ train {train_rounds}(FedAvg), cosine={use_cosine_annealing}")
        # stage 1: 每客户端独立 propensity 模型，纯本地训练（无聚合）
        self.prop_models = [self._make_model() for _ in range(self.n_clients)]
        self.propensity_model = self.prop_models[0]  # 保持兼容字段
        if self.est_rounds > 0:
            print("  stage 1: propensity (LOCAL, no aggregation)")
            for r in range(self.est_rounds):
                if use_cosine_annealing:
                    half = self.est_rounds // 2
                    if r < half:
                        self.lr = _initial_lr * (r + 1) / max(half, 1)
                    else:
                        t = (r - half) / max(self.est_rounds - half - 1, 1)
                        self.lr = _initial_lr * 0.5 * (1.0 + np.cos(np.pi * t))
                for c in range(self.n_clients):
                    ls, ns = self._train_propensity(self.clients[c], self.prop_models[c].state_dict())
                    self.prop_models[c].load_state_dict(ls)
                if (r + 1) % eval_interval == 0:
                    print(f"    est {r+1}/{self.est_rounds}")
            client_weights = [self._compute_nipw_weights(self.clients[c], self.prop_models[c])
                              for c in range(self.n_clients)]
            print("  NIPW weights done (per-client local propensity)")
        else:
            print("  est=0: skip propensity, use uniform weights (degenerate to nnPU)")
            client_weights = [torch.ones(len(c['P_y']), device=self.device) for c in self.clients]
        # stage 2: NIPW nnPU + FedAvg（与 PUEFed 相同）
        print("  stage 2: NIPW nnPU (FedAvg)")
        self.global_model = self._make_model()
        global_state = self.global_model.state_dict()
        for r in range(train_rounds):
            if use_cosine_annealing:
                half = train_rounds // 2
                if r < half:
                    self.lr = _initial_lr * (r + 1) / max(half, 1)
                else:
                    t = (r - half) / max(train_rounds - half - 1, 1)
                    self.lr = _initial_lr * 0.5 * (1.0 + np.cos(np.pi * t))
            ls_list, ns_list = [], []
            for c in range(self.n_clients):
                ls, ns = self._local_train_classifier(self.clients[c], global_state, client_weights[c])
                ls_list.append(ls); ns_list.append(ns)
            global_state = self._fedavg_aggregate(ls_list, ns_list)
            self.global_model.load_state_dict(global_state)
            total_round = self.est_rounds + r + 1
            if total_round in self.checkpoint_rounds and self.checkpoint_dir:
                ckpt_path = os.path.join(self.checkpoint_dir, f"ckpt_round{total_round}.pt")
                torch.save({
                    'model_state': self.global_model.state_dict(),
                    'propensity_state': self.prop_models[0].state_dict() if self.prop_models else None,
                    'round': total_round,
                }, ckpt_path)
                self.saved_checkpoints[total_round] = ckpt_path
            if total_round % eval_interval == 0 or r == train_rounds - 1:
                if test_X is not None:
                    metrics = self.evaluate(test_X, test_y)
                    self.per_round_results.append({'round': total_round, 'metrics': metrics})
                    print(f"    train {r+1}/{train_rounds}: AUC={metrics['AUC']:.2f}, F1={metrics['F1']:.2f}")
        self.lr = _initial_lr
        print("[PUe-LocalEst] done")
        return self

class USBFed(NNPUFed):
    """nnPUSB-Fed: nnPU 非负伪分类风险 + 本地 BEP 阈值（Kato et al., ICLR 2019）。

    注意：原论文定义 PUSB = PU损失(公式4,无截断) + BEP阈值(公式9)，
    nnPUSB = nnPU损失(公式6,非负截断) + BEP阈值(公式9)。
    本实现使用公式(6)非负截断，故严格命名为 nnPUSB。

    训练阶段：各客户端本地最小化公式(6)非负伪分类风险：
    R = -pi * E_P[log f] + max(0, pi * E_P[log(1-f)] - E_U[log(1-f)])
    模型参数通过 FedAvg 聚合。由定理3，f* = pi * r(x)，即密度比 r = f/pi。

    校准阶段：公式(9) top-π 分位数阈值（BEP: Precision-Recall Breakeven Point）。
    原论文 Section 4.3/5.3 明确允许用 held-out data / U 集计算阈值
    （"we regarded the unlabeled data as test data"）。
    各客户端用本地 U 集（~p(x)）独立计算阈值，不上传原始数据。
    定理2证明此时 precision = recall。无需真实标签，只需已知类先验 pi。
    决策方式：y_hat = 1(f(x) >= theta_pi)。
    """

    def __init__(self, lambda_reg=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_reg = lambda_reg  # 保留参数以向后兼容，公式(6)不再使用密度比约束
        self.client_thresholds = [0.5] * self.n_clients  # 默认阈值，fit结束后更新
        self.global_threshold = 0.5  # v11.0: 服务端全局统一阈值（概率空间），fit结束后更新

    # _local_train 继承自 NNPUFed：使用 pu_loss() 函数（softplus 数值稳定实现）
    # PUSB 与 nnPU 的训练损失完全一致，唯一区别是 fit() 后计算 BEP 阈值 + evaluate() 用本地阈值

    def _pre_evaluate_hook(self):
        """评估前计算BEP阈值并聚合（v11.0：服务端按样本量加权平均→全局统一阈值）。"""
        self.global_model.eval()
        self.client_thresholds = []
        with torch.no_grad():
            for c in range(self.n_clients):
                f_c = torch.sigmoid(self.global_model(self.clients[c]['U_X']).squeeze(-1)).cpu().numpy()
                n = len(f_c)
                k = max(1, int(np.ceil(self.pi * n)))
                sorted_f = np.sort(f_c)[::-1]
                self.client_thresholds.append(float(sorted_f[k - 1]))
        client_ns = np.array([self.clients[c]['n_samples'] for c in range(self.n_clients)], dtype=float)
        self.global_threshold = float(np.sum(np.array(self.client_thresholds) * client_ns) / client_ns.sum())

    def fit(self, test_X=None, test_y=None, eval_interval=10, use_cosine_annealing=False, val_ratio=0.2):
        super().fit(test_X, test_y, eval_interval, use_cosine_annealing=use_cosine_annealing)
        # 阈值已在最后一轮评估前由_pre_evaluate_hook()计算
        print(f"  PUSB local BEP thresholds: {[f'{t:.3f}' for t in self.client_thresholds]}")
        print(f"  PUSB global threshold (sample-weighted): {self.global_threshold:.3f}")
        return self

    def evaluate(self, test_X, test_y):
        """v11.0: 服务端全局统一阈值（按样本量加权平均）在测试集上评估。"""
        from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
        y_prob = self.predict_proba(test_X)
        metrics = compute_metrics(test_y, y_prob)  # AUC等排序指标
        # 全局统一阈值
        y_pred = (y_prob >= self.global_threshold).astype(int)
        metrics['F1'] = float(f1_score(test_y, y_pred, zero_division=0) * 100)
        metrics['Recall'] = float(recall_score(test_y, y_pred, zero_division=0) * 100)
        metrics['Precision'] = float(precision_score(test_y, y_pred, zero_division=0) * 100)
        metrics['ACC'] = float(accuracy_score(test_y, y_pred) * 100)
        # 阈值统计
        metrics['threshold_global'] = self.global_threshold
        metrics['threshold_mean'] = float(np.mean(self.client_thresholds))
        metrics['threshold_std'] = float(np.std(self.client_thresholds))
        metrics['threshold_min'] = float(np.min(self.client_thresholds))
        metrics['threshold_max'] = float(np.max(self.client_thresholds))
        return metrics

class NTCFed(FederatedPUBase):
    """NTC-τMI-Fed: Non-Traditional Classifier + local τMI threshold.

    训练阶段（论文公式2）：各客户端本地用 BCE 训练 NTC（P label=1, U label=0），
    模型参数通过 FedAvg 聚合。

    校准阶段（论文Algorithm 1 + 附录D）：各客户端本地用 τMI calibration
    在**仅 U 集**的 NTC **logit 输出**（before final activation）上计算阈值
    τ* = argmax_τ Î(ŷτ, z)，其中 MI 用 k-NN 估计（论文公式6）：
        MI = ψ(n) + ψ(k) − ⟨ψ(|ŷτ=yτi|)⟩ − ⟨ψ(ri)⟩
    阈值不共享（联邦适配）。
    决策方式：ŷ = 1(z(x) > τ*)，z 为 logit。
    """

    def __init__(self, tau_mi_k=5, tau_mi_m=100, **kwargs):
        super().__init__(**kwargs)
        self.tau_mi_k = tau_mi_k  # k-NN 邻居数（论文Figure 4用k=5）
        self.tau_mi_m = tau_mi_m  # 候选阈值数
        self.client_thresholds = [0.0] * self.n_clients  # logit空间阈值，fit结束后更新
        self.global_threshold = 0.0  # v11.0: 服务端全局统一阈值（logit空间），fit结束后更新

    @staticmethod
    def _tau_mi_calibrate(z, k=5, m=100):
        """论文Algorithm 1 τMI校准（k-NN MI估计，公式6）。
        z: U集NTC logit输出（before final activation），1D numpy array。
        返回最优阈值 τ*（logit空间）。
        """
        from scipy.special import digamma
        n = len(z)
        if n < 2 * k + 2:
            return float(np.median(z))
        z_sorted = np.sort(z)
        # 候选阈值范围：2%-98%分位数（避免极端值，对应论文Optimization 3/4的viable range）
        lo = np.percentile(z_sorted, 2)
        hi = np.percentile(z_sorted, 98)
        if hi <= lo:
            lo, hi = z_sorted[0], z_sorted[-1]
        taus = np.linspace(lo, hi, m)
        best_tau, best_mi = float(taus[0]), -np.inf
        psi_k = digamma(k)
        psi_n = digamma(n)
        for tau in taus:
            y_hat = (z_sorted > tau).astype(int)
            n0 = int((y_hat == 0).sum())
            n1 = n - n0
            if n0 < 1 or n1 < 1:
                continue
            # ⟨ψ(|ŷτ = yτi|)⟩ = (n0/n)ψ(n0) + (n1/n)ψ(n1)
            term_group = (n0 / n) * digamma(n0) + (n1 / n) * digamma(n1)
            # ⟨ψ(ri)⟩：Optimization 2 — 仅阈值附近2k个样本需重新计算ri，其余ri=k
            split = n0  # 0块[0,n0), 1块[n0,n)
            r_vals = np.full(n, k, dtype=np.float64)
            lo_idx = max(0, split - k)
            hi_idx = min(n, split + k)
            for pos in range(lo_idx, hi_idx):
                label = y_hat[pos]
                if label == 0:
                    avail = pos  # 左边同标签样本数
                    kth = min(k, avail)
                    if kth == 0:
                        continue
                    delta = z_sorted[pos] - z_sorted[pos - kth]
                else:
                    avail = n - 1 - pos  # 右边同标签样本数
                    kth = min(k, avail)
                    if kth == 0:
                        continue
                    delta = z_sorted[pos + kth] - z_sorted[pos]
                # ri = |{z_i: |z_i - z_pos| ≤ delta}|
                left_b = int(np.searchsorted(z_sorted, z_sorted[pos] - delta))
                right_b = int(np.searchsorted(z_sorted, z_sorted[pos] + delta, side='right'))
                r_vals[pos] = float(right_b - left_b)
            term_r = float(np.mean(digamma(np.maximum(r_vals, 1.0))))
            mi = psi_n + psi_k - term_group - term_r
            if mi > best_mi:
                best_mi, best_tau = mi, float(tau)
        return best_tau

    def _local_train(self, client, global_state, round_idx):
        model = self._make_model()
        model.load_state_dict(global_state)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        P_X, U_X = client['P_X'], client['U_X']
        n_P, n_U = len(P_X), len(U_X)
        half_bs = self.batch_size // 2
        n_batches = max(1, max(n_P, n_U) // max(half_bs, 1))
        for _ in range(self.local_epochs):
            perm_p = torch.randperm(n_P, device=self.device)
            perm_u = torch.randperm(n_U, device=self.device)
            for b in range(n_batches):
                p_start = (b * half_bs) % n_P
                p_end = min(p_start + half_bs, n_P)
                u_start = (b * half_bs) % n_U
                u_end = min(u_start + half_bs, n_U)
                p_idx = perm_p[p_start:p_end]
                u_idx = perm_u[u_start:u_end]
                if len(p_idx) == 0 or len(u_idx) == 0:
                    continue
                X = torch.cat([P_X[p_idx], U_X[u_idx]], dim=0)
                y = torch.cat([torch.ones(len(p_idx), device=self.device),
                               torch.zeros(len(u_idx), device=self.device)], dim=0)
                optimizer.zero_grad()
                F.binary_cross_entropy_with_logits(model(X).squeeze(-1), y).backward()
                optimizer.step()
        return model.state_dict(), n_P + n_U

    def _pre_evaluate_hook(self):
        """评估前计算τMI阈值并聚合（v11.0：服务端按样本量加权平均→全局统一阈值）。"""
        self.global_model.eval()
        self.client_thresholds = []
        with torch.no_grad():
            for c in range(self.n_clients):
                z_u = self.global_model(self.clients[c]['U_X']).squeeze(-1).cpu().numpy()  # logit
                tau = self._tau_mi_calibrate(z_u, k=self.tau_mi_k, m=self.tau_mi_m)
                self.client_thresholds.append(tau)
        client_ns = np.array([self.clients[c]['n_samples'] for c in range(self.n_clients)], dtype=float)
        self.global_threshold = float(np.sum(np.array(self.client_thresholds) * client_ns) / client_ns.sum())

    def fit(self, test_X=None, test_y=None, eval_interval=10, use_cosine_annealing=False, val_ratio=0.2):
        super().fit(test_X, test_y, eval_interval, use_cosine_annealing=use_cosine_annealing)
        # 阈值已在最后一轮评估前由_pre_evaluate_hook()计算
        print(f"  NTC local τMI thresholds (logit): {[f'{t:.3f}' for t in self.client_thresholds]}")
        print(f"  NTC global threshold (sample-weighted, logit): {self.global_threshold:.3f}")
        return self

    def evaluate(self, test_X, test_y):
        """v11.0: 服务端全局统一阈值（logit空间，按样本量加权平均）在测试集上评估。"""
        from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
        y_prob = self.predict_proba(test_X)
        metrics = compute_metrics(test_y, y_prob)  # AUC等排序指标
        # 全局统一阈值（logit空间）
        self.global_model.eval()
        with torch.no_grad():
            X_dev = torch.tensor(test_X, dtype=torch.float32, device=self.device)
            logits = self.global_model(X_dev).squeeze(-1).cpu().numpy()
        y_pred = (logits >= self.global_threshold).astype(int)
        metrics['F1'] = float(f1_score(test_y, y_pred, zero_division=0) * 100)
        metrics['Recall'] = float(recall_score(test_y, y_pred, zero_division=0) * 100)
        metrics['Precision'] = float(precision_score(test_y, y_pred, zero_division=0) * 100)
        metrics['ACC'] = float(accuracy_score(test_y, y_pred) * 100)
        # 阈值统计
        metrics['threshold_global'] = self.global_threshold
        metrics['threshold_mean'] = float(np.mean(self.client_thresholds))
        metrics['threshold_std'] = float(np.std(self.client_thresholds))
        metrics['threshold_min'] = float(np.min(self.client_thresholds))
        metrics['threshold_max'] = float(np.max(self.client_thresholds))
        return metrics

class UniPUFed(FederatedPUBase):
    def __init__(self, K=5, alpha_pos=0.1, alpha_neg=0.1,
                 use_lla=True, dynamic_local_pi=True, local_pi_true=None, non_negative=True, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.use_lla = use_lla  # False: 均匀权重（旧消融）
        self.dynamic_local_pi = dynamic_local_pi  # False: 固定local_pi（w/o Dynamic消融）
        self.local_pi_true = local_pi_true  # 非None: 用真实local_pi（w/o LLA消融）
        self.non_negative = non_negative  # True: nnPU（有截断）, False: uPU（无截断）
        self._fixed_local_pi = None  # 用于 fixed 模式缓存第一轮 local_pi
        self.local_pi_est = None  # 训练后保存的最终local_pi估计（跨客户端平均）

    def fit(self, test_X=None, test_y=None, eval_interval=10, use_cosine_annealing=False):
        """训练后保存最终local_pi估计（跨客户端平均）。"""
        super().fit(test_X, test_y, eval_interval, use_cosine_annealing)
        # 训练后用最终模型估计local_pi（跨客户端平均）
        if self.use_lla and self.local_pi_true is None:
            local_pi_list = []
            for c in range(self.n_clients):
                g = self._estimate_local_pi_lla(self.clients[c], self.global_model)
                local_pi_list.append(g)
            self.local_pi_est = np.mean(local_pi_list, axis=0)
            print(f"  [UniPU] final local_pi_est: {[f'{g:.3f}' for g in self.local_pi_est]}")
        return self

    def _kmeans_cluster_client(self, client_data):
        """对单个客户端的U集做k-means聚类，P分配到最近簇中心。
        返回 (U_cluster_np, P_cluster_np)。"""
        from sklearn.cluster import KMeans
        U_X_np = client_data['U_X']
        P_X_np = client_data['P_X']
        # 展平为2D（CIFAR-10是(N,3,32,32)→(N,3072)）
        U_flat = U_X_np.reshape(len(U_X_np), -1)
        P_flat = P_X_np.reshape(len(P_X_np), -1) if len(P_X_np) > 0 else P_X_np
        k = min(self.K, len(U_flat))
        if k < 2:
            # 样本太少，全归为簇0
            U_cluster = np.zeros(len(U_flat), dtype=int)
            P_cluster = np.zeros(len(P_flat), dtype=int) if len(P_flat) > 0 else np.array([], dtype=int)
            return U_cluster, P_cluster
        km = KMeans(n_clusters=k, init='k-means++', n_init=30, random_state=self.seed)
        U_cluster = km.fit_predict(U_flat).astype(int)
        # P分配到最近簇中心
        if len(P_flat) > 0:
            P_cluster = km.predict(P_flat).astype(int)
        else:
            P_cluster = np.array([], dtype=int)
        return U_cluster, P_cluster

    def setup_clients(self, clients_data):
        """重写setup_clients：如果数据中已有真实簇标签则使用，否则对每个客户端的U做k-means聚类。"""
        super().setup_clients(clients_data)
        # 检测是否已有真实簇标签（基类已从data中读取到client['P_cluster']）
        has_true_cluster = False
        if len(self.clients) > 0 and 'P_cluster' in self.clients[0]:
            # 如果P_cluster不全是0，认为有真实簇标签
            all_zero = all((c['P_cluster'] == 0).all().item() for c in self.clients)
            has_true_cluster = not all_zero

        if has_true_cluster:
            print(f"  [UniPU-Fed] 使用真实簇标签, K={self.K}")
        else:
            print(f"  [UniPU-Fed] k-means聚类: {self.n_clients}客户端, K={self.K}")
            for i, (client, data) in enumerate(zip(self.clients, clients_data)):
                U_cluster, P_cluster = self._kmeans_cluster_client(data)
                client['U_cluster'] = torch.tensor(U_cluster, dtype=torch.long, device=self.device)
                client['P_cluster'] = torch.tensor(P_cluster, dtype=torch.long, device=self.device)
                print(f"    客户端{i}: U簇分布={[int((U_cluster==k).sum()) for k in range(self.K)]}, "
                      f"P簇分布={[int((P_cluster==k).sum()) for k in range(self.K)]}")

    def _estimate_local_pi_lla(self, client, model):
        """LLA估计γ = P(C=i|Y=1)（正例中簇i的比例）。
        步骤：1) 估计簇内正例率 P(Y=1|C=i)
              2) 转换为 P(C=i|Y=1) = P(Y=1|C=i)*P(C=i)/P(Y=1)
        """
        model.eval()
        with torch.no_grad():
            U_s = torch.sigmoid(model(client['U_X']).squeeze(-1))
            P_s = torch.sigmoid(model(client['P_X']).squeeze(-1))
        pi_est = np.ones(self.K) / self.K  # 簇内正例率 P(Y=1|C=i)
        n_pa_list = np.zeros(self.K)  # v17：每簇LLA锚点数（top-α正例锚点），无效簇记0
        for k in range(self.K):
            U_mask_k = (client['U_cluster'] == k)
            P_mask_k = (client['P_cluster'] == k)
            if U_mask_k.sum() < 5 or P_mask_k.sum() < 5:
                continue
            U_s_k, P_s_k = U_s[U_mask_k], P_s[P_mask_k]
            n_pa = max(1, int(len(P_s_k) * self.alpha_pos))
            n_pa_list[k] = n_pa
            a_hat = torch.topk(P_s_k, n_pa).values.mean().item()
            n_na = max(1, int(len(U_s_k) * self.alpha_neg))
            b_hat = torch.topk(U_s_k, n_na, largest=False).values.mean().item()
            m_hat = U_s_k.mean().item()
            if abs(a_hat - b_hat) > 1e-6:
                pi_est[k] = np.clip((m_hat - b_hat) / (a_hat - b_hat), 0.01, 0.99)

        # 转换：P(C=i|Y=1) = P(Y=1|C=i) * P(C=i) / P(Y=1)
        # p_i = P(C=i) 从U集估计（U集近似总体分布）
        p = np.zeros(self.K)
        for k in range(self.K):
            p[k] = (client['U_cluster'] == k).sum().item() / max(len(client['U_cluster']), 1)
        # local_pi_i = pi_est_i * p_i / sum_j(pi_est_j * p_j)
        gp = pi_est * p
        if gp.sum() > 0:
            local_pi_est = gp / gp.sum()
        else:
            local_pi_est = np.ones(self.K) / self.K
        client['anchor_counts'] = n_pa_list  # v17：记录每簇LLA锚点数（每客户端，训练末保留最后值）
        return local_pi_est

    def _get_local_pi(self, client, model, round_idx):
        """获取 local_pi：支持消融变体。
        - local_pi_true非None: 用真实 local_pi（w/o LLA）
        - use_lla=False: 均匀 local_pi（旧消融）
        - dynamic_local_pi=False: 固定第一轮 local_pi（w/o Dynamic）
        """
        if self.local_pi_true is not None:
            return self.local_pi_true
        if not self.use_lla:
            return np.ones(self.K) / self.K
        if not self.dynamic_local_pi:
            if self._fixed_local_pi is None:
                self._fixed_local_pi = self._estimate_local_pi_lla(client, model)
            return self._fixed_local_pi
        return self._estimate_local_pi_lla(client, model)

    def _local_train(self, client, global_state, round_idx):
        """簇级加权 nnPU（对齐论文§4.3公式和Theorem 1无偏性）。
        R = pi * sum_i local_pi_i * E_{P_i}[l+] + max(0, E_U[l-] - pi * sum_i local_pi_i * E_{P_i}[l-])
        用 local_pi 直接加权簇内均值（与 PUe NIPW 同构，sum local_pi=1）。
        UniPU 和 Oracle 共用此训练循环，仅 local_pi 来源不同。
        """
        model = self._make_model()
        model.load_state_dict(global_state)
        local_pi = self._get_local_pi(client, model, round_idx)
        local_pi_tensor = torch.tensor(local_pi, dtype=torch.float32, device=self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        P_X, U_X = client['P_X'], client['U_X']
        P_cluster = client['P_cluster']
        n_P, n_U = len(P_X), len(U_X)
        half_bs = self.batch_size // 2
        # n_batches 与 nnPU/PUe 对齐：max(n_P, n_U) // half_bs
        n_batches = max(1, max(n_P, n_U) // max(half_bs, 1))
        for _ in range(self.local_epochs):
            perm_p = torch.randperm(n_P, device=self.device)
            perm_u = torch.randperm(n_U, device=self.device)
            for b in range(n_batches):
                p_start = (b * half_bs) % n_P
                p_end = min(p_start + half_bs, n_P)
                u_start = (b * half_bs) % n_U
                u_end = min(u_start + half_bs, n_U)
                p_idx = perm_p[p_start:p_end]
                u_idx = perm_u[u_start:u_end]
                if len(p_idx) == 0 or len(u_idx) == 0:
                    continue
                batch_p_X = P_X[p_idx]
                batch_p_cluster = P_cluster[p_idx]
                batch_u_X = U_X[u_idx]
                logits_p = model(batch_p_X).squeeze(-1)
                logits_u = model(batch_u_X).squeeze(-1)
                # 簇级加权：按簇分别计算 E_{P_i}[l+] 和 E_{P_i}[l-]，然后 sum_i local_pi_i * E_{P_i}[l]
                l_plus_p = F.softplus(-logits_p)
                l_minus_p = F.softplus(logits_p)
                cluster_loss_plus = torch.zeros(self.K, device=self.device)
                cluster_loss_neg = torch.zeros(self.K, device=self.device)
                for k in range(self.K):
                    mask = batch_p_cluster == k
                    if mask.sum() > 0:
                        cluster_loss_plus[k] = l_plus_p[mask].mean()
                        cluster_loss_neg[k] = l_minus_p[mask].mean()
                weighted_l_plus = (local_pi_tensor * cluster_loss_plus).sum()
                weighted_l_minus = (local_pi_tensor * cluster_loss_neg).sum()
                l_minus_u = F.softplus(logits_u).mean()
                neg_risk = l_minus_u - self.pi * weighted_l_minus
                if self.non_negative:
                    neg_risk = torch.clamp(neg_risk, min=0)
                risk = self.pi * weighted_l_plus + neg_risk
                optimizer.zero_grad()
                risk.backward()
                optimizer.step()
        return model.state_dict(), n_P + n_U

class UniPUClusterFed(UniPUFed):
    """UniPU-Fed（复制改造）：方法内部使用替代分簇方法生成簇标签。

    与 UniPUFed 的训练循环完全一致（继承 _local_train/_estimate_local_pi_lla），
    仅 setup_clients 中的分簇逻辑不同：
    - cluster_method='true'    : 使用数据自带的真实簇标签（主配置，消融基准）
    - cluster_method='random'  : 客户端本地 P/U 样本随机分配 K 簇
    - cluster_method='kmeans'  : P集锚定 KMeans(n_init=30)，U分配到最近簇中心
    - cluster_method='gmm'     : P集锚定 GaussianMixture(diag)，U用predict分配
    - cluster_method='spectral': P集锚定 SpectralClustering，U分配到最近簇中心

    领域标签不可用时的分簇原则（簇纯度·文档）：先用纯正例 P 集估计簇结构
    （对 P 聚类取簇中心），再将 U 样本（含负例）分配到最近簇中心——使 U 簇
    结构与 P 结构对齐，比独立聚类 U 更可能产生"簇内单一类别"。

    数据生成（π/η偏倚构造、P/U采样）不受影响——消融只改变方法内部的分簇。

    分簇质量评估（ground truth = 细粒度类别标签，如MNIST数字0-9）：
    Purity / Homogeneity / V-measure / NMI / ARI，逐客户端对P集、U集计算。
    """

    def __init__(self, K=5, alpha_pos=0.1, alpha_neg=0.1,
                 use_lla=True, dynamic_local_pi=True, local_pi_true=None, non_negative=True,
                 cluster_method='true', cluster_k=None, native_layout=None,
                 semantic_min_size=20, semantic_skip_unviable=True,
                 semantic_var_margin=0.2, spectral_pca=0,
                 spectral_neighbors=10, spectral_on='PU', **kwargs):
        super().__init__(K=K,
                         alpha_pos=alpha_pos, alpha_neg=alpha_neg,
                         use_lla=use_lla, dynamic_local_pi=dynamic_local_pi,
                         local_pi_true=local_pi_true, non_negative=non_negative, **kwargs)
        self.cluster_method = cluster_method
        self.cluster_k = cluster_k  # 分簇数覆盖参数（K扫描用，None=沿用self.K）
        self.native_layout = native_layout  # 【v-nat】{'n_cont':38,'blocks':[(lo,hi),...]}
        self.semantic_min_size = semantic_min_size  # 【v-sem】贪心分区的 LLA 最小簇规模
        self.semantic_skip_unviable = semantic_skip_unviable  # 属性不可行时跳过
        self.semantic_var_margin = semantic_var_margin  # 【v-sem2】连续切分方差下降要求
        self.spectral_pca = spectral_pca  # 【图像】谱聚类前 PCA 维数；None=自动(>64 维取 64)
        self.spectral_neighbors = spectral_neighbors  # 【图像】kNN 图邻居数
        self.spectral_on = spectral_on  # 【图像】'PU'=联合聚类 / 'P'=仅P聚类+最近中心
        if cluster_k is not None:
            # K扫描：下游γ估计/簇加权损失均按self.K展开（range(self.K)/zeros(self.K)），
            # 必须同步覆盖self.K，否则簇≥K的样本在估计与加权中被忽略
            self.K = cluster_k
        self.cluster_quality = None  # 分簇质量指标（跨客户端均值）
        self.cluster_quality_per_client = None  # 逐客户端明细
        self.cluster_audit = None  # 簇审计（min_cluster_size/anchors_per_cluster）
        self.cluster_gain_per_client = []  # 【v-nat/v-sem】逐客户端 gain 与分区审计

    def _assign_to_centers(self, flat_X, centers):
        """将样本分配到最近簇中心（用于谱聚类的U集分配）。"""
        return np.argmin(
            np.linalg.norm(flat_X[:, None, :] - centers[None, :, :], axis=2),
            axis=1).astype(int)

    # ------------------------------------------------------------------
    # 【v-nat / v-sem】基于 XGBoost gain 的原生类别加权与语义贪心分区
    # ------------------------------------------------------------------
    def _to_codes(self, X):
        """把 one-hot 类别块反解成整数编码（数值列保持原样）。"""
        layout = self.native_layout
        n_cont = int(layout['n_cont'])
        cols = [X[:, :n_cont]]
        for lo, hi in layout['blocks']:
            cols.append(np.argmax(X[:, lo:hi], axis=1).reshape(-1, 1))
        return np.hstack(cols).astype(np.float32)

    def _native_feature_gain(self, P_flat, U_flat):
        """【v-nat】原生类别特征上的 gain：每个类别特征只得到一个值。"""
        import pandas as pd
        from xgboost import XGBClassifier

        layout = self.native_layout
        n_cont = int(layout['n_cont'])
        blocks = list(layout['blocks'])
        Pc, Uc = self._to_codes(P_flat), self._to_codes(U_flat)
        X = np.vstack([Pc, Uc])
        y = np.concatenate([np.ones(len(Pc), dtype=np.int8),
                            np.zeros(len(Uc), dtype=np.int8)])
        names = [f'num{i}' for i in range(n_cont)] + \
                [f'cat{j}' for j in range(len(blocks))]
        df = pd.DataFrame(X, columns=names)
        for j, (lo, hi) in enumerate(blocks):
            df[f'cat{j}'] = pd.Categorical(df[f'cat{j}'].astype(int),
                                          categories=list(range(hi - lo)))
        clf = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8,
            colsample_bytree=1.0,
            scale_pos_weight=float(len(Uc)) / max(len(Pc), 1),
            tree_method='hist', enable_categorical=True, n_jobs=1,
            random_state=int(self.seed))
        clf.fit(df, y, verbose=False)
        return np.asarray(clf.feature_importances_, dtype=np.float64)

    def _native_gain_weights(self, P_flat, U_flat):
        """【v-nat】原生 gain 展开回 one-hot 块权重（每块一个值）。"""
        g = self._native_feature_gain(P_flat, U_flat)
        d = P_flat.shape[1]
        n_cont = int(self.native_layout['n_cont'])
        w = np.zeros(d, dtype=np.float64)
        w[:n_cont] = g[:n_cont]
        for j, (lo, hi) in enumerate(self.native_layout['blocks']):
            w[lo:hi] = g[n_cont + j]
        return w

    @staticmethod
    def _mask(codes, rules):
        m = np.ones(len(codes), dtype=bool)
        for r in rules:
            if r[0] == 'cat':
                m &= (codes[:, r[1]].astype(int) == r[2])
            elif r[0] == 'catin':
                m &= np.isin(codes[:, r[1]].astype(int), r[2])
            elif r[0] == 'le':
                m &= (codes[:, r[1]] <= r[2])
            elif r[0] == 'gt':
                m &= (codes[:, r[1]] > r[2])
        return m

    def _native_gain_and_probs(self, P_flat, U_flat):
        """【v-sem2】原生类别 gain + 正例上的 XGBoost 后验概率（同一个模型）。"""
        import pandas as pd
        from xgboost import XGBClassifier

        layout = self.native_layout
        n_cont = int(layout['n_cont'])
        blocks = list(layout['blocks'])
        Pc, Uc = self._to_codes(P_flat), self._to_codes(U_flat)
        X = np.vstack([Pc, Uc])
        y = np.concatenate([np.ones(len(Pc), dtype=np.int8),
                            np.zeros(len(Uc), dtype=np.int8)])
        names = [f'num{i}' for i in range(n_cont)] + \
                [f'cat{j}' for j in range(len(blocks))]
        df = pd.DataFrame(X, columns=names)
        for j, (lo, hi) in enumerate(blocks):
            df[f'cat{j}'] = pd.Categorical(df[f'cat{j}'].astype(int),
                                          categories=list(range(hi - lo)))
        clf = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8,
            colsample_bytree=1.0,
            scale_pos_weight=float(len(Uc)) / max(len(Pc), 1),
            tree_method='hist', enable_categorical=True, n_jobs=1,
            random_state=int(self.seed))
        clf.fit(df, y, verbose=False)
        g = np.asarray(clf.feature_importances_, dtype=np.float64)
        ps = clf.predict_proba(df.iloc[:len(Pc)])[:, 1]
        return g, ps

    def _cat_hierarchical(self, f, idx, rules, Pc, ps, min_size):
        """【修法A】离散特征层级二叉切分。
        先按各类别在 P 上的平均后验排序（名义特征也适用；有序年龄退化为自然序），
        再在相邻类别之间做二叉切分，两侧均须 >= min_size；小类与相邻类合并，
        不产生 catch-all，所有 P 仍在叶子里。返回 [(leaf_rules, leaf_idx), ...]。
        """
        col = Pc[idx, f].astype(int)
        vals = np.unique(col)
        if len(vals) < 2 or len(idx) < 2 * min_size:
            return [(rules + (('catin', f, tuple(int(v) for v in vals)),), idx)]
        order_vals = sorted(
            (int(v) for v in vals),
            key=lambda v: float(ps[idx[col == v]].mean()))
        return self._cat_hier_rec(f, order_vals, idx, rules, Pc, ps, min_size)

    def _cat_hier_rec(self, f, vals, idx, rules, Pc, ps, min_size):
        col = Pc[idx, f].astype(int)
        if len(vals) < 2 or len(idx) < 2 * min_size:
            return [(rules + (('catin', f, tuple(vals)),), idx)]
        best = None
        for k in range(1, len(vals)):
            lv, rv = vals[:k], vals[k:]
            in_left = np.isin(col, lv)
            li, ri = idx[in_left], idx[~in_left]
            if len(li) < min_size or len(ri) < min_size:
                continue
            vc = (len(li) * np.var(ps[li])
                  + len(ri) * np.var(ps[ri])) / len(idx)
            if best is None or vc < best[0]:
                best = (vc, li, ri, lv, rv)
        if best is None:
            return [(rules + (('catin', f, tuple(vals)),), idx)]
        _, li, ri, lv, rv = best
        return (self._cat_hier_rec(f, lv, li, rules, Pc, ps, min_size)
                + self._cat_hier_rec(f, rv, ri, rules, Pc, ps, min_size))

    def _semantic_greedy_v2(self, P_flat, U_flat):
        """【v-sem2】离散按取值分簇、连续按后验方差分簇；不设簇数上限。"""
        min_size = int(self.semantic_min_size)
        gamma = float(self.semantic_var_margin)
        g, ps = self._native_gain_and_probs(P_flat, U_flat)
        n_cont = int(self.native_layout['n_cont'])
        n_feat = len(g)
        Pc, Uc = self._to_codes(P_flat), self._to_codes(U_flat)
        order = list(np.argsort(-g))

        cells = [((), np.arange(len(Pc)))]
        used = []
        log = []
        for f in order:
            new_cells, changed = [], False
            for rules, idx in cells:
                if len(idx) < 2 * min_size:
                    new_cells.append((rules, idx))
                    continue
                if f < n_cont:      # 连续：阈值扫描 + 簇内后验方差判据
                    vals = Pc[idx, f]
                    var_p = float(np.var(ps[idx]))
                    cands = np.unique(np.quantile(vals, np.linspace(0.1, 0.9, 9)))
                    best = None
                    for thr in cands:
                        left = idx[vals <= thr]
                        right = idx[vals > thr]
                        if len(left) < min_size or len(right) < min_size:
                            continue
                        var_c = (len(left) * np.var(ps[left])
                                 + len(right) * np.var(ps[right])) / len(idx)
                        if best is None or var_c < best[0]:
                            best = (var_c, float(thr), left, right)
                    if best is not None and best[0] < (1 - gamma) * var_p:
                        vc, thr, left, right = best
                        new_cells.append((rules + (('le', f, thr),), left))
                        new_cells.append((rules + (('gt', f, thr),), right))
                        log.append((f, 'cont', thr, round(var_p, 4),
                                    round(vc, 4)))
                        changed = True
                        continue
                else:
                    # 【修法A】离散层级二叉：小类与相邻类合并，每叶 >= min_size
                    parts = self._cat_hierarchical(
                        f, idx, rules, Pc, ps, min_size)
                    if len(parts) >= 2:
                        new_cells.extend(parts)
                        log.append((f, 'cat_hier', None, None, None))
                        changed = True
                        continue
                new_cells.append((rules, idx))
            if changed:
                cells = new_cells
                used.append(int(f))
            elif not self.semantic_skip_unviable:
                break

        # 逐叶给 P 打标签（方案D：所有 P 都在 cells 里，无 loose_P）
        P_cluster = np.full(len(Pc), -1, dtype=int)
        for c, (_, idx) in enumerate(cells):
            P_cluster[idx] = c
        # U：规则精确匹配优先；未命中任何叶规则的 U 不做最近质心兜底，
        # 保留 -1 标记，由 setup_clients 直接丢弃（支撑域限制，保证叶内纯度）。
        U_cluster = np.full(len(Uc), -1, dtype=int)
        for c, (rules, _) in enumerate(cells):
            m = self._mask(Uc, rules)
            U_cluster[m] = c
        n_dropped = int((U_cluster < 0).sum())

        info = {'weighting': 'semantic_greedy_v2_hier', 'n_features': int(n_feat),
                'n_features_nonzero_gain': int((g > 0).sum()),
                'top_features': [{'index': int(i), 'gain': float(g[i])}
                                 for i in np.argsort(-g)[:10] if g[i] > 0],
                'fallback': None if used else 'no_split',
                'vector': g, 'used_attrs': used,
                'n_cells': int(len(cells)),
                'n_dropped_u': n_dropped,
                'split_log': log[:60]}
        return U_cluster, P_cluster, info

    def _semantic_greedy_cluster(self, P_flat, U_flat):
        """【v-sem】贪心语义分区（仅离散属性，保留作对照）。"""
        min_size = int(self.semantic_min_size)
        max_cells = int(self.cluster_k or self.K)
        g = self._native_feature_gain(P_flat, U_flat)
        n_cont = int(self.native_layout['n_cont'])
        n_cat = len(self.native_layout['blocks'])
        cat_cols = [n_cont + j for j in range(n_cat)]
        order = list(np.argsort(-g[n_cont:]))
        Pc, Uc = self._to_codes(P_flat), self._to_codes(U_flat)

        cells = [({}, np.arange(len(Pc)))]
        used = []
        for j in order:
            col = cat_cols[j]
            new = []
            for cons, idx in cells:
                if cons.get('__other__'):
                    new.append((cons, idx))
                    continue
                vals = Pc[idx, col].astype(int)
                for v in np.unique(vals):
                    c2 = dict(cons)
                    c2[j] = int(v)
                    new.append((c2, idx[vals == v]))
            if len(new) <= len(cells):
                continue
            big = [(c, i) for c, i in new if len(i) >= min_size]
            small = [i for c, i in new if len(i) < min_size]
            if small:
                big.append(({'__other__': True}, np.concatenate(small)))
            if len(big) < 2 or min(len(i) for _, i in big) < min_size:
                if self.semantic_skip_unviable:
                    continue
                break
            cells, used = big, used + [j]
            if len(cells) >= max_cells:
                break

        P_cluster = np.zeros(len(Pc), dtype=int)
        for c, (_, idx) in enumerate(cells):
            P_cluster[idx] = c
        catch = None
        for c, (cons, _) in enumerate(cells):
            if cons.get('__other__'):
                catch = c
        U_cluster = np.full(len(Uc), catch if catch is not None else 0, dtype=int)
        for c, (cons, _) in enumerate(cells):
            if cons.get('__other__'):
                continue
            m = np.ones(len(Uc), dtype=bool)
            for j, v in cons.items():
                m &= (Uc[:, cat_cols[j]].astype(int) == v)
            U_cluster[m] = c
        info = {'weighting': 'semantic_greedy', 'n_features': int(len(g)),
                'n_features_nonzero_gain': int((g > 0).sum()),
                'top_features': [{'index': int(i), 'gain': float(g[i])}
                                 for i in np.argsort(-g)[:10] if g[i] > 0],
                'fallback': None if used else 'no_split',
                'vector': g, 'used_attrs': [int(j) for j in used],
                'n_cells': int(len(cells))}
        return U_cluster, P_cluster, info

    def _cluster_client(self, client_data):
        """P集锚定分簇：对P聚类取簇中心，U分配到最近簇中心。
        返回 (U_cluster_np, P_cluster_np)。

        【双输入】若 clients_data 里带了 'P_X_cluster'/'U_X_cluster'（分簇专用
        特征，与训练输入解耦，例如"训练用图像、分簇用元数据"），优先用它分簇；
        否则用模型输入 X。
        """
        U_X_np = client_data.get('U_X_cluster', client_data['U_X'])
        P_X_np = client_data.get('P_X_cluster', client_data['P_X'])
        # 展平为2D（CIFAR-10是(N,3,32,32)→(N,3072)）
        U_flat = U_X_np.reshape(len(U_X_np), -1)
        P_flat = P_X_np.reshape(len(P_X_np), -1) if len(P_X_np) > 0 else P_X_np
        # K扫描：cluster_k覆盖self.K（谱聚类分簇数）；样本数不足时取min保护
        k = min(self.cluster_k if self.cluster_k is not None else self.K, len(U_flat))
        if k < 2 or len(P_flat) < 2:
            # 样本太少，全归为簇0
            U_cluster = np.zeros(len(U_flat), dtype=int)
            P_cluster = np.zeros(len(P_flat), dtype=int) if len(P_flat) > 0 else np.array([], dtype=int)
            return U_cluster, P_cluster

        method = self.cluster_method

        if method == 'semantic_v2':
            # 【v-sem2】离散按取值、连续按后验方差，不设簇数上限
            if self.native_layout is None:
                print('    [semantic_v2] 缺少 native_layout，退化为 KMeans')
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=max(2, min(k, len(P_flat))),
                            init='k-means++', n_init=30, random_state=self.seed)
                return (km.predict(U_flat).astype(int),
                        km.fit_predict(P_flat).astype(int))
            U_cl, P_cl, info = self._semantic_greedy_v2(P_flat, U_flat)
            self.cluster_gain_per_client.append(info)
            return U_cl, P_cl

        if method == 'semantic_greedy':
            # 【v-sem】仅离散属性的贪心分区（对照用）
            if self.native_layout is None:
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=max(2, min(k, len(P_flat))),
                            init='k-means++', n_init=30, random_state=self.seed)
                return (km.predict(U_flat).astype(int),
                        km.fit_predict(P_flat).astype(int))
            U_cl, P_cl, info = self._semantic_greedy_cluster(P_flat, U_flat)
            self.cluster_gain_per_client.append(info)
            return U_cl, P_cl

        if method == 'pulsnar_native':
            # 【v-nat】原生 gain 加权后再做 KMeans
            if self.native_layout is not None:
                from sklearn.cluster import KMeans
                w = self._native_gain_weights(P_flat, U_flat)
                km = KMeans(n_clusters=k, init='k-means++', n_init=30,
                            random_state=self.seed)
                P_cluster = km.fit_predict(P_flat * w).astype(int)
                U_cluster = km.predict(U_flat * w).astype(int)
                return U_cluster, P_cluster

        if method == 'random':
            # 随机簇标签：P/U样本各自随机分配K簇
            rng = np.random.RandomState(self.seed)
            U_cluster = rng.randint(0, k, size=len(U_flat)).astype(int)
            if len(P_flat) > 0:
                P_cluster = rng.randint(0, k, size=len(P_flat)).astype(int)
            else:
                P_cluster = np.array([], dtype=int)
            return U_cluster, P_cluster

        if method == 'kmeans':
            from sklearn.cluster import KMeans
            # P集锚定：P聚类取中心，U/P分配到最近簇中心
            km = KMeans(n_clusters=k, init='k-means++', n_init=30, random_state=self.seed)
            P_cluster = km.fit_predict(P_flat).astype(int)
            U_cluster = km.predict(U_flat).astype(int)
            return U_cluster, P_cluster

        if method == 'gmm':
            from sklearn.mixture import GaussianMixture
            # P集锚定：P拟合GMM(diag)，U用predict分配
            # diag协方差：784维下full协方差会奇异（每簇样本远小于维度）
            gmm = GaussianMixture(n_components=k, covariance_type='diag',
                                  random_state=self.seed)
            P_cluster = gmm.fit_predict(P_flat).astype(int)
            U_cluster = gmm.predict(U_flat).astype(int)
            return U_cluster, P_cluster

        if method == 'spectral':
            # 【图像数据集】像素上的谱聚类。两种模式由 spectral_on 控制：
            #   'PU'（默认）：直接对 P∪U 联合聚类，一次划分同时得到 P/U 的簇标签
            #   'P'         ：只在 P 上聚类，U 按最近簇中心分配
            # spectral_pca>0 时先做 PCA 降维（基只用被聚类的样本拟合）。
            from sklearn.cluster import SpectralClustering
            from sklearn.decomposition import PCA

            Xp = np.asarray(P_flat, dtype=np.float64)
            Xu = np.asarray(U_flat, dtype=np.float64)
            joint = str(self.spectral_on).upper() == 'PU'
            n_p = len(Xp)
            if joint:
                X = np.vstack([Xp, Xu]) if n_p > 0 else Xu
            else:
                X = Xp
            k_eff = max(2, min(int(k), len(X)))

            n_pca = int(self.spectral_pca or 0)
            n_pca = int(min(n_pca, X.shape[1], max(2, len(X) - 1)))
            if n_pca >= 2:
                pca = PCA(n_components=n_pca, random_state=self.seed)
                X = pca.fit_transform(X)

            nn = int(min(max(2, self.spectral_neighbors), len(X) - 1))
            sc = SpectralClustering(n_clusters=k_eff,
                                    affinity='nearest_neighbors',
                                    n_neighbors=nn, assign_labels='kmeans',
                                    n_init=10, random_state=self.seed)
            lab = sc.fit_predict(X).astype(int)

            if joint:
                P_cluster = lab[:n_p]
                U_cluster = lab[n_p:]
            else:
                P_cluster = lab
                centers = np.array([
                    X[P_cluster == j].mean(axis=0)
                    if (P_cluster == j).sum() > 0 else np.zeros(X.shape[1])
                    for j in range(k_eff)])
                U_cluster = self._assign_to_centers(Xu, centers).astype(int)

            self.cluster_gain_per_client.append({
                'weighting': f'spectral_{"joint" if joint else "P_anchored"}',
                'n_features': int(P_flat.shape[1]),
                'n_samples_clustered': int(len(X)),
                'pca_dim': int(n_pca),
                'n_neighbors': int(nn),
                'n_clusters': int(k_eff),
                'fallback': None, 'vector': None})
            return U_cluster, P_cluster

        raise ValueError(f"未知cluster_method: {method}（支持 true/random/kmeans/gmm/spectral）")

    @staticmethod
    def _purity(labels_true, labels_pred):
        """Purity：每个簇中主导真实类别占比的加权平均。"""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels_true, labels_pred)
        # cm[i,j] = 真实类i、预测簇j的样本数；每簇主导类 = cm[:,j].max()
        return float(cm.max(axis=0).sum() / cm.sum())

    def _compute_cluster_quality(self, clients_data):
        """逐客户端计算分簇质量指标（P集/U集分别），跨客户端平均。
        需要 client_data['P_digit']/['U_digit']（细粒度类别标签）。"""
        from sklearn.metrics import (
            homogeneity_score, v_measure_score,
            normalized_mutual_info_score, adjusted_rand_score,
        )
        per_client = []
        for i, (client, data) in enumerate(zip(self.clients, clients_data)):
            q = {}
            for split in ('P', 'U'):
                key_digit = f'{split}_digit'
                if key_digit not in data:
                    continue
                cluster_np = client[f'{split}_cluster'].cpu().numpy()
                digit_np = data[key_digit]
                if len(np.unique(cluster_np)) < 2 or len(np.unique(digit_np)) < 2:
                    continue
                q[split] = {
                    'purity': self._purity(digit_np, cluster_np),
                    'homogeneity': float(homogeneity_score(digit_np, cluster_np)),
                    'v_measure': float(v_measure_score(digit_np, cluster_np)),
                    'nmi': float(normalized_mutual_info_score(digit_np, cluster_np)),
                    'ari': float(adjusted_rand_score(digit_np, cluster_np)),
                }
            per_client.append(q)
        if not per_client:
            return None, None
        # 跨客户端平均（按split维度）
        splits = [k for k in per_client[0].keys()]
        agg = {}
        for split in splits:
            agg[split] = {}
            for metric in ('purity', 'homogeneity', 'v_measure', 'nmi', 'ari'):
                vals = [q[split][metric] for q in per_client if split in q]
                agg[split][metric] = float(np.mean(vals))
        return agg, per_client

    def setup_clients(self, clients_data):
        """重写setup_clients：按cluster_method在方法内部重新分簇。

        - 'true'：保留数据自带的真实簇标签（主配置）
        - 其他：忽略真实簇标签，用替代方法对每个客户端的P/U本地分簇
        分簇完成后计算簇质量指标（ground truth=细粒度类别标签）。
        """
        super().setup_clients(clients_data)
        if self.cluster_method == 'true':
            print(f"  [UniPU-Cluster] cluster_method=true：使用数据自带真实簇标签, K={self.K}")
        else:
            eff_k = self.cluster_k if self.cluster_k is not None else self.K
            print(f"  [UniPU-Cluster] 方法内部分簇 cluster_method={self.cluster_method} "
                  f"（P集锚定）: {self.n_clients}客户端, K={eff_k}"
                  f"{'' if self.cluster_k is None else f' (cluster_k覆盖={self.cluster_k})'}")
            for i, (client, data) in enumerate(zip(self.clients, clients_data)):
                U_cluster, P_cluster = self._cluster_client(data)
                # 分簇阶段支撑域限制：未命中任何叶规则的 U（标签 -1）直接丢弃，
                # 真删 U 侧所有对齐张量；训练流程与标准 LLA 不改动。
                keep = U_cluster >= 0
                n_drop_u = int((~keep).sum())
                valid_u = U_cluster[keep]
                # 簇级加权只覆盖 range(self.K)（_local_train），超出 K 的簇
                # 会被静默漏加权，故在此硬性拦截（含 P/U 两侧的最大簇号）。
                max_u = int(valid_u.max()) if len(valid_u) else -1
                max_p = int(P_cluster.max()) if len(P_cluster) else -1
                n_cells = max(max_u, max_p) + 1
                assert n_cells <= int(self.K), (
                    f'客户端{i} 分簇数 {n_cells} 超过 self.K={self.K}：'
                    f'簇级加权只覆盖 range(self.K)，簇外样本会被静默漏加权')
                keep_t = torch.tensor(keep, dtype=torch.bool, device=self.device)
                client['U_X'] = client['U_X'][keep_t]
                client['U_y'] = client['U_y'][keep_t]
                client['U_y_true'] = client['U_y_true'][keep_t]
                client['U_cluster'] = torch.tensor(valid_u, dtype=torch.long, device=self.device)
                client['U_class'] = client['U_class'][keep_t]
                client['P_cluster'] = torch.tensor(P_cluster, dtype=torch.long, device=self.device)
                print(f"    客户端{i}: U簇分布={[int((valid_u==k).sum()) for k in range(self.K)]}, "
                      f"P簇分布={[int((P_cluster==k).sum()) for k in range(self.K)]}, "
                      f"丢弃U={n_drop_u}")
        # 簇质量评估（含true变体；ground truth=细粒度类别标签）
        self.cluster_quality, self.cluster_quality_per_client = \
            self._compute_cluster_quality(clients_data)
        if self.cluster_quality:
            for split in ('P', 'U'):
                if split in self.cluster_quality:
                    q = self.cluster_quality[split]
                    print(f"  [簇质量 {split}集] Purity={q['purity']:.3f}, Homogeneity={q['homogeneity']:.3f}, "
                          f"V-measure={q['v_measure']:.3f}, NMI={q['nmi']:.3f}, ARI={q['ari']:.3f}")
        # v17 簇审计：min_cluster_size（P集每簇样本量最小值，含空簇=0）、
        # min_cluster_labeled（P集每簇正例数最小值）、anchors_per_cluster（LLA top-α锚点数均值）
        sizes, labeled = [], []
        for client in self.clients:
            P_cl = client['P_cluster'].cpu().numpy()
            P_y = client['P_y'].cpu().numpy()
            for k in range(int(P_cl.max()) + 1 if len(P_cl) else 0):
                m = P_cl == k
                sizes.append(int(m.sum()))
                labeled.append(int((m & (P_y == 1)).sum()))
        ac = [c['anchor_counts'] for c in self.clients if c.get('anchor_counts') is not None]
        self.cluster_audit = {
            'min_cluster_size': int(min(sizes)) if sizes else 0,
            'min_cluster_labeled': int(min(labeled)) if labeled else 0,
            'anchors_per_cluster': float(np.mean([np.mean(a) for a in ac])) if ac else 0.0,
        }
        print(f"  [簇审计] min_cluster_size={self.cluster_audit['min_cluster_size']}, "
              f"min_cluster_labeled={self.cluster_audit['min_cluster_labeled']}, "
              f"anchors_per_cluster={self.cluster_audit['anchors_per_cluster']:.2f}")

    def fit(self, test_X=None, test_y=None, eval_interval=10, use_cosine_annealing=False):
        """v17：训练结束后回填 anchors_per_cluster（LLA锚点数在训练期估计，setup时不可得）。"""
        super().fit(test_X, test_y, eval_interval, use_cosine_annealing)
        if self.cluster_audit is not None:
            ac = [c['anchor_counts'] for c in self.clients if c.get('anchor_counts') is not None]
            if ac:
                self.cluster_audit['anchors_per_cluster'] = float(np.mean([np.mean(a) for a in ac]))
                print(f"  [簇审计] anchors_per_cluster={self.cluster_audit['anchors_per_cluster']:.2f} (训练后)")

class UniPUVariantWeightFed(UniPUClusterFed):
    """UniPU 变体消融：真实簇标签 + 按指定来源注入真实簇权重（绕过 LLA）。

    强制 cluster_method='true'（真实簇标签、K=真实 K），仅簇权重来源不同：
    - local_pi    : 每客户端在自己 U 上数 U+ 的簇经验分布（per-client，基准）
    - global_pi   : 集中所有客户端 U 数 U+ 簇构成（所有客户端相同）
    - shared_gamma: 用数据生成 info['gamma']（共享库正例簇构成，所有客户端相同）
    - eta_inv     : 用 info['eta']，权重取 1/η 归一化（只抵消 P 侧偏倚、对齐 γ，全局固定）

    全 U 口径：分布计数只在 U 上做，P 不进入。标量 self.pi 保持 U 正例率
    （外部传入），不因变体改变。
    """

    VALID_WEIGHT_SOURCES = ('local_pi', 'global_pi', 'shared_gamma', 'eta_inv')

    def __init__(self, weight_source='local_pi', oracle_info=None,
                 eta_beta=1.0, **kwargs):
        kwargs['cluster_method'] = 'true'
        kwargs['cluster_k'] = None
        super().__init__(**kwargs)
        if weight_source not in self.VALID_WEIGHT_SOURCES:
            raise ValueError(f"Unknown weight_source: {weight_source}. "
                             f"Available: {self.VALID_WEIGHT_SOURCES}")
        self.weight_source = weight_source
        self.oracle_info = oracle_info
        self.eta_beta = float(eta_beta)

    @staticmethod
    def _normalize(vec):
        total = float(np.sum(vec))
        if total <= 0.0:
            raise ValueError("权重计数总和为 0，无法归一化")
        return vec / total

    def _assert_prob(self, vec):
        if len(vec) != self.K:
            raise ValueError(f"权重长度 {len(vec)} != K={self.K}")
        if not np.isfinite(vec).all():
            raise ValueError("权重含非有限值")
        if abs(float(np.sum(vec)) - 1.0) > 1e-9:
            raise ValueError(f"权重和 {float(np.sum(vec))} != 1")

    def setup_clients(self, clients_data):
        super().setup_clients(clients_data)
        # 每客户端 U+ 簇计数（真实簇标签 + 真实标签，全 U 口径）
        per_client_counts = []
        for client in self.clients:
            u_cluster = client['U_cluster'].cpu().numpy()
            u_y_true = client['U_y_true'].cpu().numpy()
            counts = np.zeros(self.K, dtype=np.float64)
            for k in range(self.K):
                counts[k] = int(np.sum((u_cluster == k) & (u_y_true == 1)))
            per_client_counts.append(counts)

        if self.weight_source == 'local_pi':
            # 本地正例簇构成（oracle）：w_k = U+_{mk} / Σ_j U+_{mj}
            # 注：早期版本曾尝试 w = U+_{mk}/p_k 反解簇质量，但实验显示
            # 在 p_k 跨簇变异大的数据集（NSL）上会把负例主导簇权重放大约10倍，
            # 使 dual-term loss 退化为单簇问题，AUC 反降。p_k 是本征簇结构而非
            # 采样扭曲，不应反解；保留原始 U+ 经验占比作为 local oracle。
            weights = [self._normalize(c) for c in per_client_counts]
        elif self.weight_source == 'global_pi':
            global_w = self._normalize(np.sum(per_client_counts, axis=0))
            weights = [global_w for _ in self.clients]
        elif self.weight_source == 'shared_gamma':
            if self.oracle_info is None:
                raise ValueError("shared_gamma 需要 oracle_info")
            gamma = np.asarray(self.oracle_info['gamma'], dtype=np.float64)
            self._assert_prob(gamma)
            weights = [gamma for _ in self.clients]
        else:  # eta_inv：NIPW 权重取 1/η（q=η⊙γ，1/η 只抵消偏倚、还原到 γ）
            if self.oracle_info is None:
                raise ValueError("eta_inv 需要 oracle_info")
            eta = np.asarray(self.oracle_info['eta'], dtype=np.float64)
            if eta.shape != (self.n_clients, self.K):
                raise ValueError(f"info['eta'] 形状 {eta.shape} != "
                                 f"(n_clients={self.n_clients}, K={self.K})")
            weights = []
            for m in range(self.n_clients):
                raw = 1.0 / np.clip(eta[m], 1e-12, None)
                weights.append(self._normalize(raw))

        for client, w in zip(self.clients, weights):
            w = np.asarray(w, dtype=np.float64)
            self._assert_prob(w)
            client['weight_override'] = w

    def _get_local_pi(self, client, model, round_idx):
        return client['weight_override']

class PUEOracleFed(PUEFed):
    """PUe Oracle: 用真实η(x)做NIPW加权，无估计阶段，全部轮数用于训练。

    验证：即使给PUe完美的倾向性估计，样本级NIPW训练是否仍有局限性。
    """

    def __init__(self, eta_base=None, client_coefs=None, **kwargs):
        # est_rounds=0，跳过估计阶段
        kwargs['est_rounds'] = 0
        super().__init__(**kwargs)
        self.eta_base = np.array(eta_base) if eta_base is not None else None
        self.client_coefs = np.array(client_coefs) if client_coefs is not None else None

    def _compute_oracle_nipw_weights(self, client, client_idx):
        """用真实η计算NIPW权重。"""
        if self.eta_base is None or self.client_coefs is None:
            # 没有真实η配置，退化为均匀权重
            n_P = len(client['P_y'])
            return torch.ones(n_P, device=self.device)

        P_cluster = client['P_cluster']
        c_m = self.client_coefs[client_idx]
        # 每个P样本的真实η = eta_base[簇标签] * 客户端系数
        eta = np.array([self.eta_base[k] * c_m for k in P_cluster])
        eta = np.clip(eta, 1e-6, 1.0)
        # NIPW权重 w = 1/η，归一化到Σ ω_i = 1（论文公式）
        raw_weights = 1.0 / eta
        raw_weights = torch.tensor(raw_weights, dtype=torch.float32, device=self.device)
        return raw_weights / raw_weights.sum()

    def fit(self, test_X=None, test_y=None, eval_interval=10, use_cosine_annealing=False):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        train_rounds = self.total_rounds  # 全部轮数用于训练
        _initial_lr = self.lr
        print(f"[PUe-Oracle] NIPW with true η, {train_rounds} rounds, cosine={use_cosine_annealing}")

        # 用真实η计算NIPW权重（每个客户端独立计算）
        client_weights = []
        for c in range(self.n_clients):
            w = self._compute_oracle_nipw_weights(self.clients[c], c)
            client_weights.append(w)
        print("  Oracle NIPW weights done (true η)")

        # 全部轮数用于分类器训练
        self.global_model = self._make_model()
        global_state = self.global_model.state_dict()
        for r in range(train_rounds):
            if use_cosine_annealing:
                half = train_rounds // 2
                if r < half:
                    self.lr = _initial_lr * (r + 1) / max(half, 1)
                else:
                    t = (r - half) / max(train_rounds - half - 1, 1)
                    self.lr = _initial_lr * 0.5 * (1.0 + np.cos(np.pi * t))
            ls_list, ns_list = [], []
            for c in range(self.n_clients):
                ls, ns = self._local_train_classifier(self.clients[c], global_state, client_weights[c])
                ls_list.append(ls); ns_list.append(ns)
            global_state = self._fedavg_aggregate(ls_list, ns_list)
            self.global_model.load_state_dict(global_state)
            if (r + 1) % eval_interval == 0 or r == train_rounds - 1:
                if test_X is not None:
                    metrics = self.evaluate(test_X, test_y)
                    self.per_round_results.append({'round': r + 1, 'metrics': metrics})
                    print(f"    train {r+1}/{train_rounds}: AUC={metrics['AUC']:.2f}, F1={metrics['F1']:.2f}")
        self.lr = _initial_lr
        print("[PUe-Oracle] done")
        return self

METHOD_REGISTRY = {
    'FedPU': FedPUOfficialFed,          # 官方 MPULoss_V2 的二分类移植
    'nnPU': NNPUFed,
    'PUe-LocalEst': PUELocalEstFed,
    'NTC': NTCFed,
    'UniPU-Fed': UniPUFed,
    'UniPU-Cluster': UniPUClusterFed,
    # 变体消融：四个名字共享 UniPUVariantWeightFed，由 get_method 注入 weight_source
    'UniPU-LocalPi': UniPUVariantWeightFed,
    'UniPU-GlobalPi': UniPUVariantWeightFed,
    'UniPU-SharedGamma': UniPUVariantWeightFed,
    'UniPU-EtaInv': UniPUVariantWeightFed,
}

# 变体注册名 → 簇权重来源（get_method 据此注入 weight_source）
VARIANT_WEIGHT_SOURCE = {
    'UniPU-LocalPi': 'local_pi',
    'UniPU-GlobalPi': 'global_pi',
    'UniPU-SharedGamma': 'shared_gamma',
    'UniPU-EtaInv': 'eta_inv',
}

def get_method(name, **kwargs):
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_REGISTRY.keys())}")
    if name in VARIANT_WEIGHT_SOURCE:
        kwargs.setdefault('weight_source', VARIANT_WEIGHT_SOURCE[name])
    return METHOD_REGISTRY[name](**kwargs)
