"""
评估指标 v2：完整指标体系。
包含：ACC, Precision, Recall, F1, AUC, AP, ECE, MCE, Brier Score
以及校准曲线(reliability_curve)、PR曲线(pr_curve)、概率分布(probability_distribution)
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, brier_score_loss,
)


def balance_test_set(y_true, X=None, seed=42):
    """下采样测试集使正负例数量相等（取min(n_pos, n_neg)）。
    注：v10.26起主实验使用原始测试集，此函数仅保留供历史脚本调用。
    """
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_min = min(len(pos_idx), len(neg_idx))
    if n_min == 0:
        if X is not None:
            return X, y_true
        return (y_true,)
    pos_sel = rng.choice(pos_idx, n_min, replace=False)
    neg_sel = rng.choice(neg_idx, n_min, replace=False)
    sel = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(sel)
    if X is not None:
        return X[sel], y_true[sel]
    return (y_true[sel],)


def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    计算所有评估指标（原始测试集，v10.26起不做平衡化）。

    Returns:
        dict of metrics，含标量指标 + reliability_curve + pr_curve + probability_distribution
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}
    metrics['ACC'] = accuracy_score(y_true, y_pred) * 100
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0) * 100
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0) * 100
    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0) * 100

    # AUC 和 AP
    try:
        metrics['AUC'] = roc_auc_score(y_true, y_prob) * 100
    except ValueError:
        metrics['AUC'] = 50.0
    try:
        metrics['AP'] = average_precision_score(y_true, y_prob) * 100
    except ValueError:
        metrics['AP'] = 0.0

    # ECE (15等宽桶)
    metrics['ECE'] = compute_ece(y_true, y_prob, n_bins=15) * 100

    # MCE (15桶)
    metrics['MCE'] = compute_mce(y_true, y_prob, n_bins=15) * 100

    # Brier Score
    try:
        metrics['Brier'] = brier_score_loss(y_true, y_prob)
    except ValueError:
        metrics['Brier'] = 0.25

    # 校准曲线（可靠性图，10 bin）
    metrics['reliability_curve'] = compute_reliability_curve(y_true, y_prob, n_bins=10)

    # PR 曲线（9个阈值点）
    metrics['pr_curve'] = compute_pr_curve(y_true, y_prob)

    # 预测概率分布
    metrics['probability_distribution'] = compute_probability_distribution(y_prob, n_bins=10)

    return metrics


def compute_ece(y_true, y_prob, n_bins=15):
    """计算 Expected Calibration Error (ECE)。"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if in_bin.sum() > 0:
            bin_confidence = y_prob[in_bin].mean()
            bin_accuracy = y_true[in_bin].mean()
            ece += (in_bin.sum() / n) * abs(bin_confidence - bin_accuracy)
    return ece


def compute_mce(y_true, y_prob, n_bins=15):
    """计算 Maximum Calibration Error (MCE)。"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if in_bin.sum() > 0:
            bin_confidence = y_prob[in_bin].mean()
            bin_accuracy = y_true[in_bin].mean()
            mce = max(mce, abs(bin_confidence - bin_accuracy))
    return mce


def compute_reliability_curve(y_true, y_prob, n_bins=10):
    """计算校准曲线（可靠性图）的bin级数据，任务书v10.28格式。

    Returns:
        dict: {'bin_centers': list, 'observed_positive_rate': list, 'bin_counts': list}
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    observed_positive_rate = []
    bin_counts = []
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        count = int(in_bin.sum())
        bin_centers.append(float((bin_lower + bin_upper) / 2))
        bin_counts.append(count)
        if count > 0:
            observed_positive_rate.append(float(y_true[in_bin].mean()))
        else:
            observed_positive_rate.append(0.0)
    return {
        'bin_centers': bin_centers,
        'observed_positive_rate': observed_positive_rate,
        'bin_counts': bin_counts,
    }


def compute_pr_curve(y_true, y_prob):
    """计算PR曲线，任务书v10.28格式：9个阈值点[0.1-0.9]。

    Returns:
        dict: {'thresholds': list, 'precision': list, 'recall': list}
    """
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    precision_list = []
    recall_list = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0) * 100
        r = recall_score(y_true, y_pred, zero_division=0) * 100
        precision_list.append(float(p))
        recall_list.append(float(r))
    return {
        'thresholds': thresholds,
        'precision': precision_list,
        'recall': recall_list,
    }


def compute_probability_distribution(y_prob, n_bins=10):
    """计算预测概率分布统计，任务书v10.28格式。

    Returns:
        dict: {'mean', 'median', 'percentile_25', 'percentile_75',
               'histogram_bins', 'histogram_counts'}
    """
    histogram_counts, histogram_bin_edges = np.histogram(y_prob, bins=n_bins, range=(0, 1))
    return {
        'mean': float(np.mean(y_prob)),
        'median': float(np.median(y_prob)),
        'percentile_25': float(np.percentile(y_prob, 25)),
        'percentile_75': float(np.percentile(y_prob, 75)),
        'histogram_bins': histogram_bin_edges.tolist(),
        'histogram_counts': histogram_counts.tolist(),
    }


def compute_local_pi_mae(local_pi_true, local_pi_est):
    """计算估计的本地正例簇构成与真实值的MAE。"""
    return np.mean(np.abs(np.array(local_pi_true) - np.array(local_pi_est)))


def print_metrics(metrics, prefix=""):
    """格式化打印标量指标（跳过曲线/分布字段）。"""
    print(f"{prefix}ACC={metrics['ACC']:.2f}  "
          f"Prec={metrics['Precision']:.2f}  Rec={metrics['Recall']:.2f}  "
          f"F1={metrics['F1']:.2f}  AUC={metrics['AUC']:.2f}  AP={metrics['AP']:.2f}  "
          f"ECE={metrics['ECE']:.2f}  Brier={metrics['Brier']:.4f}")
