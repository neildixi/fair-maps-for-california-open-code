"""
Aggregation utilities: percentile computation via discrete CDF and weighted power mean.

Percentiles are computed from an empirical CDF (discrete bins approximating continuous).
Power mean supports configurable lambda (AM, GM, HM, min, etc.).
"""

import numpy as np
from typing import List, Optional, Dict, Any


def build_empirical_cdf(values: np.ndarray, max_bins: int = 50000) -> np.ndarray:
    """
    Build empirical CDF from ensemble values (sorted array for percentile lookup).
    
    For large ensembles, subsample to max_bins to keep memory O(max_bins).
    Returns sorted array of values.
    """
    vals = np.asarray(values).flatten()
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.array([0.0])
    
    sorted_vals = np.sort(vals)
    n = len(sorted_vals)
    if n > max_bins:
        indices = np.linspace(0, n - 1, max_bins, dtype=int)
        return sorted_vals[indices]
    return sorted_vals


def percentile_from_empirical_cdf(value: float, sorted_values: np.ndarray) -> float:
    """
    Compute percentile (0-1) of value in empirical distribution.
    
    Percentile = (count of values <= value) / total count.
    Uses searchsorted for O(log n) lookup.
    """
    if len(sorted_values) == 0:
        return 0.5
    count_le = np.searchsorted(sorted_values, value, side='right')
    p = count_le / len(sorted_values)
    return float(np.clip(p, 0.0, 1.0))


def build_cdf_stats(ensemble_values: Dict[str, np.ndarray], max_bins: int = 50000) -> Dict[str, np.ndarray]:
    """
    Build sorted value arrays for each metric (for empirical percentile lookup).
    
    Returns dict: metric_name -> sorted array of values from ensemble.
    """
    return {
        name: build_empirical_cdf(vals, max_bins)
        for name, vals in ensemble_values.items()
    }


def power_mean(values: np.ndarray, weights: np.ndarray, lam: float) -> float:
    """
    Weighted power mean (generalized mean) with parameter lambda.
    
    M_lambda = (sum_i w_i * x_i^lambda)^(1/lambda)  for lambda != 0
    M_0 = exp(sum_i w_i * ln(x_i))                  for lambda = 0 (geometric)
    M_{-inf} = min(x_i)                             for lambda -> -inf
    M_{+inf} = max(x_i)                             for lambda -> +inf
    
    Weights should sum to 1. Values should be in [0, 1] (percentiles).
    """
    arr = np.asarray(values).astype(float)
    w = np.asarray(weights).astype(float)
    if len(w) != len(arr):
        w = np.resize(w, len(arr))
    w = w / w.sum()
    
    # Clip to avoid log(0) or 0^negative
    arr = np.clip(arr, 1e-10, 1.0)
    
    if lam == 0:
        return float(np.exp(np.sum(w * np.log(arr))))
    elif lam <= -1e6:
        return float(np.min(arr))
    elif lam >= 1e6:
        return float(np.max(arr))
    else:
        return float(np.sum(w * (arr ** lam)) ** (1.0 / lam))


def aggregate_plan_score(
    percentile_array: np.ndarray,
    weights: np.ndarray,
    lam: float = 0.0
) -> float:
    """
    Aggregate percentile scores into a single fairness score using power mean.
    
    Args:
        percentile_array: Array of percentile values (0-1) for each metric.
        weights: Weights for each metric (will be normalized).
        lam: Power mean parameter (0=GM, 1=AM, -1=HM, -inf=min).
    
    Returns:
        Single aggregated score in [0, 1].
    """
    return power_mean(percentile_array, weights, lam)
