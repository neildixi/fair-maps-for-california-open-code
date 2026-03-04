"""
Demographic / segregation metrics (Massey & Denton 1988).

All metrics operate on a partition with demographic columns (e.g. NH_Wht, Hispanic Origin).
Group columns and population column must be provided.
"""

import math
from typing import Dict, Optional


def _get_district_totals(partition, pop_col: str, group_cols: Dict[str, str]) -> Dict:
    """Compute per-district and city-wide totals."""
    city_totals = {g: 0.0 for g in group_cols}
    city_pop = 0.0
    district_data = {}

    for part in partition.parts:
        nodes = partition.parts[part]
        part_pop = sum(partition.graph.nodes[n][pop_col] for n in nodes)
        part_groups = {
            g: sum(partition.graph.nodes[n][group_cols[g]] for n in nodes)
            for g in group_cols
        }
        district_data[part] = {"pop": part_pop, "groups": part_groups}
        city_pop += part_pop
        for g in group_cols:
            city_totals[g] += part_groups[g]

    return {
        "district": district_data,
        "city_pop": city_pop,
        "city_groups": city_totals,
    }


def dissimilarity_index(
    partition,
    group1_col: str,
    group2_col: str,
    pop_col: str,
) -> float:
    """
    Dissimilarity index: D = (1/2) * sum_i |a_i/A - b_i/B|.
    Range [0,1]; lower = more even distribution.
    """
    data = _get_district_totals(partition, pop_col, {"a": group1_col, "b": group2_col})
    A = data["city_groups"]["a"]
    B = data["city_groups"]["b"]
    if A == 0 or B == 0:
        return 0.0

    total = 0.0
    for part, d in data["district"].items():
        a_i = d["groups"]["a"]
        b_i = d["groups"]["b"]
        total += abs(a_i / A - b_i / B)
    return total / 2.0


def gini_index(
    partition,
    minority_col: str,
    pop_col: str,
) -> float:
    """
    Gini coefficient of segregation (Massey-Denton).
    Mean absolute difference in minority proportions across district pairs.
    Range [0,1]; lower = more even.
    """
    data = _get_district_totals(partition, pop_col, {"x": minority_col})
    X = data["city_groups"]["x"]
    T = data["city_pop"]
    if T == 0 or X == 0:
        return 0.0
    P = X / T

    parts = list(data["district"].keys())
    n = len(parts)
    if n < 2:
        return 0.0

    weighted_sum = 0.0
    for i, p_i in enumerate(parts):
        t_i = data["district"][p_i]["pop"]
        x_i = data["district"][p_i]["groups"]["x"]
        p_i_val = x_i / t_i if t_i > 0 else 0
        for j, p_j in enumerate(parts):
            t_j = data["district"][p_j]["pop"]
            x_j = data["district"][p_j]["groups"]["x"]
            p_j_val = x_j / t_j if t_j > 0 else 0
            weighted_sum += (t_i / T) * (t_j / T) * abs(p_i_val - p_j_val)

    return weighted_sum / (2 * P * (1 - P)) if P * (1 - P) > 0 else 0.0


def entropy_index(
    partition,
    group_cols: Dict[str, str],
    pop_col: str,
) -> float:
    """
    Theil entropy index H.
    Measures deviation from metropolitan entropy (diversity).
    Range [0,1]; lower = more even.
    """
    data = _get_district_totals(partition, pop_col, group_cols)
    T = data["city_pop"]
    if T == 0:
        return 0.0

    # Metropolitan entropy E = -sum_k P_k * ln(P_k)
    groups = list(group_cols.keys())
    E = 0.0
    for g in groups:
        P_k = data["city_groups"][g] / T
        if P_k > 0:
            E -= P_k * math.log(P_k)

    if E == 0:
        return 0.0

    # H = sum_i (t_i/T) * (E - E_i) / E
    H = 0.0
    for part, d in data["district"].items():
        t_i = d["pop"]
        if t_i == 0:
            continue
        E_i = 0.0
        for g in groups:
            p_ik = d["groups"][g] / t_i
            if p_ik > 0:
                E_i -= p_ik * math.log(p_ik)
        H += (t_i / T) * (E - E_i) / E
    return H


def atkinson_index(
    partition,
    minority_col: str,
    pop_col: str,
    b: float = 0.5,
) -> float:
    """
    Atkinson index with shape parameter b in (0, 1).
    b=0.5: symmetric; b<0.5: under-representation weighted more.
    Range [0,1]; lower = more even.
    """
    data = _get_district_totals(partition, pop_col, {"x": minority_col})
    X = data["city_groups"]["x"]
    T = data["city_pop"]
    if T == 0 or X == 0:
        return 0.0
    P = X / T

    total = 0.0
    for part, d in data["district"].items():
        t_i = d["pop"]
        x_i = d["groups"]["x"]
        if t_i == 0:
            continue
        p_i = x_i / t_i
        if p_i <= 0 or p_i >= 1:
            continue
        term = 1 - (p_i / P) ** b * ((1 - p_i) / (1 - P)) ** (1 - b)
        total += (t_i / T) * term

    return total


def isolation_index(
    partition,
    group_col: str,
    pop_col: str,
) -> float:
    """
    Isolation: sum_i (x_i/X) * (x_i/t_i).
    Exposure of group to own group. Range [0,1]; context-dependent.
    """
    data = _get_district_totals(partition, pop_col, {"x": group_col})
    X = data["city_groups"]["x"]
    if X == 0:
        return 0.0

    total = 0.0
    for part, d in data["district"].items():
        t_i = d["pop"]
        x_i = d["groups"]["x"]
        if t_i > 0 and x_i > 0:
            total += (x_i / X) * (x_i / t_i)
    return total


def interaction_index(
    partition,
    group_col: str,
    pop_col: str,
    reference_col: Optional[str] = None,
) -> float:
    """
    Interaction (exposure): sum_i (x_i/X) * (y_i/t_i).
    Exposure of minority x to non-x (or to reference group).
    If reference_col is None, y_i = t_i - x_i (non-group).
    Range [0,1]; higher = more exposure/integration.
    For fairness: lower isolation and higher interaction can be desirable.
    """
    data = _get_district_totals(partition, pop_col, {"x": group_col})
    X = data["city_groups"]["x"]
    if X == 0:
        return 0.0

    # Get y (other group) per district
    if reference_col:
        data2 = _get_district_totals(partition, pop_col, {"y": reference_col})
        def y_i(part):
            return data2["district"][part]["groups"]["y"]
    else:
        def y_i(part):
            d = data["district"][part]
            return d["pop"] - d["groups"]["x"]

    total = 0.0
    for part, d in data["district"].items():
        t_i = d["pop"]
        x_i = d["groups"]["x"]
        y_i_val = y_i(part)
        if t_i > 0 and x_i > 0:
            total += (x_i / X) * (y_i_val / t_i)
    return total


def delta_index(
    partition,
    minority_col: str,
    pop_col: str,
) -> float:
    """
    Delta (Hoover): proportion of minority in districts with above-average density.
    Measures concentration. Range [0,1]; lower = less concentrated.
    """
    data = _get_district_totals(partition, pop_col, {"x": minority_col})
    X = data["city_groups"]["x"]
    T = data["city_pop"]
    if T == 0 or X == 0:
        return 0.0
    P = X / T  # average density

    surplus = 0.0
    for part, d in data["district"].items():
        t_i = d["pop"]
        x_i = d["groups"]["x"]
        if t_i == 0:
            continue
        p_i = x_i / t_i
        if p_i > P:
            surplus += (p_i - P) * t_i

    return surplus / (2 * X * (1 - P)) if P < 1 else 0.0
