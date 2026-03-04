"""
Compute all FairMaps metrics for a partition.

Provides a single entry point for the city_map_generator to get metric values.
"""

import numpy as np
from typing import Dict, Any, Optional

from .metrics.compactness import (
    polsby_popper,
    schwartzberg,
    reock,
    convex_hull_ratio,
    boundary_node_ratio,
    avg_compactness,
)
from .metrics.demographic import (
    dissimilarity_index,
    gini_index,
    entropy_index,
    atkinson_index,
    isolation_index,
    interaction_index,
    delta_index,
)


DEFAULT_POP_COL = "Population P2"
DEFAULT_GROUP_COLS = {
    "white": "NH_Wht",
    "black": "NH_Blk",
    "asian": "NH_Asn",
    "hispanic": "Hispanic Origin",
}


def compute_all_metrics(
    partition,
    pop_col: str = DEFAULT_POP_COL,
    group_cols: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Compute all compactness and demographic metrics for a partition.

    Returns dict of metric_name -> scalar value.
    Compactness metrics are averaged across districts.
    """
    if group_cols is None:
        group_cols = DEFAULT_GROUP_COLS

    w, h, a, b = group_cols["white"], group_cols["hispanic"], group_cols["asian"], group_cols["black"]

    metrics = {}

    # Compactness (higher = better)
    try:
        pp = polsby_popper(partition)
        metrics["avg_polsby_popper"] = avg_compactness(pp)
    except Exception:
        metrics["avg_polsby_popper"] = 0.0
    try:
        sch = schwartzberg(partition)
        metrics["avg_schwartzberg"] = avg_compactness(sch)
    except Exception:
        metrics["avg_schwartzberg"] = 0.0
    try:
        rk = reock(partition)
        metrics["avg_reock"] = avg_compactness(rk)
    except Exception:
        metrics["avg_reock"] = 0.0
    try:
        ch = convex_hull_ratio(partition)
        metrics["avg_convex_hull"] = avg_compactness(ch)
    except Exception:
        metrics["avg_convex_hull"] = 0.0
    try:
        bnr = boundary_node_ratio(partition)
        metrics["avg_boundary_node_ratio"] = avg_compactness(bnr)
    except Exception:
        metrics["avg_boundary_node_ratio"] = 0.0

    # Demographics (lower = better for dissimilarity, gini, entropy, atkinson, isolation, delta)
    try:
        metrics["white_hispanic_dissimilarity"] = dissimilarity_index(partition, w, h, pop_col)
    except Exception:
        metrics["white_hispanic_dissimilarity"] = 0.0
    try:
        metrics["white_asian_dissimilarity"] = dissimilarity_index(partition, w, a, pop_col)
    except Exception:
        metrics["white_asian_dissimilarity"] = 0.0
    try:
        metrics["hispanic_gini"] = gini_index(partition, h, pop_col)
    except Exception:
        metrics["hispanic_gini"] = 0.0
    try:
        metrics["asian_gini"] = gini_index(partition, a, pop_col)
    except Exception:
        metrics["asian_gini"] = 0.0
    try:
        metrics["entropy_index"] = entropy_index(partition, group_cols, pop_col)
    except Exception:
        metrics["entropy_index"] = 0.0
    try:
        metrics["hispanic_atkinson"] = atkinson_index(partition, h, pop_col, b=0.5)
    except Exception:
        metrics["hispanic_atkinson"] = 0.0
    try:
        metrics["asian_atkinson"] = atkinson_index(partition, a, pop_col, b=0.5)
    except Exception:
        metrics["asian_atkinson"] = 0.0
    try:
        metrics["hispanic_isolation"] = isolation_index(partition, h, pop_col)
    except Exception:
        metrics["hispanic_isolation"] = 0.0
    try:
        metrics["asian_isolation"] = isolation_index(partition, a, pop_col)
    except Exception:
        metrics["asian_isolation"] = 0.0
    try:
        metrics["hispanic_interaction"] = interaction_index(partition, h, pop_col)
    except Exception:
        metrics["hispanic_interaction"] = 0.0
    try:
        metrics["asian_interaction"] = interaction_index(partition, a, pop_col)
    except Exception:
        metrics["asian_interaction"] = 0.0
    try:
        metrics["hispanic_delta"] = delta_index(partition, h, pop_col)
    except Exception:
        metrics["hispanic_delta"] = 0.0
    try:
        metrics["asian_delta"] = delta_index(partition, a, pop_col)
    except Exception:
        metrics["asian_delta"] = 0.0

    return metrics
