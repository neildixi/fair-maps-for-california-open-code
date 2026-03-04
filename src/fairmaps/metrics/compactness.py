"""
Compactness metrics for redistricting plans.

Uses GerryChain definitions where available (polsby_popper).
Formulas from redistmetrics/CRAN and standard literature.
All return dict[district_id -> float] or scalar (for averages).
"""

import math
from typing import Dict, Optional, Any
from shapely.ops import unary_union


def _get_district_geometry(partition, part: int):
    """Get merged geometry for a district from partition graph."""
    nodes = partition.parts[part]
    geoms = []
    for node in nodes:
        if "geometry" in partition.graph.nodes[node]:
            geoms.append(partition.graph.nodes[node]["geometry"])
    if not geoms:
        return None
    try:
        return unary_union(geoms)
    except Exception:
        return None


def _district_area_perimeter(partition, part: int) -> tuple:
    """
    Get area and perimeter for a district.
    Prefer partition['area'] and partition['perimeter'] if available (GerryChain).
    Otherwise compute from geometry.
    """
    try:
        area = partition["area"][part]
        perimeter = partition["perimeter"][part]
        if not (math.isnan(area) or math.isnan(perimeter) or perimeter <= 0):
            return float(area), float(perimeter)
    except (KeyError, TypeError):
        pass

    geom = _get_district_geometry(partition, part)
    if geom is None or geom.is_empty:
        return 0.0, 1.0  # avoid div by zero
    area = geom.area
    perimeter = geom.length
    return float(area), float(perimeter)


def polsby_popper(partition, part: Optional[int] = None) -> Dict[int, float] | float:
    """
    Polsby-Popper compactness: 4*pi*area / perimeter^2.
    Range [0,1]; higher = more compact.
    Uses GerryChain formula when partition has area/perimeter updaters.
    """
    scores = {}
    for p in partition.parts:
        area, perimeter = _district_area_perimeter(partition, p)
        if perimeter > 0:
            scores[p] = 4 * math.pi * area / (perimeter ** 2)
        else:
            scores[p] = 0.0
    if part is not None:
        return scores.get(part, 0.0)
    return scores


def schwartzberg(partition, part: Optional[int] = None) -> Dict[int, float] | float:
    """
    Schwartzberg compactness: 2*sqrt(pi*area) / perimeter.
    Equals sqrt(polsby_popper). Range [0,1]; higher = more compact.
    """
    pp = polsby_popper(partition)
    if isinstance(pp, dict):
        scores = {p: math.sqrt(v) if v > 0 else 0.0 for p, v in pp.items()}
        return scores.get(part, 0.0) if part is not None else scores
    return math.sqrt(pp) if pp > 0 else 0.0


def reock(partition, part: Optional[int] = None) -> Dict[int, float] | float:
    """
    Reock compactness: area / area(minimum bounding circle).
    Range [0,1]; higher = more compact.
    """
    try:
        from shapely import minimum_bounding_circle
    except ImportError:
        try:
            from shapely.creation import minimum_bounding_circle
        except ImportError:
            from shapely.ops import minimum_bounding_circle

    scores = {}
    for p in partition.parts:
        geom = _get_district_geometry(partition, p)
        if geom is None or geom.is_empty:
            scores[p] = 0.0
            continue
        area = geom.area
        if area <= 0:
            scores[p] = 0.0
            continue
        mbc = minimum_bounding_circle(geom)
        mbc_area = mbc.area
        scores[p] = float(area / mbc_area) if mbc_area > 0 else 0.0

    if part is not None:
        return scores.get(part, 0.0)
    return scores


def convex_hull_ratio(partition, part: Optional[int] = None) -> Dict[int, float] | float:
    """
    Convex hull compactness: area / area(convex_hull).
    Range [0,1]; higher = more compact.
    """
    scores = {}
    for p in partition.parts:
        geom = _get_district_geometry(partition, p)
        if geom is None or geom.is_empty:
            scores[p] = 0.0
            continue
        area = geom.area
        hull = geom.convex_hull
        hull_area = hull.area
        scores[p] = float(area / hull_area) if hull_area > 0 else 0.0

    if part is not None:
        return scores.get(part, 0.0)
    return scores


def boundary_node_ratio(partition, part: Optional[int] = None) -> Dict[int, float] | float:
    """
    Ratio of boundary nodes to total nodes per district.
    Graph-theoretic compactness; lower = more compact (fewer boundary nodes).
    Returns 1 - ratio so that higher = better (consistent with other compactness).
    """
    boundary = set()
    for node in partition.graph.nodes:
        if partition.graph.nodes[node].get("boundary_node", False):
            boundary.add(node)

    scores = {}
    for p in partition.parts:
        nodes = partition.parts[p]
        n_total = len(nodes)
        n_boundary = sum(1 for n in nodes if n in boundary)
        if n_total > 0:
            ratio = n_boundary / n_total
            scores[p] = 1.0 - ratio  # invert so higher = more compact
        else:
            scores[p] = 0.0

    if part is not None:
        return scores.get(part, 0.0)
    return scores


def avg_compactness(scores: Dict[int, float]) -> float:
    """Average compactness across districts."""
    if not scores:
        return 0.0
    return sum(scores.values()) / len(scores)
