"""
FairMaps metrics: compactness and demographic segregation measures.
"""

from .compactness import (
    polsby_popper,
    schwartzberg,
    reock,
    convex_hull_ratio,
    boundary_node_ratio,
)
from .demographic import (
    dissimilarity_index,
    gini_index,
    entropy_index,
    atkinson_index,
    isolation_index,
    interaction_index,
    delta_index,
)

__all__ = [
    "polsby_popper",
    "schwartzberg",
    "reock",
    "convex_hull_ratio",
    "boundary_node_ratio",
    "dissimilarity_index",
    "gini_index",
    "entropy_index",
    "atkinson_index",
    "isolation_index",
    "interaction_index",
    "delta_index",
]
