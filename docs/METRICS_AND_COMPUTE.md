# FairMaps Metrics and Compute Notes

## Metric Time Complexity

All metrics are designed to be **efficient** (no O(n²) or worse on block count):

| Metric | Complexity | Notes |
|--------|------------|-------|
| **Compactness** | O(D × B̄) | D = districts, B̄ = avg blocks per district. Geometry union per district. |
| Polsby-Popper | O(D) | Uses area/perimeter from partition or geometry |
| Schwartzberg | O(D) | √(Polsby-Popper) |
| Reock | O(D × B̄) | Shapely `minimum_bounding_circle` |
| Convex Hull | O(D × B̄) | Shapely `convex_hull` |
| Boundary Node | O(N) | N = total nodes, simple count |
| **Demographic** | O(D) or O(D²) | D is small (typically 4–15) |
| Dissimilarity | O(D) | |
| Gini | O(D²) | D² is trivial for small D |
| Entropy, Atkinson | O(D) | |
| Isolation, Interaction | O(D) | |
| Delta | O(D) | |

**Bottom line:** The slow part is generating 100k plans via MCMC, not computing metrics. Metric evaluation is fast.

## GerryChain Built-in Metrics

GerryChain 0.3.x includes:
- `polsby_popper` (compactness)
- Partisan metrics (mean_median, efficiency_gap, etc.) — **not used** (local elections)

FairMaps implements compactness and demographic metrics separately because:
1. GerryChain has no demographic segregation metrics (dissimilarity, isolation, etc.)
2. We need partition geometry; GerryChain's `polsby_popper` requires `area` and `perimeter` updaters (GeographicPartition). We fall back to geometry-based computation when those are not available.
