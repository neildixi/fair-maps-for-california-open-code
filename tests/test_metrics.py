"""
Test FairMaps metrics with 1 generated map.

Creates minimal synthetic data (small grid) and runs all metrics to verify they work.
Run: python -m pytest tests/test_metrics.py -v
Or:  python tests/test_metrics.py

Note: Requires geopandas. If geopandas fails (e.g. shapely version mismatch),
run the fallback unit test: python tests/test_metrics.py --unit-only
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def make_synthetic_grid(nx=4, ny=4):
    """Create a small grid of blocks with dummy demographics."""
    import geopandas as gpd
    from shapely.geometry import box

    rows = []
    for i in range(ny):
        for j in range(nx):
            idx = i * nx + j
            geom = box(j * 1000, i * 1000, (j + 1) * 1000, (i + 1) * 1000)
            # Vary demographics: top-left more Hispanic, bottom-right more Asian
            pop = 100 + (idx % 5) * 20
            hispanic = int(pop * (0.1 + 0.3 * (i / max(1, ny - 1))))
            asian = int(pop * (0.1 + 0.3 * (j / max(1, nx - 1))))
            white = pop - hispanic - asian
            white = max(0, white)
            black = 0
            rows.append({
                "GEOID20": f"06{i:02d}{j:02d}{idx:04d}",
                "PLACE20": 69084,
                "geometry": geom,
                "Population P2": pop,
                "NH_Wht": white,
                "NH_Blk": black,
                "NH_Asn": asian,
                "Hispanic Origin": hispanic,
            })
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:26910")
    return gdf


class MockPartition:
    """Minimal partition-like object for testing (no gerrychain)."""
    def __init__(self):
        self.parts = {0: [0, 1, 2], 1: [3, 4, 5]}
        self.graph = type('G', (), {})()
        nodes = {
            0: {"Population P2": 100, "NH_Wht": 60, "Hispanic Origin": 25, "NH_Asn": 15, "NH_Blk": 0},
            1: {"Population P2": 100, "NH_Wht": 60, "Hispanic Origin": 25, "NH_Asn": 15, "NH_Blk": 0},
            2: {"Population P2": 100, "NH_Wht": 60, "Hispanic Origin": 25, "NH_Asn": 15, "NH_Blk": 0},
            3: {"Population P2": 100, "NH_Wht": 40, "Hispanic Origin": 35, "NH_Asn": 15, "NH_Blk": 0},
            4: {"Population P2": 100, "NH_Wht": 40, "Hispanic Origin": 35, "NH_Asn": 15, "NH_Blk": 0},
            5: {"Population P2": 100, "NH_Wht": 40, "Hispanic Origin": 35, "NH_Asn": 15, "NH_Blk": 0},
        }
        self.graph.nodes = nodes


def run_unit_test_mock():
    """Test metrics with a mock partition (no gerrychain/geopandas)."""
    from fairmaps.metrics import dissimilarity_index, gini_index, isolation_index, interaction_index, delta_index
    from fairmaps.metrics.demographic import entropy_index, atkinson_index
    from fairmaps.aggregation import power_mean, percentile_from_empirical_cdf

    partition = MockPartition()
    pop_col = "Population P2"
    d = dissimilarity_index(partition, "NH_Wht", "Hispanic Origin", pop_col)
    g = gini_index(partition, "Hispanic Origin", pop_col)
    iso = isolation_index(partition, "Hispanic Origin", pop_col)
    inter = interaction_index(partition, "Hispanic Origin", pop_col)
    delta = delta_index(partition, "Hispanic Origin", pop_col)
    atk = atkinson_index(partition, "Hispanic Origin", pop_col, b=0.5)
    group_cols = {"white": "NH_Wht", "hispanic": "Hispanic Origin", "asian": "NH_Asn"}
    ent = entropy_index(partition, group_cols, pop_col)

    print("Unit test (mock partition):")
    print(f"  Dissimilarity: {d:.4f}, Gini: {g:.4f}")
    print(f"  Isolation: {iso:.4f}, Interaction: {inter:.4f}")
    print(f"  Delta: {delta:.4f}, Atkinson: {atk:.4f}, Entropy: {ent:.4f}")

    pm = power_mean(np.array([0.5, 0.7]), np.ones(2)/2, 0.0)
    pct = percentile_from_empirical_cdf(0.5, np.sort(np.random.rand(100)))
    print(f"  Power mean (GM): {pm:.4f}, Empirical pctl: {pct:.2f}")
    print("Unit test passed.")


def run_metrics_test():
    """Generate 1 map and run all FairMaps metrics."""
    from gerrychain import Graph, Partition, updaters
    from gerrychain.tree import recursive_tree_part
    from fairmaps.metrics import (
        polsby_popper,
        schwartzberg,
        reock,
        convex_hull_ratio,
        boundary_node_ratio,
        dissimilarity_index,
        gini_index,
        isolation_index,
        interaction_index,
        delta_index,
    )
    from fairmaps.metrics.demographic import entropy_index, atkinson_index
    from fairmaps.metrics.compactness import avg_compactness
    from fairmaps.aggregation import percentile_from_empirical_cdf, power_mean

    print("=" * 60)
    print("FairMaps Metrics Test - 1 Map")
    print("=" * 60)

    # Build synthetic grid
    gdf = make_synthetic_grid(4, 4)
    print(f"\nSynthetic grid: {len(gdf)} blocks")

    graph = Graph.from_geodataframe(gdf)
    num_districts = 4
    total_pop = sum(graph.nodes[n]["Population P2"] for n in graph.nodes)
    ideal_pop = total_pop // num_districts

    # Initial partition
    np.random.seed(42)
    assignment = recursive_tree_part(
        graph,
        range(num_districts),
        ideal_pop,
        pop_col="Population P2",
        epsilon=0.2,
    )

    partition = Partition(
        graph,
        assignment=assignment,
        updaters={"population": updaters.Tally("Population P2", alias="population")},
    )

    pop_col = "Population P2"
    white_col = "NH_Wht"
    hispanic_col = "Hispanic Origin"
    asian_col = "NH_Asn"

    # --- Compactness ---
    print("\n--- Compactness ---")
    pp = polsby_popper(partition)
    sch = schwartzberg(partition)
    rk = reock(partition)
    ch = convex_hull_ratio(partition)
    bnr = boundary_node_ratio(partition)

    print(f"  Polsby-Popper (avg): {avg_compactness(pp):.4f}")
    print(f"  Schwartzberg (avg):  {avg_compactness(sch):.4f}")
    print(f"  Reock (avg):         {avg_compactness(rk):.4f}")
    print(f"  Convex Hull (avg):   {avg_compactness(ch):.4f}")
    print(f"  Boundary Node (avg): {avg_compactness(bnr):.4f}")

    # --- Demographics ---
    print("\n--- Demographics ---")
    d_wh = dissimilarity_index(partition, white_col, hispanic_col, pop_col)
    d_wa = dissimilarity_index(partition, white_col, asian_col, pop_col)
    g_h = gini_index(partition, hispanic_col, pop_col)
    g_a = gini_index(partition, asian_col, pop_col)
    iso_h = isolation_index(partition, hispanic_col, pop_col)
    iso_a = isolation_index(partition, asian_col, pop_col)
    int_h = interaction_index(partition, hispanic_col, pop_col)
    int_a = interaction_index(partition, asian_col, pop_col)
    delta_h = delta_index(partition, hispanic_col, pop_col)
    delta_a = delta_index(partition, asian_col, pop_col)

    # Entropy and Atkinson need group_cols dict
    group_cols = {"white": white_col, "hispanic": hispanic_col, "asian": asian_col}
    try:
        ent = entropy_index(partition, group_cols, pop_col)
        atk_h = atkinson_index(partition, hispanic_col, pop_col, b=0.5)
        atk_a = atkinson_index(partition, asian_col, pop_col, b=0.5)
        print(f"  Dissimilarity (W-H): {d_wh:.4f}")
        print(f"  Dissimilarity (W-A): {d_wa:.4f}")
        print(f"  Gini (Hispanic):     {g_h:.4f}")
        print(f"  Gini (Asian):        {g_a:.4f}")
        print(f"  Entropy:             {ent:.4f}")
        print(f"  Atkinson (H):        {atk_h:.4f}")
        print(f"  Atkinson (A):        {atk_a:.4f}")
        print(f"  Isolation (H):       {iso_h:.4f}")
        print(f"  Isolation (A):       {iso_a:.4f}")
        print(f"  Interaction (H):     {int_h:.4f}")
        print(f"  Interaction (A):     {int_a:.4f}")
        print(f"  Delta (H):           {delta_h:.4f}")
        print(f"  Delta (A):           {delta_a:.4f}")
    except Exception as e:
        print(f"  Error in some metric: {e}")
        raise

    # --- Aggregation ---
    print("\n--- Aggregation ---")
    # Fake percentiles
    percentiles = np.array([0.6, 0.7, 0.5, 0.8])
    weights = np.ones(4) / 4
    pm_am = power_mean(percentiles, weights, lam=1.0)
    pm_gm = power_mean(percentiles, weights, lam=0.0)
    pm_hm = power_mean(percentiles, weights, lam=-1.0)
    print(f"  Power mean (AM, lam=1): {pm_am:.4f}")
    print(f"  Power mean (GM, lam=0): {pm_gm:.4f}")
    print(f"  Power mean (HM, lam=-1): {pm_hm:.4f}")

    sorted_vals = np.sort(np.random.rand(100))
    pct = percentile_from_empirical_cdf(0.5, sorted_vals)
    print(f"  Empirical percentile(0.5 in 100 vals): {pct:.2f} (expect ~0.5)")

    # Sanity checks
    assert 0 <= avg_compactness(pp) <= 1, "Polsby-Popper out of range"
    assert 0 <= d_wh <= 1, "Dissimilarity out of range"
    assert 0 <= iso_h <= 1, "Isolation out of range"
    assert 0 <= pm_gm <= 1, "Power mean out of range"

    print("\n" + "=" * 60)
    print("All metrics computed successfully.")
    print("=" * 60)


def test_unit_metrics():
    """Pytest: metrics with mock partition (no gerrychain/geopandas)."""
    run_unit_test_mock()


def test_full_metrics():
    """Pytest: full pipeline with synthetic grid and gerrychain."""
    run_metrics_test()


if __name__ == "__main__":
    unit_only = "--unit-only" in sys.argv
    if unit_only:
        run_unit_test_mock()
    else:
        try:
            run_metrics_test()
        except Exception as e:
            print(f"\nFull test failed: {e}")
            print("Running unit-only test (no geopandas)...")
            run_unit_test_mock()
