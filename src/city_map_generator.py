import warnings
warnings.filterwarnings("ignore")

# Ensure fairmaps is importable (when run without pip install -e .)
import sys
import os
_src = os.path.dirname(os.path.abspath(__file__))
if _src not in sys.path:
    sys.path.insert(0, _src)

import pandas as pd
import geopandas as gpd
import numpy as np
from gerrychain import Graph, Partition, updaters
from gerrychain.constraints import single_flip_contiguous
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept
from gerrychain import MarkovChain
from gerrychain.tree import recursive_tree_part
from scipy.stats import norm
import os
import time

# =============================================================================
# CONFIGURATION - EASILY SWITCH BETWEEN ANY CITY OR COUNTY IN CALIFORNIA
# =============================================================================
#
# Set JURISDICTION_NAME to match what you used in extract_city_data.py.
# Paths are derived automatically: {jurisdiction}_outputs/{jurisdiction}_merged_data.geojson
# See docs/CALIFORNIA_JURISDICTIONS.md for PLACE20/COUNTY20 codes.
#
# =============================================================================

# Jurisdiction name (must match extract_city_data.py JURISDICTION_NAME)
# Examples: "Palo Alto", "San Jose", "San Mateo County", "Santa Clara County"
JURISDICTION_NAME = "Palo Alto"

# Override input file path (None = auto: {jurisdiction}_outputs/{jurisdiction}_merged_data.geojson)
INPUT_GEOJSON_FILE = None  # e.g. "palo_alto_outputs/palo_alto_merged_data.geojson"

# Number of districts to create (cities often 4-7; counties vary by population)
NUM_DISTRICTS = 6

# Ensemble configuration
# Default values - will be updated by user input
TEST_ENSEMBLE_SIZE = 100  # Size of test ensemble to generate statistics
MAIN_ENSEMBLE_SIZE = 1000  # Size of main ensemble for final analysis
TOP_N_PLANS = 100  # Number of top plans to save block assignment CSVs for

# File paths (auto-derived from JURISDICTION_NAME)
def _jurisdiction_slug(name):
    return name.lower().replace(" ", "_").replace(",", "")
_slug = _jurisdiction_slug(JURISDICTION_NAME)
OUTPUT_DIRECTORY = _slug + "_outputs"
if INPUT_GEOJSON_FILE is None:
    INPUT_GEOJSON_FILE = os.path.join(OUTPUT_DIRECTORY, _slug + "_merged_data.geojson")
TEST_STATS_FILE = "test_ensemble_stats.json"  # File to save test ensemble statistics

# Processing settings
MARKOV_CHAIN_STEPS = 1000  # Number of steps in Markov chain for each plan
POPULATION_EPSILON = 0.15  # Population deviation tolerance (15% = 0.15) - increased for larger cities

# Population column name (modify if your data uses different column names)
POPULATION_COLUMN = "Population P2"  # Column name for total population
WHITE_POPULATION_COLUMN = "NH_Wht"  # Column name for white population
BLACK_POPULATION_COLUMN = "NH_Blk"  # Column name for black population
ASIAN_POPULATION_COLUMN = "NH_Asn"  # Column name for Asian population
HISPANIC_POPULATION_COLUMN = "Hispanic Origin"  # Column name for Hispanic population
GEOID_COLUMN = "GEOID20"  # Column name for geographic ID

# Metric configuration (modify if you want different metrics or weights)
# All metrics from fairmaps compute_all_metrics
METRIC_ORDER = [
    # Compactness (higher = better)
    "avg_polsby_popper",
    "avg_schwartzberg",
    "avg_reock",
    "avg_convex_hull",
    "avg_boundary_node_ratio",
    # Demographics - evenness (lower = better)
    "white_hispanic_dissimilarity",
    "white_asian_dissimilarity",
    "hispanic_gini",
    "asian_gini",
    "entropy_index",
    "hispanic_atkinson",
    "asian_atkinson",
    # Demographics - exposure (isolation lower, interaction higher)
    "hispanic_isolation",
    "asian_isolation",
    "hispanic_interaction",
    "asian_interaction",
    # Demographics - concentration (lower = better)
    "hispanic_delta",
    "asian_delta",
]
NUM_METRICS = len(METRIC_ORDER)

# Weights for scoring (must sum to 1.0)
# Default: all metrics equal weight
METRIC_WEIGHTS = [1 / NUM_METRICS] * NUM_METRICS

# FairMaps aggregation: use empirical CDF (discrete) and configurable power mean
USE_FAIRMAPS = True           # Use fairmaps metrics + empirical percentile + power mean
POWER_MEAN_LAMBDA = 0.0       # 0=GM, 1=AM, -1=HM, -inf=min

# Communities of Interest (COI): graph contraction
# Each COI is a list of block GEOIDs (GEOID20) that will be merged into one "super-node"
# with population = sum of block populations and adjacency = boundary of the COI.
# Leave empty [] to disable COI contraction.
# Example: COI_LIST = [ ["060855113024003", "060855113024004"], ["060855110002001"] ]
COI_LIST = []

# =============================================================================
# END CONFIGURATION - DO NOT MODIFY BELOW THIS LINE
# =============================================================================

# Global variable to store metric statistics (will be loaded from test ensemble)
metric_stats = None

def load_test_ensemble_stats():
    """Load metric statistics from test ensemble file."""
    global metric_stats
    stats_path = os.path.join(OUTPUT_DIRECTORY, TEST_STATS_FILE)
    
    if os.path.exists(stats_path):
        import json
        with open(stats_path, 'r') as f:
            metric_stats = json.load(f)
        print(f"[OK] Loaded test ensemble statistics from {stats_path}")
        return True
    else:
        print(f"[!] Test ensemble statistics not found at {stats_path}")
        print("Please run the test ensemble first to generate statistics.")
        metric_stats = None
        return False

def save_test_ensemble_stats(stats):
    """Save metric statistics from test ensemble to file."""
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    stats_path = os.path.join(OUTPUT_DIRECTORY, TEST_STATS_FILE)
    
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Saved test ensemble statistics to {stats_path}")


def contract_graph_for_cois(graph, coi_list, geoid_to_node):
    """
    Contract each COI (list of GEOIDs) into a single super-node:
    - Population and demographics = sum over blocks in the COI.
    - Adjacency = boundary: super-node is adjacent to any block outside the COI
      that was adjacent to any block inside the COI.
    - Geometry = unary_union of block geometries (for compactness metrics).

    Returns:
        graph_contracted: Graph with COI nodes merged into super-nodes.
        contraction_map: dict original_node_id -> contracted_node_id (original or ("coi", i)).
        original_nodes_for_export: list of (original_node_id, geoid) for every block (for CSV export).
    """
    from shapely.ops import unary_union

    # Columns to sum for super-nodes
    sum_columns = [
        POPULATION_COLUMN,
        WHITE_POPULATION_COLUMN,
        BLACK_POPULATION_COLUMN,
        ASIAN_POPULATION_COLUMN,
        HISPANIC_POPULATION_COLUMN,
    ]
    # Copy any other numeric/attribute columns from first node (except geometry)
    sample_node = next(iter(graph.nodes))
    all_attr = list(graph.nodes[sample_node].keys()) if graph.nodes else []
    for col in all_attr:
        if col not in sum_columns and col != "geometry" and col != GEOID_COLUMN:
            try:
                _ = sum(graph.nodes[n].get(col, 0) for n in list(graph.nodes)[:2])
                sum_columns.append(col)
            except (TypeError, ValueError):
                pass

    # Resolve COI lists of GEOIDs -> sets of node ids; skip missing GEOIDs
    coi_node_sets = []
    for i, geoid_list in enumerate(coi_list):
        nodes_in_coi = set()
        for g in geoid_list:
            if g in geoid_to_node:
                nodes_in_coi.add(geoid_to_node[g])
            else:
                print(f"[!] COI {i}: GEOID {g} not found in graph, skipping")
        if nodes_in_coi:
            coi_node_sets.append(nodes_in_coi)
        else:
            print(f"[!] COI {i}: no valid nodes, skipping")

    if not coi_node_sets:
        # No valid COIs; return identity contraction
        contraction_map = {n: n for n in graph.nodes}
        original_nodes_for_export = [
            (n, graph.nodes[n].get(GEOID_COLUMN, n)) for n in graph.nodes
        ]
        return graph, contraction_map, original_nodes_for_export

    # Map: original node -> coi index (None if not in any COI)
    node_to_coi = {}
    for n in graph.nodes:
        node_to_coi[n] = None
    for i, node_set in enumerate(coi_node_sets):
        for n in node_set:
            if node_to_coi[n] is not None:
                print(f"[!] Node {n} appears in multiple COIs; assigning to first only")
            else:
                node_to_coi[n] = i

    # Build contracted graph (gerrychain Graph is a networkx-based graph)
    graph_contracted = Graph()
    # Add non-COI nodes with same attributes
    for n in graph.nodes:
        if node_to_coi[n] is not None:
            continue
        attrs = dict(graph.nodes[n])
        graph_contracted.add_node(n, **attrs)

    # Add super-nodes for each COI
    for i, node_set in enumerate(coi_node_sets):
        coi_id = ("coi", i)
        attrs = {}
        for col in sum_columns:
            if col in graph.nodes[sample_node]:
                attrs[col] = sum(
                    graph.nodes[n].get(col, 0) or 0 for n in node_set
                )
            else:
                attrs[col] = 0
        attrs[GEOID_COLUMN] = f"COI_{i}"
        # Geometry = union of all block geometries in this COI
        geoms = [
            graph.nodes[n]["geometry"]
            for n in node_set
            if "geometry" in graph.nodes[n] and graph.nodes[n]["geometry"] is not None
        ]
        if geoms:
            attrs["geometry"] = unary_union(geoms)
        else:
            attrs["geometry"] = None
        graph_contracted.add_node(coi_id, **attrs)

    # Edges: (u, v) in original -> add (contracted(u), contracted(v)) if different
    def contract(n):
        return n if node_to_coi[n] is None else ("coi", node_to_coi[n])

    seen_edges = set()
    for u, v in graph.edges:
        cu, cv = contract(u), contract(v)
        if cu == cv:
            continue
        key = frozenset([cu, cv])
        if key not in seen_edges:
            seen_edges.add(key)
            graph_contracted.add_edge(cu, cv)

    contraction_map = {n: contract(n) for n in graph.nodes}
    original_nodes_for_export = [
        (n, graph.nodes[n].get(GEOID_COLUMN, n)) for n in graph.nodes
    ]

    print(f"[OK] COI contraction: {len(coi_node_sets)} COI(s), "
          f"{len(graph.nodes)} -> {len(graph_contracted.nodes)} nodes, "
          f"{len(graph.edges)} -> {len(graph_contracted.edges)} edges")
    return graph_contracted, contraction_map, original_nodes_for_export


def generate_test_ensemble_stats(graph, num_districts, test_ensemble_size=None):
    """Generate test ensemble and calculate metric statistics."""
    if test_ensemble_size is None:
        test_ensemble_size = TEST_ENSEMBLE_SIZE
        
    print(f"\n{'='*60}")
    print(f"GENERATING TEST ENSEMBLE ({test_ensemble_size} plans)")
    print(f"{'='*60}")
    
    test_plans = []
    successful_plans = 0
    
    for i in range(test_ensemble_size):
        plan_data = generate_plan_without_scoring(graph, num_districts, i)
        
        if plan_data is not None:
            test_plans.append(plan_data)
            successful_plans += 1
            
            if (i + 1) % 5 == 0:
                print(f"Generated {i+1}/{test_ensemble_size} test plans ({successful_plans} successful)")
    
    if len(test_plans) == 0:
        print("ERROR: No successful plans generated in test ensemble!")
        return None
    
    print(f"\nTest ensemble complete: {successful_plans}/{test_ensemble_size} successful plans")
    
    # Calculate statistics for each metric
    df_test = pd.DataFrame(test_plans)
    stats = {}
    
    if USE_FAIRMAPS:
        from fairmaps.aggregation import build_empirical_cdf
        for metric in METRIC_ORDER:
            if metric in df_test.columns:
                values = df_test[metric].dropna().values
                if len(values) > 0:
                    sorted_vals = build_empirical_cdf(values)
                    stats[metric] = {"sorted_values": sorted_vals.tolist()}
                else:
                    stats[metric] = {"sorted_values": [0.0]}
            else:
                print(f"Warning: Metric {metric} not found in test ensemble data")
                stats[metric] = {"sorted_values": [0.0]}
    else:
        for metric in METRIC_ORDER:
            if metric in df_test.columns:
                values = df_test[metric].dropna()
                if len(values) > 0:
                    stats[metric] = {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "count": int(len(values))
                    }
                else:
                    stats[metric] = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0, "count": 0}
            else:
                stats[metric] = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0, "count": 0}
    
    # Save statistics
    save_test_ensemble_stats(stats)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST ENSEMBLE STATISTICS SUMMARY")
    print(f"{'='*60}")
    for metric, stat in stats.items():
        print(f"{metric}:")
        if "sorted_values" in stat:
            sv = np.array(stat["sorted_values"])
            print(f"  Count: {len(sv)}, Range: [{sv.min():.6f}, {sv.max():.6f}]")
        else:
            print(f"  Mean: {stat['mean']:.6f}")
            print(f"  Std:  {stat['std']:.6f}")
            print(f"  Range: [{stat['min']:.6f}, {stat['max']:.6f}]")
            print(f"  Count: {stat['count']}")
        print()
    
    return stats

# Compactness metrics: higher is better. Demographics: lower is better.
HIGHER_IS_BETTER = ["avg_polsby_popper", "avg_schwartzberg", "avg_reock", "avg_convex_hull", "avg_boundary_node_ratio", "hispanic_interaction", "asian_interaction"]


def plan_score_array(row):
    if metric_stats is None:
        raise ValueError("metric_stats is None. Please ensure test ensemble statistics are loaded.")
    
    if USE_FAIRMAPS:
        from fairmaps.aggregation import percentile_from_empirical_cdf
        arr = []
        for i, metric in enumerate(METRIC_ORDER):
            if metric not in row or pd.isna(row[metric]):
                arr.append(0.5)
                continue
            value = row[metric]
            stat = metric_stats.get(metric, {})
            sorted_vals = stat.get("sorted_values")
            if sorted_vals is not None and len(sorted_vals) > 0:
                sv = np.array(sorted_vals)
                percentile = percentile_from_empirical_cdf(float(value), sv)
                if metric not in HIGHER_IS_BETTER:
                    percentile = 1.0 - percentile  # lower raw value -> better -> higher percentile
            else:
                # Fallback: parametric (old stats format)
                mean = stat.get("mean", 0.5)
                std = stat.get("std", 1.0)
                if std == 0:
                    percentile = 0.5
                else:
                    z = (float(value) - mean) / std
                    percentile = norm.cdf(z) if metric in HIGHER_IS_BETTER else 1 - norm.cdf(z)
            arr.append(float(np.clip(percentile, 0, 1)))
        return np.array(arr)
    
    # Legacy: parametric (normal) approximation
    arr = []
    for i, metric in enumerate(METRIC_ORDER):
        mean = metric_stats[metric]["mean"]
        std = metric_stats[metric]["std"]
        value = row[metric]
        if std == 0:
            percentile = 0.5
        else:
            z = (value - mean) / std
            if metric in HIGHER_IS_BETTER or i == 0:
                percentile = norm.cdf(z)
            else:
                percentile = 1 - norm.cdf(z)
        arr.append(np.clip(percentile, 0, 1))
    return np.array(arr)


def weighted_aggregate(arr, lam=None):
    """Power mean aggregation. lam=0 is geometric mean."""
    if lam is None:
        lam = POWER_MEAN_LAMBDA
    if USE_FAIRMAPS:
        from fairmaps.aggregation import power_mean
        return power_mean(np.array(arr), np.array(METRIC_WEIGHTS), lam)
    # Legacy geometric mean
    arr = np.array(arr)
    result = 1.0
    for i, weight in enumerate(METRIC_WEIGHTS):
        if i < len(arr):
            result *= arr[i] ** weight
    return result


def weighted_geom_mean(arr):
    return weighted_aggregate(arr, lam=0.0)

def calculate_demographic_metrics(partition):
    metrics = {}
    district_populations = {}
    for district in partition.parts:
        district_populations[district] = sum(
            partition.graph.nodes[node][POPULATION_COLUMN] 
            for node in partition.parts[district]
        )
    for district in partition.parts:
        district_nodes = partition.parts[district]
        total_pop = sum(partition.graph.nodes[node][POPULATION_COLUMN] for node in district_nodes)
        white_pop = sum(partition.graph.nodes[node][WHITE_POPULATION_COLUMN] for node in district_nodes)
        hispanic_pop = sum(partition.graph.nodes[node][HISPANIC_POPULATION_COLUMN] for node in district_nodes)
        asian_pop = sum(partition.graph.nodes[node][ASIAN_POPULATION_COLUMN] for node in district_nodes)
        black_pop = sum(partition.graph.nodes[node][BLACK_POPULATION_COLUMN] for node in district_nodes)
        if total_pop > 0:
            metrics[f'district_{district}_white_pct'] = white_pop / total_pop * 100
            metrics[f'district_{district}_hispanic_pct'] = hispanic_pop / total_pop * 100
            metrics[f'district_{district}_asian_pct'] = asian_pop / total_pop * 100
            metrics[f'district_{district}_black_pct'] = black_pop / total_pop * 100
            metrics[f'district_{district}_total_pop'] = total_pop
        else:
            metrics[f'district_{district}_white_pct'] = 0
            metrics[f'district_{district}_hispanic_pct'] = 0
            metrics[f'district_{district}_asian_pct'] = 0
            metrics[f'district_{district}_black_pct'] = 0
            metrics[f'district_{district}_total_pop'] = 0
    all_nodes = list(partition.graph.nodes)
    city_total_pop = sum(partition.graph.nodes[node][POPULATION_COLUMN] for node in all_nodes)
    city_white_pop = sum(partition.graph.nodes[node][WHITE_POPULATION_COLUMN] for node in all_nodes)
    city_hispanic_pop = sum(partition.graph.nodes[node][HISPANIC_POPULATION_COLUMN] for node in all_nodes)
    city_asian_pop = sum(partition.graph.nodes[node][ASIAN_POPULATION_COLUMN] for node in all_nodes)
    city_black_pop = sum(partition.graph.nodes[node][BLACK_POPULATION_COLUMN] for node in all_nodes)
    if city_total_pop > 0:
        metrics['city_white_pct'] = city_white_pop / city_total_pop * 100
        metrics['city_hispanic_pct'] = city_hispanic_pop / city_total_pop * 100
        metrics['city_asian_pct'] = city_asian_pop / city_total_pop * 100
        metrics['city_black_pct'] = city_black_pop / city_total_pop * 100
        metrics['city_total_pop'] = city_total_pop
    else:
        metrics['city_white_pct'] = 0
        metrics['city_hispanic_pct'] = 0
        metrics['city_asian_pct'] = 0
        metrics['city_black_pct'] = 0
        metrics['city_total_pop'] = 0
    metrics.update(calculate_segregation_indices(partition))
    return metrics

def calculate_segregation_indices(partition):
    indices = {}
    all_nodes = list(partition.graph.nodes)
    city_total_pop = sum(partition.graph.nodes[node][POPULATION_COLUMN] for node in all_nodes)
    city_white_pop = sum(partition.graph.nodes[node][WHITE_POPULATION_COLUMN] for node in all_nodes)
    city_hispanic_pop = sum(partition.graph.nodes[node][HISPANIC_POPULATION_COLUMN] for node in all_nodes)
    city_asian_pop = sum(partition.graph.nodes[node][ASIAN_POPULATION_COLUMN] for node in all_nodes)
    city_black_pop = sum(partition.graph.nodes[node][BLACK_POPULATION_COLUMN] for node in all_nodes)
    white_hispanic_dissimilarity = calculate_dissimilarity_index(
        partition, WHITE_POPULATION_COLUMN, HISPANIC_POPULATION_COLUMN, city_white_pop, city_hispanic_pop
    )
    white_asian_dissimilarity = calculate_dissimilarity_index(
        partition, WHITE_POPULATION_COLUMN, ASIAN_POPULATION_COLUMN, city_white_pop, city_asian_pop
    )
    hispanic_separation = calculate_separation_index(
        partition, HISPANIC_POPULATION_COLUMN, city_hispanic_pop
    )
    asian_separation = calculate_separation_index(
        partition, ASIAN_POPULATION_COLUMN, city_asian_pop
    )
    hispanic_isolation = calculate_isolation_index(
        partition, HISPANIC_POPULATION_COLUMN, city_hispanic_pop
    )
    asian_isolation = calculate_isolation_index(
        partition, ASIAN_POPULATION_COLUMN, city_asian_pop
    )
    indices.update({
        'white_hispanic_dissimilarity': white_hispanic_dissimilarity,
        'white_asian_dissimilarity': white_asian_dissimilarity,
        'hispanic_separation': hispanic_separation,
        'asian_separation': asian_separation,
        'hispanic_isolation': hispanic_isolation,
        'asian_isolation': asian_isolation
    })
    return indices

def calculate_dissimilarity_index(partition, group1_col, group2_col, city_group1_pop, city_group2_pop):
    if city_group1_pop == 0 or city_group2_pop == 0:
        return 0
    total_diff = 0
    for district in partition.parts:
        district_nodes = partition.parts[district]
        district_total_pop = sum(partition.graph.nodes[node][POPULATION_COLUMN] for node in district_nodes)
        if district_total_pop > 0:
            district_group1_pop = sum(partition.graph.nodes[node][group1_col] for node in district_nodes)
            district_group2_pop = sum(partition.graph.nodes[node][group2_col] for node in district_nodes)
            group1_pct = district_group1_pop / city_group1_pop
            group2_pct = district_group2_pop / city_group2_pop
            total_diff += abs(group1_pct - group2_pct)
    return total_diff / 2

def calculate_separation_index(partition, group_col, city_group_pop):
    if city_group_pop == 0:
        return 0
    separation_sum = 0
    for district in partition.parts:
        district_nodes = partition.parts[district]
        district_total_pop = sum(partition.graph.nodes[node][POPULATION_COLUMN] for node in district_nodes)
        if district_total_pop > 0:
            district_group_pop = sum(partition.graph.nodes[node][group_col] for node in district_nodes)
            district_non_group_pop = district_total_pop - district_group_pop
            if district_group_pop > 0:
                separation_sum += (district_group_pop / city_group_pop) * (district_non_group_pop / district_total_pop)
    return separation_sum

def calculate_isolation_index(partition, group_col, city_group_pop):
    if city_group_pop == 0:
        return 0
    isolation_sum = 0
    for district in partition.parts:
        district_nodes = partition.parts[district]
        district_total_pop = sum(partition.graph.nodes[node][POPULATION_COLUMN] for node in district_nodes)
        if district_total_pop > 0:
            district_group_pop = sum(partition.graph.nodes[node][group_col] for node in district_nodes)
            if district_group_pop > 0:
                isolation_sum += (district_group_pop / city_group_pop) * (district_group_pop / district_total_pop)
    return isolation_sum

def calculate_polsby_popper_scores(partition):
    """Calculate Polsby-Popper compactness scores for each district."""
    polsby_scores = []
    
    for district in partition.parts:
        district_nodes = partition.parts[district]
        
        # Get the geometry for this district
        district_geometries = []
        for node in district_nodes:
            if 'geometry' in partition.graph.nodes[node]:
                district_geometries.append(partition.graph.nodes[node]['geometry'])
        
        if not district_geometries:
            # Fallback if no geometries available
            polsby_scores.append(0.1)
            continue
            
        # Combine geometries for the district
        from shapely.ops import unary_union
        try:
            district_geom = unary_union(district_geometries)
            
            # Calculate area and perimeter
            area = district_geom.area
            perimeter = district_geom.length
            
            # Polsby-Popper = (4π * area) / (perimeter²)
            if perimeter > 0:
                polsby_popper = (4 * np.pi * area) / (perimeter ** 2)
                polsby_scores.append(polsby_popper)
            else:
                polsby_scores.append(0.0)
                
        except Exception as e:
            # Fallback if geometry operations fail
            print(f"Warning: Could not calculate Polsby-Popper for district {district}: {e}")
            polsby_scores.append(0.1)
    
    return polsby_scores

def generate_plan_without_scoring(graph, num_districts, index):
    timings = {}
    t0 = time.time()
    total_population = sum(graph.nodes[node][POPULATION_COLUMN] for node in graph.nodes)
    ideal_population = total_population // num_districts
    t1 = time.time()
    timings['population_calc'] = t1 - t0
    
    # Debug: Check graph connectivity
    if index == 0:  # Only check once
        print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        print(f"Total population: {total_population}, Ideal per district: {ideal_population}")
        
        # Check if graph is connected
        import networkx as nx
        if not nx.is_connected(graph):
            print("WARNING: Graph is not connected! This will cause partitioning failures.")
            # Find connected components
            components = list(nx.connected_components(graph))
            print(f"Graph has {len(components)} connected components:")
            for i, comp in enumerate(components):
                comp_pop = sum(graph.nodes[node][POPULATION_COLUMN] for node in comp)
                print(f"  Component {i+1}: {len(comp)} nodes, {comp_pop} population")
        else:
            print("Graph is connected [OK]")

    try:
        # Suppress the BipartitionWarning
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="gerrychain.tree")
            
            initial_assignment = recursive_tree_part(
                graph,
                parts=range(num_districts),
                pop_target=ideal_population,
                pop_col=POPULATION_COLUMN,
                epsilon=POPULATION_EPSILON
            )
        t2 = time.time()
        timings['recursive_tree_part'] = t2 - t1
    except RuntimeError as e:
        if "Could not find a possible cut" in str(e):
            return None
        else:
            raise e

    # Check if initial partition is valid before creating Markov chain
    try:
        partition = Partition(
            graph,
            assignment=initial_assignment,
            updaters={"population": updaters.Tally(POPULATION_COLUMN, alias="population")}
        )
        t3 = time.time()
        timings['partition_creation'] = t3 - t2

        # Test if the partition satisfies constraints
        from gerrychain.constraints import Validator
        validator = Validator([single_flip_contiguous])
        if not validator(partition):
            return None
            
        chain = MarkovChain(
            proposal=propose_random_flip,
            constraints=[single_flip_contiguous],
            accept=always_accept,
            initial_state=partition,
            total_steps=MARKOV_CHAIN_STEPS
        )
        for partition in chain:
            break
        t4 = time.time()
        timings['markov_chain'] = t4 - t3
        
    except ValueError as e:
        if "not valid according" in str(e):
            return None
        else:
            raise e
    except Exception as e:
        return None

    # Calculate metrics for this plan
    t6 = time.time()
    if USE_FAIRMAPS:
        from fairmaps.compute import compute_all_metrics
        group_cols = {
            "white": WHITE_POPULATION_COLUMN,
            "black": BLACK_POPULATION_COLUMN,
            "asian": ASIAN_POPULATION_COLUMN,
            "hispanic": HISPANIC_POPULATION_COLUMN,
        }
        metrics = compute_all_metrics(partition, pop_col=POPULATION_COLUMN, group_cols=group_cols)
        total_pop = sum(partition.graph.nodes[n][POPULATION_COLUMN] for n in partition.graph.nodes)
        plan_data = {
            'plan_number': index + 1,
            'num_districts': num_districts,
            'total_population': total_pop,
        }
        plan_data.update(metrics)
        if 'hispanic_interaction' not in plan_data:
            plan_data['hispanic_interaction'] = plan_data.get('hispanic_separation', 0)
        if 'asian_interaction' not in plan_data:
            plan_data['asian_interaction'] = plan_data.get('asian_separation', 0)
    else:
        metrics = calculate_demographic_metrics(partition)
        polsby_scores = calculate_polsby_popper_scores(partition)
        plan_data = {
            'plan_number': index + 1,
            'num_districts': num_districts,
            'total_population': sum(metrics[f'district_{d}_total_pop'] for d in range(num_districts)),
            'avg_polsby_popper': np.mean(polsby_scores) if polsby_scores else 0,
            'min_polsby_popper': np.min(polsby_scores) if polsby_scores else 0,
            'max_polsby_popper': np.max(polsby_scores) if polsby_scores else 0
        }
        plan_data.update(metrics)
        plan_data['hispanic_interaction'] = plan_data.get('hispanic_separation', plan_data.get('hispanic_interaction', 0))
        plan_data['asian_interaction'] = plan_data.get('asian_separation', plan_data.get('asian_interaction', 0))
    t7 = time.time()
    timings['calculate_metrics'] = t7 - t6
    
    return plan_data

def generate_and_save_plan(graph, num_districts, index, timing_log, contraction_map=None, original_nodes_for_export=None):
    timings = {}
    t0 = time.time()
    total_population = sum(graph.nodes[node][POPULATION_COLUMN] for node in graph.nodes)
    ideal_population = total_population // num_districts
    t1 = time.time()
    timings['population_calc'] = t1 - t0

    try:
        # Suppress the BipartitionWarning
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="gerrychain.tree")
            
            initial_assignment = recursive_tree_part(
                graph,
                parts=range(num_districts),
                pop_target=ideal_population,
                pop_col=POPULATION_COLUMN,
                epsilon=POPULATION_EPSILON
            )
        t2 = time.time()
        timings['recursive_tree_part'] = t2 - t1
    except RuntimeError as e:
        if "Could not find a possible cut" in str(e):
            return None
        else:
            raise e

    # Check if initial partition is valid before creating Markov chain
    try:
        partition = Partition(
            graph,
            assignment=initial_assignment,
            updaters={"population": updaters.Tally(POPULATION_COLUMN, alias="population")}
        )
        t3 = time.time()
        timings['partition_creation'] = t3 - t2

        # Test if the partition satisfies constraints
        from gerrychain.constraints import Validator
        validator = Validator([single_flip_contiguous])
        if not validator(partition):
            return None
            
        chain = MarkovChain(
            proposal=propose_random_flip,
            constraints=[single_flip_contiguous],
            accept=always_accept,
            initial_state=partition,
            total_steps=MARKOV_CHAIN_STEPS
        )
        for partition in chain:
            break
        t4 = time.time()
        timings['markov_chain'] = t4 - t3
        
    except ValueError as e:
        if "not valid according" in str(e):
            return None
        else:
            raise e
    except Exception as e:
        return None

    # Save block assignments (expand COI super-nodes back to original blocks when applicable)
    block_assignments = []
    if contraction_map is not None and original_nodes_for_export is not None:
        for node_id, geoid in original_nodes_for_export:
            contracted = contraction_map[node_id]
            district = partition.assignment[contracted] + 1  # 1-based
            block_assignments.append({'GEOID20': geoid, 'District': district})
    else:
        for node in partition.graph.nodes:
            geoid = partition.graph.nodes[node].get(GEOID_COLUMN, node)
            district = partition.assignment[node] + 1  # 1-based
            block_assignments.append({'GEOID20': geoid, 'District': district})
    assignment_df = pd.DataFrame(block_assignments)
    assignment_df.to_csv(f"{OUTPUT_DIRECTORY}/plan_{index+1}_block_assignment.csv", index=False)
    t5 = time.time()
    timings['save_block_assignments'] = t5 - t4

    # Calculate metrics for this plan
    t6 = time.time()
    if USE_FAIRMAPS:
        from fairmaps.compute import compute_all_metrics
        group_cols = {
            "white": WHITE_POPULATION_COLUMN,
            "black": BLACK_POPULATION_COLUMN,
            "asian": ASIAN_POPULATION_COLUMN,
            "hispanic": HISPANIC_POPULATION_COLUMN,
        }
        metrics = compute_all_metrics(partition, pop_col=POPULATION_COLUMN, group_cols=group_cols)
        total_pop = sum(partition.graph.nodes[n][POPULATION_COLUMN] for n in partition.graph.nodes)
        plan_data = {
            'plan_number': index + 1,
            'num_districts': num_districts,
            'total_population': total_pop,
        }
        plan_data.update(metrics)
    else:
        metrics = calculate_demographic_metrics(partition)
        polsby_scores = calculate_polsby_popper_scores(partition)
        plan_data = {
            'plan_number': index + 1,
            'num_districts': num_districts,
            'total_population': sum(metrics[f'district_{d}_total_pop'] for d in range(num_districts)),
            'avg_polsby_popper': np.mean(polsby_scores) if polsby_scores else 0,
            'min_polsby_popper': np.min(polsby_scores) if polsby_scores else 0,
            'max_polsby_popper': np.max(polsby_scores) if polsby_scores else 0
        }
        plan_data.update(metrics)
        plan_data['hispanic_interaction'] = plan_data.get('hispanic_separation', plan_data.get('hispanic_interaction', 0))
        plan_data['asian_interaction'] = plan_data.get('asian_separation', plan_data.get('asian_interaction', 0))
    t7 = time.time()
    timings['calculate_metrics'] = t7 - t6
    
    # Calculate plan score (percentiles -> power mean aggregate)
    t10 = time.time()
    score_arr = plan_score_array(plan_data)
    plan_data['plan_score'] = weighted_aggregate(score_arr, POWER_MEAN_LAMBDA)
    plan_data['score_array'] = score_arr
    t11 = time.time()
    timings['calculate_plan_score'] = t11 - t10
    
    # Log timing information
    timing_log.append({
        'plan_number': index + 1,
        **timings
    })
    
    return plan_data

def main():
    print(f"{'='*60}")
    print(f"REDISTRICTING ANALYSIS FOR {JURISDICTION_NAME.upper()}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Load data
    t0 = time.time()
    gdf = gpd.read_file(INPUT_GEOJSON_FILE)
    t1 = time.time()
    graph = Graph.from_geodataframe(gdf)
    t2 = time.time()
    print(f"Time to load GeoDataFrame: {t1-t0:.3f}s")
    print(f"Time to create Graph: {t2-t1:.3f}s")
    
    # Remove islands (degree-0 nodes) that cause partitioning failures
    import networkx as nx
    islands = [node for node in graph.nodes() if graph.degree(node) == 0]
    if islands:
        print(f"Removing {len(islands)} islands: {islands}")
        graph.remove_nodes_from(islands)
        print(f"Graph now has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Ensure graph is connected
    if not nx.is_connected(graph):
        print("WARNING: Graph still not connected after removing islands!")
        components = list(nx.connected_components(graph))
        print(f"Graph has {len(components)} connected components:")
        for i, comp in enumerate(components):
            comp_pop = sum(graph.nodes[node][POPULATION_COLUMN] for node in comp)
            print(f"  Component {i+1}: {len(comp)} nodes, {comp_pop} population")
        # Use only the largest component
        largest_component = max(components, key=len)
        print(f"Using largest component with {len(largest_component)} nodes")
        graph = graph.subgraph(largest_component).copy()
        print(f"Final graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    # Optional: contract COIs into super-nodes (population sum, boundary adjacency)
    contraction_map = None
    original_nodes_for_export = None
    if COI_LIST:
        geoid_to_node = {
            graph.nodes[n].get(GEOID_COLUMN): n
            for n in graph.nodes
            if graph.nodes[n].get(GEOID_COLUMN) is not None
        }
        graph, contraction_map, original_nodes_for_export = contract_graph_for_cois(
            graph, COI_LIST, geoid_to_node
        )
    
    # Check if test ensemble statistics exist
    if not load_test_ensemble_stats():
        print(f"\nNo test ensemble statistics found.")
        test_ensemble_size = int(input(f"Enter the size of test ensemble to generate statistics (default {TEST_ENSEMBLE_SIZE}): ") or TEST_ENSEMBLE_SIZE)
        print(f"Generating test ensemble of {test_ensemble_size} plans...")
        stats = generate_test_ensemble_stats(graph, NUM_DISTRICTS, test_ensemble_size)
        if stats is None:
            print("ERROR: Failed to generate test ensemble statistics!")
            return
        # Update the global metric_stats
        globals()['metric_stats'] = stats
    else:
        # Use loaded statistics (already set in load_test_ensemble_stats)
        pass
    
    # Ask user for main ensemble size
    print(f"\nTest ensemble statistics loaded successfully.")
    print(f"Available metrics: {', '.join(METRIC_ORDER)}")
    
    ensemble_size = int(input(f"Enter the number of redistricting plans to generate (default {MAIN_ENSEMBLE_SIZE}): ") or MAIN_ENSEMBLE_SIZE)
    top_n = int(input(f"How many top plans to keep block assignment CSVs for? (default {TOP_N_PLANS}): ") or TOP_N_PLANS)

    plans = []
    timing_log = []
    top_plans = {}  # filename -> (plan_number, plan_score)
    top_plan_scores = []  # list of (plan_score, filename, plan_number)

    print(f"\n{'='*60}")
    print(f"GENERATING MAIN ENSEMBLE ({ensemble_size} plans)")
    print(f"{'='*60}")
    
    for i in range(ensemble_size):
        plan_data = generate_and_save_plan(
            graph, NUM_DISTRICTS, i, timing_log,
            contraction_map=contraction_map,
            original_nodes_for_export=original_nodes_for_export,
        )
        
        # Skip if plan generation failed
        if plan_data is None:
            continue
            
        plans.append(plan_data)
        plan_score = plan_data["plan_score"]
        plan_number = plan_data["plan_number"]

        # For block assignment CSVs, keep only the best top_n
        if len(top_plans) < top_n:
            filename = f"{OUTPUT_DIRECTORY}/plan_{plan_number}_block_assignment.csv"
            top_plans[filename] = (plan_number, plan_score)
            top_plan_scores.append((plan_score, filename, plan_number))
        else:
            # Find the worst plan in top_plans
            worst_idx, (worst_score, worst_filename, worst_plan_number) = min(enumerate(top_plan_scores), key=lambda x: x[1][0])
            if plan_score > worst_score:
                # Overwrite the CSV file
                new_filename = f"{OUTPUT_DIRECTORY}/plan_{plan_number}_block_assignment.csv"
                # Remove the old file
                if os.path.exists(worst_filename):
                    os.remove(worst_filename)
                # Rename new file to the old filename for consistency (optional)
                os.rename(new_filename, worst_filename)
                # Update the dictionary and list
                top_plans.pop(worst_filename)
                top_plans[worst_filename] = (plan_number, plan_score)
                top_plan_scores[worst_idx] = (plan_score, worst_filename, plan_number)
            else:
                # Remove the just-created CSV since it's not in the top_n
                filename = f"{OUTPUT_DIRECTORY}/plan_{plan_number}_block_assignment.csv"
                if os.path.exists(filename):
                    os.remove(filename)

        if (i + 1) % 50 == 0 or (i + 1) == ensemble_size:
            print(f"Generated plan {i+1}")

    # Save all plans to CSV
    df_plans = pd.DataFrame(plans)
    df_plans.to_csv(f"{OUTPUT_DIRECTORY}/{_slug}_redistricting_plans.csv", index=False)

    # Save timing log to CSV
    timing_df = pd.DataFrame(timing_log)
    timing_df.to_csv(f"{OUTPUT_DIRECTORY}/timing_log.csv", index=False)

    # Compute and save summary statistics
    metrics = {m: df_plans[m] for m in METRIC_ORDER if m in df_plans.columns}
    with open(f"{OUTPUT_DIRECTORY}/{_slug}_summary.txt", "w") as f:
        f.write(f"{JURISDICTION_NAME.upper()} REDISTRICTING ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"total_plans: {len(df_plans)}\n\n")
        for metric_name, values in metrics.items():
            arr = np.array(values)
            summary = {
                "Lowest value": np.min(arr),
                "Q1": np.percentile(arr, 25),
                "Median": np.median(arr),
                "Q3": np.percentile(arr, 75),
                "Highest value": np.max(arr),
                "Mean": np.mean(arr),
                "Standard deviation": np.std(arr, ddof=1) if len(arr) > 1 else float('nan'),
            }
            f.write(f"{metric_name}:\n")
            for k, v in summary.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    # After df_plans is created:
    df_plans["score_array"] = df_plans.apply(plan_score_array, axis=1)
    df_plans["plan_score"] = df_plans["score_array"].apply(weighted_geom_mean)

    # Sort and print/save the ranking of the saved plans
    # Ensure all items are tuples before sorting
    top_plan_scores_clean = []
    for item in top_plan_scores:
        if isinstance(item, (list, tuple)):
            if len(item) >= 3:
                top_plan_scores_clean.append((float(item[0]), str(item[1]), int(item[2])))
        else:
            print(f"Warning: Skipping invalid item in top_plan_scores: {item}")
    
    top_plan_scores_sorted = sorted(top_plan_scores_clean, key=lambda x: -x[0])
    print("\nTop saved plans ranking (by plan_score):")
    for rank, (score, filename, plan_number) in enumerate(top_plan_scores_sorted, 1):
        print(f"#{rank}: Plan {plan_number} (file: {os.path.basename(filename)}) score: {score:.4f}")

    # Optionally, save the ranking to a file
    with open(f"{OUTPUT_DIRECTORY}/top_saved_plans_ranking.txt", "w") as f:
        for rank, (score, filename, plan_number) in enumerate(top_plan_scores_sorted, 1):
            f.write(f"#{rank}: Plan {plan_number} (file: {os.path.basename(filename)}) score: {score:.4f}\n")

if __name__ == "__main__":
    main() 