# Redistricting Analysis Project

A Python-based redistricting analysis tool that generates and evaluates redistricting plans for California cities using Markov Chain Monte Carlo (MCMC) methods.

## Features

- **Multi-City Support**: Configurable for any California city
- **Two-Phase Ensemble**: Test ensemble for statistics, main ensemble for analysis
- **Comprehensive Metrics**: 
  - Polsby-Popper compactness scores
  - Demographic segregation indices (Dissimilarity, Separation, Isolation)
  - Population balance validation
- **Robust Error Handling**: Graceful handling of partitioning failures and contiguity violations
- **Flexible Configuration**: Easy-to-modify parameters for different cities and requirements

## Requirements

- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`):
  - gerrychain
  - geopandas
  - pandas
  - numpy
  - shapely
  - networkx

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd redistricting-analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure for your city**:
   Edit the configuration variables at the top of `city_map_generator.py`:
   ```python
   CITY_NAME = "Your City"
   NUM_DISTRICTS = 6
   INPUT_GEOJSON_FILE = "path/to/your/data.geojson"
   ```

4. **Run the analysis**:
   ```bash
   python city_map_generator.py
   ```

## Configuration
### You need to download "california_merged_data.geojson," which is available on our website. Visit fairmapsforcalifornia.com/
### City Configuration
- `CITY_NAME`: Name of your city (used for output filenames)
- `INPUT_GEOJSON_FILE`: Path to your city's GeoJSON data file
- `NUM_DISTRICTS`: Number of districts to create

### Ensemble Configuration
- `TEST_ENSEMBLE_SIZE`: Number of plans for test ensemble (default: 100)
- `MAIN_ENSEMBLE_SIZE`: Number of plans for main analysis (default: 1000)
- `TOP_N_PLANS`: Number of top plans to save block assignments for (default: 100)

### Processing Settings
- `MARKOV_CHAIN_STEPS`: Steps in Markov chain per plan (default: 1000)
- `POPULATION_EPSILON`: Population deviation tolerance (default: 0.01 = 1%)

## Output Files

The script generates several output files in the city-specific output directory:

- `*_redistricting_plans.csv`: All generated plans with metrics
- `plan_*_block_assignment.csv`: Block-to-district assignments for top plans
- `*_summary.txt`: Statistical summary of all metrics
- `timing_log.csv`: Performance timing data
- `top_saved_plans_ranking.txt`: Ranking of best plans
- `test_ensemble_stats.json`: Statistical norms from test ensemble

## Methodology

### Two-Phase Ensemble Approach
1. **Test Ensemble**: Generates a smaller set of plans to establish statistical norms for each metric
2. **Main Ensemble**: Uses those norms to score and rank a larger set of plans

### Metrics Calculated
- **Compactness**: Polsby-Popper scores for each district
- **Demographic Segregation**: 
  - Dissimilarity Index (White-Hispanic, White-Asian)
  - Separation Index (Hispanic, Asian)
  - Isolation Index (Hispanic, Asian)

### Plan Scoring
Plans are scored using a weighted geometric mean of Z-scores, with weights favoring:
- Compactness (2/7 weight)
- Demographic balance (5/7 weight distributed among segregation metrics)

## Error Handling

The script includes robust error handling for:
- **Partitioning Failures**: Skips plans when `recursive_tree_part` fails
- **Contiguity Violations**: Validates district contiguity before Markov chain
- **Graph Connectivity**: Removes isolated nodes and ensures connected components
- **Constraint Violations**: Handles Markov chain constraint failures gracefully

## Example Usage

```python
# For a small city (6 districts)
CITY_NAME = "Palo Alto"
NUM_DISTRICTS = 6
POPULATION_EPSILON = 0.01

# For a large city (10 districts)
CITY_NAME = "San Jose" 
NUM_DISTRICTS = 10
POPULATION_EPSILON = 0.15  # More flexible for larger cities
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]

## Acknowledgments

- Built with [GerryChain](https://github.com/mggg/GerryChain) for redistricting algorithms
- Uses [GeoPandas](https://geopandas.org/) for geospatial data processing
- Inspired by modern redistricting research and best practices 
