# 🗺️ Complete Guide to Replicating Redistricting Analysis

This guide will walk you through the complete process of replicating our redistricting analysis for your own city, county, or jurisdiction. Whether you're a data scientist, community organizer, or concerned citizen, this guide will help you bring fair redistricting to your community.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Finding Your Jurisdiction Codes](#finding-your-jurisdiction-codes)
3. [Setting Up Your Environment](#setting-up-your-environment)
4. [Preparing Your Data](#preparing-your-data)
5. [Running the Analysis](#running-the-analysis)
6. [Understanding the Results](#understanding-the-results)
7. [Customizing for Your Needs](#customizing-for-your-needs)
8. [Troubleshooting](#troubleshooting)
9. [Next Steps](#next-steps)

---

## 🎯 Prerequisites

Before you begin, you'll need:

- **Basic Python knowledge** (variables, functions, running scripts)
- **Command line experience** (navigating directories, running commands)
- **Geographic data** for your jurisdiction (we'll help you get this)
- **Patience** - redistricting analysis can take several hours to complete

---

## 🔍 Finding Your Jurisdiction Codes

### What Are These Codes?

The redistricting analysis uses specific codes to identify geographic areas:

- **PLACE20**: Identifies cities and incorporated places
- **COUNTY20**: Identifies counties
- **GEOID20**: Unique identifier for each census block

### How to Find Your Codes

#### Method 1: Using the Census Bureau Website (Recommended)

1. **Go to [Census Bureau TIGER/Line Files](https://www.census.gov/cgi-bin/geo/shapefiles/index.php)**
2. **Select your state** from the dropdown
3. **Choose "Places (Incorporated Places)"** for cities or "Counties" for counties
4. **Download the shapefile** for your state
5. **Open in QGIS or similar GIS software** to view the codes

#### Method 2: Using Our Data Extraction Tool

1. **Run the data extraction script** with a sample of your state's data
2. **Look at the debug output** to see available PLACE20 codes
3. **Identify your jurisdiction** from the list

#### Method 3: Online Census Tools

1. **Visit [Census Bureau QuickFacts](https://www.census.gov/quickfacts/)**
2. **Search for your city/county**
3. **Look for the "FIPS Code"** in the detailed information

### Common California Codes (Examples)

| Jurisdiction | Type | PLACE20 Code | COUNTY20 Code |
|--------------|------|--------------|---------------|
| San Jose | City | 68000 | 085 |
| Sunnyvale | City | 77000 | 085 |
| Santa Clara | City | 69084 | 085 |
| Palo Alto | City | 55282 | 081 |
| Los Angeles | City | 44000 | 037 |
| San Francisco | City | 67000 | 075 |

### Finding Your Specific Code

1. **For Cities**: Look for the PLACE20 code in the "Places" shapefile
2. **For Counties**: Look for the COUNTY20 code in the "Counties" shapefile
3. **For Special Districts**: You may need to use a different approach

---

## 🛠️ Setting Up Your Environment

### Step 1: Install Python

1. **Download Python 3.8+** from [python.org](https://python.org)
2. **Verify installation**:
   ```bash
   python --version
   pip --version
   ```

### Step 2: Clone the Repository

```bash
git clone <your-repository-url>
cd Fair-maps-for-Santa-Clara-main
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import gerrychain, geopandas, pandas, numpy; print('All packages installed successfully!')"
```

---

## 📊 Preparing Your Data

### Option 1: Use Our California Data (Recommended for California)

If you're working in California, we've already prepared the data:

1. **Download the merged California data** (contact us for access)
2. **Place it in your project directory** as `california_merged_data.geojson`
3. **Skip to the next section**

### Option 2: Prepare Your Own Data

If you're working outside California or need custom data:

#### Required Data Format

Your data must be in **GeoJSON format** with these columns:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {...},
      "properties": {
        "GEOID20": "06085000001",
        "PLACE20": 68000,
        "COUNTY20": 085,
        "Population P2": 1250,
        "NH_Wht": 800,
        "NH_Blk": 50,
        "NH_Asn": 200,
        "Hispanic Origin": 200
      }
    }
  ]
}
```

#### Data Sources

1. **Census Bureau TIGER/Line Files**: [census.gov](https://www.census.gov/cgi-bin/geo/shapefiles/index.php)
2. **American Community Survey (ACS)**: [census.gov/programs-surveys/acs](https://www.census.gov/programs-surveys/acs)
3. **Local Government Data**: Check your city/county's open data portal

#### Data Processing Steps

1. **Download census block boundaries** (TIGER/Line)
2. **Download demographic data** (ACS 5-year estimates)
3. **Merge the datasets** using a GIS tool or Python
4. **Export as GeoJSON** with the required column structure

---

## 🚀 Running the Analysis

### Step 1: Configure for Your Jurisdiction

Edit `extract_city_data.py`:

```python
# City configuration
PLACE_CODE = 68000  # Replace with your PLACE20 code
CITY_NAME = "San Jose"  # Replace with your city name
```

Edit `city_map_generator.py`:

```python
# City configuration
CITY_NAME = "San Jose"  # Replace with your city name
NUM_DISTRICTS = 10  # Replace with your desired number of districts

# Population column names (verify these match your data)
POPULATION_COLUMN = "Population P2"
WHITE_POPULATION_COLUMN = "NH_Wht"
BLACK_POPULATION_COLUMN = "NH_Blk"
ASIAN_POPULATION_COLUMN = "NH_Asn"
HISPANIC_POPULATION_COLUMN = "Hispanic Origin"
GEOID_COLUMN = "GEOID20"
```

### Step 2: Extract Your City Data

```bash
python extract_city_data.py
```

This will:
- Read the merged California data
- Extract only the data for your jurisdiction
- Save it as `[city_name]_merged_data.geojson`

### Step 3: Run the Redistricting Analysis

```bash
python city_map_generator.py
```

The script will prompt you for:
- **Test ensemble size** (default: 100) - for establishing statistical norms
- **Main ensemble size** (default: 1000) - for final analysis
- **Top plans to save** (default: 100) - number of best plans to export

### Step 4: Monitor Progress

The analysis will show:
- Progress updates every 1000 steps
- Time estimates for completion
- Any errors or warnings
- Final results summary

---

## 📈 Understanding the Results

### Output Files Generated

Your analysis will create these files in `[city_name]_outputs/`:

#### Core Results
- **`[city]_redistricting_plans.csv`**: All generated plans with metrics
- **`plan_[X]_block_assignment.csv`**: District assignments for top plans
- **`[city]_summary.txt`**: Statistical summary of all metrics

#### Analysis Data
- **`test_ensemble_stats.json`**: Statistical norms from test ensemble
- **`top_saved_plans_ranking.txt`**: Ranking of best plans
- **`timing_log.csv`**: Performance timing data

### Key Metrics Explained

#### Compactness (Polsby-Popper)
- **Range**: 0.0 to 1.0
- **Higher is better**: More compact, logical shapes
- **Good**: 0.7+ (compact districts)
- **Poor**: 0.3- (irregular shapes)

#### Demographic Segregation
- **Dissimilarity Index**: How evenly distributed groups are between districts
- **Separation Index**: How isolated minority communities are
- **Isolation Index**: How concentrated minority populations are
- **Lower is better**: More balanced representation

#### Population Equality
- **Target**: 0% deviation (perfect equality)
- **Acceptable**: ±1% deviation
- **Legal limit**: ±5% deviation (varies by jurisdiction)

### Interpreting Your Results

1. **Look at the top plans**: Focus on plans ranked #1-10
2. **Check compactness scores**: Aim for average scores above 0.6
3. **Review demographic metrics**: Lower segregation scores are better
4. **Verify population balance**: All districts should be within ±1% of target

---

## ⚙️ Customizing for Your Needs

### Adjusting District Count

```python
NUM_DISTRICTS = 6  # Change this to your desired number
```

**Considerations**:
- **Even numbers** work better for most algorithms
- **Too few districts**: May not capture community diversity
- **Too many districts**: May create very small, unrepresentative areas

### Modifying Metrics and Weights

```python
METRIC_ORDER = [
    "avg_polsby_popper",           # Compactness
    "white_hispanic_dissimilarity", # Demographic balance
    "white_asian_dissimilarity",    # Demographic balance
    "hispanic_separation",          # Minority representation
    "asian_separation",             # Minority representation
    "hispanic_isolation",           # Minority concentration
    "asian_isolation"               # Minority concentration
]

METRIC_WEIGHTS = [2/7, 25/98, 25/98, 5/98, 5/98, 5/98, 5/98]
```

**Current weights**:
- **Compactness**: 28.6% (2/7)
- **Demographic balance**: 71.4% (5/7)

**To adjust**:
- Increase compactness weight for more regular shapes
- Increase demographic weights for better representation
- Add new metrics (e.g., partisan fairness, incumbent protection)

### Population Tolerance

```python
POPULATION_EPSILON = 0.15  # 15% tolerance
```

**Recommendations**:
- **Small jurisdictions** (< 100k people): 0.01-0.05 (1-5%)
- **Medium jurisdictions** (100k-1M people): 0.05-0.10 (5-10%)
- **Large jurisdictions** (> 1M people): 0.10-0.20 (10-20%)

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue: "Module not found" errors
**Solution**: Install missing packages
```bash
pip install [package_name]
```

#### Issue: "No features found" error
**Solution**: Check your PLACE20 code and data file
```python
# Verify your code exists in the data
python -c "
import json
with open('california_merged_data.geojson', 'r') as f:
    data = json.load(f)
place_codes = set(f['properties']['PLACE20'] for f in data['features'])
print('Available PLACE20 codes:', sorted(list(place_codes))[:20])
"
```

#### Issue: "Partitioning failed" errors
**Solution**: Increase population tolerance or check data quality
```python
POPULATION_EPSILON = 0.20  # Increase tolerance
```

#### Issue: Very slow performance
**Solution**: Reduce ensemble sizes for testing
```python
TEST_ENSEMBLE_SIZE = 50    # Smaller test ensemble
MAIN_ENSEMBLE_SIZE = 500   # Smaller main ensemble
```

#### Issue: Memory errors
**Solution**: Process smaller areas or use more efficient data structures
```python
# Add memory optimization
import gc
gc.collect()  # Force garbage collection
```

### Performance Optimization Tips

1. **Start small**: Use small ensemble sizes for testing
2. **Monitor memory**: Watch system resources during analysis
3. **Use SSD storage**: Faster I/O for large data files
4. **Close other applications**: Free up system resources

---

## 🎯 Next Steps

### After Running Your Analysis

1. **Review the results**: Examine top plans and metrics
2. **Validate manually**: Check a few districts on a map
3. **Share findings**: Present to local officials and community groups
4. **Iterate**: Adjust parameters and run again if needed

### Community Engagement

1. **Create visualizations**: Use the block assignment files to create maps
2. **Write a report**: Document your methodology and findings
3. **Present to stakeholders**: Local government, community organizations
4. **Build support**: Educate others about fair redistricting

### Advanced Customization

1. **Add new metrics**: Partisan fairness, incumbent protection
2. **Custom constraints**: Respect existing boundaries, preserve communities
3. **Multi-objective optimization**: Balance competing goals
4. **Real-time analysis**: Web interface for interactive exploration

---

## 📚 Additional Resources

### Documentation
- **GerryChain Documentation**: [gerrychain.readthedocs.io](https://gerrychain.readthedocs.io)
- **GeoPandas Documentation**: [geopandas.org](https://geopandas.org)
- **Census Bureau API**: [census.gov/data/developers](https://www.census.gov/data/developers)

### Community Support
- **GitHub Issues**: Report bugs and request features
- **Discussion Forums**: Connect with other redistricting analysts
- **Local Organizations**: Find groups working on fair redistricting

### Legal Considerations
- **Voting Rights Act**: Ensure compliance with federal law
- **State Requirements**: Check your state's redistricting laws
- **Local Ordinances**: Review city/county redistricting rules

---

## 🆘 Getting Help

### When You're Stuck

1. **Check this guide**: Review the relevant section
2. **Look at examples**: Examine the Santa Clara County analysis
3. **Search issues**: Look for similar problems in GitHub issues
4. **Ask the community**: Post questions in discussion forums

### Contact Information

- **Technical Support**: GitHub issues for code problems
- **Methodology Questions**: Community forums for best practices
- **Data Access**: Contact us for California data files

---

## 🎉 Congratulations!

You've successfully set up your own redistricting analysis! This tool gives you the power to:

- **Generate fair redistricting plans** for your community
- **Evaluate existing districts** for fairness and compactness
- **Advocate for better representation** with data-driven evidence
- **Contribute to democracy** by ensuring fair electoral boundaries

Remember: **Fair redistricting is a fundamental part of democracy**. Your work helps ensure that every voice is heard and every vote counts equally.

**Good luck with your analysis, and thank you for working to make democracy fairer for everyone!** 🗽✨
