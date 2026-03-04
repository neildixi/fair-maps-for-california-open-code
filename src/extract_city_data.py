import json
import os
import time
from decimal import Decimal

# =============================================================================
# CONFIGURATION - EASILY SWITCH BETWEEN ANY CITY OR COUNTY IN CALIFORNIA
# =============================================================================
#
# Set JURISDICTION_TYPE to "city" or "county", then set the code and name.
# See docs/CALIFORNIA_JURISDICTIONS.md for a reference of PLACE20 and COUNTY20 codes.
#
# =============================================================================

# Jurisdiction type: "city" (filter by PLACE20) or "county" (filter by COUNTY20)
JURISDICTION_TYPE = "city"  # "city" | "county"

# For cities: PLACE20 code (e.g. 55282=Palo Alto, 68000=San Jose, 69084=Santa Clara)
# For counties: COUNTY20 code, 3-digit FIPS (e.g. 081=San Mateo, 085=Santa Clara, 037=LA)
JURISDICTION_CODE = 55282  # e.g. 55282 for Palo Alto (city), or 081 for San Mateo (county)

# Display name for outputs (used in filenames and directory)
JURISDICTION_NAME = "Palo Alto"  # e.g. "Palo Alto", "San Mateo County"

# File paths - use path relative to project root or absolute path
# Download california_merged_data.geojson from fairmapsforcalifornia.com
INPUT_DATA_FILE = "california_merged_data.geojson"
OUTPUT_DIRECTORY = None  # None = auto: {jurisdiction_name}_outputs

# Processing settings
CHECKPOINT_INTERVAL = 1000  # Show progress every N rows
DEBUG_SAMPLE_SIZE = 100  # Number of sample PLACE20 values to collect for debugging

# =============================================================================
# END CONFIGURATION - DO NOT MODIFY BELOW THIS LINE
# =============================================================================

def _slug(name):
    """Convert jurisdiction name to filesystem-friendly slug."""
    return name.lower().replace(" ", "_").replace(",", "")


def extract_jurisdiction_data(jurisdiction_type, jurisdiction_code, jurisdiction_name, input_path=None, output_dir=None):
    """
    Extract data for a city or county from the merged California data.

    Args:
        jurisdiction_type: "city" (filter by PLACE20) or "county" (filter by COUNTY20)
        jurisdiction_code: PLACE20 code (city) or COUNTY20 code (county, 3-digit)
        jurisdiction_name: Display name for output filenames
        input_path: Path to california_merged_data.geojson (default: INPUT_DATA_FILE)
        output_dir: Output directory (default: {jurisdiction_name}_outputs)

    Returns:
        GeoJSON dict or None
    """
    input_path = input_path or INPUT_DATA_FILE
    output_dir = output_dir or (_slug(jurisdiction_name) + "_outputs")
    filter_col = "PLACE20" if jurisdiction_type == "city" else "COUNTY20"

    # Normalize county code: accept 81 or 081
    code_str = str(jurisdiction_code).zfill(3) if jurisdiction_type == "county" else str(jurisdiction_code)

    print("=" * 60)
    print(f"EXTRACTING {jurisdiction_type.upper()} DATA")
    print(f"  {filter_col} = {code_str}")
    print(f"  Name: {jurisdiction_name}")
    print("=" * 60)
    
    start_time = time.time()

    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found!")
        print("Please download california_merged_data.geojson (see fairmapsforcalifornia.com)")
        return None

    slug = _slug(jurisdiction_name)
    output_filename = f"{slug}_merged_data.geojson"
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Streaming through {input_path}...")
    print(f"Looking for {filter_col} = {code_str}")
    print(f"Output will be saved to: {output_path}")
    
    # Initialize counters and data structures
    total_rows_processed = 0
    matching_rows = 0
    checkpoint_interval = CHECKPOINT_INTERVAL
    
    # Debug: Track some sample values for the filter column
    sample_values = set()
    debug_count = 0
    
    # Initialize the output GeoJSON structure
    output_geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    # Try streaming with ijson first (handles multi-GB files)
    try:
        import ijson
        print("Using streaming parser (ijson) for large file...")
        with open(input_path, 'rb') as infile:
            parser = ijson.items(infile, 'features.item')
            for feature in parser:
                total_rows_processed += 1
                if total_rows_processed % checkpoint_interval == 0:
                    print(f"Processed {total_rows_processed} rows, found {matching_rows} matches")
                properties = feature.get('properties', {})
                raw_value = properties.get(filter_col)
                if debug_count < DEBUG_SAMPLE_SIZE:
                    sample_values.add(raw_value)
                    debug_count += 1
                val_str = str(raw_value).zfill(3) if jurisdiction_type == "county" and raw_value is not None else str(raw_value)
                if val_str == code_str:
                    matching_rows += 1
                    output_geojson['features'].append(feature)
                    if matching_rows % 100 == 0:
                        print(f"Found {matching_rows} matching blocks")
    except ImportError:
        # Fallback: load entire file (fails on very large files)
        print("Loading GeoJSON file (ijson not installed - install with: pip install ijson)...")
        with open(input_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        print(f"Loaded GeoJSON with {len(data.get('features', []))} features")
        for feature in data.get('features', []):
            total_rows_processed += 1
            if total_rows_processed % checkpoint_interval == 0:
                print(f"Processed {total_rows_processed} rows, found {matching_rows} matches")
            properties = feature.get('properties', {})
            place_value = properties.get('PLACE20')
            if debug_count < DEBUG_SAMPLE_SIZE:
                sample_place_values.add(place_value)
                debug_count += 1
            if str(place_value) == str(place_code):
                matching_rows += 1
                output_geojson['features'].append(feature)
                if matching_rows % 100 == 0:
                    print(f"Found {matching_rows} matching blocks")
    
    print(f"\nProcessing complete!")
    print(f"Total rows processed: {total_rows_processed}")
    print(f"Matching rows found: {matching_rows}")
    
    # Debug: Show sample values
    print(f"\nDEBUG: Sample {filter_col} values in data:")
    for val in sorted(sample_values, key=lambda x: (x is None, str(x)))[:20]:
        print(f"  {val} (type: {type(val)})")

    if matching_rows == 0:
        print(f"\nWARNING: No data found for {filter_col} = {code_str}")
        return None
    
    # Save the filtered data (convert Decimal to float for JSON serialization)
    def _convert(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    output_geojson = _convert(output_geojson)
    print(f"\nSaving {matching_rows} rows to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(output_geojson, outfile)
    
    print("City data saved successfully")
    
    # Create a summary
    print(f"\nCITY DATA SUMMARY:")
    print(f"City PLACE20 code: {place_code}")
    if city_name:
        print(f"City name: {city_name}")
    print(f"Total blocks: {matching_rows}")
    
    # Calculate population if available
    total_population = 0
    white_population = 0
    black_population = 0
    asian_population = 0
    hispanic_population = 0
    
    for feature in output_geojson['features']:
        properties = feature.get('properties', {})
        total_population += properties.get('Population P2', 0)
        white_population += properties.get('NH_Wht', 0)
        black_population += properties.get('NH_Blk', 0)
        asian_population += properties.get('NH_Asn', 0)
        hispanic_population += properties.get('Hispanic Origin', 0)
    
    print(f"Total population: {total_population}")
    print(f"White population: {white_population}")
    print(f"Black population: {black_population}")
    print(f"Asian population: {asian_population}")
    print(f"Hispanic population: {hispanic_population}")
    
    end_time = time.time()
    print(f"\nExtraction completed in {end_time - start_time:.2f} seconds")
    
    return output_geojson

def extract_city_data(place_code, city_name=None):
    """Convenience wrapper for city extraction (backward compatible)."""
    return extract_jurisdiction_data("city", place_code, city_name or f"Place_{place_code}")


def main():
    """Main function to extract jurisdiction data (city or county)."""
    print("FairMaps Jurisdiction Data Extraction")
    print("Extracts city or county data from merged California GeoJSON.")
    print("Change JURISDICTION_TYPE, JURISDICTION_CODE, JURISDICTION_NAME at top of script.")
    print()

    output_dir = OUTPUT_DIRECTORY or (_slug(JURISDICTION_NAME) + "_outputs")

    data = extract_jurisdiction_data(
        JURISDICTION_TYPE,
        JURISDICTION_CODE,
        JURISDICTION_NAME,
        input_path=INPUT_DATA_FILE,
        output_dir=output_dir,
    )

    if data is not None:
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"File: {output_dir}/{_slug(JURISDICTION_NAME)}_merged_data.geojson")
        print("\nRun city_map_generator.py with the same JURISDICTION_NAME to generate plans.")
    else:
        print("Extraction failed.")

if __name__ == "__main__":
    main() 