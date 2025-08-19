import json
import os
import time

# =============================================================================
# CONFIGURATION VARIABLES - MODIFY THESE FOR YOUR CITY
# =============================================================================

# City configuration
PLACE_CODE = 69084  # PLACE20 code for your city (e.g., 55282 for Palo Alto)
CITY_NAME = "Santa Clara"  # Name of your city (used for output filename)

# File paths
INPUT_DATA_FILE = "california_merged_data.geojson"  # Source merged California data
OUTPUT_DIRECTORY = CITY_NAME.lower().replace(" ", "_") + "_outputs"  # Directory to save extracted city data

# Processing settings
CHECKPOINT_INTERVAL = 1000  # Show progress every N rows
DEBUG_SAMPLE_SIZE = 100  # Number of sample PLACE20 values to collect for debugging

# =============================================================================
# END CONFIGURATION - DO NOT MODIFY BELOW THIS LINE
# =============================================================================

def extract_city_data(place_code, city_name=None):
    """
    Extract data for a specific city from the merged California data
    by streaming through the GeoJSON file line by line
    
    Args:
        place_code (int): The PLACE20 code for the city
        city_name (str, optional): Name of the city for output filename. 
                                 If None, will use the place_code
    """
    print("=" * 60)
    print(f"EXTRACTING CITY DATA FOR PLACE20 CODE: {place_code}")
    if city_name:
        print(f"CITY NAME: {city_name}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Input and output paths
    input_path = INPUT_DATA_FILE
    
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found!")
        print("Please run merge_california_data.py first to create the merged data file.")
        return None
    
    # Create output filename
    if city_name:
        output_filename = f"{city_name.lower().replace(' ', '_')}_merged_data.geojson"
    else:
        output_filename = f"place_{place_code}_merged_data.geojson"
    
    output_path = f"{OUTPUT_DIRECTORY}/{output_filename}"
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    print(f"Streaming through {input_path}...")
    print(f"Looking for PLACE20 = {place_code}")
    print(f"Output will be saved to: {output_path}")
    
    # Initialize counters and data structures
    total_rows_processed = 0
    matching_rows = 0
    checkpoint_interval = CHECKPOINT_INTERVAL
    
    # Debug: Track some sample PLACE20 values
    sample_place_values = set()
    debug_count = 0
    
    # Initialize the output GeoJSON structure
    output_geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    # Load the entire GeoJSON file
    print("Loading GeoJSON file...")
    with open(input_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    print(f"Loaded GeoJSON with {len(data.get('features', []))} features")
    
    # Process each feature
    for feature in data.get('features', []):
        total_rows_processed += 1
        
        # Check progress checkpoint
        if total_rows_processed % checkpoint_interval == 0:
            print(f"Processed {total_rows_processed} rows, found {matching_rows} matches")
        
        # Check if this feature has the matching PLACE20 code
        properties = feature.get('properties', {})
        place_value = properties.get('PLACE20')
        
        # Debug: Collect some sample PLACE20 values
        if debug_count < DEBUG_SAMPLE_SIZE:
            sample_place_values.add(place_value)
            debug_count += 1
        
        # Convert both to strings for comparison to handle type mismatches
        if str(place_value) == str(place_code):
            matching_rows += 1
            output_geojson['features'].append(feature)
            
            # Show progress for matching rows
            if matching_rows % 100 == 0:
                print(f"Found {matching_rows} matching blocks")
    
    print(f"\nProcessing complete!")
    print(f"Total rows processed: {total_rows_processed}")
    print(f"Matching rows found: {matching_rows}")
    
    # Debug: Show sample PLACE20 values found
    print(f"\nDEBUG: Sample PLACE20 values found in data:")
    for val in sorted(sample_place_values):
        print(f"  {val} (type: {type(val)})")
    
    if matching_rows == 0:
        print(f"\nWARNING: No data found for PLACE20 code {place_code}")
        print(f"Looking for: {place_code} (type: {type(place_code)})")
        return None
    
    # Save the filtered data
    print(f"\nSaving {matching_rows} rows to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(output_geojson, outfile)
    
    print(f"✓ City data saved successfully")
    
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

def main():
    """Main function to extract city data"""
    print("City Data Extraction Tool")
    print("This tool extracts data for a specific city from the merged California data.")
    print()
    
    # Use configuration variables
    place_code = PLACE_CODE
    city_name = CITY_NAME
    
    print(f"Extracting data for {city_name} (PLACE20: {place_code})...")
    
    city_data = extract_city_data(place_code, city_name)
    
    if city_data is not None:
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"File created: {OUTPUT_DIRECTORY}/{city_name.lower().replace(' ', '_')}_merged_data.geojson")
        print("\nYou can now use this file for city-specific redistricting analysis.")
        print("\nTo extract data for other cities, modify the configuration variables at the top of this script.")
    else:
        print("Failed to extract city data.")

if __name__ == "__main__":
    main() 