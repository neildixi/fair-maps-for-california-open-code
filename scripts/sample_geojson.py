"""Sample GeoJSON to check structure and PLACE20 values."""
import json
import sys

path = r"C:\Users\neild\OneDrive\Desktop\california_merged_data.geojson"
with open(path, 'r', encoding='utf-8') as f:
    chunk = f.read(200000)

# Find first feature
i = chunk.find('"features":')
if i >= 0:
    print("Has features array")
i = chunk.find('"properties"')
if i >= 0:
    print("Has properties")
# Get first complete feature
start = chunk.find('"type":')
if start < 0:
    start = chunk.find('"features":')
print("Start idx:", start)
# Skip to first { of a feature
start = chunk.find('{', chunk.find('"features":'))
if start >= 0:
    depth = 0
    for j in range(start, min(start + 5000, len(chunk))):
        c = chunk[j]
        if c == '{': depth += 1
        elif c == '}': depth -= 1
        if depth == 0:
            try:
                feat = json.loads(chunk[start:j+1])
                props = feat.get('properties', {})
                print("Property keys:", list(props.keys())[:20])
                print("PLACE20:", props.get('PLACE20'), type(props.get('PLACE20')))
                print("GEOID20:", props.get('GEOID20'))
                print("Population P2:", props.get('Population P2'))
            except Exception as e:
                print("Parse error:", e)
            break
