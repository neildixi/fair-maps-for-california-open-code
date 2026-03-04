# California Jurisdiction Reference

Use this guide to configure FairMaps for any **city** or **county** in California.

## Quick setup

1. **Choose jurisdiction type**: `"city"` or `"county"`
2. **Look up the code** in the tables below
3. **Edit `extract_city_data.py`**:
   ```python
   JURISDICTION_TYPE = "city"   # or "county"
   JURISDICTION_CODE = 55282    # PLACE20 (city) or COUNTY20 (county)
   JURISDICTION_NAME = "Palo Alto"
   ```
4. **Edit `city_map_generator.py`**:
   ```python
   JURISDICTION_NAME = "Palo Alto"  # must match extract script
   NUM_DISTRICTS = 6
   ```
5. **Run extraction** then **run generator**

---

## Cities (PLACE20 codes)

| City            | PLACE20 | County     |
|-----------------|---------|------------|
| Palo Alto       | 55282   | San Mateo  |
| San Jose        | 68000   | Santa Clara|
| Santa Clara     | 69084   | Santa Clara|
| Sunnyvale       | 77000   | Santa Clara|
| Mountain View   | 49670   | Santa Clara|
| San Francisco   | 67000   | San Francisco|
| Los Angeles     | 44000   | Los Angeles|
| San Diego       | 66000   | San Diego  |
| Sacramento      | 64000   | Sacramento |
| Oakland         | 53000   | Alameda    |
| Berkeley        | 06078   | Alameda    |
| Fremont         | 26000   | Alameda    |
| Long Beach      | 43080   | Los Angeles|
| Fresno          | 27000   | Fresno     |
| Irvine          | 36770   | Orange     |
| Anaheim         | 02000   | Orange     |

**Finding more codes**: Census TIGER/Line "Places" shapefile or [Census Bureau place codes](https://www.census.gov/geographies/reference-files/time-series/geo/place.html).

---

## Counties (COUNTY20 codes, 3-digit FIPS)

| County          | COUNTY20 |
|-----------------|----------|
| Alameda         | 001      |
| Los Angeles     | 037      |
| Orange          | 059      |
| Riverside       | 065      |
| San Bernardino  | 071      |
| San Diego       | 073      |
| San Francisco   | 075      |
| San Mateo       | 081      |
| Santa Barbara   | 083      |
| Santa Clara     | 085      |
| Sacramento      | 067      |
| Fresno          | 019      |
| Kern            | 029      |
| Ventura         | 111      |

**Note**: Use 3 digits (e.g. `81` or `081` for San Mateo). Leading zeros are optional.

---

## Full California county list (COUNTY20)

| Code | County    | Code | County   | Code | County      |
|------|-----------|------|----------|------|-------------|
| 001  | Alameda   | 039  | Marin    | 077  | San Luis Obispo |
| 003  | Alpine    | 041  | Mariposa | 079  | Santa Barbara |
| 005  | Amador    | 043  | Mendocino| 081  | San Mateo     |
| 007  | Butte     | 045  | Merced   | 083  | Santa Barbara |
| 009  | Calaveras | 047  | Modoc    | 085  | Santa Clara  |
| 011  | Colusa    | 049  | Mono     | 087  | Shasta      |
| 013  | Contra Costa | 051 | Monterey | 089  | Sierra      |
| 015  | Del Norte | 053  | Napa     | 091  | Siskiyou    |
| 017  | El Dorado | 055  | Nevada   | 093  | Solano      |
| 019  | Fresno    | 057  | Orange   | 095  | Sonoma      |
| 021  | Glenn     | 059  | Placer   | 097  | Stanislaus  |
| 023  | Humboldt  | 061  | Plumas   | 099  | Sutter      |
| 025  | Imperial  | 063  | Riverside| 101  | Tehama      |
| 027  | Inyo      | 065  | Sacramento | 103 | Trinity     |
| 029  | Kern      | 067  | San Benito | 105 | Tulare      |
| 031  | Kings     | 069  | San Bernardino | 107 | Tuolumne |
| 033  | Lake      | 071  | San Diego | 109 | Ventura    |
| 035  | Lassen    | 073  | San Francisco | 111 | Yolo      |
| 037  | Los Angeles | 075  | San Joaquin | 113 | Yuba      |

*Verify codes against [Census TIGER](https://www.census.gov/geographies/reference-files/time-series/geo/tiger-line-file.html) if unsure.

---

## Workflow summary

```
1. extract_city_data.py  →  JURISDICTION_TYPE, JURISDICTION_CODE, JURISDICTION_NAME
                           Output: {name}_outputs/{name}_merged_data.geojson

2. city_map_generator.py →  JURISDICTION_NAME (same as step 1)
                           Output: {name}_outputs/*.csv, *.json, etc.
```

Same `JURISDICTION_NAME` in both scripts = automatic path matching.
