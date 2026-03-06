# Wildfire Analysis Project

## **Main Analysis Script**
*   `wildfire_analysis_ca.py`: **This is the main file.** It performs the complete analysis (Clustering and Network Analysis) restricted to California with the proper discrepancy fixes (0.74° buffer).

### How to Run
```bash
# Run Clustering Analysis
python3 wildfire_analysis_ca.py --mode clustering

# Run Network Analysis
python3 wildfire_analysis_ca.py --mode network

# Run Both
python3 wildfire_analysis_ca.py --mode both
```

## Data Files
*   `data/wildfire_data.geojson`: Main dataset (2010-2024).
*   `data/us_states.json`: US State boundaries for filtering.

## Helper/Investigation Scripts (Can be ignored or deleted)
These scripts were created to investigate the fire count discrepancy:
*   `investigate_discrepancy.py`: Compared strict vs. relaxed filtering.
*   `find_exact_buffer.py`: Calculated the exact 0.74-degree buffer.
*   `check_dates.py` & `inspect_ids.py`: Verified the date range of the dataset.
*   `inspect_columns.py`, `check_sums.py`, `check_sizes.py`, `hack_the_count_v2.py`: Various debugging tools.

---
*Last updated: March 2026*
