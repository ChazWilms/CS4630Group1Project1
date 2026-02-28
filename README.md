# Project 1: Urban Data Cleaning, Integration, and Enrichment with Python

**Course:** CS 4/5630 — Python for Computational and Data Sciences
**Instructor:** Dr. Arijit Khan
**Group 1:** Quang Minh Nguyen, Rachel Stevenson, Jack Handley, Isaac Avila, Vaughn Gugger, Chaz Wilms

---

## Project Overview

This project implements a complete Python data pipeline that integrates structured data from Philadelphia 311 Service Requests with unstructured text data from the Yelp Open Dataset. The pipeline produces a cleaned, enriched, and integrated dataset that supports analysis of urban service complaints in relation to the local business environment.

---

## Datasets

**You must obtain the raw data files before running the pipeline.** The data is not stored in this repository due to file size. Download it from the shared OneDrive folder:

**[Download Raw Data (OneDrive)](https://falconbgsu-my.sharepoint.com/:f:/g/personal/cwilms_bgsu_edu/IgDahgf74i67QbGsfH4FNaNIASx8FIoQh3wMvXKL6MUgNWw?e=VFsBsg)**

Once downloaded, place the files in the repository as follows:

```
data/
├── raw/
│   ├── public_cases_fc.csv              ← Philadelphia 311 Service Requests
│   └── Yelp JSON/
│       ├── yelp_academic_dataset_business.json
│       └── yelp_academic_dataset_review.json    (~5 GB)
└── processed/                           ← created automatically by the scripts
```

---

## Repository Structure

```
CS4630Group1Project1/
├── data/
│   ├── raw/                        # Raw input files (not committed — download from OneDrive)
│   │   ├── public_cases_fc.csv
│   │   └── Yelp JSON/
│   │       ├── yelp_academic_dataset_business.json
│   │       └── yelp_academic_dataset_review.json
│   └── processed/                  # All intermediate and final output CSVs (auto-generated)
├── src/
│   ├── acquisition/                # Phase 1: Load and filter raw data
│   │   ├── load_yelp_business.py
│   │   └── load_yelp_reviews.py
│   ├── cleaning/                   # Phase 2: Clean and normalize datasets
│   │   ├── clean_311.py
│   │   ├── clean_yelp_business.py
│   │   └── clean_yelp_reviews.py
│   ├── text/                       # Phase 2: Text/NLP — VADER sentiment analysis
│   │   └── yelp_review_sentiment.py
│   ├── integration/                # Phase 3: Geospatial + semantic integration
│   │   └── match_311_yelp.py
│   ├── analysis/                   # Phase 4: K-Means clustering analysis
│   │   └── k-means_clustering.py
│   └── Phase4/                     # Phase 4: Visualization scripts
│       ├── ComparisonsofPatternsbyBusiness.py
│       ├── ComplaintHotspotsByNeighborhood.py
│       ├── ComplaintHotspotsByNeighborhood(2).py
│       ├── TopComplaintsByMonth.py
│       └── business_vs_complaints_.py
├── results/                        # Output images and HTML files
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Python version

Python 3.9 or higher is recommended.

### 2. Create a virtual environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `folium` is required for interactive heatmaps but is not in `requirements.txt`. Install it separately:
> ```bash
> pip install folium
> ```

---

## How to Run the Pipeline

Run all scripts from the **project root directory** (`CS4630Group1Project1/`) unless noted otherwise.

---

### Phase 1 — Data Acquisition (Load & Filter)

These scripts load the raw JSON datasets and produce filtered CSVs in `data/processed/`.

**Load and filter Yelp businesses to Philadelphia only:**
```bash
python src/acquisition/load_yelp_business.py
```
- Input: `data/raw/Yelp JSON/yelp_academic_dataset_business.json`
- Output: `data/processed/yelp_philly_business_filtered.csv`

**Load and filter Yelp reviews to Philadelphia businesses only:**
```bash
python src/acquisition/load_yelp_reviews.py
```
- Input: `data/raw/Yelp JSON/yelp_academic_dataset_review.json` (~5 GB, takes several minutes)
- Requires: `data/processed/yelp_philly_business_clean.csv` (run cleaning first — see below)
- Output: `data/processed/yelp_philly_reviews_filtered.csv`

---

### Phase 2 — Data Cleaning

**Clean the 311 dataset:**
```bash
python src/cleaning/clean_311.py --dir data/raw/ --filename "public_cases_fc.csv"
```
- `-d` / `--dir`: path to the directory containing the 311 CSV (must end with `/`)
- `-f` / `--filename`: name of the CSV file
- Operations: removes unnecessary columns, drops rows missing lat/lon or service name, deduplicates by service request ID and location, normalizes complaint types to lowercase, and validates ZIP codes

**Clean the Yelp business dataset:**
```bash
python src/cleaning/clean_yelp_business.py
```
- Input: `data/processed/yelp_philly_business_filtered.csv`
- Output: `data/processed/yelp_philly_business_clean.csv`
- Operations: standardizes column names, validates coordinates to Philadelphia bounds, deduplicates by `business_id` and fuzzy name+address matching, normalizes categories to a 14-group taxonomy

**Clean the Yelp review dataset:**
```bash
python src/cleaning/clean_yelp_reviews.py
```
- Input: `data/processed/yelp_philly_reviews_filtered.csv`
- Output: `data/processed/yelp_philly_reviews_clean.csv`
- Operations: standardizes column names, drops reviews missing text/business_id/date, deduplicates by review ID and identical text, normalizes review text (lowercase, removes URLs and punctuation)

**Run VADER sentiment analysis on reviews:**
```bash
python src/text/yelp_review_sentiment.py
```
- Input: `data/processed/yelp_philly_reviews_clean.csv`
- Output: `data/processed/yelp_review_sentiment.csv` — per-business aggregated sentiment scores
- Output: `data/processed/yelp_sentiment_stats.txt` — summary statistics

---

### Phase 3 — Data Integration

Matches each 311 complaint to nearby Yelp businesses using a hybrid geospatial + semantic similarity approach.

```bash
python src/integration/match_311_yelp.py
```

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--radius` / `-r` | `250` | Search radius in metres |
| `--geo-weight` | `0.5` | Weight for proximity score in hybrid score |
| `--cat-weight` | `0.5` | Weight for category similarity in hybrid score |

**Example with custom radius:**
```bash
python src/integration/match_311_yelp.py --radius 500 --geo-weight 0.6 --cat-weight 0.4
```

- Input: `data/processed/public_cases_fc.csv` (cleaned 311 data)
- Input: `data/processed/yelp_philly_business_clean.csv`
- Output: `data/processed/integrated_311_yelp_250m.csv` — all complaint-business pairs within radius
- Output: `data/processed/integrated_311_yelp_closest_250m.csv` — one closest business per complaint

**How integration works:**
1. **Geospatial search:** BallTree with Haversine metric finds all Yelp businesses within the search radius of each 311 complaint
2. **Semantic similarity:** A domain-specific lookup table maps each complaint type to relevant Yelp business categories (scores 0.0–1.0)
3. **Hybrid score:** `hybrid_score = 0.5 × proximity_score + 0.5 × category_similarity`

---

### Phase 4 — Analysis & Visualization

The Phase 4 scripts all read from `data/integrated_311_yelp.csv`. Before running them, copy or rename the integration output:

```bash
cp "data/processed/integrated_311_yelp_250m.csv" data/integrated_311_yelp.csv
```

**Important:** The Phase 4 scripts use relative paths and must be run from the `src/Phase4/` directory:

```bash
cd src/Phase4
```

**Complaint hotspot heatmap (with neighborhood boundaries):**
```bash
python "ComplaintHotspotsByNeighborhood(2).py"
```
- Output: `results/hotspot_hexbin.png` — static hexbin density map
- Output: `results/complaint_heatmap.html` — interactive folium heatmap with ZIP boundaries

**Complaint hotspot heatmap (simple version):**
```bash
python ComplaintHotspotsByNeighborhood.py
```
- Output: `results/hotspot_hexbin.png` and `results/complaint_heatmap.html`

**Business type vs. complaint type comparison:**
```bash
python ComparisonsofPatternsbyBusiness.py
```
- Output: `results/category_comparison.png` — top 5 complaints grouped by business category

**Business density vs. complaint volume:**
```bash
python business_vs_complaints_.py
```
- Output: `results/business_vs_complaints.png` — scatter/regression plot by ZIP code

**Monthly complaint trends:**
```bash
python TopComplaintsByMonth.py
```
- Output: `results/monthly_trends.png` — total complaints per month
- Output: `results/top_5_per_month.png` — top 5 complaint types for each month

**Return to project root when done:**
```bash
cd ../..
```

**K-Means clustering analysis:**
```bash
python src/analysis/k-means_clustering.py
```
- Input: `data/processed/integrated_311_yelp_closest_250m.csv` (update the path inside the script if needed)
- Produces elbow method and cluster scatter plots (displayed inline, not saved to file)

---

## Output Files Summary

| File | Description |
|---|---|
| `data/processed/yelp_philly_business_filtered.csv` | Yelp businesses filtered to Philadelphia ZIP codes |
| `data/processed/yelp_philly_business_clean.csv` | Cleaned and normalized Yelp business data |
| `data/processed/yelp_philly_reviews_filtered.csv` | Yelp reviews for Philadelphia businesses |
| `data/processed/yelp_philly_reviews_clean.csv` | Cleaned review text |
| `data/processed/yelp_review_sentiment.csv` | Per-business VADER sentiment scores |
| `data/processed/integrated_311_yelp_250m.csv` | All complaint-business pairs within 250m |
| `data/processed/integrated_311_yelp_closest_250m.csv` | One closest business per complaint |
| `data/integrated_311_yelp.csv` | Copy used by Phase 4 visualization scripts |
| `results/*.png` | Static visualization outputs |
| `results/*.html` | Interactive folium heatmap outputs |

---

## Key Findings

- **187,988** matched complaint-business pairs (250m radius)
- **Average match distance:** 101.8m (median 90.6m)
- **Average hybrid score:** 0.3667
- **Top complaint categories:** Sanitation 30%, Property & Buildings 26%, Infrastructure 25%
- **Chi-Square test** (complaint distribution by ZIP): χ² = 14,010, df = 265, p ≈ 0.0 → complaint patterns differ significantly across Philadelphia neighborhoods
