# Project 1:Urban Data Cleaning, Integration, and Enrichment with Python

**Course:** CS 4/5630, PYTHON FOR COMPUTATIONAL AND DATA SCIENCES  
**Instructor:** Dr. Arijit Khan  
**Group 1** Quang Minh Nguyen, Rachel Stevenson, Jack Handley, Isaac Avila, Vaughn Gugger, Chaz Wilms

## Project Overview
This project features a complete Python pipeline that integrates structured data from NYC Philadelphia 311 Service Requests and unstructured text data from the Yelp Open Dataset. The final output is a cleaned, enriched, and integrated dataset supporting the analysis of urban service complaints in relation to local business environments. 

## Datasets
1. **Philadelphia 311 Service Requests:** Contains complaint types, descriptions, timestamps, and locations.
2. **Yelp Open Dataset:** Contains business metadata, categories, geolocation, and review text.

## Repository Structure
* `data/raw/`: Directory for the original, downloaded datasets. *(Note: Data files are ignored via .gitignore due to size)*
* `data/processed/`: Directory for the cleaned and integrated final datasets. *(Note: Data files are ignored via .gitignore)*
* `src/`: Directory containing all Python scripts and Jupyter Notebooks for the data pipeline.
* `Project-1_Report.pdf`: The final 10-page project report detailing problems studied, solutions proposed, and key findings.

## How to Run the Code

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Access
Contact Chaz to receive the dataset ZIP files (the open-source datasets update continuously; the specific snapshots we used are hosted in a shared team OneDrive folder). Once received, extract files as follows:
- Philadelphia 311 CSV → `data/raw/public_cases_fc.csv`
- Yelp JSON files → `data/raw/Yelp JSON/`

### 3. Data Cleaning
Clean the 311 dataset:
```bash
python src/cleaning/clean_311.py -d data/raw/ -f public_cases_fc.csv
```
Clean the Yelp business and review datasets:
```bash
python src/cleaning/clean_yelp_business.py
python src/cleaning/clean_yelp_reviews.py
```
Cleaned files are saved to `data/processed/`.

### 4. Text Enrichment (NLP)
Run VADER sentiment and complaint category classification on 311 data:
```bash
python src/text/complaint_nlp.py
```
Aggregate Yelp review sentiment per business:
```bash
python src/text/yelp_review_sentiment.py
```

### 5. Data Integration
Run the hybrid integration (Strategy C: geospatial + semantic similarity):
```bash
python src/integration/match_311_yelp.py
```
Outputs saved to `data/processed/`:
- `integrated_311_yelp.csv` — all complaint-business pairs within 250m radius
- `integrated_311_yelp_closest.csv` — one row per complaint (nearest business only)

### 6. Analysis & Visualizations
```bash
python src/analysis/analysis.py
```
All figures saved to `data/processed/figures/`. Summary statistics saved to `data/processed/analysis_summary.txt`.

## How to Comprehend the Outputs

### Integrated Dataset Columns
| Column | Description |
|---|---|
| `service_request_id` | Unique 311 complaint ID |
| `service_name` | Raw complaint type (47 types) |
| `complaint_category` | Broad category (6 groups, ML-derived) |
| `business_name` / `primary_category` | Matched Yelp business details |
| `distance_m` | Distance in meters from complaint to business |
| `proximity_score` | Geospatial score: 1.0 = same location, 0.0 = at 250m radius edge |
| `category_similarity` | Semantic similarity score (domain map, 0–1) |
| `hybrid_score` | Final match score: 0.5 × proximity + 0.5 × category similarity |
| `sentiment_compound` | VADER compound score on 311 status notes (−1 to 1) |
| `biz_avg_sentiment` | Avg VADER score across all Yelp reviews for matched business |
| `stars` | Yelp star rating of matched business |

### Figures
| Figure | What it shows |
|---|---|
| `complaint_category_distribution.png` | Volume by complaint category |
| `complaints_by_business_category.png` | Heatmap of complaint vs. business category co-occurrence |
| `complaint_volume_over_time.png` | Monthly complaint trend across 2025 |
| `sentiment_by_complaint_category.png` | VADER sentiment on 311 status notes by category |
| `business_rating_vs_complaints.png` | Yelp star ratings of businesses near each complaint type |
| `hybrid_score_distribution.png` | Distribution of integration match quality scores |
| `top_zipcodes_complaints.png` | Top 15 Philadelphia ZIP codes by complaint volume |
| `complaint_category_by_zipcode.png` | Complaint type mix per ZIP code |
| `biz_sentiment_vs_complaint_cat.png` | Yelp review sentiment of nearby businesses by complaint type |
| `distance_distribution.png` | Distribution of complaint-to-business match distances |

---