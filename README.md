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
1. Ensure you have Python installed, along with the required libraries:
2. **Data Access:** To get the exact raw data we used, please contact Chaz to receive the dataset ZIP files, as the open source data sets are continuouslly updating, the specific datasets we used are hosted in a shared team OneDrive folder. Once received, extract the CSV and JSON files directly into the `data/raw/` folder.
3. Run the data cleaning pipeline by executing:
     a. 311 Cleaning: python src/cleaning/clean_311.py
     b. Yelp Business Cleaning: python src/cleaning/clean_yelp_business.py
     c. Yelp Review Cleaning: python src/cleaning/clean_yelp_reviews.py
   Each script generates cleaned datasets in the `data/processed/` folder.
4. Run the data integration script by executing: python src/integration/match_311_yelp.py
   This will perform spatial matching between complaints and businesses, enrich complaints with meta datasets, and produce the final integrated datasets, integrated_311_yelp(geospatial + semantic map).csv and integrated_311_yelp(geospatial + TF-IDF).csv, which can be found in the `data/processed/` folder.
5. Run the data analysis scripts by executing:
     a. K-Mean Analysis: python src/analysis/k-mean-analysis.py
     b. K-Mean Clustering: python src/analysis/k-mean-clustering.py
     c. Comparison of Patterns by Business: python src/Phase4/ComparisonsofPatternsbyBusiness.py
     d. Complaint Hotspots: python src/Phase4/ComplaintHotspotsByNeighborhood.py
     e. Businesses vs. Complaints: python src/Phase4/business_vs_complaints_.py
   This will generate graphs (hotmap, barcharts, cluster plots, etc.) that can be found in the `Graphs` folder in the shared OneDrive. 

## How to Comprehend the Outputs
1. Cleaned 311 Dataset: Gives standardized column names, removes duplicate records, cleans timestamps and geographic fields, and normalizes complaint types. This provides reliable, structured complaint data ready for integration. 
2. Cleaned Yelp Datasets: Gives normalized business categories, removes incomplete or invalid records, and preprocesses review texts. This provides structured business metadata and feature-engineered text inputs.
3. Integrated Datasets:
     a. Geospatial + Semantic Map: Matches complaints to nearby businesses within a defined radius and uses similarity to measures between complaint descriptions and business categories.
     b. Geospatial + TF-IDF: Uses TF-IDF vectorization for complaint text and applies cosine similarity for feature-based integration.
   Each row represents a complaint enriched with nearby business information, category similarity metrics, and distance metrics.
5. Analysis Outputs:
     a. Heatmap: Shows complaint density by neighborhood
     b. Cluster Plots: Show grouped complaint patterns using K-Means
     c. Bar Charts: Compares complaint frequency by business type
     d. Correlation Plots: Shows relations between business density and complaint volume.
   These outputs support the interpretation of Urban complaint hotspots, patterns near different business categories, and structural trends in service requests.
