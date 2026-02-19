"""
Load and explore Yelp business data
This script loads the Yelp academic dataset and provides initial data exploration
"""

import json
import pandas as pd
from pathlib import Path


def load_yelp_businesses(file_path, nrows=None):
    """
    Load Yelp business data from JSON file

    Parameters:
    -----------
    file_path : str or Path
        Path to yelp_academic_dataset_business.json
    nrows : int, optional
        Number of rows to load (None for all)

    Returns:
    --------
    pd.DataFrame
        DataFrame containing business data
    """
    businesses = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if nrows and i >= nrows:
                break
            businesses.append(json.loads(line))

    df = pd.DataFrame(businesses)
    print(f"Loaded {len(df)} businesses")
    return df


def explore_yelp_data(df):
    """
    Explore Yelp business data and print summary statistics

    Parameters:
    -----------
    df : pd.DataFrame
        Yelp business DataFrame
    """
    print("\n" + "="*80)
    print("YELP BUSINESS DATA EXPLORATION")
    print("="*80)

    # Basic info
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Data types
    print("\n--- Data Types ---")
    print(df.dtypes)

    # Missing values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False))

    # City distribution
    print("\n--- Top 20 Cities ---")
    print(df['city'].value_counts().head(20))

    # State distribution
    print("\n--- State Distribution ---")
    print(df['state'].value_counts())

    # Sample record
    print("\n--- Sample Business Record ---")
    print(df.iloc[0].to_dict())

    return df


def filter_philadelphia_area(df):
    """
    Filter for Philadelphia city businesses using ZIP codes.

    The 311 service request dataset covers Philadelphia city limits only,
    which corresponds to ZIP codes in the 191xx range. Filtering Yelp to
    match ensures all businesses can be joined to 311 records.

    Parameters:
    -----------
    df : pd.DataFrame
        Full Yelp business DataFrame

    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame for Philadelphia city
    """
    # Philadelphia city ZIP codes all start with 191
    philly_df = df[
        df['postal_code'].astype(str).str.strip().str.startswith('191')
    ].copy()

    print(f"\n--- Philadelphia City Filtering (by ZIP code) ---")
    print(f"Original businesses: {len(df)}")
    print(f"Philadelphia city businesses: {len(philly_df)}")
    print(f"Percentage retained: {len(philly_df)/len(df)*100:.2f}%")

    print("\n--- ZIP Code Distribution (top 20) ---")
    print(philly_df['postal_code'].value_counts().head(20))

    return philly_df


if __name__ == "__main__":
    # Set paths
    data_dir = Path(__file__).parent.parent.parent / "data"
    yelp_file = data_dir / "raw" / "Yelp JSON" / "yelp_academic_dataset_business.json"

    # Load data
    print("Loading Yelp business data...")
    df = load_yelp_businesses(yelp_file)

    # Explore data
    explore_yelp_data(df)

    # Filter for Philadelphia area
    philly_df = filter_philadelphia_area(df)

    # Save filtered data for cleaning
    output_file = data_dir / "processed" / "yelp_philly_business_filtered.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    philly_df.to_csv(output_file, index=False)
    print(f"\nFiltered data saved to: {output_file}")
