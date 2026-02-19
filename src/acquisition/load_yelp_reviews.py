"""
Load Yelp Review Data for Philadelphia Area Businesses
Filters the 5GB review dataset down to only Philadelphia businesses
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def count_lines(file_path):
    """Count total lines in file for progress bar"""
    count = 0
    with open(file_path, 'r') as f:
        for _ in f:
            count += 1
    return count


def load_philly_reviews(review_file, philly_business_ids, output_file):
    """
    Stream through the full review JSON and keep only Philadelphia reviews.

    Parameters:
    -----------
    review_file : str or Path
        Path to yelp_academic_dataset_review.json
    philly_business_ids : set
        Set of business_ids from the cleaned Philadelphia business dataset
    output_file : str or Path
        Where to save the filtered reviews CSV

    Returns:
    --------
    pd.DataFrame
        Filtered reviews for Philadelphia businesses
    """
    print(f"Streaming review file (5GB) — filtering for {len(philly_business_ids):,} Philadelphia businesses...")
    print("This will take a few minutes.\n")

    reviews = []
    total_read = 0
    philly_found = 0

    with open(review_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading reviews", unit=" reviews"):
            total_read += 1
            record = json.loads(line)

            if record['business_id'] in philly_business_ids:
                reviews.append({
                    'review_id':   record['review_id'],
                    'business_id': record['business_id'],
                    'stars':       record['stars'],
                    'text':        record['text'],
                    'date':        record['date'],
                    'useful':      record['useful'],
                })
                philly_found += 1

    df = pd.DataFrame(reviews)

    print(f"\nTotal reviews read:       {total_read:,}")
    print(f"Philadelphia reviews kept: {philly_found:,}")
    print(f"Retention rate:           {philly_found/total_read*100:.1f}%")

    # Save raw filtered reviews
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return df


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "data"
    review_file  = data_dir / "raw" / "Yelp JSON" / "yelp_academic_dataset_review.json"
    business_file = data_dir / "processed" / "yelp_philly_business_clean.csv"
    output_file  = data_dir / "processed" / "yelp_philly_reviews_filtered.csv"

    # Load Philadelphia business IDs
    philly_biz = pd.read_csv(business_file)
    philly_ids = set(philly_biz['business_id'])
    print(f"Loaded {len(philly_ids):,} Philadelphia business IDs")

    # Filter reviews
    df = load_philly_reviews(review_file, philly_ids, output_file)

    print("\nReview dataset overview:")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Star distribution:\n{df['stars'].value_counts().sort_index()}")
    print(f"  Avg review length: {df['text'].str.len().mean():.0f} characters")
