"""
Yelp Review Data Cleaning Script
Phase 2: Data Cleaning for Yelp Reviews Dataset

This script implements cleaning tasks for review text:
1. Standardize column names
2. Handle missing values
3. Deduplicate reviews
4. Clean and normalize review text (lowercase, remove URLs/punctuation)
5. Save cleaned reviews for downstream sentiment analysis
"""

import pandas as pd
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class YelpReviewCleaner:
    """Clean and normalize Yelp review data"""

    def __init__(self, input_file):
        """
        Parameters:
        -----------
        input_file : str or Path
            Path to yelp_philly_reviews_raw.csv
        """
        self.input_file = Path(input_file)
        self.df = None
        self.cleaning_stats = {}

    def load_data(self):
        """Load the raw filtered review data"""
        print("Loading review data...")
        self.df = pd.read_csv(self.input_file)
        print(f"Loaded {len(self.df):,} reviews")
        self.cleaning_stats['original_count'] = len(self.df)
        return self

    def standardize_column_names(self):
        """
        Task 1: Standardize column names to snake_case
        """
        print("\n--- Standardizing Column Names ---")
        new_cols = {}
        for col in self.df.columns:
            new_col = col.lower().strip()
            new_col = re.sub(r'[^\w\s]', '', new_col)
            new_col = re.sub(r'\s+', '_', new_col)
            new_cols[col] = new_col

        self.df.rename(columns=new_cols, inplace=True)
        print(f"Columns: {list(self.df.columns)}")
        self.cleaning_stats['columns'] = list(self.df.columns)
        return self

    def handle_missing_values(self):
        """
        Task 2: Handle missing values

        Strategy:
        - Drop reviews with missing text (cannot be used for any analysis)
        - Drop reviews with missing business_id (cannot be linked to a business)
        - Drop reviews with missing date (cannot be used for time-series analysis)
        - Keep missing useful score (not critical)
        """
        print("\n--- Handling Missing Values ---")

        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        print("\nMissing value percentages:")
        for col in self.df.columns:
            if missing[col] > 0:
                print(f"  {col}: {missing[col]:,} ({missing_pct[col]:.2f}%)")

        before = len(self.df)

        # Drop reviews missing critical fields
        self.df = self.df.dropna(subset=['text', 'business_id', 'date'])
        dropped = before - len(self.df)
        print(f"\nDropped {dropped:,} reviews with missing critical fields")
        print(f"Reviews remaining: {len(self.df):,}")

        self.cleaning_stats['dropped_missing'] = dropped
        self.cleaning_stats['count_after_missing'] = len(self.df)
        return self

    def deduplicate_records(self):
        """
        Task 3: Deduplicate reviews

        Rules:
        1. Remove exact duplicate review_ids (primary key)
        2. Remove reviews with identical text from the same business
           (copy-paste duplicates / bot reviews)
        """
        print("\n--- Deduplicating Records ---")
        initial = len(self.df)

        # Rule 1: Exact review_id duplicates
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=['review_id'], keep='first')
        exact_dupes = before - len(self.df)
        print(f"Exact review_id duplicates removed: {exact_dupes:,}")

        # Rule 2: Same business_id + identical text (bot/copy-paste reviews)
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=['business_id', 'text'], keep='first')
        text_dupes = before - len(self.df)
        print(f"Duplicate text on same business removed: {text_dupes:,}")

        total_removed = initial - len(self.df)
        print(f"Total duplicates removed: {total_removed:,}")
        print(f"Reviews remaining: {len(self.df):,}")

        self.cleaning_stats['duplicates_removed'] = total_removed
        self.cleaning_stats['count_after_deduplication'] = len(self.df)
        return self

    def clean_text(self):
        """
        Task 4: Clean and normalize review text

        Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove punctuation and special characters
        4. Collapse whitespace
        """
        print("\n--- Cleaning Review Text ---")

        def normalize_text(text):
            if pd.isna(text) or not isinstance(text, str):
                return ""
            text = text.lower()
            text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
            text = re.sub(r'[^\w\s]', ' ', text)          # remove punctuation
            text = re.sub(r'\s+', ' ', text).strip()      # collapse whitespace
            return text

        self.df['text_clean'] = self.df['text'].apply(normalize_text)

        # Drop any reviews that are empty after cleaning
        before = len(self.df)
        self.df = self.df[self.df['text_clean'] != ''].copy()
        empty_dropped = before - len(self.df)

        avg_len = self.df['text_clean'].str.len().mean()
        print(f"Empty reviews after cleaning: {empty_dropped:,}")
        print(f"Reviews remaining: {len(self.df):,}")
        print(f"Average cleaned review length: {avg_len:.0f} characters")

        self.cleaning_stats['empty_after_cleaning'] = empty_dropped
        self.cleaning_stats['count_after_text_cleaning'] = len(self.df)
        self.cleaning_stats['avg_review_length_chars'] = round(avg_len, 1)
        return self

    def save_cleaned_data(self, output_file):
        """Save cleaned reviews to CSV"""
        print("\n--- Saving Cleaned Data ---")

        output_columns = [
            'review_id', 'business_id', 'stars', 'date',
            'useful', 'text', 'text_clean'
        ]

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df[output_columns].to_csv(output_path, index=False)
        print(f"Cleaned reviews saved to: {output_path}")
        print(f"Final dataset size: {len(self.df):,} reviews")

        # Save cleaning statistics
        stats_file = output_path.parent / 'yelp_review_cleaning_stats.txt'
        with open(stats_file, 'w') as f:
            f.write("YELP REVIEW CLEANING STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            for key, value in self.cleaning_stats.items():
                f.write(f"{key}: {value}\n")
        print(f"Stats saved to: {stats_file}")
        return self

    def print_cleaning_summary(self):
        """Print summary of cleaning operations"""
        print("\n" + "=" * 80)
        print("REVIEW CLEANING SUMMARY")
        print("=" * 80)
        for key, value in self.cleaning_stats.items():
            print(f"{key}: {value}")

        print("\nStar distribution in cleaned data:")
        print(self.df['stars'].value_counts().sort_index().to_string())
        print("=" * 80)


def main():
    data_dir    = Path(__file__).parent.parent.parent / "data"
    input_file  = data_dir / "processed" / "yelp_philly_reviews_filtered.csv"
    output_file = data_dir / "processed" / "yelp_philly_reviews_clean.csv"

    cleaner = YelpReviewCleaner(input_file)

    (cleaner
     .load_data()
     .standardize_column_names()
     .handle_missing_values()
     .deduplicate_records()
     .clean_text()
     .save_cleaned_data(output_file)
     .print_cleaning_summary())

    print("\n✓ Yelp review cleaning complete!")


if __name__ == "__main__":
    main()
