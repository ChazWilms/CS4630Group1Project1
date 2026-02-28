"""
Phase 2 — Text Processing: Yelp Review Sentiment Aggregation

Runs VADER sentiment on each cleaned Yelp review, then aggregates scores
per business_id to produce a per-business sentiment summary that can be
joined into the integrated dataset for analysis.

Outputs:
  data/processed/yelp_review_sentiment.csv   — per-business aggregated scores
  data/processed/yelp_sentiment_stats.txt    — summary statistics
"""

import time
import warnings
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE        = Path(__file__).parent
DATA_DIR     = _HERE.parent.parent / "data" / "processed"
PATH_REVIEWS = DATA_DIR / "yelp_philly_reviews_clean.csv"
PATH_OUT     = DATA_DIR / "yelp_review_sentiment.csv"
PATH_STATS   = DATA_DIR / "yelp_sentiment_stats.txt"

BATCH_SIZE = 10_000   # print progress every N rows


# ---------------------------------------------------------------------------
# Step 1: Load reviews
# ---------------------------------------------------------------------------
def load_reviews():
    print("Loading cleaned Yelp reviews...")
    df = pd.read_csv(PATH_REVIEWS)
    print(f"  Reviews loaded: {len(df):,}")
    print(f"  Unique businesses: {df['business_id'].nunique():,}")
    return df


# ---------------------------------------------------------------------------
# Step 2: VADER on review text
# ---------------------------------------------------------------------------
def score_reviews(df):
    print("\nScoring reviews with VADER...")
    analyzer = SentimentIntensityAnalyzer()

    # Use cleaned text where available, fall back to raw text
    text_col = "text_clean" if "text_clean" in df.columns else "text"
    texts = df[text_col].fillna("").astype(str).tolist()

    compounds = []
    t0 = time.time()
    for i, text in enumerate(texts):
        compounds.append(analyzer.polarity_scores(text)["compound"])
        if (i + 1) % BATCH_SIZE == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i+1:,} / {len(texts):,}  ({rate:,.0f} reviews/s)")

    df["review_sentiment"] = compounds
    print(f"  Scoring complete — {len(df):,} reviews in {time.time()-t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Step 3: Aggregate per business_id
# ---------------------------------------------------------------------------
def aggregate_by_business(df):
    print("\nAggregating sentiment per business...")

    agg = df.groupby("business_id").agg(
        avg_sentiment   = ("review_sentiment", "mean"),
        median_sentiment= ("review_sentiment", "median"),
        std_sentiment   = ("review_sentiment", "std"),
        review_count    = ("review_sentiment", "count"),
        pct_positive    = ("review_sentiment", lambda x: (x >= 0.05).mean() * 100),
        pct_negative    = ("review_sentiment", lambda x: (x <= -0.05).mean() * 100),
    ).reset_index()

    agg["avg_sentiment"]    = agg["avg_sentiment"].round(4)
    agg["median_sentiment"] = agg["median_sentiment"].round(4)
    agg["std_sentiment"]    = agg["std_sentiment"].round(4)
    agg["pct_positive"]     = agg["pct_positive"].round(1)
    agg["pct_negative"]     = agg["pct_negative"].round(1)

    print(f"  Businesses with sentiment data: {len(agg):,}")
    print(f"  Avg sentiment across all businesses: {agg['avg_sentiment'].mean():.4f}")
    print(f"  Median reviews per business: {agg['review_count'].median():.0f}")
    return agg


# ---------------------------------------------------------------------------
# Step 4: Save
# ---------------------------------------------------------------------------
def save(agg):
    agg.to_csv(PATH_OUT, index=False)
    size_mb = PATH_OUT.stat().st_size / 1_048_576
    print(f"\nSaved -> {PATH_OUT}")
    print(f"  Shape: {agg.shape}  |  Size: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Step 5: Stats file
# ---------------------------------------------------------------------------
def save_stats(df_reviews, agg):
    with open(PATH_STATS, "w") as f:
        f.write("YELP REVIEW SENTIMENT STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total reviews scored: {len(df_reviews):,}\n")
        f.write(f"Unique businesses:    {agg['business_id'].nunique():,}\n\n")

        f.write("Review-level sentiment distribution:\n")
        pos = (df_reviews["review_sentiment"] >= 0.05).sum()
        neg = (df_reviews["review_sentiment"] <= -0.05).sum()
        neu = len(df_reviews) - pos - neg
        f.write(f"  Positive (>= 0.05) : {pos:,} ({pos/len(df_reviews)*100:.1f}%)\n")
        f.write(f"  Neutral             : {neu:,} ({neu/len(df_reviews)*100:.1f}%)\n")
        f.write(f"  Negative (<= -0.05) : {neg:,} ({neg/len(df_reviews)*100:.1f}%)\n\n")

        f.write("Business-level aggregated sentiment:\n")
        f.write(f"  Mean avg_sentiment:   {agg['avg_sentiment'].mean():.4f}\n")
        f.write(f"  Median avg_sentiment: {agg['avg_sentiment'].median():.4f}\n")
        f.write(f"  Std avg_sentiment:    {agg['avg_sentiment'].std():.4f}\n\n")

        f.write("Top 10 most positively reviewed businesses:\n")
        top = agg.nlargest(10, "avg_sentiment")[["business_id", "avg_sentiment", "review_count"]]
        f.write(top.to_string(index=False) + "\n\n")

        f.write("Top 10 most negatively reviewed businesses:\n")
        bot = agg.nsmallest(10, "avg_sentiment")[["business_id", "avg_sentiment", "review_count"]]
        f.write(bot.to_string(index=False) + "\n")

    print(f"Stats saved -> {PATH_STATS}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_total = time.time()
    print("=" * 70)
    print("Phase 2 — Yelp Review Sentiment Aggregation")
    print("=" * 70)

    df_reviews = load_reviews()
    df_reviews = score_reviews(df_reviews)
    agg        = aggregate_by_business(df_reviews)
    save(agg)
    save_stats(df_reviews, agg)

    print(f"\nTotal runtime: {time.time() - t_total:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
