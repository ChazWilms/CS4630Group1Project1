"""
Phase 4 Analysis Script

Loads the integrated closest-match dataset, joins NLP enrichment
(VADER sentiment, complaint categories) and Yelp review sentiment,
then produces visualizations and summary statistics.

Input files (data/processed/):
  integrated_311_yelp_closest.csv   — one row per complaint, nearest business
  311_complaints_enriched.csv       — VADER sentiment + complaint_category
  yelp_review_sentiment.csv         — per-business aggregated review sentiment

Outputs (data/processed/figures/):
  1.  complaint_category_distribution.png
  2.  complaints_by_business_category.png
  3.  complaint_volume_over_time.png
  4.  sentiment_by_complaint_category.png
  5.  business_rating_vs_complaints.png
  6.  hybrid_score_distribution.png
  7.  top_zipcodes_complaints.png
  8.  complaint_category_by_zipcode.png
  9.  biz_sentiment_vs_complaint_cat.png
  10. distance_distribution.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_HERE       = Path(__file__).parent
DATA_DIR    = _HERE.parent.parent / "data" / "processed"
FIGURES_DIR = DATA_DIR / "figures"

PATH_INTEGRATED = DATA_DIR / "integrated_311_yelp_closest.csv"
PATH_NLP        = DATA_DIR / "311_complaints_enriched.csv"
PATH_SENTIMENT  = DATA_DIR / "yelp_review_sentiment.csv"

sns.set_theme(style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------------
# Load and join
# ---------------------------------------------------------------------------
def load_data():
    print("Loading integrated dataset...")
    df = pd.read_csv(PATH_INTEGRATED)
    print(f"  Integrated rows: {len(df):,}")

    # Join NLP enrichment (sentiment + complaint_category) on service_request_id
    print("Joining NLP enrichment...")
    nlp_cols = ["service_request_id", "sentiment_compound", "sentiment_label",
                "complaint_category"]
    nlp = pd.read_csv(PATH_NLP, usecols=nlp_cols)
    df = df.merge(nlp, on="service_request_id", how="left")

    # Join Yelp review sentiment on business_id
    print("Joining Yelp review sentiment...")
    sent = pd.read_csv(PATH_SENTIMENT)
    sent = sent.rename(columns={"avg_sentiment": "biz_avg_sentiment",
                                "pct_positive":  "biz_pct_positive",
                                "pct_negative":  "biz_pct_negative"})
    df = df.merge(sent[["business_id", "biz_avg_sentiment",
                         "biz_pct_positive", "biz_pct_negative"]],
                  on="business_id", how="left")

    # Parse datetime and ZIP
    df['requested_datetime'] = pd.to_datetime(df['requested_datetime'], utc=True)
    df['year_month'] = df['requested_datetime'].dt.to_period('M')
    df['zipcode'] = (
        df['zipcode_311'].astype(str)
        .str.extract(r'(\d{5})')[0]
    )

    print(f"  Final rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Figure 1: Complaint category distribution
# ---------------------------------------------------------------------------
def plot_complaint_distribution(df, out_dir):
    counts = df['complaint_category'].value_counts()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=sns.color_palette("Blues_r", len(counts)))
    ax.set_xlabel("Number of Complaints")
    ax.set_title("311 Complaint Category Distribution")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va='center', fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "complaint_category_distribution.png", dpi=150)
    plt.close(fig)
    print("  [1] complaint_category_distribution.png")


# ---------------------------------------------------------------------------
# Figure 2: Complaint category vs Yelp business category heatmap
# ---------------------------------------------------------------------------
def plot_complaint_vs_biz_heatmap(df, out_dir):
    top_biz = df['primary_category'].value_counts().head(10).index
    sub = df[df['primary_category'].isin(top_biz)]

    pivot = (
        sub.groupby(['complaint_category', 'primary_category'])
        .size()
        .unstack(fill_value=0)
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_pct, annot=True, fmt=".1f", cmap="Blues",
                linewidths=0.5, ax=ax,
                cbar_kws={'label': '% of complaints in category'})
    ax.set_title("Complaint Category vs. Nearby Business Category (%)")
    ax.set_xlabel("Yelp Business Category")
    ax.set_ylabel("311 Complaint Category")
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    fig.savefig(out_dir / "complaints_by_business_category.png", dpi=150)
    plt.close(fig)
    print("  [2] complaints_by_business_category.png")


# ---------------------------------------------------------------------------
# Figure 3: Complaint volume over time
# ---------------------------------------------------------------------------
def plot_volume_over_time(df, out_dir):
    monthly = df.groupby('year_month').size().reset_index(name='count')
    monthly['date'] = monthly['year_month'].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly['date'], monthly['count'], marker='o', linewidth=1.5,
            color='steelblue', markersize=4)
    ax.fill_between(monthly['date'], monthly['count'], alpha=0.15, color='steelblue')
    ax.set_title("Monthly 311 Complaint Volume (Matched Complaints)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Complaints")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(out_dir / "complaint_volume_over_time.png", dpi=150)
    plt.close(fig)
    print("  [3] complaint_volume_over_time.png")


# ---------------------------------------------------------------------------
# Figure 4: 311 complaint sentiment (VADER) by category
# ---------------------------------------------------------------------------
def plot_sentiment_by_category(df, out_dir):
    avg_sent = (
        df.groupby('complaint_category')['sentiment_compound']
        .mean()
        .sort_values()
    )
    colors = ['#d73027' if v < 0 else '#4393c3' for v in avg_sent.values]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(avg_sent.index, avg_sent.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Avg VADER Compound Score")
    ax.set_title("311 Complaint Sentiment by Category (VADER on status_notes)")
    for bar, val in zip(bars, avg_sent.values):
        ax.text(val + (0.001 if val >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va='center',
                ha='left' if val >= 0 else 'right', fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "sentiment_by_complaint_category.png", dpi=150)
    plt.close(fig)
    print("  [4] sentiment_by_complaint_category.png")


# ---------------------------------------------------------------------------
# Figure 5: Nearby business Yelp star rating by complaint category
# ---------------------------------------------------------------------------
def plot_biz_stars_by_complaint(df, out_dir):
    order = (
        df.groupby('complaint_category')['stars']
        .median()
        .sort_values()
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x='complaint_category', y='stars',
                order=order, palette="Blues", ax=ax)
    ax.set_title("Nearby Business Yelp Star Rating by Complaint Category")
    ax.set_xlabel("Complaint Category")
    ax.set_ylabel("Business Star Rating")
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    fig.savefig(out_dir / "business_rating_vs_complaints.png", dpi=150)
    plt.close(fig)
    print("  [5] business_rating_vs_complaints.png")


# ---------------------------------------------------------------------------
# Figure 6: Hybrid score distribution
# ---------------------------------------------------------------------------
def plot_hybrid_score_distribution(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(df['hybrid_score'], bins=50, color='steelblue', edgecolor='white')
    ax.axvline(df['hybrid_score'].mean(), color='red', linestyle='--',
               label=f"Mean: {df['hybrid_score'].mean():.3f}")
    ax.axvline(df['hybrid_score'].median(), color='orange', linestyle='--',
               label=f"Median: {df['hybrid_score'].median():.3f}")
    ax.set_title("Hybrid Integration Score Distribution (Closest Match)")
    ax.set_xlabel("Hybrid Score (0.5 × proximity + 0.5 × category similarity)")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "hybrid_score_distribution.png", dpi=150)
    plt.close(fig)
    print("  [6] hybrid_score_distribution.png")


# ---------------------------------------------------------------------------
# Figure 7: Top ZIP codes by complaint volume
# ---------------------------------------------------------------------------
def plot_top_zipcodes(df, out_dir):
    top_zips = (
        df.dropna(subset=['zipcode'])
        .groupby('zipcode')
        .size()
        .sort_values(ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(top_zips.index, top_zips.values,
                  color=sns.color_palette("Blues_r", len(top_zips)))
    ax.set_title("Top 15 ZIP Codes by 311 Complaint Volume")
    ax.set_xlabel("ZIP Code")
    ax.set_ylabel("Number of Complaints")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, val in zip(bars, top_zips.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}", ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(out_dir / "top_zipcodes_complaints.png", dpi=150)
    plt.close(fig)
    print("  [7] top_zipcodes_complaints.png")


# ---------------------------------------------------------------------------
# Figure 8: Complaint category mix per top ZIP (stacked bar)
# ---------------------------------------------------------------------------
def plot_category_by_zipcode(df, out_dir):
    top_zips = (
        df.dropna(subset=['zipcode'])
        .groupby('zipcode')
        .size()
        .sort_values(ascending=False)
        .head(12)
        .index.tolist()
    )
    sub = df[df['zipcode'].isin(top_zips)]
    pivot = (
        sub.groupby(['zipcode', 'complaint_category'])
        .size()
        .unstack(fill_value=0)
    )
    pivot = pivot.loc[top_zips]
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(13, 5))
    pivot_pct.plot(kind='bar', stacked=True, ax=ax,
                   color=sns.color_palette("Set2", len(pivot_pct.columns)), width=0.7)
    ax.set_title("Complaint Category Mix by ZIP Code (Top 12 ZIPs by Volume)")
    ax.set_xlabel("ZIP Code")
    ax.set_ylabel("% of Complaints")
    ax.legend(title="Category", bbox_to_anchor=(1.01, 1),
              loc='upper left', fontsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(out_dir / "complaint_category_by_zipcode.png",
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [8] complaint_category_by_zipcode.png")


# ---------------------------------------------------------------------------
# Figure 9: Business review sentiment by complaint category
# ---------------------------------------------------------------------------
def plot_biz_sentiment_by_complaint(df, out_dir):
    avg = (
        df.groupby('complaint_category')['biz_avg_sentiment']
        .mean()
        .sort_values()
    )
    colors = ['#d73027' if v < 0 else '#4393c3' for v in avg.values]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(avg.index, avg.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Avg Business Review Sentiment (VADER Compound)")
    ax.set_title("Nearby Business Review Sentiment by Complaint Category")
    for bar, val in zip(bars, avg.values):
        ax.text(val + (0.003 if val >= 0 else -0.003),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va='center',
                ha='left' if val >= 0 else 'right', fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "biz_sentiment_vs_complaint_cat.png", dpi=150)
    plt.close(fig)
    print("  [9] biz_sentiment_vs_complaint_cat.png")


# ---------------------------------------------------------------------------
# Figure 10: Match distance distribution (closest match)
# ---------------------------------------------------------------------------
def plot_distance_distribution(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(df['distance_m'], bins=40, color='steelblue', edgecolor='white')
    ax.axvline(df['distance_m'].mean(), color='red', linestyle='--',
               label=f"Mean: {df['distance_m'].mean():.0f}m")
    ax.axvline(df['distance_m'].median(), color='orange', linestyle='--',
               label=f"Median: {df['distance_m'].median():.0f}m")
    ax.set_title("Distribution of Complaint-to-Business Match Distance (Nearest)")
    ax.set_xlabel("Distance (meters)")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "distance_distribution.png", dpi=150)
    plt.close(fig)
    print(" [10] distance_distribution.png")


# ---------------------------------------------------------------------------
# Chi-square test: complaint distribution by ZIP
# ---------------------------------------------------------------------------
def run_chi_square(df, out_dir):
    print("\n--- Chi-Square Test: Complaint Distribution by ZIP Code ---")
    clean = df.dropna(subset=['zipcode']).copy()
    contingency = pd.crosstab(clean['zipcode'], clean['complaint_category'])
    chi2, p, dof, _ = chi2_contingency(contingency)

    print(f"Chi-Square statistic : {chi2:,.4f}")
    print(f"p-value              : {p:.2e}")
    print(f"Degrees of freedom   : {dof}")
    result = 'REJECT H0' if p < 0.05 else 'FAIL TO REJECT H0'
    print(f"Result               : {result}")

    stats_path = out_dir.parent / "chi_square_results.txt"
    with open(stats_path, 'w') as f:
        f.write("CHI-SQUARE TEST: COMPLAINT DISTRIBUTION BY ZIP CODE\n")
        f.write("=" * 60 + "\n\n")
        f.write("H0: Complaint category distribution is the same across all ZIP codes\n")
        f.write("H1: Complaint category distribution differs across ZIP codes\n\n")
        f.write(f"Chi-Square statistic : {chi2:,.4f}\n")
        f.write(f"p-value              : {p:.2e}\n")
        f.write(f"Degrees of freedom   : {dof}\n")
        f.write(f"ZIP codes tested     : {contingency.shape[0]}\n")
        f.write(f"Complaint categories : {contingency.shape[1]}\n\n")
        f.write(f"Result: {result}\n\n")
        f.write("Complaint totals per ZIP (top 20 by volume):\n")
        zip_totals = contingency.sum(axis=1).sort_values(ascending=False).head(20)
        for z, total in zip_totals.items():
            f.write(f"  {z}: {total:,}\n")
        f.write("\nContingency table (counts):\n")
        f.write(contingency.to_string())
    print(f"Results saved to: {stats_path}")
    return chi2, p, dof


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def print_summary(df):
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total matched complaint-business pairs : {len(df):,}")
    print(f"Date range: {df['requested_datetime'].min().date()} "
          f"→ {df['requested_datetime'].max().date()}")
    print(f"Unique ZIP codes: {df['zipcode'].nunique()}")

    print(f"\nComplaint categories:")
    for cat, count in df['complaint_category'].value_counts().items():
        print(f"  {cat}: {count:,} ({count/len(df)*100:.1f}%)")

    print(f"\nTop 5 Yelp business categories matched:")
    for cat, count in df['primary_category'].value_counts().head(5).items():
        print(f"  {cat}: {count:,}")

    print(f"\nMatch quality (closest business per complaint):")
    print(f"  Avg distance  : {df['distance_m'].mean():.1f}m")
    print(f"  Median distance: {df['distance_m'].median():.1f}m")
    print(f"  Avg hybrid score: {df['hybrid_score'].mean():.4f}")

    print(f"\nSentiment (311 status_notes, VADER):")
    for label, count in df['sentiment_label'].value_counts().items():
        print(f"  {label}: {count:,} ({count/len(df)*100:.1f}%)")

    print(f"\nAvg Yelp star rating by complaint category:")
    for cat, stars in df.groupby('complaint_category')['stars'].mean().sort_values().items():
        print(f"  {cat}: {stars:.2f} stars")

    print(f"\nAvg business review sentiment by complaint category:")
    for cat, sent in df.groupby('complaint_category')['biz_avg_sentiment'].mean().sort_values().items():
        print(f"  {cat}: {sent:.4f}")
    print("=" * 70)


def save_summary(df, out_dir):
    stats_path = out_dir.parent / "analysis_summary.txt"
    with open(stats_path, 'w') as f:
        f.write("PHASE 4 ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total matched pairs: {len(df):,}\n")
        f.write(f"Date range: {df['requested_datetime'].min().date()} "
                f"to {df['requested_datetime'].max().date()}\n\n")

        f.write("Complaint category distribution:\n")
        for cat, count in df['complaint_category'].value_counts().items():
            f.write(f"  {cat}: {count:,} ({count/len(df)*100:.1f}%)\n")

        f.write("\nTop 10 Yelp business categories matched:\n")
        for cat, count in df['primary_category'].value_counts().head(10).items():
            f.write(f"  {cat}: {count:,}\n")

        f.write("\nMatch quality:\n")
        f.write(f"  Avg distance:    {df['distance_m'].mean():.1f}m\n")
        f.write(f"  Median distance: {df['distance_m'].median():.1f}m\n")
        f.write(f"  Avg hybrid score: {df['hybrid_score'].mean():.4f}\n")

        f.write("\nAvg Yelp star rating by complaint category:\n")
        for cat, stars in df.groupby('complaint_category')['stars'].mean().sort_values().items():
            f.write(f"  {cat}: {stars:.2f}\n")

        f.write("\nAvg business review sentiment by complaint category:\n")
        for cat, sent in df.groupby('complaint_category')['biz_avg_sentiment'].mean().sort_values().items():
            f.write(f"  {cat}: {sent:.4f}\n")

        f.write("\nTop 15 ZIP codes by complaint volume:\n")
        for z, count in (df.dropna(subset=['zipcode'])
                           .groupby('zipcode').size()
                           .sort_values(ascending=False).head(15).items()):
            f.write(f"  {z}: {count:,}\n")

    print(f"Summary saved to: {stats_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    print("\nGenerating figures...")
    plot_complaint_distribution(df, FIGURES_DIR)
    plot_complaint_vs_biz_heatmap(df, FIGURES_DIR)
    plot_volume_over_time(df, FIGURES_DIR)
    plot_sentiment_by_category(df, FIGURES_DIR)
    plot_biz_stars_by_complaint(df, FIGURES_DIR)
    plot_hybrid_score_distribution(df, FIGURES_DIR)
    plot_top_zipcodes(df, FIGURES_DIR)
    plot_category_by_zipcode(df, FIGURES_DIR)
    plot_biz_sentiment_by_complaint(df, FIGURES_DIR)
    plot_distance_distribution(df, FIGURES_DIR)

    run_chi_square(df, FIGURES_DIR)
    print_summary(df)
    save_summary(df, FIGURES_DIR)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
