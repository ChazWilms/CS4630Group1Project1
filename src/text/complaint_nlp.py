"""
Phase 2 — Text Processing: 311 Complaint NLP Enrichment

Produces an enriched version of the cleaned 311 dataset with three additions:

1. VADER sentiment scores on subject text
     sentiment_compound  : float  [-1, 1]
     sentiment_label     : str    (positive / neutral / negative)

2. Complaint category labels (broad groupings of the 47 service_name values)
     complaint_category  : str    (6 broad categories)

3. Random Forest classifier trained to predict complaint_category from
   TF-IDF features of the complaint subject text — demonstrates classical ML
   text classification as required by the rubric.
     predicted_category  : str    (RF prediction)
     prediction_correct  : bool   (predicted_category == complaint_category)

Output: data/processed/311_complaints_enriched.csv
Stats:  data/processed/311_nlp_stats.txt
"""

import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).parent
DATA_DIR = _HERE.parent.parent / "data"
PATH_IN  = DATA_DIR / "processed" / "public_cases_fc 2.csv"
PATH_OUT = DATA_DIR / "processed" / "311_complaints_enriched.csv"
PATH_STATS = DATA_DIR / "processed" / "311_nlp_stats.txt"

# ---------------------------------------------------------------------------
# Broad category mapping (47 service_name values -> 6 categories)
# Used as ground-truth labels for the ML classifier
# ---------------------------------------------------------------------------
CATEGORY_MAP = {
    # Sanitation & Waste
    "rubbish/recyclable material collection": "Sanitation & Waste",
    "illegal dumping":                        "Sanitation & Waste",
    "sanitation violation":                   "Sanitation & Waste",
    "dumpster violation":                     "Sanitation & Waste",
    "sanitation / dumpster violation":        "Sanitation & Waste",
    "dead animal in street":                  "Sanitation & Waste",
    "inlet cleaning":                         "Sanitation & Waste",
    "plastic bag complaint":                  "Sanitation & Waste",

    # Infrastructure & Streets
    "street defect":                          "Infrastructure & Streets",
    "street light outage":                    "Infrastructure & Streets",
    "alley light outage":                     "Infrastructure & Streets",
    "street paving":                          "Infrastructure & Streets",
    "dangerous sidewalk":                     "Infrastructure & Streets",
    "manhole cover":                          "Infrastructure & Streets",
    "right of way unit":                      "Infrastructure & Streets",
    "shoveling":                              "Infrastructure & Streets",
    "salting":                                "Infrastructure & Streets",
    "hydrant request":                        "Infrastructure & Streets",
    "complaint (streets)":                    "Infrastructure & Streets",
    "other (streets)":                        "Infrastructure & Streets",
    "street trees":                           "Infrastructure & Streets",
    "traffic signal emergency":               "Infrastructure & Streets",
    "traffic (other)":                        "Infrastructure & Streets",
    "traffic calming request":                "Infrastructure & Streets",
    "stop sign repair":                       "Infrastructure & Streets",
    "line striping":                          "Infrastructure & Streets",

    # Property & Buildings
    "maintenance complaint":                  "Property & Buildings",
    "maintenance residential or commercial":  "Property & Buildings",
    "dangerous building complaint":           "Property & Buildings",
    "construction complaints":                "Property & Buildings",
    "graffiti removal":                       "Property & Buildings",
    "smoke detector":                         "Property & Buildings",
    "license complaint":                      "Property & Buildings",
    "li escalation":                          "Property & Buildings",

    # Vehicles & Parking
    "abandoned vehicle":                      "Vehicles & Parking",
    "abandoned bike":                         "Vehicles & Parking",

    # Community & Social Services
    "homeless encampment request":            "Community & Social Services",
    "opioid response unit":                   "Community & Social Services",
    "digital navigator request":              "Community & Social Services",
    "eclipse help":                           "Community & Social Services",
    "fire safety complaint":                  "Community & Social Services",
    "police complaint":                       "Community & Social Services",
    "complaints against fire or ems":         "Community & Social Services",
    "parks and rec safety and maintenance":   "Community & Social Services",
    "information request":                    "Community & Social Services",
    "kb escalations":                         "Community & Social Services",

    # Other
    "miscellaneous":                          "Other",
}


# ---------------------------------------------------------------------------
# Step 1: Load
# ---------------------------------------------------------------------------
def load_data():
    print("Loading 311 complaints...")
    df = pd.read_csv(PATH_IN, dtype={"zipcode": str})
    print(f"  Rows loaded: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Step 2: VADER sentiment on subject
# ---------------------------------------------------------------------------
def add_sentiment(df):
    print("\nRunning VADER sentiment on subject...")
    analyzer = SentimentIntensityAnalyzer()

    text = df["subject"].fillna("").astype(str)

    scores = [analyzer.polarity_scores(t) for t in text]
    df["sentiment_compound"] = [s["compound"] for s in scores]

    def label(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        return "neutral"

    df["sentiment_label"] = df["sentiment_compound"].apply(label)

    dist = df["sentiment_label"].value_counts()
    print(f"  Positive : {dist.get('positive', 0):,}")
    print(f"  Neutral  : {dist.get('neutral',  0):,}")
    print(f"  Negative : {dist.get('negative', 0):,}")
    print(f"  Avg compound score: {df['sentiment_compound'].mean():.4f}")
    return df


# ---------------------------------------------------------------------------
# Step 3: Map service_name -> broad complaint_category
# ---------------------------------------------------------------------------
def add_complaint_category(df):
    print("\nMapping service_name -> complaint_category...")
    df["complaint_category"] = (
        df["service_name"]
        .str.lower()
        .str.strip()
        .map(CATEGORY_MAP)
        .fillna("Other")
    )
    print(df["complaint_category"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# Step 4: TF-IDF + Random Forest classifier
# Trains on subject text to predict complaint_category.
# Demonstrates classical ML text classification (rubric requirement).
# ---------------------------------------------------------------------------
def train_classifier(df):
    print("\nTraining TF-IDF + Random Forest complaint classifier...")

    # Use subject as the text feature — it's always populated
    text = df["subject"].fillna("").astype(str).str.lower().str.strip()
    labels = df["complaint_category"]

    # Drop rows where label is Other (too few / noisy)
    mask = labels != "Other"
    text_filtered  = text[mask]
    labels_filtered = labels[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        text_filtered, labels_filtered,
        test_size=0.2, random_state=42, stratify=labels_filtered
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"  Training time: {time.time() - t0:.1f}s")

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy: {acc:.4f}")
    print("\nClassification report:")
    report = classification_report(y_test, y_pred)
    print(report)

    # Predict on full dataset
    df["predicted_category"] = pipeline.predict(text)
    df["prediction_correct"] = df["predicted_category"] == df["complaint_category"]

    return df, acc, report


# ---------------------------------------------------------------------------
# Step 5: Save enriched dataset
# ---------------------------------------------------------------------------
def save(df):
    print(f"\nSaving enriched dataset to {PATH_OUT}...")
    PATH_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PATH_OUT, index=False)
    size_mb = PATH_OUT.stat().st_size / 1_048_576
    print(f"  Shape: {df.shape}  |  Size: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Step 6: Write stats file
# ---------------------------------------------------------------------------
def save_stats(df, acc, report):
    with open(PATH_STATS, "w") as f:
        f.write("311 COMPLAINT NLP ENRICHMENT STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total complaints processed: {len(df):,}\n\n")

        f.write("Sentiment (VADER on subject):\n")
        for label, count in df["sentiment_label"].value_counts().items():
            f.write(f"  {label}: {count:,} ({count/len(df)*100:.1f}%)\n")
        f.write(f"  Avg compound score: {df['sentiment_compound'].mean():.4f}\n\n")

        f.write("Complaint category distribution:\n")
        for cat, count in df["complaint_category"].value_counts().items():
            f.write(f"  {cat}: {count:,} ({count/len(df)*100:.1f}%)\n")

        f.write(f"\nRandom Forest classifier accuracy: {acc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)

    print(f"Stats saved to {PATH_STATS}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_total = time.time()
    print("=" * 70)
    print("Phase 2 — 311 Complaint NLP Enrichment")
    print("=" * 70)

    df = load_data()
    df = add_sentiment(df)
    df = add_complaint_category(df)
    df, acc, report = train_classifier(df)
    save(df)
    save_stats(df, acc, report)

    print(f"\nTotal runtime: {time.time() - t_total:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
