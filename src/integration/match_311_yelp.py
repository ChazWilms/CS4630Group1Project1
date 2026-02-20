"""
Phase 3 — Data Integration: Strategy C (Hybrid)

Strategy C combines two complementary integration methods:
  1. Geospatial proximity  — restrict candidates to Yelp businesses within a
                             configurable radius of each 311 complaint location
  2. Feature-based similarity — among the nearby candidates, rank by TF-IDF
                                 cosine similarity between the 311 service_name
                                 and the Yelp primary_category

Design rationale
----------------
- Geospatial filtering first dramatically prunes the search space from
  231 k × 14 k = 3.3 B pairs down to a manageable candidate set.
- BallTree (Haversine) gives O(n log m) nearest-neighbour queries instead
  of O(n·m) brute-force.
- TF-IDF on short label strings captures shared vocabulary between complaint
  types (e.g. "sanitation violation") and business categories
  (e.g. "Food & Restaurants") — no LLMs required.
- The final hybrid score is a weighted sum, allowing tuning via GEO_WEIGHT
  and CAT_WEIGHT.

Inputs  (data/)
-------
  public_cases_fc.csv              — cleaned Philadelphia 311 complaints
  yelp_philly_business_clean.csv   — cleaned Yelp Philadelphia businesses

Output  (data/)
-------
  integrated_311_yelp.csv          — one row per (complaint, matched business)
                                     with distance, similarity, and hybrid scores
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import BallTree

# ── Configuration ─────────────────────────────────────────────────────────────

RADIUS_M       = 250      # geospatial search radius (metres)
EARTH_RADIUS_M = 6_371_000
GEO_WEIGHT     = 0.5      # weight for proximity score in hybrid score
CAT_WEIGHT     = 0.5      # weight for category similarity in hybrid score

_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.normpath(os.path.join(_HERE, "..", "..", "data"))

PATH_311   = os.path.join(DATA_DIR, "public_cases_fc.csv")
PATH_YELP  = os.path.join(DATA_DIR, "yelp_philly_business_clean.csv")
PATH_OUT   = os.path.join(DATA_DIR, "integrated_311_yelp.csv")


# ── Step 1: Load data ─────────────────────────────────────────────────────────

def load_data():
    """Load and minimally validate both cleaned datasets."""
    print("Loading data…")

    df_311  = pd.read_csv(PATH_311,  dtype={"zipcode": str})
    df_yelp = pd.read_csv(PATH_YELP, dtype={"postal_code": str})

    # Safety: drop any rows that lost coordinates after cleaning
    df_311  = df_311.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    df_yelp = df_yelp.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    print(f"  311 complaints : {len(df_311):>10,}")
    print(f"  Yelp businesses: {len(df_yelp):>10,}")
    return df_311, df_yelp


# ── Step 2: TF-IDF category similarity ───────────────────────────────────────

def build_category_similarity(df_311, df_yelp):
    """
    Build a lookup dict:
        sim_lookup[service_name][primary_category] → cosine similarity (float)

    TF-IDF is fit on the union of all complaint-type and category labels.
    Bi-grams are included so that phrases like "street light" score better than
    purely token-level matching.
    """
    print("Building TF-IDF category similarity…")

    complaint_types = (
        df_311["service_name"].fillna("unknown").str.lower().unique().tolist()
    )
    yelp_categories = (
        df_yelp["primary_category"].fillna("unknown").str.lower().unique().tolist()
    )

    all_labels = complaint_types + yelp_categories

    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
    tfidf_all  = vectorizer.fit_transform(all_labels)

    n_ct = len(complaint_types)
    complaint_vecs = tfidf_all[:n_ct]
    category_vecs  = tfidf_all[n_ct:]

    sim_matrix = cosine_similarity(complaint_vecs, category_vecs)  # (n_ct × n_cat)

    # Build lookup: original (not lowercased) keys for safe lookup at query time
    original_cts   = df_311["service_name"].fillna("unknown").unique().tolist()
    original_cats  = df_yelp["primary_category"].fillna("unknown").unique().tolist()

    sim_lookup = {}
    for i, ct in enumerate(original_cts):
        ct_lower = ct.lower()
        # find index of the lower-cased version in complaint_types list
        try:
            row = complaint_types.index(ct_lower)
        except ValueError:
            row = 0
        sim_lookup[ct] = {
            cat: float(sim_matrix[row, j])
            for j, cat in enumerate(original_cats)
        }

    print(f"  Complaint types : {len(complaint_types)}")
    print(f"  Yelp categories : {len(yelp_categories)}")
    return sim_lookup


# ── Step 3: Geospatial BallTree search ───────────────────────────────────────

def geospatial_match(df_311, df_yelp, radius_m=RADIUS_M):
    """
    For every 311 complaint, return all Yelp businesses within `radius_m`
    metres using a BallTree with the Haversine metric.

    Returns
    -------
    idx_311   : list[int]   positional indices into df_311
    idx_yelp  : list[int]   positional indices into df_yelp
    distances : np.ndarray  distances in metres
    """
    print(f"Running BallTree geospatial search (radius = {radius_m} m)…")
    t0 = time.time()

    # BallTree expects coordinates in radians
    yelp_rad  = np.radians(df_yelp[["latitude", "longitude"]].values)
    query_rad = np.radians(df_311[["lat", "lon"]].values)
    radius_rad = radius_m / EARTH_RADIUS_M

    tree = BallTree(yelp_rad, metric="haversine")
    indices_arr, dists_arr = tree.query_radius(
        query_rad, r=radius_rad, return_distance=True, sort_results=True
    )

    idx_311_list  = []
    idx_yelp_list = []
    dist_list     = []

    for i311, (idxs, dists) in enumerate(zip(indices_arr, dists_arr)):
        for yelp_pos, dist_rad in zip(idxs, dists):
            idx_311_list.append(i311)
            idx_yelp_list.append(int(yelp_pos))
            dist_list.append(dist_rad * EARTH_RADIUS_M)

    n_matched = sum(1 for idxs in indices_arr if len(idxs) > 0)
    print(f"  Pairs found           : {len(idx_311_list):,}")
    print(f"  Complaints with ≥1 match: {n_matched:,} / {len(df_311):,}")
    print(f"  Elapsed               : {time.time() - t0:.1f} s")

    return (
        idx_311_list,
        idx_yelp_list,
        np.array(dist_list, dtype=np.float64),
    )


# ── Step 4: Build integrated dataset ─────────────────────────────────────────

def build_integrated_dataset(df_311, df_yelp, idx_311, idx_yelp, distances, sim_lookup):
    """
    Assemble the final DataFrame for every (complaint, business) pair and
    compute the three integration scores.

    Scores
    ------
    proximity_score      = 1 − (distance_m / RADIUS_M)
                           → 1.0 at same location, 0.0 at the radius edge
    category_similarity  = TF-IDF cosine similarity (service_name vs primary_category)
    hybrid_score         = GEO_WEIGHT × proximity_score
                         + CAT_WEIGHT × category_similarity
    """
    print("Assembling integrated dataset…")

    if not idx_311:
        print("  No matches found — returning empty DataFrame.")
        return pd.DataFrame()

    # Vectorised row selection (much faster than iterating row by row)
    df_311m  = df_311.iloc[idx_311].reset_index(drop=True)
    df_yelpm = df_yelp.iloc[idx_yelp].reset_index(drop=True)

    service_names = df_311m["service_name"].fillna("unknown")
    primary_cats  = df_yelpm["primary_category"].fillna("unknown")

    # Proximity score
    prox_scores = 1.0 - (distances / RADIUS_M)
    prox_scores = np.clip(prox_scores, 0.0, 1.0)

    # Category similarity (vectorised lookup)
    cat_sims = np.array(
        [sim_lookup.get(sn, {}).get(pc, 0.0)
         for sn, pc in zip(service_names, primary_cats)],
        dtype=np.float64,
    )

    hybrid_scores = GEO_WEIGHT * prox_scores + CAT_WEIGHT * cat_sims

    df_out = pd.DataFrame({
        # ── 311 fields ──────────────────────────────────────────────────────
        "service_request_id":  df_311m["service_request_id"].values,
        "service_name":        service_names.values,
        "subject":             df_311m["subject"].values,
        "agency_responsible":  df_311m["agency_responsible"].values,
        "requested_datetime":  df_311m["requested_datetime"].values,
        "address_311":         df_311m["address"].values,
        "zipcode_311":         df_311m["zipcode"].values,
        "lat_311":             df_311m["lat"].values,
        "lon_311":             df_311m["lon"].values,
        # ── Yelp fields ─────────────────────────────────────────────────────
        "business_id":         df_yelpm["business_id"].values,
        "business_name":       df_yelpm["name"].values,
        "address_yelp":        df_yelpm["address"].values,
        "postal_code_yelp":    df_yelpm["postal_code"].values,
        "lat_yelp":            df_yelpm["latitude"].values,
        "lon_yelp":            df_yelpm["longitude"].values,
        "stars":               df_yelpm["stars"].values,
        "review_count":        df_yelpm["review_count"].values,
        "primary_category":    primary_cats.values,
        "categories_normalized": df_yelpm["categories_normalized"].values,
        # ── Integration scores ───────────────────────────────────────────────
        "distance_m":          np.round(distances, 2),
        "proximity_score":     np.round(prox_scores, 4),
        "category_similarity": np.round(cat_sims, 4),
        "hybrid_score":        np.round(hybrid_scores, 4),
    })

    # Rank each complaint's matches by hybrid_score (rank 1 = best)
    df_out["match_rank"] = (
        df_out.groupby("service_request_id")["hybrid_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    df_out = df_out.sort_values(
        ["service_request_id", "match_rank"]
    ).reset_index(drop=True)

    print(f"  Output rows            : {len(df_out):,}")
    print(f"  Unique 311 complaints  : {df_out['service_request_id'].nunique():,}")
    print(f"  Unique Yelp businesses : {df_out['business_id'].nunique():,}")
    print(f"  Avg matches/complaint  : {len(df_out) / df_out['service_request_id'].nunique():.1f}")
    return df_out


# ── Step 5: Save ──────────────────────────────────────────────────────────────

def save_output(df_out, path=PATH_OUT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_out.to_csv(path, index=False)
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"Saved → {path}")
    print(f"  Shape: {df_out.shape}  |  Size: {size_mb:.1f} MB")


# ── Summary stats ─────────────────────────────────────────────────────────────

def print_summary(df_out):
    print("\n── Integration Summary ──────────────────────────────────────")
    print(f"Hybrid score range : {df_out['hybrid_score'].min():.4f} – "
          f"{df_out['hybrid_score'].max():.4f}  "
          f"(mean {df_out['hybrid_score'].mean():.4f})")
    print(f"Distance range (m) : {df_out['distance_m'].min():.1f} – "
          f"{df_out['distance_m'].max():.1f}  "
          f"(mean {df_out['distance_m'].mean():.1f})")

    print("\nTop-5 complaint → category pairings (by frequency):")
    pairing_counts = (
        df_out.groupby(["service_name", "primary_category"])
        .size()
        .sort_values(ascending=False)
        .head(5)
    )
    print(pairing_counts.to_string())

    print("\nSample best matches (rank 1, highest hybrid score):")
    cols = ["service_request_id", "service_name", "business_name",
            "primary_category", "distance_m", "hybrid_score"]
    sample = (
        df_out[df_out["match_rank"] == 1]
        .nlargest(5, "hybrid_score")[cols]
    )
    print(sample.to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("Phase 3 — Strategy C: Hybrid Integration")
    print(f"  Radius    : {RADIUS_M} m")
    print(f"  Geo weight: {GEO_WEIGHT}  |  Category weight: {CAT_WEIGHT}")
    print("=" * 62)
    t_total = time.time()

    df_311, df_yelp       = load_data()
    sim_lookup            = build_category_similarity(df_311, df_yelp)
    idx_311, idx_yelp, distances = geospatial_match(df_311, df_yelp)
    df_out                = build_integrated_dataset(
                                df_311, df_yelp, idx_311, idx_yelp,
                                distances, sim_lookup
                            )
    save_output(df_out)
    print_summary(df_out)

    print(f"\nTotal runtime: {time.time() - t_total:.1f} s")


if __name__ == "__main__":
    main()
