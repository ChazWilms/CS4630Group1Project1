"""
Phase 3 — Data Integration: Strategy C (Hybrid)

Strategy C combines two complementary integration methods:
  1. Geospatial proximity    — restrict candidates to Yelp businesses within a
                               configurable radius of each 311 complaint location
                               using a BallTree with Haversine metric.
  2. Semantic category similarity — among nearby candidates, rank by a
                               domain-specific similarity score that maps each
                               311 complaint type to relevant Yelp business
                               categories using a hand-crafted lookup table.

Design rationale
----------------
- BallTree (Haversine) gives O(n log m) nearest-neighbour queries instead
  of O(n·m) brute-force across 3.3 B candidate pairs.
- SEMANTIC_MAP scores are graded (0.0–1.0): 1.0 = direct match (abandoned
  vehicle -> Automotive), lower scores encode partial relevance.
- The final hybrid score is a weighted sum, tunable via GEO_WEIGHT / CAT_WEIGHT.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# Configuration

RADIUS_M       = 250      # geospatial search radius (metres)
EARTH_RADIUS_M = 6_371_000
GEO_WEIGHT     = 0.5      # weight for proximity score in hybrid score
CAT_WEIGHT     = 0.5      # weight for category similarity in hybrid score

_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.normpath(os.path.join(_HERE, "..", "..", "data"))

PATH_311   = os.path.join(DATA_DIR, "public_cases_fc.csv")
PATH_YELP  = os.path.join(DATA_DIR, "yelp_philly_business_clean.csv")
PATH_OUT   = os.path.join(DATA_DIR, "integrated_311_yelp.csv")


# Step 1: Load data

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


# Step 2: Domain semantic similarity map

# Maps each 311 complaint type -> {Yelp primary_category: similarity score (0–1)}.
# Scores are graded: 1.0 = direct semantic match, lower = partial relevance.
# Any complaint type / category pair not listed receives score 0.0.
# Justification: TF-IDF produces zero cosine similarity for all pairs because
# complaint labels ("rubbish/recyclable material collection") and Yelp category
# labels ("Food & Restaurants") share no vocabulary tokens. A domain map is
# the appropriate classical alternative.
SEMANTIC_MAP = {
    # Vehicles
    "abandoned vehicle":                     {"Automotive": 1.0,
                                              "Local & Community Services": 0.2},
    "abandoned bike":                        {"Active Life & Fitness": 0.7,
                                              "Local & Community Services": 0.3},
    "traffic signal emergency":              {"Automotive": 0.7,
                                              "Local & Community Services": 0.4},
    "traffic (other)":                       {"Automotive": 0.6,
                                              "Local & Community Services": 0.3},
    "traffic calming request":               {"Automotive": 0.5,
                                              "Local & Community Services": 0.4},
    "stop sign repair":                      {"Automotive": 0.4,
                                              "Local & Community Services": 0.5},
    "line striping":                         {"Automotive": 0.4,
                                              "Local & Community Services": 0.5},
    # Sanitation / waste
    "rubbish/recyclable material collection":{"Food & Restaurants": 0.4,
                                              "Home Services": 0.5,
                                              "Local & Community Services": 0.4},
    "illegal dumping":                       {"Home Services": 0.5,
                                              "Local & Community Services": 0.6},
    "sanitation violation":                  {"Food & Restaurants": 0.8,
                                              "Health & Medical": 0.4,
                                              "Shopping & Retail": 0.3},
    "dumpster violation":                    {"Food & Restaurants": 0.5,
                                              "Home Services": 0.5,
                                              "Local & Community Services": 0.3},
    "sanitation / dumpster violation":       {"Food & Restaurants": 0.6,
                                              "Home Services": 0.5},
    "dead animal in street":                 {"Local & Community Services": 0.5,
                                              "Pets": 0.4},
    "inlet cleaning":                        {"Local & Community Services": 0.6,
                                              "Home Services": 0.3},
    "plastic bag complaint":                 {"Shopping & Retail": 0.7,
                                              "Food & Restaurants": 0.6},
    # Infrastructure / streets
    "street defect":                         {"Local & Community Services": 0.6,
                                              "Home Services": 0.4},
    "street light outage":                   {"Local & Community Services": 0.7,
                                              "Home Services": 0.3},
    "alley light outage":                    {"Local & Community Services": 0.7,
                                              "Home Services": 0.3},
    "street paving":                         {"Home Services": 0.5,
                                              "Local & Community Services": 0.5},
    "dangerous sidewalk":                    {"Local & Community Services": 0.6,
                                              "Home Services": 0.4},
    "manhole cover":                         {"Local & Community Services": 0.6,
                                              "Home Services": 0.3},
    "right of way unit":                     {"Local & Community Services": 0.6,
                                              "Professional Services": 0.3},
    "shoveling":                             {"Home Services": 0.7,
                                              "Local & Community Services": 0.4},
    "salting":                               {"Home Services": 0.6,
                                              "Local & Community Services": 0.4},
    "hydrant request":                       {"Local & Community Services": 0.6,
                                              "Home Services": 0.3},
    "complaint (streets)":                   {"Local & Community Services": 0.6,
                                              "Home Services": 0.3},
    "other (streets)":                       {"Local & Community Services": 0.5},
    # Buildings / property
    "maintenance complaint":                 {"Home Services": 0.8,
                                              "Professional Services": 0.4},
    "maintenance residential or commercial": {"Home Services": 0.9,
                                              "Professional Services": 0.5},
    "dangerous building complaint":          {"Home Services": 0.7,
                                              "Professional Services": 0.4},
    "construction complaints":               {"Home Services": 0.8,
                                              "Professional Services": 0.5},
    "graffiti removal":                      {"Local & Community Services": 0.5,
                                              "Entertainment & Arts": 0.4,
                                              "Home Services": 0.4},
    "smoke detector":                        {"Home Services": 0.7,
                                              "Health & Medical": 0.3},
    # Licensing / legal / safety
    "license complaint":                     {"Professional Services": 0.7,
                                              "Food & Restaurants": 0.5,
                                              "Shopping & Retail": 0.4},
    "li escalation":                         {"Professional Services": 0.6,
                                              "Local & Community Services": 0.3},
    "fire safety complaint":                 {"Health & Medical": 0.5,
                                              "Local & Community Services": 0.5},
    "police complaint":                      {"Local & Community Services": 0.6},
    "complaints against fire or ems":        {"Health & Medical": 0.6,
                                              "Local & Community Services": 0.4},
    # Health / social services
    "homeless encampment request":           {"Health & Medical": 0.5,
                                              "Local & Community Services": 0.7},
    "opioid response unit":                  {"Health & Medical": 0.9,
                                              "Local & Community Services": 0.3},
    "digital navigator request":             {"Education": 0.5,
                                              "Local & Community Services": 0.5},
    "eclipse help":                          {"Local & Community Services": 0.5,
                                              "Education": 0.4},
    # Parks / environment
    "street trees":                          {"Active Life & Fitness": 0.5,
                                              "Local & Community Services": 0.5,
                                              "Home Services": 0.3},
    "parks and rec safety and maintenance":  {"Active Life & Fitness": 0.9,
                                              "Entertainment & Arts": 0.4,
                                              "Local & Community Services": 0.3},
    # General
    "information request":                   {"Local & Community Services": 0.5},
    "kb escalations":                        {"Local & Community Services": 0.4},
    "miscellaneous":                         {"Local & Community Services": 0.3},
}


def build_semantic_similarity(df_311, df_yelp):
    """
    Return a lookup dict:
        sim_lookup[service_name][primary_category] -> similarity score (float)

    Scores come from SEMANTIC_MAP. Any pair not in the map defaults to 0.0.
    """
    print("Building domain semantic similarity map…")

    complaint_types = df_311["service_name"].fillna("unknown").unique().tolist()
    yelp_categories = df_yelp["primary_category"].fillna("unknown").unique().tolist()

    sim_lookup = {}
    for ct in complaint_types:
        cat_scores = SEMANTIC_MAP.get(ct.lower(), SEMANTIC_MAP.get(ct, {}))
        sim_lookup[ct] = {pc: cat_scores.get(pc, 0.0) for pc in yelp_categories}

    covered = sum(1 for ct in complaint_types
                  if ct.lower() in SEMANTIC_MAP or ct in SEMANTIC_MAP)
    print(f"  Complaint types : {len(complaint_types)}")
    print(f"  Yelp categories : {len(yelp_categories)}")
    print(f"  Types with non-zero scores: {covered} / {len(complaint_types)}")
    return sim_lookup


# Step 3: Geospatial BallTree search

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
    print(f"  Complaints with >=1 match: {n_matched:,} / {len(df_311):,}")
    print(f"  Elapsed               : {time.time() - t0:.1f} s")

    return (
        idx_311_list,
        idx_yelp_list,
        np.array(dist_list, dtype=np.float64),
    )


# Step 4: Build integrated dataset

def build_integrated_dataset(df_311, df_yelp, idx_311, idx_yelp, distances, sim_lookup):
    """
    Assemble the final DataFrame for every (complaint, business) pair and
    compute the three integration scores.

    Scores
    ------
    proximity_score      = 1 − (distance_m / RADIUS_M)
                           -> 1.0 at same location, 0.0 at the radius edge
    category_similarity  = domain semantic map score (service_name vs primary_category)
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
        # 311 fields
        "service_request_id":  df_311m["service_request_id"].values,
        "service_name":        service_names.values,
        "subject":             df_311m["subject"].values,
        "agency_responsible":  df_311m["agency_responsible"].values,
        "requested_datetime":  df_311m["requested_datetime"].values,
        "address_311":         df_311m["address"].values,
        "zipcode_311":         df_311m["zipcode"].values,
        "lat_311":             df_311m["lat"].values,
        "lon_311":             df_311m["lon"].values,
        # Yelp fields
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
        # Integration scores
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


# Step 5: Save

def save_output(df_out, path=PATH_OUT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_out.to_csv(path, index=False)
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"Saved -> {path}")
    print(f"  Shape: {df_out.shape}  |  Size: {size_mb:.1f} MB")


# Summary stats

def print_summary(df_out):
    print("\n── Integration Summary ──────────────────────────────────────")
    print(f"Hybrid score range : {df_out['hybrid_score'].min():.4f} – "
          f"{df_out['hybrid_score'].max():.4f}  "
          f"(mean {df_out['hybrid_score'].mean():.4f})")
    print(f"Distance range (m) : {df_out['distance_m'].min():.1f} – "
          f"{df_out['distance_m'].max():.1f}  "
          f"(mean {df_out['distance_m'].mean():.1f})")

    print("\nTop-5 complaint -> category pairings (by frequency):")
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


# Main

def main():
    print("=" * 62)
    print("Phase 3 — Strategy C: Hybrid Integration")
    print(f"  Radius    : {RADIUS_M} m")
    print(f"  Geo weight: {GEO_WEIGHT}  |  Category weight: {CAT_WEIGHT}")
    print("=" * 62)
    t_total = time.time()

    df_311, df_yelp       = load_data()
    sim_lookup            = build_semantic_similarity(df_311, df_yelp)
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
