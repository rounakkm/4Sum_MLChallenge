"""Candidate generation (blocking) stage.

Uses character n-gram TF-IDF + nearest neighbors as a strong, dependency-light
baseline. For larger datasets, swap NearestNeighbors for an ANN index (e.g.
faiss) -- the interface below stays the same.

Deliberately does NOT hard-filter by country: the problem statement notes the
test set includes France, which never appears in training, so any pipeline
that assumes {US, India} will silently drop those entities.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .normalize import normalize_name, normalize_address


def _build_text(df):
    return df["business_name"].map(normalize_name) + " " + df["business_address"].map(normalize_address)


def generate_candidates(s1_df, other_df, top_k=20):
    """Return dict: source1_entity_id -> list of candidate entity_ids from other_df."""
    if len(other_df) == 0:
        return {eid: [] for eid in s1_df["entity_id"]}

    text1 = _build_text(s1_df).tolist()
    text2 = _build_text(other_df).tolist()

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    vectorizer.fit(text1 + text2)

    X1 = vectorizer.transform(text1)
    X2 = vectorizer.transform(text2)

    k = min(top_k, len(other_df))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X2)
    _, indices = nn.kneighbors(X1)

    other_ids = other_df["entity_id"].tolist()
    candidates = {}
    for i, eid in enumerate(s1_df["entity_id"]):
        candidates[eid] = [other_ids[j] for j in indices[i]]
    return candidates


def generate_all_candidates(s1_df, s2_df, s3_df, top_k=20):
    """Combine candidates from Source 2 and Source 3 for every Source 1 entity."""
    cand2 = generate_candidates(s1_df, s2_df, top_k=top_k)
    cand3 = generate_candidates(s1_df, s3_df, top_k=top_k)
    combined = {}
    for eid in s1_df["entity_id"]:
        combined[eid] = cand2.get(eid, []) + cand3.get(eid, [])
    return combined
