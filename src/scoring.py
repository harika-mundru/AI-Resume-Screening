"""
scoring.py

Combines TF-IDF similarity and skill match into a single, transparent
Final Match Score, and ranks candidates by that score.

The weights are configurable constants (not hidden anywhere in the UI
code), matching the requirement that the scoring methodology be visible
and explainable to HR users in the About / Methodology page.
"""

from typing import List, Dict, Any

import pandas as pd

# Default weighting: TF-IDF similarity counts for 70% of the final score,
# explicit skill matching counts for 30%. Change these two numbers (they
# must sum to 1.0) to adjust how much weight skills get vs. overall text
# similarity.
DEFAULT_WEIGHTS = {
    "tfidf": 0.70,
    "skill": 0.30,
}


def compute_final_score(tfidf_score: float, skill_score: float, weights: Dict[str, float] = None) -> float:
    """
    Combine a TF-IDF similarity score and a skill match score (both in
    [0, 1]) into a single final score in [0, 1], using the configured
    weights.
    """
    weights = weights or DEFAULT_WEIGHTS
    return (weights["tfidf"] * tfidf_score) + (weights["skill"] * skill_score)


def to_percent(fraction: float) -> int:
    """Convert a [0, 1] fraction into a rounded whole-number percentage,
    e.g. 0.91 -> 91."""
    return round(max(0.0, min(1.0, fraction)) * 100)


def build_results_dataframe(candidates: List[Dict[str, Any]], weights: Dict[str, float] = None) -> pd.DataFrame:
    """
    Take a list of per-candidate result dicts (each already containing
    tfidf_score and skill_score as [0,1] fractions, plus parsed resume
    fields) and return a ranked pandas DataFrame with a Rank column,
    sorted by Final Match Score (highest first).

    Ranking is calculated purely from the Final Match Score computed
    here — nothing is hardcoded or pre-sorted by the caller.
    """
    weights = weights or DEFAULT_WEIGHTS
    rows = []
    for c in candidates:
        final = compute_final_score(c["tfidf_score"], c["skill_score"], weights)
        rows.append({
            "Candidate": c["name"],
            "Email": c["email"],
            "Phone": c["phone"],
            "Final Match Score": to_percent(final),
            "TF-IDF Similarity": to_percent(c["tfidf_score"]),
            "Skill Match": to_percent(c["skill_score"]),
            "Experience": c["experience"],
            "Education": c["education"],
            "Degree": c["degree"],
            "University": c["university"],
            "Skills": ", ".join(c["skills"]) if c["skills"] else "N/A",
            "Matched Skills": ", ".join(c["matched_skills"]) if c["matched_skills"] else "None",
            "Missing Skills": ", ".join(c["missing_skills"]) if c["missing_skills"] else "None",
            "Certifications": "; ".join(c["certifications"]) if c["certifications"] else "N/A",
            "Filename": c["filename"],
            "_final_fraction": final,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("_final_fraction", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    df = df.drop(columns=["_final_fraction"])
    return df
