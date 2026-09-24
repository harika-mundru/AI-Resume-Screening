"""
matcher.py

Implements the two building blocks of the matching methodology:

1. TF-IDF + Cosine Similarity — measures how textually similar each
   resume is to the job description as a whole.
2. Skill Matching — compares the explicit "Required Skills" list against
   each candidate's extracted skills.

Both are combined into a final score in scoring.py.
"""

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.text_processor import clean_for_tfidf


def compute_tfidf_similarity(job_text: str, resume_texts: List[str]) -> List[float]:
    """
    Vectorize the job description and every resume together using
    TF-IDF, then compute the cosine similarity of each resume against the
    job description.

    Returns a list of similarity scores in [0, 1], one per resume, in the
    same order as resume_texts. These are calculated directly from the
    actual text supplied — nothing here is randomly generated.
    """
    if not resume_texts:
        return []

    job_clean = clean_for_tfidf(job_text)
    resumes_clean = [clean_for_tfidf(t) for t in resume_texts]

    corpus = [job_clean] + resumes_clean

    # min_df=1 keeps this robust for small batches (a handful of resumes);
    # stop_words='english' removes common filler words that add noise to
    # the similarity comparison without carrying real signal.
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)

    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # Happens if, after cleaning, every document is empty (e.g. the
        # job description and all resumes had no extractable text at all).
        return [0.0] * len(resume_texts)

    job_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(job_vector, resume_vectors)[0]
    return [float(s) for s in similarities]


def compute_skill_match(required_skills: List[str], candidate_skills: List[str]) -> Tuple[List[str], List[str], float]:
    """
    Compare a job's required skills against a candidate's extracted
    skills.

    Returns (matched_skills, missing_skills, skill_match_fraction) where
    skill_match_fraction is in [0, 1]. If no required skills were
    supplied, the skill match is treated as 100% (not applicable / no
    skill requirement was given by HR), so it doesn't unfairly drag down
    the final score.
    """
    if not required_skills:
        return [], [], 1.0

    required_set = {s.lower(): s for s in required_skills}
    candidate_set = {s.lower() for s in candidate_skills}

    matched = [orig for key, orig in required_set.items() if key in candidate_set]
    missing = [orig for key, orig in required_set.items() if key not in candidate_set]

    fraction = len(matched) / len(required_set)
    return sorted(matched), sorted(missing), fraction
