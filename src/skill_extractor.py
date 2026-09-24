"""
skill_extractor.py

A practical, keyword/regex-based skill extractor. This is deliberately NOT
a machine-learned classifier — it matches against a modular vocabulary of
known technical skills and normalizes common spelling/format variations to
a single canonical name (e.g. "scikit learn" and "scikit-learn" both become
"Scikit-learn").

The vocabulary is a plain Python list/dict at the top of this file so it is
easy to extend without touching any matching logic.
"""

import re
from typing import List, Set

# Canonical skill names. Add new skills here — everything below builds off
# this list automatically.
SKILL_VOCABULARY = [
    "Python", "C", "C++", "C#", "Java", "JavaScript", "TypeScript",
    "HTML", "CSS", "React", "Angular", "Vue.js", "Node.js", "Express.js",
    "Flask", "Django", "FastAPI",
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "SQLite",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes",
    "Git", "GitHub", "GitLab", "CI/CD", "Linux",
    "Machine Learning", "Deep Learning", "Artificial Intelligence", "NLP",
    "Computer Vision", "TensorFlow", "PyTorch", "Keras", "Scikit-learn",
    "Pandas", "NumPy", "Matplotlib", "Seaborn",
    "Power BI", "Tableau", "Excel", "Data Analytics", "Data Science",
    "REST API", "GraphQL", "Microservices",
    "Computer Networks", "Operating Systems", "Data Structures", "Algorithms",
]

# Maps a lowercased, loosely-normalized variant -> canonical skill name.
# Every entry in SKILL_VOCABULARY is included automatically below; this
# dict only needs to hold the *extra* aliases/misspellings/variants.
_ALIASES = {
    "scikit learn": "Scikit-learn",
    "scikitlearn": "Scikit-learn",
    "sklearn": "Scikit-learn",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "expressjs": "Express.js",
    "express js": "Express.js",
    "vuejs": "Vue.js",
    "vue js": "Vue.js",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "artificial intelligence": "Artificial Intelligence",
    "natural language processing": "NLP",
    "computer vision": "Computer Vision",
    "rest apis": "REST API",
    "restful api": "REST API",
    "restful apis": "REST API",
    "rest api": "REST API",
    "power bi": "Power BI",
    "powerbi": "Power BI",
    "data science": "Data Science",
    "data analytics": "Data Analytics",
    "c plus plus": "C++",
    "c sharp": "C#",
    "ci cd": "CI/CD",
    "cicd": "CI/CD",
    "postgres": "PostgreSQL",
    "amazon web services": "AWS",
    "google cloud": "GCP",
    "google cloud platform": "GCP",
    "microsoft azure": "Azure",
}


def _normalize_key(text: str) -> str:
    """Lowercase and collapse whitespace/hyphens into single spaces to build
    a lookup key. Unlike a plain alphanumeric strip, this DELIBERATELY keeps
    symbols such as '+', '#', and '.' so that distinct skills like 'C',
    'C++', and 'C#' (or 'Node.js' vs a plain 'nodejs' alias) do not collide
    into the same key."""
    text = text.lower().strip()
    text = re.sub(r"[\s\-]+", " ", text)
    return text


def _build_lookup():
    lookup = {}
    for skill in SKILL_VOCABULARY:
        lookup[_normalize_key(skill)] = skill
    for alias, canonical in _ALIASES.items():
        lookup[_normalize_key(alias)] = canonical
    return lookup


_SKILL_LOOKUP = _build_lookup()

# Build a regex pattern per canonical skill/alias so multi-word and
# symbol-containing skills (C++, C#, Node.js) are matched with correct
# word boundaries instead of naive substring search.
def _build_patterns():
    patterns = []
    seen_keys = set()
    all_terms = list(SKILL_VOCABULARY) + list(_ALIASES.keys())
    for term in all_terms:
        key = _normalize_key(term)
        if key in seen_keys or not key:
            continue
        seen_keys.add(key)
        canonical = _SKILL_LOOKUP[key]
        escaped = re.escape(term)
        # Use lookaround word boundaries that also work for symbols like
        # C++ / C# / Node.js, where \b behaves inconsistently. '+' and '#'
        # are excluded from both sides too, so a bare 'C' pattern cannot
        # match as a false substring inside 'C++' or 'C#'.
        pattern = re.compile(
            r"(?<![a-zA-Z0-9\+\#])" + escaped.replace(r"\ ", r"[\s\-]+") + r"(?![a-zA-Z0-9\+\#])",
            re.IGNORECASE,
        )
        patterns.append((pattern, canonical))
    return patterns


_SKILL_PATTERNS = _build_patterns()


def extract_skills(text: str) -> List[str]:
    """
    Find every known skill mentioned in `text` and return the canonical,
    de-duplicated, sorted list of matches. Only matches skills present in
    SKILL_VOCABULARY / its aliases — this will never invent a skill that
    isn't in the vocabulary, and avoids treating random words as skills.
    """
    if not text:
        return []
    found: Set[str] = set()
    for pattern, canonical in _SKILL_PATTERNS:
        if pattern.search(text):
            found.add(canonical)
    return sorted(found)


def parse_required_skills(raw_input: str) -> List[str]:
    """
    Parse a comma-separated "Required Skills" input from the HR user into a
    normalized, canonical skill list — reusing the same vocabulary/alias
    matching as extract_skills, so "sklearn, node js" becomes
    ["Node.js", "Scikit-learn"].
    """
    if not raw_input:
        return []
    return extract_skills(raw_input)
