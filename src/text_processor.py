"""
text_processor.py

Handles text cleaning and spaCy-based NLP processing.

Two separate cleaning paths are used on purpose:

1. clean_for_tfidf() — light, non-destructive cleaning used before TF-IDF
   vectorization. It lowercases and normalizes whitespace but deliberately
   avoids aggressive lemmatization, because lemmatizing technical terms
   (e.g. splitting "Node.js" or mangling "AWS") would hurt matching
   accuracy rather than help it. Symbols that matter in tech terms
   (+, #, ., -) are preserved.

2. spaCy-based processing (get_nlp_model, get_doc, extract_sentences,
   extract_person_entities) — used specifically where spaCy adds real
   value: sentence segmentation, named entity recognition for candidate
   name extraction, and tokenization for keyword-level skill matching.
   This keeps spaCy meaningfully involved without letting it damage the
   TF-IDF text representation.
"""

import re
from typing import List, Optional

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

SPACY_MODEL_NAME = "en_core_web_sm"

_WHITESPACE_RE = re.compile(r"\s+")
# Keep letters, numbers, and the symbols that matter inside tech terms.
_ALLOWED_CHARS_RE = re.compile(r"[^a-zA-Z0-9\+\#\.\-\@\s]")


def clean_for_tfidf(text: str) -> str:
    """Light cleaning used before TF-IDF vectorization (see module docstring)."""
    if not text:
        return ""
    text = text.lower()
    text = _ALLOWED_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_display_text(text: str) -> str:
    """Basic whitespace cleanup for text that will still be shown to the user."""
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


_nlp_model = None
_nlp_load_attempted = False
_nlp_load_error = None


def get_nlp_model():
    """
    Load and cache the spaCy English model. Returns None if spaCy or the
    model is not installed, so callers can fall back to regex-only logic
    instead of crashing. app.py wraps this with st.cache_resource so the
    (relatively slow) model load only happens once per app session.
    """
    global _nlp_model, _nlp_load_attempted, _nlp_load_error

    if _nlp_load_attempted:
        return _nlp_model

    _nlp_load_attempted = True

    if not SPACY_AVAILABLE:
        _nlp_load_error = (
            "spaCy is not installed. Run 'pip install spacy' and "
            f"'python -m spacy download {SPACY_MODEL_NAME}'."
        )
        return None

    try:
        _nlp_model = spacy.load(SPACY_MODEL_NAME)
    except OSError:
        _nlp_load_error = (
            f"The spaCy model '{SPACY_MODEL_NAME}' is not installed. Run: "
            f"python -m spacy download {SPACY_MODEL_NAME}"
        )
        _nlp_model = None

    return _nlp_model


def get_nlp_load_error() -> Optional[str]:
    """Human-readable reason the spaCy model isn't available, if any."""
    return _nlp_load_error


def extract_person_entities(text: str, nlp=None) -> List[str]:
    """Return PERSON entities spaCy finds in the given text (usually just the
    first ~10 lines of a resume are passed in, to keep this fast and
    relevant to the candidate's own name rather than names mentioned
    elsewhere in the document)."""
    if nlp is None:
        return []
    try:
        doc = nlp(text)
    except Exception:
        return []
    return [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON" and ent.text.strip()]


def tokenize_lemmatize(text: str, nlp=None) -> List[str]:
    """Tokenize and lemmatize text using spaCy, skipping stopwords and
    punctuation. Used for auxiliary keyword-style comparisons (not for the
    main TF-IDF vectors, which use clean_for_tfidf instead)."""
    if nlp is None or not text:
        return []
    try:
        doc = nlp(text)
    except Exception:
        return []
    tokens = [
        t.lemma_.lower() for t in doc
        if not t.is_stop and not t.is_punct and not t.is_space and len(t.text) > 1
    ]
    return tokens
