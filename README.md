# AI-Powered Resume Screening Tool

Automated Resume Parsing & Job Matching using NLP

## Overview

This is a Streamlit web application that helps HR professionals and recruiters
do an initial screening pass over a batch of resumes against a job
description. HR enters (or pastes) a job description, uploads multiple PDF
resumes, and the tool extracts each candidate's information, compares it
against the job description using TF-IDF + cosine similarity plus explicit
skill matching, and produces a ranked, exportable results table.

**This is a resume screening and ranking aid — not an autonomous hiring
system.** It does not claim that a candidate is objectively suitable for a
role; it only reports a transparent, text-based matching score.

## Problem Statement

Manually screening a stack of resumes against a job description is slow and
inconsistent — a recruiter has to read every resume and mentally compare it
against the requirements. This tool automates the repetitive first pass:
extracting candidate information and computing a consistent, explainable
match score for every resume, so a recruiter can focus their time on the
most relevant candidates.

## Objectives

- Extract text from PDF resumes (single or multi-page, various layouts).
- Parse candidate name, email, phone, skills, education, and experience.
- Convert the job description and every resume into TF-IDF vectors.
- Calculate cosine similarity between the job description and each resume.
- Explicitly compare required skills against each candidate's skills.
- Combine both into one transparent, weighted final score.
- Rank candidates by that score.
- Provide a dashboard for filtering, inspecting, and exporting results.

## Features

- PDF resume upload (multiple files at once)
- Text extraction with PyMuPDF, with graceful handling of unreadable/empty PDFs
- Candidate info extraction: name, email, phone, skills, education, degree,
  university, experience, job titles, certifications
- TF-IDF + cosine similarity text matching (scikit-learn)
- Explicit skill matching with a modular, expandable skill vocabulary
- Transparent, configurable final scoring (TF-IDF weight + skill weight)
- Candidate ranking (highest matching score first)
- Screening dashboard with metrics (total candidates, average/highest score,
  candidates meeting a configurable threshold)
- Filtering by score, skill, and education keyword
- Candidate detail view with matched/missing skills and an explanation
- CSV export and formatted Excel export
- About / Methodology page explaining the full pipeline and scoring formula

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Web UI | Streamlit |
| Data processing | Pandas, NumPy |
| NLP | spaCy |
| Text matching | scikit-learn (TF-IDF + cosine similarity) |
| PDF processing | PyMuPDF (fitz) |
| Excel export | openpyxl |

No OpenAI, Claude, Gemini, or any other paid/cloud AI API is used or
required. The entire application runs locally, and resume content is never
sent to an external service.

## System Architecture

```
Resume PDFs
     |
PDF Text Extraction  (src/pdf_extractor.py)
     |
NLP Processing        (src/text_processor.py)
     |
Resume Info Extraction (src/resume_parser.py, src/skill_extractor.py)
     |
TF-IDF Vectorization   (src/matcher.py)
     |
Cosine Similarity      (src/matcher.py)
     |
Skill Matching         (src/matcher.py)
     |
Final Score            (src/scoring.py)
     |
Candidate Ranking      (src/scoring.py)
     |
HR Dashboard           (app.py)
```

## How It Works

1. **Resume Upload** — HR uploads one or more PDF resumes through the dashboard.
2. **PDF Text Extraction** — each PDF is opened with PyMuPDF and its text is
   extracted page by page. Unreadable or empty PDFs are skipped with a
   warning instead of crashing the whole run.
3. **NLP Processing** — spaCy is used for sentence handling and named-entity
   recognition (to help identify the candidate's name), while a separate,
   lighter cleaning step (lowercase + whitespace normalization) is used
   before TF-IDF vectorization so technical terms like `Node.js` or `C++`
   are not damaged by lemmatization.
4. **Resume Information Extraction** — regex-based extractors pull out
   email, phone, education/degree/university, experience, job titles, and
   certifications. A modular skill vocabulary (`src/skill_extractor.py`)
   is used to detect known technical skills and normalize variants (e.g.
   `scikit learn` / `scikit-learn` → `Scikit-learn`).
5. **TF-IDF Vectorization & Cosine Similarity** — the job description and
   every resume are vectorized together with `TfidfVectorizer`, and each
   resume's cosine similarity to the job description is calculated with
   `cosine_similarity`.
6. **Skill Matching** — the HR-provided "Required Skills" list is compared
   against each candidate's extracted skills to produce a skill match
   percentage, plus explicit matched/missing skill lists.
7. **Final Score & Ranking** — see below.
8. **Export** — results can be downloaded as CSV or a formatted Excel file.

## TF-IDF

TF-IDF (Term Frequency–Inverse Document Frequency) represents each document
(the job description, or a resume) as a vector where common, uninformative
words are weighted down and distinctive words are weighted up. This project
uses `sklearn.feature_extraction.text.TfidfVectorizer` to build these
vectors for the job description and all uploaded resumes together.

## Cosine Similarity

Cosine similarity measures how close two vectors point in the same
direction, regardless of their length — a value from 0 (completely
different) to 1 (identical direction). This project uses
`sklearn.metrics.pairwise.cosine_similarity` to compare each resume's
TF-IDF vector against the job description's vector, and converts the result
into a percentage for display.

## Skill Matching

In addition to the overall TF-IDF similarity, the required skills entered
by HR are compared directly against each candidate's extracted skills.

```
Skill Match % = (Number of Required Skills Found in Resume / Total Required Skills) × 100
```

If no required skills are entered, skill match is treated as 100% (not
applicable) so it doesn't unfairly reduce the final score.

## Scoring Methodology

```
Final Score = 0.70 × TF-IDF Similarity + 0.30 × Skill Match
```

Both weights are plain constants in `src/scoring.py` (`DEFAULT_WEIGHTS`)
and can be changed there. The same formula is shown to the user on the
About / Methodology page, so the scoring is never a hidden calculation.

## Project Structure

```
AI-Resume-Screening/
│
├── app.py                     # Streamlit application (UI + orchestration)
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── pdf_extractor.py       # PDF text extraction (PyMuPDF)
│   ├── resume_parser.py       # Name/email/phone/education/experience extraction
│   ├── text_processor.py      # spaCy loading + text cleaning
│   ├── skill_extractor.py     # Skill vocabulary, normalization, matching
│   ├── matcher.py             # TF-IDF + cosine similarity, skill matching
│   ├── scoring.py             # Final score + candidate ranking
│   └── utils.py                # CSV / Excel export helpers
│
├── data/
│   └── sample_resumes/        # Synthetic sample resumes + sample job description
│
├── exports/                   # Default local folder for exported files (empty in repo)
│
└── assets/
    └── logo.png
```

## Installation

```bash
git clone <this-repository-url>
cd AI-Resume-Screening
```

## Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
```

## Running the Application

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Example Usage

1. Enter a job title (e.g. "Python Developer") and paste a job description.
2. Optionally enter required skills, comma-separated (e.g.
   `Python, Flask, Django, SQL, REST API, Git, Machine Learning`).
3. Upload one or more PDF resumes.
4. Click **Screen Resumes**.
5. Review the ranked results table and dashboard metrics.
6. Open **Candidate Analysis** to inspect an individual candidate's matched
   and missing skills.
7. Export the results as CSV or Excel from the dashboard.

Sample synthetic resumes and a sample job description are provided under
`data/sample_resumes/` for trying the tool out end-to-end — see
`data/sample_resumes/README.md` for details.

## Exporting Results

Both export options include: rank, candidate name, email, phone, skills,
education, experience, matched skills, missing skills, TF-IDF score, skill
match score, and final match score. The Excel export additionally applies
bold headers, auto-sized columns, and percentage formatting on the score
columns.

## Privacy

- No resume content is sent to any external API — all processing (PDF
  extraction, NLP, TF-IDF, matching) runs locally.
- No cloud LLM (OpenAI, Claude, Gemini, etc.) is used anywhere in this
  project.
- Uploaded resumes are processed in memory for the current session and are
  not written to disk or logged.
- No personal information is hardcoded anywhere in the source code.

## Limitations

- PDF extraction quality depends on how the original PDF was produced;
  scanned/image-only PDFs are not supported without adding OCR.
- TF-IDF measures textual similarity, not true semantic/contextual
  understanding.
- Skill extraction depends on the configured skill vocabulary in
  `src/skill_extractor.py`.
- Resume formatting/layout can affect extraction accuracy (e.g. unusual
  section headings or heavily graphical resumes).
- Matching scores are a screening aid and must not be treated as a final
  or sole hiring decision.

## Future Scope

- OCR support for scanned resumes
- Semantic/embedding-based matching in addition to TF-IDF
- Transformer-based NLP for deeper parsing
- A larger, ontology-based skill vocabulary
- Multilingual resume support
- More experience-aware and education-aware scoring
- ATS compatibility analysis
- Recruiter authentication and multi-user support
- Database integration for persisting screening runs
- Side-by-side candidate comparison view
- Cloud deployment
- Resume anonymization for bias-aware screening
- Basic bias/fairness monitoring on scoring outcomes

## Author

[HARIKA MUNDRU]
