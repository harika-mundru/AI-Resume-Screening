"""
resume_parser.py

Extracts structured candidate information out of raw resume text:
name, email, phone, skills, education, degree, university, experience,
job titles, and certifications.

Every extractor here is regex/heuristic-based and returns "N/A" when it
cannot confidently find a value, rather than raising an exception or
guessing. This module never crashes the screening run for one resume —
worst case, some fields come back as "N/A" and the resume still gets
scored on whatever was found.
"""

import re
from dataclasses import dataclass, field
from typing import List

from src.skill_extractor import extract_skills
from src.text_processor import extract_person_entities

NA = "N/A"

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Supports formats like: +91 9876543210, 9876543210, +1 234 567 8900,
# (123) 456-7890, 123-456-7890
PHONE_RE = re.compile(
    r"(?:\+\d{1,3}[\s\-]?)?(?:\(\d{2,4}\)[\s\-]?)?\d{3,5}[\s\-]?\d{3,4}[\s\-]?\d{0,4}"
)

DEGREE_KEYWORDS = [
    "B.Tech", "B\\.E\\.", "BE ", "M.Tech", "M\\.E\\.", "MBA", "BCA", "MCA",
    "B.Sc", "M.Sc", "Ph.D", "PhD", "Bachelor of", "Master of", "Bachelor's",
    "Master's", "B.Com", "M.Com", "Diploma",
]
DEGREE_RE = re.compile("(" + "|".join(DEGREE_KEYWORDS) + ")", re.IGNORECASE)

EDU_LINE_HINTS_RE = re.compile(r"(University|College|Institute|Polytechnic)", re.IGNORECASE)

EXPERIENCE_YEARS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*years?", re.IGNORECASE
)

JOB_TITLE_KEYWORDS = [
    "Software Engineer", "Software Developer", "Developer", "Data Analyst",
    "Data Scientist", "Machine Learning Engineer", "ML Engineer",
    "Backend Developer", "Frontend Developer", "Full Stack Developer",
    "Web Developer", "Intern", "Analyst", "Consultant", "Engineer", "Manager",
]
JOB_TITLE_RE = re.compile("(" + "|".join(re.escape(t) for t in JOB_TITLE_KEYWORDS) + ")", re.IGNORECASE)

CERT_LINE_RE = re.compile(r"(certified|certificate|certification)", re.IGNORECASE)

# Lines that look like a resume section header, not a candidate name.
NAME_BLOCKLIST_RE = re.compile(
    r"^(resume|curriculum vitae|cv|profile|objective|summary|contact|address)\b",
    re.IGNORECASE,
)


@dataclass
class Candidate:
    filename: str
    name: str = NA
    email: str = NA
    phone: str = NA
    skills: List[str] = field(default_factory=list)
    education: str = NA
    degree: str = NA
    university: str = NA
    experience: str = NA
    job_titles: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    raw_text: str = ""


def extract_email(text: str) -> str:
    match = EMAIL_RE.search(text)
    return match.group(0) if match else NA


def extract_phone(text: str) -> str:
    for match in PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        # A real phone number has at least 10 digits; this filters out
        # incidental short numbers (e.g. a graduation year or a PIN code).
        if 10 <= len(digits) <= 13:
            return match.group(0).strip()
    return NA


def extract_name(text: str, filename: str, nlp=None) -> str:
    """
    Try to identify the candidate's name using, in order:
    1. spaCy PERSON entities found in the first few lines (most reliable
       when the model is available).
    2. The first meaningful line of the resume that isn't an email, phone
       number, address, or a known section heading.
    3. The PDF filename, as a last-resort fallback identifier.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    header_block = "\n".join(lines[:8])

    if nlp is not None:
        persons = extract_person_entities(header_block, nlp)
        if persons:
            return persons[0]

    for line in lines[:8]:
        if EMAIL_RE.search(line) or PHONE_RE.search(line):
            continue
        if NAME_BLOCKLIST_RE.match(line):
            continue
        if len(line) > 60 or len(line) < 2:
            continue
        # A plausible name line: mostly letters/spaces, 1-4 words.
        words = line.split()
        if 1 <= len(words) <= 4 and all(re.match(r"^[A-Za-z.\-']+$", w) for w in words):
            return line.title()

    # Fallback: use the filename (without extension) as the identifier.
    stem = re.sub(r"\.(pdf|docx?)$", "", filename, flags=re.IGNORECASE)
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    return stem if stem else NA


def extract_education(text: str) -> str:
    match = DEGREE_RE.search(text)
    if match:
        # Return the sentence/line containing the degree keyword for context.
        for line in text.splitlines():
            if match.group(0).lower() in line.lower():
                return line.strip()[:120]
        return match.group(0)
    return NA


def extract_degree(text: str) -> str:
    match = DEGREE_RE.search(text)
    return match.group(0) if match else NA


def extract_university(text: str) -> str:
    for line in text.splitlines():
        if EDU_LINE_HINTS_RE.search(line):
            return line.strip()[:120]
    return NA


def extract_experience(text: str) -> str:
    matches = EXPERIENCE_YEARS_RE.findall(text)
    if matches:
        years = [float(m) for m in matches]
        return f"{max(years):g} years"
    return NA


def extract_job_titles(text: str) -> List[str]:
    titles = set()
    for match in JOB_TITLE_RE.finditer(text):
        titles.add(match.group(0).title())
    return sorted(titles)


def extract_certifications(text: str) -> List[str]:
    certs = []
    for line in text.splitlines():
        if CERT_LINE_RE.search(line):
            cleaned = line.strip()[:150]
            if cleaned:
                certs.append(cleaned)
    return certs


def parse_resume(text: str, filename: str, nlp=None) -> Candidate:
    """Run every field extractor on one resume's raw text and return a
    populated Candidate object."""
    return Candidate(
        filename=filename,
        name=extract_name(text, filename, nlp),
        email=extract_email(text),
        phone=extract_phone(text),
        skills=extract_skills(text),
        education=extract_education(text),
        degree=extract_degree(text),
        university=extract_university(text),
        experience=extract_experience(text),
        job_titles=extract_job_titles(text),
        certifications=extract_certifications(text),
        raw_text=text,
    )
