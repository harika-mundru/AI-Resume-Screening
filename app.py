"""
app.py

AI-Powered Resume Screening Tool
Automated Resume Parsing & Job Matching using NLP

Main Streamlit application. This file wires together the modules in
src/ into three views (Screening Dashboard, Candidate Analysis, and
About / Methodology), selected from the sidebar.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from src.pdf_extractor import extract_text_from_pdf, PYMUPDF_AVAILABLE
from src.text_processor import get_nlp_model, get_nlp_load_error, clean_display_text
from src.resume_parser import parse_resume
from src.skill_extractor import parse_required_skills
from src.matcher import compute_tfidf_similarity, compute_skill_match
from src.scoring import build_results_dataframe, DEFAULT_WEIGHTS
from src.utils import dataframe_to_csv_bytes, dataframe_to_excel_bytes, format_warning_list

st.set_page_config(
    page_title="AI-Powered Resume Screening Tool",
    page_icon="🧾",
    layout="wide",
)


# --------------------------------------------------------------------------
# Cached resource loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_nlp_model():
    """Load the spaCy model once per app session (expensive to reload)."""
    return get_nlp_model()


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "candidates_raw" not in st.session_state:
    st.session_state.candidates_raw = None
if "screening_warnings" not in st.session_state:
    st.session_state.screening_warnings = []


def reset_session():
    st.session_state.results_df = None
    st.session_state.candidates_raw = None
    st.session_state.screening_warnings = []


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.title("🧾 Resume Screening Tool")
page = st.sidebar.radio(
    "Navigate",
    ["Screening Dashboard", "Candidate Analysis", "About / Methodology"],
    label_visibility="collapsed",
)
st.sidebar.divider()
if st.sidebar.button("🔄 Clear / Start Over", use_container_width=True):
    reset_session()
    st.rerun()

if not PYMUPDF_AVAILABLE:
    st.sidebar.warning("PyMuPDF is not installed. Run `pip install PyMuPDF` to enable PDF extraction.")
if get_nlp_model() is None and get_nlp_load_error():
    st.sidebar.info(f"spaCy note: {get_nlp_load_error()}")


# ==========================================================================
# PAGE 1: SCREENING DASHBOARD
# ==========================================================================
if page == "Screening Dashboard":
    st.title("AI-Powered Resume Screening Tool")
    st.caption("Screen multiple resumes against a job description using NLP, TF-IDF and cosine similarity.")

    # ---- Job description section ----
    st.subheader("Job Description")
    col1, col2 = st.columns([1, 1])
    with col1:
        job_title = st.text_input("Job Title", placeholder="e.g. Python Developer")
    with col2:
        required_skills_input = st.text_input(
            "Required Skills (comma-separated, optional)",
            placeholder="Python, Flask, Django, SQL, REST API, Git, Machine Learning",
        )
    job_description = st.text_area(
        "Job Description",
        height=160,
        placeholder="Paste or type the full job description here...",
    )

    st.divider()

    # ---- Resume upload section ----
    st.subheader("Upload Candidate Resumes")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} resume(s) uploaded**")
        with st.expander("View uploaded filenames"):
            for f in uploaded_files:
                st.write(f"- {f.name}")

    st.divider()
    screen_clicked = st.button("🔍 Screen Resumes", type="primary", use_container_width=True)

    if screen_clicked:
        if not job_description.strip():
            st.error("Please enter a job description before screening resumes.")
        elif not uploaded_files:
            st.error("Please upload at least one resume (PDF) before screening.")
        else:
            required_skills = parse_required_skills(required_skills_input)
            full_job_text = f"{job_title}\n{job_description}\n{required_skills_input}"

            nlp = load_nlp_model()
            warnings = []
            candidates = []
            resume_texts = []

            progress = st.progress(0, text="Extracting resumes...")
            for i, uploaded_file in enumerate(uploaded_files):
                file_bytes = uploaded_file.read()
                extraction = extract_text_from_pdf(file_bytes, uploaded_file.name)
                if not extraction.success:
                    warnings.append(extraction.warning)
                    progress.progress((i + 1) / len(uploaded_files))
                    continue
                candidates.append((uploaded_file.name, extraction.text))
                resume_texts.append(extraction.text)
                progress.progress((i + 1) / len(uploaded_files), text="Extracting resumes...")

            if not candidates:
                progress.empty()
                st.error("None of the uploaded files could be processed. Please check the warnings below.")
                if warnings:
                    st.warning(format_warning_list(warnings))
            else:
                progress.progress(0.4, text="Parsing candidate information...")
                parsed_candidates = []
                for filename, text in candidates:
                    try:
                        parsed_candidates.append(parse_resume(text, filename, nlp))
                    except Exception as exc:
                        warnings.append(f"Could not fully parse '{filename}' ({exc}). Some fields may show N/A.")

                progress.progress(0.7, text="Calculating similarity...")
                tfidf_scores = compute_tfidf_similarity(full_job_text, [c.raw_text for c in parsed_candidates])

                progress.progress(0.9, text="Ranking candidates...")
                rows = []
                for cand, tfidf in zip(parsed_candidates, tfidf_scores):
                    matched, missing, skill_frac = compute_skill_match(required_skills, cand.skills)
                    rows.append({
                        "name": cand.name, "email": cand.email, "phone": cand.phone,
                        "tfidf_score": tfidf, "skill_score": skill_frac,
                        "experience": cand.experience, "education": cand.education,
                        "degree": cand.degree, "university": cand.university,
                        "skills": cand.skills, "matched_skills": matched, "missing_skills": missing,
                        "certifications": cand.certifications, "filename": cand.filename,
                    })

                df = build_results_dataframe(rows)
                progress.progress(1.0, text="Done.")
                progress.empty()

                st.session_state.results_df = df
                st.session_state.candidates_raw = {c.filename: c for c in parsed_candidates}
                st.session_state.screening_warnings = warnings

                st.success(f"Screening complete. {len(df)} resume(s) processed successfully.")

    # ---- Results ----
    if st.session_state.screening_warnings:
        st.warning(format_warning_list(st.session_state.screening_warnings))

    df = st.session_state.results_df
    if df is not None and not df.empty:
        st.divider()
        st.subheader("Dashboard Metrics")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Candidates", len(df))
        m2.metric("Average Match Score", f"{df['Final Match Score'].mean():.0f}%")
        m3.metric("Highest Match Score", f"{df['Final Match Score'].max():.0f}%")

        threshold = st.slider("Shortlisting Threshold (%)", 0, 100, 70, key="threshold_slider")
        above_threshold = (df["Final Match Score"] >= threshold).sum()
        m4.metric("Candidates Meeting Threshold", int(above_threshold))
        st.caption(f"Candidates meeting the selected score threshold ({threshold}%).")

        st.divider()
        st.subheader("Screening Results")

        # ---- Filters ----
        with st.expander("Filters"):
            f1, f2, f3 = st.columns(3)
            with f1:
                min_score_filter = st.slider("Minimum Match Score", 0, 100, 0)
            with f2:
                skill_filter = st.text_input("Filter by Skill (optional)", placeholder="e.g. Python")
            with f3:
                education_filter = st.text_input("Filter by Education keyword (optional)", placeholder="e.g. B.Tech")

        filtered_df = df[df["Final Match Score"] >= min_score_filter]
        if skill_filter.strip():
            filtered_df = filtered_df[filtered_df["Skills"].str.contains(skill_filter.strip(), case=False, na=False)]
        if education_filter.strip():
            filtered_df = filtered_df[filtered_df["Education"].str.contains(education_filter.strip(), case=False, na=False)]

        display_cols = ["Rank", "Candidate", "Email", "Final Match Score", "Skill Match", "Experience", "Education"]
        st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)

        st.caption("Ranked by matching score. This reflects text and skill similarity to the job description, not a hiring recommendation.")

        # ---- Export ----
        st.divider()
        st.subheader("Export Results")
        e1, e2 = st.columns(2)
        with e1:
            st.download_button(
                "⬇️ Download CSV",
                data=dataframe_to_csv_bytes(df),
                file_name="resume_screening_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with e2:
            st.download_button(
                "⬇️ Download Excel",
                data=dataframe_to_excel_bytes(df),
                file_name="resume_screening_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    elif df is None:
        st.info("Enter a job description, upload resumes, and click **Screen Resumes** to get started.")


# ==========================================================================
# PAGE 2: CANDIDATE ANALYSIS
# ==========================================================================
elif page == "Candidate Analysis":
    st.title("Candidate Analysis")

    df = st.session_state.results_df
    if df is None or df.empty:
        st.info("No screening results yet. Go to **Screening Dashboard**, run a screening, then come back here.")
    else:
        candidate_names = df["Candidate"].tolist()
        selected_name = st.selectbox("Select a candidate", candidate_names)
        row = df[df["Candidate"] == selected_name].iloc[0]

        st.divider()
        st.subheader("Candidate Profile")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Name:** {row['Candidate']}")
            st.write(f"**Email:** {row['Email']}")
            st.write(f"**Phone:** {row['Phone']}")
            st.write(f"**Education:** {row['Education']}")
        with c2:
            st.write(f"**Degree:** {row['Degree']}")
            st.write(f"**University:** {row['University']}")
            st.write(f"**Experience:** {row['Experience']}")
            st.write(f"**Certifications:** {row['Certifications']}")

        st.write(f"**Skills:** {row['Skills']}")

        st.divider()
        st.subheader("Matching Analysis")
        s1, s2, s3 = st.columns(3)
        s1.metric("Overall Match", f"{row['Final Match Score']}%")
        s2.metric("TF-IDF Similarity", f"{row['TF-IDF Similarity']}%")
        s3.metric("Skill Match", f"{row['Skill Match']}%")

        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**Matched Skills**")
            if row["Matched Skills"] != "None":
                for s in row["Matched Skills"].split(", "):
                    st.write(f"✓ {s}")
            else:
                st.write("None")
        with m2:
            st.markdown("**Missing Skills**")
            if row["Missing Skills"] != "None":
                for s in row["Missing Skills"].split(", "):
                    st.write(f"• {s}")
            else:
                st.write("None")

        st.divider()
        st.subheader("Explanation")
        st.write(
            "The candidate's resume contains several terms and skills that overlap with the "
            "provided job description. The Overall Match score above is a text-based similarity "
            "and skill-overlap score, calculated as a weighted combination of TF-IDF similarity "
            f"({DEFAULT_WEIGHTS['tfidf']*100:.0f}%) and explicit skill matching ({DEFAULT_WEIGHTS['skill']*100:.0f}%). "
            "It is not a guarantee of candidate suitability and should be used as a screening aid, "
            "not a final hiring decision."
        )

        candidates_raw = st.session_state.candidates_raw or {}
        raw_candidate = candidates_raw.get(row["Filename"])
        if raw_candidate is not None:
            with st.expander("View extracted resume text"):
                st.text(clean_display_text(raw_candidate.raw_text)[:3000])


# ==========================================================================
# PAGE 3: ABOUT / METHODOLOGY
# ==========================================================================
else:
    st.title("About / Methodology")
    st.write(
        "This tool helps HR professionals and recruiters do an initial screening pass over a "
        "batch of resumes against a job description. It is a screening and ranking aid, "
        "**not** an autonomous hiring system, and it does not claim that any candidate is "
        "objectively suitable for employment — it only reports transparent, text-based "
        "matching scores."
    )

    st.subheader("How It Works")
    steps = [
        "Resume Upload — HR uploads one or more PDF resumes.",
        "PDF Text Extraction — text is extracted from each PDF using PyMuPDF.",
        "NLP Processing — spaCy is used for sentence handling and named-entity recognition (for candidate name extraction).",
        "Resume Information Extraction — name, email, phone, skills, education, experience and certifications are parsed out.",
        "TF-IDF Vectorization — the job description and every resume are converted into TF-IDF vectors.",
        "Cosine Similarity — each resume's vector is compared against the job description's vector.",
        "Skill Matching — the required skills list is compared against each candidate's extracted skills.",
        "Final Score Calculation — TF-IDF similarity and skill match are combined into one weighted score.",
        "Candidate Ranking — candidates are sorted by final score, highest first.",
        "Export — results can be downloaded as CSV or a formatted Excel file.",
    ]
    for i, s in enumerate(steps, start=1):
        st.write(f"**{i}.** {s}")

    st.subheader("Process Flow")
    st.code(
        "Resume PDFs\n"
        "     |\n"
        "PDF Text Extraction\n"
        "     |\n"
        "NLP Processing\n"
        "     |\n"
        "Resume Information Extraction\n"
        "     |\n"
        "TF-IDF Vectorization\n"
        "     |\n"
        "Cosine Similarity\n"
        "     |\n"
        "Skill Matching\n"
        "     |\n"
        "Final Score\n"
        "     |\n"
        "Candidate Ranking\n"
        "     |\n"
        "HR Dashboard",
        language=None,
    )

    st.subheader("Scoring Methodology")
    st.write(
        f"**Final Score = {DEFAULT_WEIGHTS['tfidf']:.2f} × TF-IDF Similarity + {DEFAULT_WEIGHTS['skill']:.2f} × Skill Match**"
    )
    st.write(
        "TF-IDF similarity measures how textually similar a resume is to the overall job "
        "description. Skill match measures what fraction of the explicitly required skills "
        "were found in the resume. Both weights are configurable in `src/scoring.py`."
    )

    st.subheader("Limitations")
    st.write(
        "- PDF extraction quality depends on how the original PDF was created.\n"
        "- Scanned / image-only PDFs are not supported without OCR.\n"
        "- TF-IDF measures textual similarity, not true semantic understanding.\n"
        "- Skill extraction depends on the configured skill vocabulary.\n"
        "- Resume formatting can affect extraction accuracy.\n"
        "- Matching scores are a screening aid and should not be treated as a final hiring decision."
    )
