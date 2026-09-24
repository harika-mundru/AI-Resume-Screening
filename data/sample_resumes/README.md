# Sample Test Data (Synthetic)

The files in this folder are **synthetic, made-up resumes** created only to
test and demonstrate the screening pipeline (extraction, parsing, TF-IDF
matching, skill matching, ranking, and export). They are not real people's
resumes and are not a machine-learning training dataset — this project does
not train any model; it processes whatever resumes a user uploads at
runtime.

Each candidate has a `.txt` source (for readability) and a matching `.pdf`
(for uploading into the app):

| File | Profile |
|---|---|
| candidate_1_python_developer | Python Developer |
| candidate_2_data_analyst | Data Analyst |
| candidate_3_frontend_developer | Frontend Developer |
| candidate_4_ml_engineer | Machine Learning Engineer |
| candidate_5_backend_developer | Backend Developer |

A matching sample job description is provided at
`../sample_job_description.txt` (Python Developer role). Uploading all five
sample PDFs against this job description will produce different, calculated
(not hardcoded) match scores for each candidate, with the Python Developer
and Backend/ML profiles ranking above the Frontend and Data Analyst
profiles, since the sample job description asks for Python, Flask, Django,
SQL, REST API, Git, and Machine Learning.
