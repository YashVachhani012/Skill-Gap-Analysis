# ==============================================
# 💼 AI JOB RECOMMENDATION STREAMLIT WEB APP
# ==============================================

import streamlit as st
import re
import pdfplumber
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

# ==============================================
# PAGE CONFIG
# ==============================================

st.set_page_config(
    page_title="AI Job Recommender",
    page_icon="💼",
    layout="wide"
)

# ==============================================
# LOAD MODEL & FILES
# ==============================================

@st.cache_resource
def load_all_models():

    try:
        model = tf.keras.models.load_model(
            "job_recommendation_model.h5"
        )
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    try:
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading vectorizer: {e}")
        st.stop()

    try:
        label_df = pd.read_csv(
            "label_mapping.csv",
            header=None
        )

        label_mapping = dict(
            zip(label_df[0], label_df[1])
        )

    except Exception as e:
        st.error(f"❌ Error loading label mapping: {e}")
        st.stop()

    try:
        job_data = pd.read_excel(
            "Jobdata.xlsx",
            engine="openpyxl"
        )

        job_data.fillna("", inplace=True)

    except Exception as e:
        st.error(f"❌ Error loading job dataset: {e}")
        st.stop()

    return model, vectorizer, label_mapping, job_data


model, vectorizer, label_mapping, job_data = load_all_models()

# ==============================================
# PDF TEXT EXTRACTION
# ==============================================

def extract_text_from_pdf(uploaded_file):

    text = ""

    try:
        with pdfplumber.open(uploaded_file) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

    except Exception as e:
        st.error(f"❌ PDF Extraction Error: {e}")

    return text


# ==============================================
# NAME EXTRACTION
# ==============================================

def extract_candidate_name(text):

    lines = text.strip().split("\n")

    for line in lines[:10]:

        line = line.strip()

        if not line:
            continue

        if any(
            word in line.lower()
            for word in [
                "resume",
                "curriculum",
                "vitae",
                "cv",
                "email",
                "phone",
                "contact"
            ]
        ):
            continue

        if (
            len(line.split()) <= 3
            and re.match(r"^[A-Za-z\s\.-]+$", line)
        ):
            return line.title()

    return "Name Not Found"


# ==============================================
# SKILL EXTRACTION
# ==============================================

def extract_skills(text):

    common_skills = [

        # Programming
        "python",
        "java",
        "c++",
        "sql",
        "html",
        "css",
        "javascript",

        # Data Analytics
        "excel",
        "power bi",
        "tableau",
        "data analysis",
        "data visualization",
        "statistics",

        # ML / AI
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "ai",
        "nlp",
        "tensorflow",
        "keras",
        "scikit-learn",

        # Libraries
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",

        # Database
        "mysql",
        "postgresql",
        "mongodb",

        # Big Data
        "spark",
        "hadoop",

        # Web
        "flask",
        "django",
        "react",

        # Cloud
        "aws",
        "azure",
        "cloud",

        # Other
        "linux",
        "communication",
        "leadership",
        "teamwork",
        "project management",
        "data science"
    ]

    text = text.lower()

    skills_found = []

    for skill in common_skills:

        if re.search(
            r"\b" + re.escape(skill) + r"\b",
            text
        ):
            skills_found.append(skill)

    return sorted(list(set(skills_found)))


# ==============================================
# FIND SKILL GAP
# ==============================================

def find_missing_skills(
    job_title,
    resume_skills
):

    try:

        row = job_data[
            job_data["Job Title"]
            .str.lower()
            == job_title.lower()
        ]

        if row.empty:
            return []

        skills_required = row.iloc[0].get(
            "Skills Required",
            ""
        )

        job_skills = [

            skill.strip().lower()

            for skill in skills_required.split(",")

            if skill.strip()
        ]

        missing = [

            skill

            for skill in job_skills

            if skill not in resume_skills
        ]

        return missing

    except Exception:
        return []


# ==============================================
# UI
# ==============================================

st.title("💼 Skill Gap Analysis & Job Recommendation")

st.markdown(
    """
Upload your **Resume (PDF)**

The system will:

✅ Extract Name  
✅ Extract Skills  
✅ Predict Best Job Role  
✅ Show Skill Gap Analysis  
✅ Recommend Top Jobs
"""
)

st.markdown("---")

uploaded_file = st.file_uploader(
    "📄 Upload Resume",
    type=["pdf"]
)

# ==============================================
# MAIN PROCESS
# ==============================================

if uploaded_file:

    with st.spinner(
        "Analyzing Resume..."
    ):

        resume_text = extract_text_from_pdf(
            uploaded_file
        )

        candidate_name = (
            extract_candidate_name(
                resume_text
            )
        )

        extracted_skills = (
            extract_skills(
                resume_text
            )
        )

    if len(extracted_skills) == 0:

        st.error(
            "⚠️ No recognizable skills found."
        )

    else:

        st.success(
            f"Skills Found: {', '.join(extracted_skills)}"
        )

        try:

            resume_input = " ".join(
                extracted_skills
            )

            X_resume = (
                vectorizer
                .transform(
                    [resume_input]
                )
                .toarray()
            )

            prediction = (
                model.predict(
                    X_resume,
                    verbose=0
                )
            )

            probs = prediction[0]

            probs = probs / np.sum(probs)

            best_index = int(
                np.argmax(probs)
            )

            predicted_job = (
                label_mapping.get(
                    best_index,
                    "Unknown Role"
                )
            )

            best_match = (
                probs[best_index]
                * 100
            )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )

            st.stop()

        # ======================================
        # SKILL GAP
        # ======================================

        missing_skills = (
            find_missing_skills(
                predicted_job,
                [
                    skill.lower()
                    for skill in extracted_skills
                ]
            )
        )

        # ======================================
        # RESULT
        # ======================================

        st.markdown("---")

        st.subheader(
            "🏆 Job Recommendation Result"
        )

        st.write(
            f"👤 **Candidate:** {candidate_name}"
        )

        st.write(
            f"🎯 **Predicted Role:** {predicted_job}"
        )

        st.write(
            f"📈 **Match Score:** {best_match:.2f}%"
        )

        # Skills

        st.markdown(
            "### 🧠 Extracted Skills"
        )

        st.success(
            ", ".join(extracted_skills)
        )

        # Missing Skills

        if missing_skills:

            st.markdown(
                "### ❌ Missing Skills"
            )

            st.warning(
                ", ".join(missing_skills)
            )

        else:

            st.markdown(
                "### ✅ Excellent Match"
            )

            st.success(
                "No major skill gaps found."
            )

        # ======================================
        # TOP JOBS
        # ======================================

        st.markdown("---")

        st.markdown(
            "### 🔝 Top 5 Recommended Jobs"
        )

        top_indices = np.argsort(
            probs
        )[::-1][:5]

        for idx in top_indices:

            role = label_mapping.get(
                int(idx),
                "Unknown Role"
            )

            score = (
                probs[idx]
                * 100
            )

            st.write(
                f"✅ **{role}** — {score:.2f}% Match"
            )

else:

    st.info(
        "📥 Upload your resume to start."
    )

# ==============================================
# FOOTER
# ==============================================

st.markdown("---")

st.caption(
    "Developed using Streamlit, TensorFlow, Scikit-Learn, Pandas & NLP"
)