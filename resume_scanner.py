"""
Resume Scanner & Ranker
========================
Scans resumes (plain text or .txt files), matches them against a job description,
extracts key information, scores and ranks candidates.

Features:
  - Resume parsing  : extracts name, email, phone, skills, education, experience
  - JD matching     : TF-IDF cosine similarity between resume and job description
  - Skill gap       : highlights required skills present / missing per resume
  - Scoring engine  : weighted score across skill match, experience, education
  - Ranking output  : sorted candidate shortlist with detailed breakdown
  - Visualization   : score distribution and skill coverage charts

Requirements:
    pip install scikit-learn pandas matplotlib nltk
    python -m nltk.downloader stopwords punkt
"""

import re
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data silently
for pkg in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

STOP_WORDS = set(stopwords.words("english"))


# ══════════════════════════════════════════════════════════════
# 1.  SAMPLE DATA  (replace with real .txt file paths or strings)
# ══════════════════════════════════════════════════════════════

JOB_DESCRIPTION = """
We are looking for a Senior Data Scientist with strong Python skills.
The ideal candidate should have experience in machine learning, deep learning,
and natural language processing. Proficiency in SQL, TensorFlow, PyTorch,
scikit-learn, and data visualization tools like Tableau or Power BI is required.
A Bachelor's or Master's degree in Computer Science, Statistics, or a related field
is expected. Experience with cloud platforms (AWS, GCP, Azure) and MLOps pipelines
is a strong plus. Excellent communication and teamwork skills are essential.
Minimum 3 years of relevant industry experience required.
"""

SAMPLE_RESUMES = {
    "Alice Johnson": """
        Alice Johnson | alice@email.com | +91-9876543210
        Senior Data Scientist with 6 years of experience.
        Education: M.Sc. Computer Science, IIT Delhi
        Skills: Python, Machine Learning, Deep Learning, NLP, TensorFlow, PyTorch,
                scikit-learn, SQL, AWS, Docker, Tableau, Git
        Experience:
        - Data Scientist at TechCorp (4 years): Built NLP pipelines, deployed ML models on AWS.
        - Junior Data Analyst at StartupX (2 years): SQL reporting, data visualisation.
        Certifications: AWS Certified ML Specialty
    """,
    "Bob Smith": """
        Bob Smith | bob@email.com | +91-9123456789
        Data Analyst with 2 years of experience.
        Education: B.Sc. Statistics, Mumbai University
        Skills: Python, SQL, Excel, Power BI, pandas, numpy
        Experience:
        - Data Analyst at FinanceCo (2 years): Dashboard creation, SQL queries, Excel reports.
        No machine learning or deep learning experience.
    """,
    "Carol White": """
        Carol White | carol@email.com | +91-9988776655
        Machine Learning Engineer with 4 years of experience.
        Education: B.Tech Computer Science, NIT Trichy
        Skills: Python, scikit-learn, TensorFlow, SQL, GCP, MLOps, Docker, Kubernetes, Git
        Experience:
        - ML Engineer at AILabs (3 years): Developed production ML pipelines on GCP, MLOps.
        - Intern at DataCo (1 year): Data preprocessing, model evaluation.
        Publications: 2 papers on computer vision.
    """,
    "David Lee": """
        David Lee | david@email.com | +91-9001122334
        Software Engineer transitioning into Data Science.
        Education: B.Tech Software Engineering, VIT
        Skills: Python, Java, SQL, basic pandas, basic scikit-learn
        Experience:
        - Software Engineer at WebCo (3 years): Backend development, REST APIs.
        Currently learning machine learning and deep learning online.
    """,
    "Eva Patel": """
        Eva Patel | eva@email.com | +91-9812345678
        Data Scientist with 5 years of experience in NLP and ML.
        Education: Ph.D. Computational Linguistics, IISc Bangalore
        Skills: Python, NLP, PyTorch, TensorFlow, scikit-learn, SQL, Azure, Tableau,
                Hugging Face, spaCy, BERT, communication, teamwork
        Experience:
        - NLP Researcher at ResearchOrg (3 years): LLM fine-tuning, NLP pipelines.
        - Data Scientist at BankCo (2 years): Fraud detection ML models, Azure deployment.
    """,
}

# ── Required skills extracted from JD (can also be auto-extracted) ──────────
REQUIRED_SKILLS = [
    "python", "machine learning", "deep learning", "nlp", "sql",
    "tensorflow", "pytorch", "scikit-learn", "tableau", "power bi",
    "aws", "gcp", "azure", "mlops", "docker",
]

EDUCATION_KEYWORDS = {
    "phd":      4,
    "ph.d":     4,
    "master":   3,
    "m.sc":     3,
    "m.tech":   3,
    "mba":      3,
    "bachelor": 2,
    "b.sc":     2,
    "b.tech":   2,
    "b.e":      2,
    "diploma":  1,
}

WEIGHTS = {
    "jd_similarity":   0.35,   # TF-IDF cosine similarity vs JD
    "skill_match":     0.35,   # % of required skills present
    "experience":      0.20,   # years of experience (capped at 10)
    "education":       0.10,   # education level score
}


# ══════════════════════════════════════════════════════════════
# 2.  TEXT UTILITIES
# ══════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Lowercase, remove special chars, strip extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_email(text: str) -> str:
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "Not found"

def extract_phone(text: str) -> str:
    match = re.search(r"(\+?\d[\d\s\-]{8,14}\d)", text)
    return match.group(0).strip() if match else "Not found"

def extract_experience_years(text: str) -> float:
    """Pull the largest year number adjacent to 'year' / 'years' / 'experience'."""
    patterns = [
        r"(\d+)\s*\+?\s*years?\s+of\s+experience",
        r"(\d+)\s*\+?\s*years?\s+experience",
        r"experience[^\n]*?(\d+)\s*years?",
    ]
    found = []
    for pat in patterns:
        for m in re.finditer(pat, text.lower()):
            found.append(int(m.group(1)))
    return float(max(found)) if found else 0.0

def extract_education_score(text: str) -> int:
    """Return highest education level score found in text."""
    text_lower = text.lower()
    best = 0
    for kw, score in EDUCATION_KEYWORDS.items():
        if kw in text_lower:
            best = max(best, score)
    return best

def extract_skills(text: str, skill_list: list) -> list:
    """Return which required skills are present in the resume text."""
    text_lower = text.lower()
    return [s for s in skill_list if s in text_lower]


# ══════════════════════════════════════════════════════════════
# 3.  RESUME LOADER
# ══════════════════════════════════════════════════════════════

def load_resumes(source) -> dict:
    """
    Accept either:
      - dict {name: text}  — in-memory resumes (used here)
      - str/Path           — folder containing .txt resume files
    Returns dict {name: raw_text}.
    """
    if isinstance(source, dict):
        return source

    folder = Path(source)
    resumes = {}
    for f in folder.glob("*.txt"):
        resumes[f.stem] = f.read_text(encoding="utf-8", errors="ignore")
    return resumes


# ══════════════════════════════════════════════════════════════
# 4.  SCORING ENGINE
# ══════════════════════════════════════════════════════════════

def compute_jd_similarity(resumes: dict, jd: str) -> dict:
    """TF-IDF cosine similarity of each resume vs the job description."""
    names = list(resumes.keys())
    texts = [clean_text(resumes[n]) for n in names]
    jd_clean = clean_text(jd)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([jd_clean] + texts)

    jd_vec      = matrix[0]
    resume_vecs = matrix[1:]
    similarities = cosine_similarity(jd_vec, resume_vecs)[0]

    return {name: round(float(sim), 4) for name, sim in zip(names, similarities)}


def score_resume(name: str, text: str, jd_sim: float) -> dict:
    """Compute a composite score and extract key metadata for one resume."""
    skills_found   = extract_skills(text, REQUIRED_SKILLS)
    skill_match    = len(skills_found) / len(REQUIRED_SKILLS)
    experience     = extract_experience_years(text)
    edu_score      = extract_education_score(text)
    edu_normalised = edu_score / 4.0   # max score = 4 (PhD)
    exp_normalised = min(experience / 10.0, 1.0)

    composite = (
        WEIGHTS["jd_similarity"] * jd_sim
        + WEIGHTS["skill_match"]   * skill_match
        + WEIGHTS["experience"]    * exp_normalised
        + WEIGHTS["education"]     * edu_normalised
    )

    skills_missing = [s for s in REQUIRED_SKILLS if s not in [x.lower() for x in skills_found]]

    return {
        "Name":             name,
        "Email":            extract_email(text),
        "Phone":            extract_phone(text),
        "Experience (yrs)": experience,
        "Education Score":  edu_score,
        "Skills Found":     len(skills_found),
        "Skills Missing":   len(skills_missing),
        "Skill Match %":    round(skill_match * 100, 1),
        "JD Similarity":    round(jd_sim * 100, 1),
        "Composite Score":  round(composite * 100, 1),
        "Matched Skills":   skills_found,
        "Missing Skills":   skills_missing,
        "Shortlisted":      composite >= 0.50,
    }


# ══════════════════════════════════════════════════════════════
# 5.  MAIN SCANNER
# ══════════════════════════════════════════════════════════════

def scan_resumes(source, jd: str = JOB_DESCRIPTION) -> pd.DataFrame:
    """Full pipeline: load → score → rank."""
    resumes    = load_resumes(source)
    jd_scores  = compute_jd_similarity(resumes, jd)
    results    = [score_resume(name, text, jd_scores[name]) for name, text in resumes.items()]
    df         = pd.DataFrame(results).sort_values("Composite Score", ascending=False)
    df["Rank"] = range(1, len(df) + 1)
    cols       = ["Rank"] + [c for c in df.columns if c != "Rank"]
    return df[cols].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# 6.  REPORTING
# ══════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame):
    print(f"\n{'═'*65}")
    print("  RESUME SCAN RESULTS — CANDIDATE RANKING")
    print(f"{'═'*65}")
    display_cols = ["Rank", "Name", "Experience (yrs)", "Skill Match %",
                    "JD Similarity", "Composite Score", "Shortlisted"]
    print(df[display_cols].to_string(index=False))

def print_detailed(df: pd.DataFrame):
    print(f"\n{'═'*65}")
    print("  DETAILED BREAKDOWN PER CANDIDATE")
    print(f"{'═'*65}")
    for _, row in df.iterrows():
        status = "✅ SHORTLISTED" if row["Shortlisted"] else "❌ Not Shortlisted"
        print(f"\n  #{row['Rank']}  {row['Name']}  —  {status}")
        print(f"       Email       : {row['Email']}")
        print(f"       Phone       : {row['Phone']}")
        print(f"       Experience  : {row['Experience (yrs)']} years")
        print(f"       Education   : {row['Education Score']}/4")
        print(f"       Skill Match : {row['Skill Match %']}%  "
              f"({row['Skills Found']}/{row['Skills Found']+row['Skills Missing']} required skills)")
        print(f"       JD Similarity: {row['JD Similarity']}%")
        print(f"       COMPOSITE   : {row['Composite Score']}/100")
        if row["Matched Skills"]:
            print(f"       ✔ Skills    : {', '.join(row['Matched Skills'])}")
        if row["Missing Skills"]:
            print(f"       ✘ Missing   : {', '.join(row['Missing Skills'])}")


# ══════════════════════════════════════════════════════════════
# 7.  VISUALIZATION
# ══════════════════════════════════════════════════════════════

def plot_results(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Resume Scanner — Candidate Analysis", fontsize=14, fontweight="bold")

    names  = df["Name"].tolist()
    colors = ["#378ADD" if s else "#B5D4F4" for s in df["Shortlisted"]]

    # Chart 1: Composite Score
    bars = axes[0].barh(names[::-1], df["Composite Score"].tolist()[::-1],
                        color=colors[::-1], height=0.5)
    axes[0].set_title("Composite Score (out of 100)", fontsize=11)
    axes[0].set_xlabel("Score")
    axes[0].axvline(50, color="gray", linestyle="--", linewidth=1, label="Threshold (50)")
    axes[0].legend(fontsize=8)
    for bar, val in zip(bars, df["Composite Score"].tolist()[::-1]):
        axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{val}", va="center", fontsize=9)

    # Chart 2: Skill Match %
    axes[1].barh(names[::-1], df["Skill Match %"].tolist()[::-1],
                 color="#9FE1CB", height=0.5)
    axes[1].set_title("Required Skill Match (%)", fontsize=11)
    axes[1].set_xlabel("%")
    axes[1].axvline(60, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlim(0, 105)

    # Chart 3: Skill coverage heatmap-style bar
    ax3 = axes[2]
    for i, (_, row) in enumerate(df[::-1].iterrows()):
        matched = row["Skills Found"]
        missing = row["Skills Missing"]
        total   = matched + missing
        ax3.barh(row["Name"], matched / total * 100, color="#639922", height=0.5)
        ax3.barh(row["Name"], missing / total * 100, left=matched / total * 100,
                 color="#F09595", height=0.5)
    ax3.set_title("Skills: Matched vs Missing", fontsize=11)
    ax3.set_xlabel("% of Required Skills")
    ax3.set_xlim(0, 100)
    # Legend
    from matplotlib.patches import Patch
    ax3.legend(handles=[Patch(color="#639922", label="Matched"),
                         Patch(color="#F09595", label="Missing")],
               fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig("resume_scan_results.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → resume_scan_results.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 8.  EXPORT
# ══════════════════════════════════════════════════════════════

def export_csv(df: pd.DataFrame, path: str = "resume_scan_results.csv"):
    export_cols = ["Rank", "Name", "Email", "Phone", "Experience (yrs)",
                   "Education Score", "Skill Match %", "JD Similarity",
                   "Composite Score", "Shortlisted"]
    df[export_cols].to_csv(path, index=False)
    print(f"Results exported → {path}")


# ══════════════════════════════════════════════════════════════
# 9.  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("   RESUME SCANNER & RANKER")
    print("=" * 65)
    print(f"\nJob Description snippet:\n  {JOB_DESCRIPTION[:120].strip()}...")
    print(f"\nRequired skills ({len(REQUIRED_SKILLS)}): {', '.join(REQUIRED_SKILLS)}")
    print(f"\nScanning {len(SAMPLE_RESUMES)} resumes...")

    df = scan_resumes(SAMPLE_RESUMES, JOB_DESCRIPTION)

    print_summary(df)
    print_detailed(df)

    shortlisted = df[df["Shortlisted"]]
    print(f"\n{'─'*65}")
    print(f"  {len(shortlisted)} of {len(df)} candidates shortlisted.")
    print(f"  Top candidate: {df.iloc[0]['Name']}  "
          f"(Score: {df.iloc[0]['Composite Score']}/100)")
    print(f"{'─'*65}")

    print("\nGenerating charts...")
    plot_results(df)
    export_csv(df)

    print("\n✅ Resume scan complete.")
    print("   To scan your own resumes: replace SAMPLE_RESUMES with a folder path")
    print("   containing .txt files, or pass your own dict {name: text}.\n")

    # ── HOW TO USE WITH REAL FILES ─────────────────────────────
    # Place your resume .txt files in a folder, then call:
    #
    #   df = scan_resumes("path/to/resume/folder", JOB_DESCRIPTION)
    #
    # Or build your own JD string and pass it as the second argument.
    # ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    main()
