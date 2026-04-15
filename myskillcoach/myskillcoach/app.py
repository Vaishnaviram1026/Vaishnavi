import os
import json
import requests
import pdfplumber
import spacy
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_session import Session
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
JSEARCH_API_KEY = os.environ.get("JSEARCH_API_KEY", "")
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY",    "")

app = Flask(__name__)
app.secret_key = "myskillcoach-secret-2024"
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = os.path.join(os.path.dirname(__file__), ".flask_session")
Session(app)

groq_client = Groq(api_key=GROQ_API_KEY)

# ── NLP setup ─────────────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")

SKILL_LIST = [
    "Python", "SQL", "Tableau", "Excel", "R", "Power BI",
    "Machine Learning", "Statistics", "Java", "JavaScript",
    "HTML", "CSS", "Communication", "Presentation", "Teamwork",
    "Leadership", "Problem Solving", "Data Visualization",
    "Financial Modeling", "Project Management",
]
SKILL_LIST_LOWER = {s.lower(): s for s in SKILL_LIST}

CAREER_GOALS = [
    "Data Analyst",
    "Business Analyst",
    "Software Engineer",
    "Financial Analyst",
    "Marketing Analyst",
]

LEVEL_RANK = {"Beginner": 1, "Mid": 2, "Advanced": 3, "Not Sure": 0}


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_skills_from_text(text: str) -> dict[str, int]:
    """Return {skill: mention_count} found in text."""
    text_lower = text.lower()
    counts: dict[str, int] = {}
    for skill_lower, skill_canonical in SKILL_LIST_LOWER.items():
        # Count occurrences (simple word-boundary-aware count)
        count = text_lower.count(skill_lower)
        if count > 0:
            counts[skill_canonical] = count
    return counts


def fetch_jsearch_jobs(career_goal: str) -> list[dict]:
    """Call JSearch API and return list of job dicts."""
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": JSEARCH_API_KEY,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }
    params = {
        "query": f"{career_goal} jobs",
        "num_pages": "2",
        "date_posted": "month",
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        print(f"JSearch error: {e}")
        return []


def top_required_skills(jobs: list[dict], n: int = 10) -> list[str]:
    """Extract top-N skills mentioned across all job descriptions."""
    all_text = " ".join(j.get("job_description", "") for j in jobs)
    counts = extract_skills_from_text(all_text)
    top = sorted(counts, key=lambda s: counts[s], reverse=True)[:n]
    return top


def suggest_student_level(skill: str, mention_count: int) -> str:
    """Heuristic: suggest student level from resume mention frequency."""
    if mention_count >= 3:
        return "Mid"
    return "Beginner"


def get_required_level_from_groq(skill: str, career_goal: str, job_descriptions_sample: str) -> str:
    """Ask Groq what level is required for a skill in job postings."""
    prompt = (
        f"Based on these real job descriptions for {career_goal}:\n"
        f"{job_descriptions_sample}\n\n"
        f"For the skill '{skill}', what level is typically required?\n"
        "Reply with only one word: Beginner, Mid, or Advanced."
    )
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        level = resp.choices[0].message.content.strip().capitalize()
        if level not in ("Beginner", "Mid", "Advanced"):
            level = "Mid"
        return level
    except Exception as e:
        print(f"Groq level error: {e}")
        return "Mid"


def get_soft_skills_from_groq(career_goal: str, student_skills: list[str]) -> list[str]:
    """Ask Groq for 3 soft skills to highlight."""
    prompt = (
        f"A student is targeting a {career_goal} role.\n"
        f"Their technical skills: {', '.join(student_skills)}\n"
        "Based on real job postings, list 3 soft skills they should highlight.\n"
        'Return as JSON array like: ["Communication", "Teamwork", "Presentation"]\n'
        "Return ONLY the JSON array, nothing else."
    )
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        return json.loads(raw[start:end]) if start != -1 else ["Communication", "Teamwork", "Presentation"]
    except Exception as e:
        print(f"Groq soft skills error: {e}")
        return ["Communication", "Teamwork", "Presentation"]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("upload"))


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html", career_goals=CAREER_GOALS)

    # ── POST: process resume ──────────────────────────────────────────────────
    career_goal = request.form.get("career_goal", "Data Analyst")
    resume_file = request.files.get("resume")

    if not resume_file or not resume_file.filename.lower().endswith(".pdf"):
        return render_template(
            "upload.html",
            career_goals=CAREER_GOALS,
            error="Please upload a valid PDF file.",
        )

    # Extract text from PDF
    resume_text = ""
    with pdfplumber.open(resume_file) as pdf:
        for page in pdf.pages:
            resume_text += (page.extract_text() or "") + "\n"

    # Find student skills
    skill_counts = extract_skills_from_text(resume_text)
    student_skills = list(skill_counts.keys())

    # Fetch jobs from JSearch
    jobs = fetch_jsearch_jobs(career_goal)

    # Derive required skills from job descriptions
    required_skills = top_required_skills(jobs, n=10) if jobs else SKILL_LIST[:10]

    # Persist to session
    session["career_goal"]     = career_goal
    session["student_skills"]  = student_skills
    session["skill_counts"]    = skill_counts
    session["required_skills"] = required_skills
    # Store truncated job descriptions for OpenAI calls later
    job_descriptions_sample = "\n\n".join(
        j.get("job_description", "")[:300] for j in jobs[:5]
    )
    session["job_descriptions_sample"] = job_descriptions_sample

    return render_template(
        "upload.html",
        career_goals=CAREER_GOALS,
        selected_goal=career_goal,
        student_skills=student_skills,
        skill_counts=skill_counts,
        required_skills=required_skills,
        jobs_found=len(jobs),
        show_results=True,
    )


@app.route("/match")
def match():
    career_goal            = session.get("career_goal", "Data Analyst")
    student_skills         = session.get("student_skills", [])
    skill_counts           = session.get("skill_counts", {})
    required_skills        = session.get("required_skills", [])
    job_descriptions_sample = session.get("job_descriptions_sample", "")

    # Build table rows
    # All required skills go in; student skills not in required go in too
    all_skills = list(dict.fromkeys(required_skills + student_skills))

    rows = []
    for skill in all_skills:
        mention_count  = skill_counts.get(skill, 0)
        student_level  = suggest_student_level(skill, mention_count) if mention_count > 0 else None
        required_level = get_required_level_from_groq(skill, career_goal, job_descriptions_sample)

        if student_level is None:
            status = "Missing"
        elif LEVEL_RANK.get(student_level, 0) >= LEVEL_RANK.get(required_level, 2):
            status = "Met"
        elif LEVEL_RANK.get(student_level, 0) == LEVEL_RANK.get(required_level, 2) - 1:
            status = "Gap"
        else:
            status = "Gap"

        rows.append({
            "skill":          skill,
            "student_level":  student_level or "Not Sure",
            "required_level": required_level,
            "status":         status,
            "in_resume":      mention_count > 0,
        })

    # Soft skills via Groq
    soft_skills = get_soft_skills_from_groq(career_goal, student_skills)

    # Readiness score
    met_count   = sum(1 for r in rows if r["status"] == "Met")
    total_req   = len(required_skills)
    readiness   = round((met_count / total_req) * 100) if total_req > 0 else 0

    session["rows"]      = rows
    session["readiness"] = readiness

    return render_template(
        "match.html",
        career_goal=career_goal,
        rows=rows,
        soft_skills=soft_skills,
        readiness=readiness,
        levels=["Beginner", "Mid", "Advanced", "Not Sure"],
        level_rank=LEVEL_RANK,
    )


@app.route("/api/recalculate", methods=["POST"])
def recalculate():
    """Called by JS when student changes a skill level dropdown."""
    data           = request.get_json()
    skill_levels   = data.get("skill_levels", {})   # {skill: student_level}
    required_skills = session.get("required_skills", [])

    # We stored rows in session; update statuses
    rows = session.get("rows", [])
    for row in rows:
        skill         = row["skill"]
        new_level     = skill_levels.get(skill, row["student_level"])
        row["student_level"] = new_level
        req_rank = LEVEL_RANK.get(row["required_level"], 2)
        stu_rank = LEVEL_RANK.get(new_level, 0)
        if new_level == "Not Sure" or stu_rank == 0:
            row["status"] = "Missing"
        elif stu_rank >= req_rank:
            row["status"] = "Met"
        else:
            row["status"] = "Gap"

    session["rows"] = rows

    met_count = sum(1 for r in rows if r["status"] == "Met")
    total_req  = len(required_skills) or 1
    readiness  = round((met_count / total_req) * 100)
    session["readiness"] = readiness

    return jsonify({"readiness": readiness, "rows": rows})


if __name__ == "__main__":
    os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
    app.run(debug=True, port=5000)
