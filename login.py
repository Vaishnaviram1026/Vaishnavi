import re
import os
import json
import math
import requests
import pdfplumber
import spacy
from groq import Groq
from flask import Flask, render_template, request, redirect, url_for, flash, session

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

nlp = spacy.load('en_core_web_sm')

JSEARCH_API_KEY = os.environ.get('JSEARCH_API_KEY', '')
GROQ_API_KEY    = os.environ.get('GROQ_API_KEY',    '')

client = Groq(api_key=GROQ_API_KEY)

# Server-side store for resume text (too large for cookie session)
user_resume_store = {}

LEVEL_MAP = {'Beginner': 1, 'Mid': 2, 'Advanced': 3}

CAREER_GOALS = [
    'Business Analyst',
    'Marketing Analyst',
    'Finance Analyst',
    'Project Manager',
    'Prompt Engineer',
    'Data Analyst',
    'Product Manager',
    'Operations Manager',
]

SKILLS_LIST = {
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust', 'swift',
    'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'sqlite', 'redis',
    'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'fastapi', 'spring boot',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins',
    'machine learning', 'deep learning', 'nlp', 'computer vision', 'tensorflow', 'pytorch',
    'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'tableau', 'power bi',
    'git', 'github', 'gitlab', 'ci/cd', 'agile', 'scrum', 'jira', 'confluence',
    'html', 'css', 'rest api', 'graphql', 'microservices', 'linux', 'bash',
    'excel', 'powerpoint', 'google analytics', 'seo', 'salesforce', 'hubspot',
    'figma', 'sketch', 'adobe xd', 'photoshop', 'illustrator',
    'financial modeling', 'budgeting', 'forecasting',
    'statistics', 'probability', 'data visualization', 'a/b testing', 'etl',
    'spark', 'hadoop', 'kafka', 'airflow', 'snowflake', 'dbt', 'looker',
}

# Context words that suggest skill level when found near a skill mention
SKILL_CONTEXT_BOOSTS = {
    'python': {
        'advanced': ['tensorflow', 'pytorch', 'fastapi', 'asyncio', 'multiprocess', 'deploy', 'production'],
        'mid':      ['pandas', 'matplotlib', 'sklearn', 'flask', 'django', 'numpy', 'automation', 'scripting'],
    },
    'sql': {
        'advanced': ['stored procedure', 'index optim', 'query tuning', 'partitioning', 'execution plan'],
        'mid':      ['join', 'aggregat', 'subquery', 'view', 'trigger', 'group by'],
    },
    'machine learning': {
        'advanced': ['deploy', 'production', 'mlops', 'distributed training', 'custom model', 'pipeline'],
        'mid':      ['train', 'classif', 'regression', 'cluster', 'feature engineering', 'cross-validation'],
    },
    'javascript': {
        'advanced': ['webpack', 'typescript', 'performance optim', 'architect', 'node', 'microservice'],
        'mid':      ['react', 'vue', 'angular', 'dom', 'async', 'promise', 'fetch', 'api'],
    },
    'data analysis': {
        'advanced': ['predictive model', 'statistical significance', 'a/b test', 'regression analysis'],
        'mid':      ['pivot', 'dashboard', 'visualiz', 'trend', 'insight', 'report'],
    },
}

GENERIC_ADVANCED = [
    'lead', 'senior', 'principal', 'architect', 'expert', 'advanced',
    'production', 'scalab', 'optimiz', 'mentor', 'enterprise',
    '5+ year', '4+ year', '3+ year', 'extensive experience',
]
GENERIC_MID = [
    'develop', 'built', 'implement', 'design', 'creat',
    'proficient', 'experience with', 'applied', 'project', 'familiar',
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def validate_password(password):
    errors = []
    if len(password) < 8:
        errors.append('Password must be at least 8 characters.')
    if not re.search(r'[A-Z]', password):
        errors.append('Password must contain at least one uppercase letter.')
    if not re.search(r'[a-z]', password):
        errors.append('Password must contain at least one lowercase letter.')
    if not re.search(r'[0-9]', password):
        errors.append('Password must contain at least one number.')
    if not re.search(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]', password):
        errors.append('Password must contain at least one special character.')
    return errors


def skill_in_resume(skill, resume_text):
    """Return True if the skill name appears in the resume (word-boundary, case-insensitive)."""
    pattern = r'(?<![a-z0-9])' + re.escape(skill.lower()) + r'(?![a-z0-9])'
    return bool(re.search(pattern, resume_text.lower()))


def suggest_skill_level(skill, resume_text):
    """Return Beginner / Mid / Advanced based on context clues in the resume."""
    text_lower = resume_text.lower()
    skill_lower = skill.lower()

    # Count total occurrences
    occurrences = len(re.findall(r'(?<![a-z])' + re.escape(skill_lower) + r'(?![a-z])', text_lower))

    # Extract context window around first mention
    idx = text_lower.find(skill_lower)
    context = ''
    if idx != -1:
        start = max(0, idx - 250)
        end   = min(len(text_lower), idx + 250)
        context = text_lower[start:end]

    adv_score = sum(1 for w in GENERIC_ADVANCED if w in context)
    mid_score = sum(1 for w in GENERIC_MID      if w in context)

    # Skill-specific boosts (search full text so co-occurring tools count)
    for key, boosts in SKILL_CONTEXT_BOOSTS.items():
        if key in skill_lower or skill_lower in key:
            adv_score += sum(2 for w in boosts['advanced'] if w in text_lower)
            mid_score += sum(1 for w in boosts['mid']      if w in text_lower)
            break

    if occurrences >= 4 or adv_score >= 3:
        return 'Advanced'
    elif occurrences >= 2 or mid_score >= 2 or adv_score >= 1:
        return 'Mid'
    else:
        return 'Beginner'


def fetch_jsearch_descriptions(job_title, num=5):
    """Call JSearch API and return a list of job description strings."""
    url = 'https://jsearch.p.rapidapi.com/search'
    headers = {
        'X-RapidAPI-Key':  JSEARCH_API_KEY,
        'X-RapidAPI-Host': 'jsearch.p.rapidapi.com',
    }
    params = {
        'query':       f'{job_title} jobs',
        'num_pages':   '1',
        'date_posted': 'month',
    }
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    jobs = resp.json().get('data', [])

    descriptions = []
    for job in jobs[:num]:
        parts = []
        desc = job.get('job_description', '')
        if desc:
            parts.append(desc[:2000])
        quals = job.get('job_highlights', {}).get('Qualifications', [])
        if quals:
            parts.append('Qualifications:\n' + '\n'.join(quals))
        req_skills = job.get('job_required_skills') or []
        if req_skills:
            parts.append('Required skills: ' + ', '.join(req_skills))
        if parts:
            descriptions.append('\n'.join(parts))
    return descriptions


def get_required_skills(job_title):
    """Fetch required skills.
    Primary: JSearch real job descriptions → OpenAI normalization.
    Fallback: OpenAI simulation only (if JSearch key missing or fails).
    """
    if not GROQ_API_KEY:
        return None

    job_descriptions = []
    jsearch_used = False
    if JSEARCH_API_KEY:
        try:
            job_descriptions = fetch_jsearch_descriptions(job_title)
            jsearch_used = bool(job_descriptions)
        except Exception:
            pass  # fall through to OpenAI-only path

    if jsearch_used:
        combined = '\n\n---\n\n'.join(job_descriptions)
        prompt = (
            f'Below are real job descriptions scraped from LinkedIn, Indeed, and Handshake '
            f'for {job_title} roles.\n\n{combined[:6000]}\n\n'
            'Extract the TOP 10 most frequently required technical and data visualization skills '
            'across these listings. Rules: '
            '(1) NO soft skills — exclude communication, teamwork, leadership, problem-solving, etc. '
            '(2) Only tools, technologies, programming languages, platforms, BI/visualization tools. '
            '(3) Normalize names — "MS Excel"→"Excel", "Power BI"→"Power BI", no duplicates. '
            '(4) Each skill must appear ONCE with a unique lowercase-normalized name. '
            'Return ONLY a valid JSON array of exactly 10 unique items. Each element: '
            '"skill" (string), "level" (Beginner | Intermediate | Advanced), '
            '"category" (Technical | Visualization). No markdown, no explanation.'
        )
    else:
        prompt = (
            f'List the top 10 technical and data visualization skills most commonly required '
            f'for a "{job_title}" role based on real job postings on LinkedIn, Indeed, and Handshake in 2025. '
            'Rules: '
            '(1) NO soft skills — exclude communication, teamwork, leadership, problem-solving, etc. '
            '(2) Only tools, technologies, programming languages, platforms, BI/visualization tools. '
            '(3) No duplicates — each skill name must be unique. '
            'Return ONLY a valid JSON array of exactly 10 unique items. Each element: '
            '"skill" (string), "level" (Beginner | Intermediate | Advanced), '
            '"category" (Technical | Visualization). No markdown, no explanation.'
        )

    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```[a-z]*\n?', '', raw)
    raw = re.sub(r'\n?```$',       '', raw)
    return json.loads(raw)


def get_ai_gap_analysis(career_goal, skill_levels, required_skills):
    """Return readiness score, assessment, strengths, and per-gap free resources."""
    if not GROQ_API_KEY:
        return None

    gap_skills = []
    for item in required_skills:
        req   = LEVEL_MAP.get(item['level'].replace('Intermediate', 'Mid'), 2)
        yours = LEVEL_MAP.get(skill_levels.get(item['skill'], 'Beginner'), 1)
        if yours < req:
            gap_skills.append(item['skill'])

    skills_summary = ', '.join(f"{s} ({l})" for s, l in skill_levels.items())

    prompt = f"""A student is targeting a {career_goal} role.
Their current skills and levels: {skills_summary}
Skills still needed: {gap_skills}

Return ONLY a valid JSON object with these exact keys:
- "score": integer 1-10 (honest readiness score)
- "assessment": string (2 sentences: one honest assessment, one encouraging next step)
- "strengths": list of 2-3 skill names they already have that are most relevant
- "resources": object where each key is a gap skill name and the value is:
  {{"title": "resource name", "url": "real working URL", "type": "Course|Video|Project|Docs"}}

Use only genuinely free resources: YouTube, freeCodeCamp, Coursera (audit), Khan Academy, official docs, roadmap.sh.
No markdown, no explanation — only the JSON object."""

    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```[a-z]*\n?', '', raw)
    raw = re.sub(r'\n?```$',       '', raw)
    return json.loads(raw)


def gap_status(your_level_str, required_level_str):
    """Return (label, css_class) for the gap badge."""
    if your_level_str == 'Missing':
        return 'Missing', 'missing'
    req_norm = required_level_str.replace('Intermediate', 'Mid')
    yours = LEVEL_MAP.get(your_level_str, 1)
    req   = LEVEL_MAP.get(req_norm, 2)
    if yours > req:
        return 'Strong', 'strong'
    elif yours == req:
        return 'Met', 'met'
    else:
        return 'Gap', 'gap'


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        errors = []
        if not username:
            errors.append('Username is required.')
        errors.extend(validate_password(password))

        if errors:
            for e in errors:
                flash(e, 'error')
            return render_template('login.html', username=username)

        session['username'] = username
        return redirect(url_for('upload'))

    return render_template('login.html', username='')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))

    error           = None
    api_key_missing = not GROQ_API_KEY

    if request.method == 'POST':
        career_goal = request.form.get('career_goal', '')
        file        = request.files.get('resume')

        if not file or file.filename == '':
            error = 'Please upload a PDF resume.'
        elif not file.filename.lower().endswith('.pdf'):
            error = 'Only PDF files are supported.'
        else:
            try:
                with pdfplumber.open(file) as pdf:
                    resume_text = ' '.join(page.extract_text() or '' for page in pdf.pages)

                if not resume_text.strip():
                    error = 'Could not extract text from the PDF. Ensure it is not a scanned image.'
                else:
                    # Step 1: Get top required skills from job postings
                    required_skills = get_required_skills(career_goal) if (career_goal and not api_key_missing) else []

                    # Step 2: Deduplicate by skill name (case-insensitive)
                    seen, deduped = set(), []
                    for item in (required_skills or []):
                        key = item['skill'].lower().strip()
                        if key not in seen:
                            seen.add(key)
                            deduped.append(item)
                    required_skills = deduped

                    # Step 3: For each required skill, check if present in resume
                    skill_presence = {}
                    for item in required_skills:
                        skill   = item['skill']
                        found   = skill_in_resume(skill, resume_text)
                        skill_presence[skill] = {
                            'in_resume':       found,
                            'suggested_level': suggest_skill_level(skill, resume_text) if found else None,
                        }

                    # Store resume text server-side (too large for session cookie)
                    username = session['username']
                    user_resume_store[username] = resume_text

                    session['career_goal']     = career_goal
                    session['required_skills'] = required_skills
                    session['skill_presence']  = skill_presence

                    return redirect(url_for('skills'))

            except Exception as e:
                error = f'Error processing file: {str(e)}'

    return render_template(
        'upload.html',
        username=session['username'],
        career_goals=CAREER_GOALS,
        error=error,
        api_key_missing=api_key_missing,
    )


@app.route('/skills', methods=['GET', 'POST'])
def skills():
    if 'username' not in session:
        return redirect(url_for('login'))
    if 'required_skills' not in session:
        return redirect(url_for('upload'))

    username        = session['username']
    career_goal     = session.get('career_goal', '')
    required_skills = session.get('required_skills', [])
    skill_presence  = session.get('skill_presence', {})

    if request.method == 'POST':
        skill_levels = {}
        for item in required_skills:
            skill    = item['skill']
            presence = skill_presence.get(skill, {})
            if presence.get('in_resume'):
                skill_levels[skill] = request.form.get(f'level_{skill}', 'Beginner')
            else:
                skill_levels[skill] = 'Missing'
        session['skill_levels'] = skill_levels
        return redirect(url_for('analysis'))

    # Build rows from required skills only (sourced from job postings)
    rows = []
    for item in required_skills:
        skill     = item['skill']
        req_level = item['level'].replace('Intermediate', 'Mid')
        presence  = skill_presence.get(skill, {})
        in_resume = presence.get('in_resume', False)
        suggested = presence.get('suggested_level') or 'Beginner'
        rows.append({
            'skill':     skill,
            'in_resume': in_resume,
            'suggested': suggested,
            'req_level': req_level,
            'category':  item.get('category', 'Technical'),
        })

    return render_template(
        'skills.html',
        username=username,
        career_goal=career_goal,
        rows=rows,
        levels=['Beginner', 'Mid', 'Advanced'],
    )


@app.route('/analysis')
def analysis():
    if 'username' not in session:
        return redirect(url_for('login'))
    if 'skill_levels' not in session:
        return redirect(url_for('skills'))

    username        = session['username']
    career_goal     = session.get('career_goal', '')
    required_skills = session.get('required_skills', [])
    skill_levels    = session.get('skill_levels', {})

    ai_result = get_ai_gap_analysis(career_goal, skill_levels, required_skills)

    # Build per-skill breakdown for bar chart
    breakdown = []
    for item in required_skills:
        skill     = item['skill']
        req_norm  = item['level'].replace('Intermediate', 'Mid')
        req_val   = LEVEL_MAP.get(req_norm, 2)
        your_val  = LEVEL_MAP.get(skill_levels.get(skill, 'Beginner'), 1)
        label, css = gap_status(skill_levels.get(skill, 'Beginner'), item['level'])
        breakdown.append({
            'skill':      skill,
            'your_val':   your_val,
            'req_val':    req_val,
            'your_label': skill_levels.get(skill, 'Not in resume'),
            'req_label':  req_norm,
            'status':     label,
            'css':        css,
            'your_pct':   int((your_val / 3) * 100),
            'req_pct':    int((req_val  / 3) * 100),
        })

    score     = ai_result['score']   if ai_result else 0
    resources = ai_result.get('resources', {}) if ai_result else {}
    strengths = ai_result.get('strengths', [])  if ai_result else []
    assessment= ai_result.get('assessment', '') if ai_result else ''

    # SVG ring math (r=54 → circumference ≈ 339)
    circumference = 2 * math.pi * 54
    dash_offset   = circumference * (1 - score / 10)

    return render_template(
        'analysis.html',
        username=username,
        career_goal=career_goal,
        score=score,
        assessment=assessment,
        strengths=strengths,
        breakdown=breakdown,
        resources=resources,
        circumference=round(circumference, 2),
        dash_offset=round(dash_offset, 2),
        api_key_missing=not GROQ_API_KEY,
    )


@app.route('/logout')
def logout():
    username = session.get('username')
    if username:
        user_resume_store.pop(username, None)
    session.clear()
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
