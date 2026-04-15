import re
import os
import json
import math
import requests
import pdfplumber
import spacy
from datetime import datetime
from groq import Groq
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'myskillcoach-dev-secret')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///myskillcoach.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
nlp = spacy.load('en_core_web_sm')

JSEARCH_API_KEY = os.environ.get('JSEARCH_API_KEY', '')
GROQ_API_KEY    = os.environ.get('GROQ_API_KEY',    '')
client          = Groq(api_key=GROQ_API_KEY)

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


# ── Database Models ────────────────────────────────────────────────────────────

class User(db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    major         = db.Column(db.String(100), default='Business')
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    profile       = db.relationship('UserProfile', backref='user', uselist=False,
                                    cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class UserProfile(db.Model):
    __tablename__   = 'user_profiles'
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    career_goal     = db.Column(db.String(100))
    resume_text     = db.Column(db.Text)
    required_skills = db.Column(db.Text)   # JSON string
    skill_presence  = db.Column(db.Text)   # JSON string
    skill_levels    = db.Column(db.Text)   # JSON string
    updated_at      = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def get_required_skills(self):
        return json.loads(self.required_skills or '[]')

    def get_skill_presence(self):
        return json.loads(self.skill_presence or '{}')

    def get_skill_levels(self):
        return json.loads(self.skill_levels or '{}')


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
    pattern = r'(?<![a-z0-9])' + re.escape(skill.lower()) + r'(?![a-z0-9])'
    return bool(re.search(pattern, resume_text.lower()))


def suggest_skill_level(skill, resume_text):
    text_lower  = resume_text.lower()
    skill_lower = skill.lower()
    occurrences = len(re.findall(r'(?<![a-z])' + re.escape(skill_lower) + r'(?![a-z])', text_lower))
    idx = text_lower.find(skill_lower)
    context = ''
    if idx != -1:
        start = max(0, idx - 250)
        end   = min(len(text_lower), idx + 250)
        context = text_lower[start:end]
    adv_score = sum(1 for w in GENERIC_ADVANCED if w in context)
    mid_score = sum(1 for w in GENERIC_MID      if w in context)
    for key, boosts in SKILL_CONTEXT_BOOSTS.items():
        if key in skill_lower or skill_lower in key:
            adv_score += sum(2 for w in boosts['advanced'] if w in text_lower)
            mid_score += sum(1 for w in boosts['mid']      if w in text_lower)
            break
    if occurrences >= 4 or adv_score >= 3:
        return 'Advanced'
    elif occurrences >= 2 or mid_score >= 2 or adv_score >= 1:
        return 'Mid'
    return 'Beginner'


def fetch_jsearch_descriptions(job_title, num=5):
    url     = 'https://jsearch.p.rapidapi.com/search'
    headers = {'X-RapidAPI-Key': JSEARCH_API_KEY, 'X-RapidAPI-Host': 'jsearch.p.rapidapi.com'}
    params  = {'query': f'{job_title} jobs', 'num_pages': '1', 'date_posted': 'month'}
    resp    = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    descriptions = []
    for job in resp.json().get('data', [])[:num]:
        parts = []
        if job.get('job_description'):
            parts.append(job['job_description'][:2000])
        quals = job.get('job_highlights', {}).get('Qualifications', [])
        if quals:
            parts.append('Qualifications:\n' + '\n'.join(quals))
        if job.get('job_required_skills'):
            parts.append('Required skills: ' + ', '.join(job['job_required_skills']))
        if parts:
            descriptions.append('\n'.join(parts))
    return descriptions


def get_required_skills(job_title):
    if not GROQ_API_KEY:
        return None
    job_descriptions, jsearch_used = [], False
    if JSEARCH_API_KEY:
        try:
            job_descriptions = fetch_jsearch_descriptions(job_title)
            jsearch_used     = bool(job_descriptions)
        except Exception:
            pass

    if jsearch_used:
        combined = '\n\n---\n\n'.join(job_descriptions)
        prompt = (
            f'Below are real job descriptions for {job_title} roles.\n\n{combined[:6000]}\n\n'
            'Extract the TOP 10 most frequently required technical and data visualization skills. '
            'NO soft skills. Only tools, technologies, languages, platforms, BI/visualization tools. '
            'Normalize names ("MS Excel"→"Excel"). No duplicates. '
            'Return ONLY a valid JSON array of exactly 10 unique items. '
            'Each element: "skill" (string), "level" (Beginner|Intermediate|Advanced), '
            '"category" (Technical|Visualization). No markdown, no explanation.'
        )
    else:
        prompt = (
            f'List the top 10 technical and data visualization skills most commonly required '
            f'for a "{job_title}" role based on real job postings in 2025. '
            'NO soft skills. Only tools, technologies, languages, platforms, BI/visualization tools. '
            'No duplicates. '
            'Return ONLY a valid JSON array of exactly 10 unique items. '
            'Each element: "skill" (string), "level" (Beginner|Intermediate|Advanced), '
            '"category" (Technical|Visualization). No markdown, no explanation.'
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
- "score": integer 1-10
- "assessment": string (2 sentences)
- "strengths": list of 2-3 skill names
- "resources": object where each key is a gap skill and value is:
  {{"title": "...", "url": "...", "type": "Course|Video|Project|Docs"}}

Free resources only: YouTube, freeCodeCamp, Coursera (audit), Khan Academy, official docs, roadmap.sh.
No markdown — only JSON."""
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
    if your_level_str == 'Missing':
        return 'Missing', 'missing'
    req_norm = required_level_str.replace('Intermediate', 'Mid')
    yours = LEVEL_MAP.get(your_level_str, 1)
    req   = LEVEL_MAP.get(req_norm, 2)
    if yours > req:   return 'Strong', 'strong'
    elif yours == req: return 'Met',   'met'
    else:              return 'Gap',   'gap'


def load_profile_into_session(user):
    """Load a user's saved profile from DB into session."""
    profile = user.profile
    if not profile:
        return False
    session['career_goal']     = profile.career_goal
    session['required_skills'] = profile.get_required_skills()
    session['skill_presence']  = profile.get_skill_presence()
    session['skill_levels']    = profile.get_skill_levels()
    # Store resume text in server-side dict
    if profile.resume_text:
        user_resume_store[user.username] = profile.resume_text
    return True


# Server-side store for resume text (too large for session cookie)
user_resume_store = {}


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        errors = []
        if not username:
            errors.append('Username is required.')

        user = User.query.filter_by(username=username).first()

        if user:
            # Existing user — verify password (no format check needed)
            if not user.check_password(password):
                flash('Incorrect password.', 'error')
                return render_template('login.html', username=username)
            # Login success — load profile
            session['username'] = username
            session['user_id']  = user.id
            has_profile = load_profile_into_session(user)
            return redirect(url_for('dashboard') if has_profile else url_for('upload'))
        else:
            # New user — validate password format then register
            errors.extend(validate_password(password))
            if errors:
                for e in errors:
                    flash(e, 'error')
                return render_template('login.html', username=username)
            new_user = User(username=username)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            session['username'] = username
            session['user_id']  = new_user.id
            return redirect(url_for('upload'))

    return render_template('login.html', username='')


@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    user    = User.query.get(session['user_id'])
    profile = user.profile if user else None
    return render_template(
        'dashboard.html',
        username=session['username'],
        career_goal=session.get('career_goal', ''),
        updated_at=profile.updated_at if profile else None,
    )


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
                    required_skills = get_required_skills(career_goal) if (career_goal and not api_key_missing) else []

                    # Deduplicate
                    seen, deduped = set(), []
                    for item in (required_skills or []):
                        key = item['skill'].lower().strip()
                        if key not in seen:
                            seen.add(key)
                            deduped.append(item)
                    required_skills = deduped

                    # Check each required skill in resume
                    skill_presence = {}
                    for item in required_skills:
                        skill = item['skill']
                        found = skill_in_resume(skill, resume_text)
                        skill_presence[skill] = {
                            'in_resume':       found,
                            'suggested_level': suggest_skill_level(skill, resume_text) if found else None,
                        }

                    # ── Save to database ──────────────────────────────────────
                    user    = User.query.get(session['user_id'])
                    profile = user.profile or UserProfile(user_id=user.id)
                    profile.career_goal     = career_goal
                    profile.resume_text     = resume_text
                    profile.required_skills = json.dumps(required_skills)
                    profile.skill_presence  = json.dumps(skill_presence)
                    profile.skill_levels    = None   # reset on new upload
                    profile.updated_at      = datetime.utcnow()
                    db.session.add(profile)
                    db.session.commit()

                    # Store in session + server-side store
                    user_resume_store[session['username']] = resume_text
                    session['career_goal']     = career_goal
                    session['required_skills'] = required_skills
                    session['skill_presence']  = skill_presence
                    session.pop('skill_levels', None)

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
            skill_levels[skill] = (
                request.form.get(f'level_{skill}', 'Beginner')
                if presence.get('in_resume') else 'Missing'
            )

        # ── Save skill levels to database ─────────────────────────────────────
        user    = User.query.get(session['user_id'])
        profile = user.profile
        if profile:
            profile.skill_levels = json.dumps(skill_levels)
            profile.updated_at   = datetime.utcnow()
            db.session.commit()

        session['skill_levels'] = skill_levels
        return redirect(url_for('analysis'))

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

    career_goal     = session.get('career_goal', '')
    required_skills = session.get('required_skills', [])
    skill_levels    = session.get('skill_levels', {})

    ai_result = get_ai_gap_analysis(career_goal, skill_levels, required_skills)

    breakdown = []
    for item in required_skills:
        skill      = item['skill']
        req_norm   = item['level'].replace('Intermediate', 'Mid')
        req_val    = LEVEL_MAP.get(req_norm, 2)
        your_level = skill_levels.get(skill, 'Beginner')
        your_val   = LEVEL_MAP.get(your_level, 1) if your_level != 'Missing' else 0
        label, css = gap_status(your_level, item['level'])
        breakdown.append({
            'skill':      skill,
            'your_val':   your_val,
            'req_val':    req_val,
            'your_label': your_level,
            'req_label':  req_norm,
            'status':     label,
            'css':        css,
            'your_pct':   int((your_val / 3) * 100),
            'req_pct':    int((req_val  / 3) * 100),
        })

    score         = ai_result['score']            if ai_result else 0
    resources     = ai_result.get('resources', {}) if ai_result else {}
    strengths     = ai_result.get('strengths', []) if ai_result else []
    assessment    = ai_result.get('assessment', '') if ai_result else ''
    circumference = 2 * math.pi * 54
    dash_offset   = circumference * (1 - score / 10)

    return render_template(
        'analysis.html',
        username=session['username'],
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
    with app.app_context():
        db.create_all()   # creates tables if they don't exist
    app.run(debug=True)
