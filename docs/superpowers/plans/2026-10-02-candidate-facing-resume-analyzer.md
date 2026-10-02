# Candidate-Facing Resume Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or inline execution to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the existing recruiter-facing resume ranker into a candidate-facing resume analysis and career optimization platform with Resume Health Score, ATS Readiness, Section Quality, Experience/Bullet Quality, Optional Job Match, Actionable Recommendations, and Resume Preview.

**Architecture:** Build a high-cohesion, modular candidate analysis subsystem (`resume_ranker/models/candidate_analysis.py`, `resume_ranker/analyzer/`) that reuses the existing PDF extraction, section detection, canonical skill taxonomy, temporal engine, and TF-IDF matcher. Provide a modern candidate dashboard UI (`templates/dashboard.html`, `templates/index.html`, `static/css/style.css`, `static/js/main.js`) while preserving backwards compatibility so that all existing 143 tests continue passing.

**Tech Stack:** Python 3.12, Flask, Pydantic v2, SpaCy, Scikit-learn, PyPDF, HTML5/CSS3/Vanilla JS, Pytest.

**Spec:** MASTER PROMPT — CONVERT RESUME RANKER INTO CANDIDATE-FACING RESUME ANALYZER (Phases 0 through 27).

---

## Global Constraints
- NO new external or heavy dependencies (no LangChain, LangGraph, LLM APIs, embeddings, paid APIs). Pure deterministic NLP/rules-based logic with existing stack.
- All existing 143 tests must continue to pass.
- Job Description must be 100% OPTIONAL.
- No fabricated scores or fake achievements for candidates. Missing sections trigger dynamic weight renormalization and are marked as "Not detected".
- Security & Privacy: User-controlled text escaped; no sensitive PII exposed.
- All recruiter-facing terminology replaced in the candidate user flow.

---

### Task 1: Candidate Analysis Data Models (`resume_ranker/models/candidate_analysis.py`)
- Define typed Pydantic models: `ResumeHealthScore`, `ATSAnalysis`, `ATSCheckItem`, `SectionQualityDetail`, `BulletQuality`, `ExperienceQualityAnalysis`, `SummaryQualityAnalysis`, `ProjectQualityAnalysis`, `SkillRelevanceAnalysis`, `JobMatchAnalysis`, `CandidateRecommendation`, `CandidateResumeAnalysis`.
- Add unit tests for serialization, validation, and dynamic weight formulas.

### Task 2: ATS Readiness Assessor (`resume_ranker/analyzer/ats_assessor.py`)
- Implement ATS checks: contact details (email, phone, links), standard section headers, date parseability, text readability/glyph issues, table/column risk, excessive special chars, missing core sections.
- Return `ATSAnalysis` with score (0-100), checklist, and actionable fix tips.

### Task 3: Content & Section Quality Evaluator (`resume_ranker/analyzer/content_evaluator.py`)
- Implement bullet analysis: detect weak verbs ("responsible for", "worked on", "helped with"), detect quantifiable metrics, suggest Action + Technology + Task + Result framework with labeled examples.
- Implement summary analysis: length, role clarity, tech keyword density, generic buzzword detection.
- Implement project analysis: tech stack clarity, problem/solution/impact structure.
- Implement section-by-section breakdown (Contact, Summary, Experience, Education, Skills, Projects, Certifications).

### Task 4: Candidate Recommendation Engine & Health Scorer (`resume_ranker/analyzer/recommendation_engine.py` & `resume_ranker/analyzer/candidate_scorer.py`)
- Generate prioritized recommendations: HIGH (missing contact, weak bullets, zero metrics, prompt injection flags), MEDIUM (summary improvements, project clarity, tech context), LOW (formatting, wording).
- Implement 7-component Resume Health Score:
  ATS Compatibility (20%), Content Quality (20%), Skills Relevance (20%), Experience Quality (15%), Projects/Achievements (10%), Education (5%), Structure & Completeness (10%).
- Implement dynamic renormalization when sections (e.g. Education, Projects) are not detected.

### Task 5: Candidate Analysis Orchestrator & Job Matcher (`resume_ranker/analyzer/resume_analyzer.py`)
- Orchestrate parsing (`ResumeParser`), ATS assessment, content evaluation, skill relevance, optional JD matching (using existing `RequirementMatcher` and `JDParser`), scoring, and recommendations.
- Return complete `CandidateResumeAnalysis`.

### Task 6: Candidate CSV Export Generator (`resume_ranker/reporting/candidate_report.py`)
- Generate candidate-friendly audit CSV: Candidate Name, Overall Health Score, ATS Score, Content Score, Skills Score, Experience Score, Education Score, Project Score, Structure Score, Detected Skills, Missing Skills, Warnings, Recommendations, Job Match Score.

### Task 7: Candidate Web Controller & Routes (`resume_ranker/web/routes.py`)
- Add `/analyze` route (handles resume upload + optional JD, processes candidate analysis, stores result).
- Add `/dashboard` route (renders modern candidate dashboard).
- Add `/download-analysis/<filename>` route (serves candidate CSV report).
- Maintain backwards compatibility for `/`, `/upload`, `/results`, `/review_jd`, `/download-report/<filename>` so existing route tests pass.

### Task 8: Modern SaaS Frontend Templates & Styling (`templates/`, `static/`)
- Update `templates/base.html` with candidate-facing brand and navigation.
- Redesign `templates/index.html` with drag-and-drop resume upload zone, file validation, optional JD collapsible section.
- Create `templates/dashboard.html` with card layout, score rings/bars, tabbed navigation, issue highlight tags, and resume preview.
- Update `static/css/style.css` and `static/js/main.js` with modern SaaS styling and interactions.

### Task 9: Comprehensive Candidate-Facing Test Suite (`tests/test_candidate_analyzer.py`)
- Write tests for:
  1. Resume-only analysis (no JD)
  2. Resume + JD analysis (job match)
  3. Missing sections & dynamic renormalization
  4. ATS readiness scoring and checks
  5. Bullet quality and metric detection
  6. Skill category grouping and missing keyword context
  7. High/Medium/Low priority recommendations
  8. Synthetic candidate profiles (Python, Designer, AI/ML)
  9. Candidate CSV export
  10. Candidate web routes (`/analyze`, `/dashboard`, `/download-analysis`)

### Task 10: Full Verification & Git Commits
- Run `pytest -q` to verify all existing 143 tests + new candidate tests pass.
- Verify Flask app runs cleanly with no errors.
- Create atomic Git commits on `feature/candidate-facing-resume-analyzer`.
