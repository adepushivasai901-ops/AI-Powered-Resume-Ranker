# AI-Powered Resume Analyzer & Career Optimization Platform

A modular, explainable decision-support resume analysis and career optimization platform built with Python, Flask, Pydantic, SpaCy, and Scikit-learn.

---

> [!IMPORTANT]
> **Evidence vs. Verification Notice:**
> Resumes are self-reported documents. In this platform, **Resume-Evidenced** means the resume text contains practical contextual evidence of a skill (such as demonstration within work experience achievements or project descriptions); **it does not constitute independent or third-party background verification.**

---

## Overview

The **AI-Powered Resume Analyzer** is an interpretable, candidate-facing decision-support platform designed to help job seekers evaluate and strengthen their resumes before applying.

Instead of relying on superficial keyword stuffing or opaque "black-box" models, the system acts as a personal resume coach:
1. **Ingests PDF Resumes:** Cascading document extraction pipeline with automated quality assessment and glyph validation.
2. **ATS Readiness Audit:** Checks contact details, standard section headings, font readability, chronological consistency, and table/symbol risks.
3. **Actionable Impact Bullets:** Detects passive openers (`Responsible for`, `Worked on`) and guides candidates using the **Action Verb + Technology + Task + Measurable Result** framework with clear examples.
4. **Canonical Skill Ontology:** Reuses a verified 53+ canonical skill ontology with boundary protection (`Java` != `JavaScript`, `C` != `C++` != `C#`, `Go` != `Google`, `R` != words containing 'r', `.NET` != email addresses).
5. **Evidenced vs. Claimed Skills:** Distinguishes skills demonstrated in roles/projects from skills merely listed in a standalone skills box.
6. **7-Factor Resume Health Score:** Deterministic scoring with **Dynamic Weight Renormalization** for missing sections (e.g. education or projects marked as "Not detected" without score fabrication).
7. **Optional Job Description Match:** Enables candidates to paste an optional target job description to reveal missing required skills, preferred competencies, and keyword coverage.
8. **Prioritized Action Plan:** Delivers HIGH, MEDIUM, and LOW priority recommendations with concrete before/after examples.
9. **Exportable Audit Report:** Downloads candidate analysis report in CSV format for offline tracking.

---

## System Architecture

```
CANDIDATE PDF RESUME
      ↓
PDF EXTRACTION & DOCUMENT QUALITY ASSESSMENT (pypdf, glyph check, scanned PDF check)
      ↓
STRUCTURED RESUME REPRESENTATION (Pydantic models, Candidate Name Extraction)
      ↓
DETERMINISTIC SECTION DETECTION (Summary, Experience, Skills, Education, Projects)
      ↓
CANONICAL SKILLS TAXONOMY & EVIDENCE CLASSIFICATION (Evidenced vs Claimed)
      ↓
CONTENT & BULLET EVALUATION (Action verbs, metric detection, summary depth)
      ↓
ATS READINESS AUDIT (Contact info, standard headers, text readability, dates)
      ↓
OPTIONAL JOB MATCH (JD Parsing, requirement-level matching, missing skills) [OPTIONAL]
      ↓
DYNAMICALLY RENORMALIZED RESUME HEALTH SCORE (7 Factors: ATS, Content, Skills, Exp, Proj, Edu, Struct)
      ↓
RECOMMENDATION ENGINE (Prioritized High/Medium/Low actionable improvements)
      ↓
CANDIDATE DASHBOARD & EXPORTABLE AUDIT CSV
```

---

## Technology Stack

- **Backend:** Python 3.10+ (tested on Python 3.12), Flask
- **Document Processing:** pypdf
- **NLP & Linguistics:** SpaCy (`en_core_web_sm`), custom boundary-safe regex engines
- **Vectorization & Lexical Overlap:** Scikit-learn (TF-IDF Vectorizer with Sublinear Term Frequency)
- **Data Modeling & Validation:** Pydantic v2
- **Frontend / UI:** HTML5, CSS3, Vanilla JavaScript, Jinja2 Templates
- **Testing:** Pytest (157 automated unit, integration, route, and synthetic evaluation tests)

---

## Directory Layout

```
AI powered Resume Ranker/
├── config.py                                   # Centralized configuration settings
├── app.py                                      # Flask application factory and entry point
├── requirements.txt                            # Python package dependencies
├── ARCHITECTURE.md                             # Architectural specification
├── README.md                                   # Project documentation
├── uploads/                                    # Uploaded resumes runtime directory
├── reports/                                    # Generated candidate CSV analysis reports
├── templates/                                  # Jinja2 HTML templates
│   ├── base.html                               # Modern base SaaS layout with navigation
│   ├── index.html                              # Candidate landing page & upload dropzone
│   ├── dashboard.html                          # Candidate analysis dashboard & tabs
│   ├── confirm_requirements.html               # Recruiter confirmation workflow (compatibility)
│   └── results.html                            # Multi-candidate screening results (compatibility)
├── static/                                     # CSS & JavaScript assets
│   ├── css/style.css                           # Professional SaaS stylesheet & progress indicators
│   └── js/main.js                              # Drag-and-drop, tab switching, and filters
├── resume_ranker/
│   ├── data/
│   │   └── skills_taxonomy.json                # Canonical external skill taxonomy (53+ audited skills)
│   ├── analyzer/                               # Candidate analysis subsystem
│   │   ├── ats_assessor.py                     # ATS readiness evaluation & checklist
│   │   ├── content_evaluator.py                # Bullet impact, verbs, metrics & summary analysis
│   │   ├── candidate_scorer.py                 # 7-factor Resume Health Score with renormalization
│   │   ├── recommendation_engine.py            # Prioritized actionable recommendations
│   │   └── resume_analyzer.py                  # End-to-end candidate analysis orchestrator
│   ├── extraction/                             # PDF extractor, quality assessor, section detector
│   ├── models/                                 # Pydantic schemas (candidate_analysis, resume, jd, matching)
│   ├── nlp/                                    # Preprocessor, skill intelligence, parsers, temporal engine
│   ├── matching/                               # Requirement-to-evidence matching engine
│   ├── scoring/                                # Recruiter scorer & anomaly detector
│   ├── ranking/                                # Deterministic sorting & tie-breaking
│   ├── reporting/                              # Candidate & HR report generators
│   │   ├── candidate_report.py                 # Candidate analysis CSV export
│   │   └── report_generator.py                 # HR screening report export
│   ├── vectorization/                          # TF-IDF lexical vectorizer
│   └── web/                                    # Flask routes & view controllers
└── tests/                                      # Automated test suite (157 passing tests)
    ├── test_candidate_analyzer.py              # Candidate workflow, ATS, bullets, CSV export
    ├── test_phase3_features.py                 # Technical requirements verification
    ├── test_synthetic_evaluation.py            # Candidates A through F benchmarks
    ├── test_scoring_integrity.py               # Weight renormalization integrity
    └── test_routes.py                          # Flask route validations
```

---

## Resume Health Scoring Methodology

The **Resume Health Score** is calculated across 7 measurable components:

| Component | Default Weight | Description |
| :--- | :---: | :--- |
| **ATS Readiness** | 20% | Contact details, parseability, standard headings, glyph readability |
| **Content Quality** | 20% | Active verb phrasing, quantifiable outcomes, summary substance |
| **Skills Relevance** | 20% | Skill breadth, domain signals, evidenced vs. claimed ratio |
| **Experience Quality** | 15% | Bullet clarity, career duration, measurable achievement density |
| **Projects & Portfolio** | 10% | Practical demonstrations, technologies utilized, repository links |
| **Education** | 5% | Degree credentials, institutions, and completion dates |
| **Structure & Completeness** | 10% | Coverage across standard resume sections (6 core sections) |

### Dynamic Weight Renormalization
When a component cannot be detected (e.g. no education or projects section present), **the system does NOT penalize the user with a fabricated zero or default score.**
Instead, the missing component is marked as `Not Detected`, and the remaining active weights dynamically renormalize to sum to 100%:

$$\text{Health Score} = \frac{\sum_{k \in \text{Active}} w_k \cdot S_k}{\sum_{k \in \text{Active}} w_k}$$

---

## ATS Readiness Audit Methodology

The ATS Readiness Assessor tests candidate documents against deterministic parsing checks:
- **Contact Information:** Detects presence of plain text email and phone numbers.
- **Section Headings:** Identifies conventional headers (`Experience`, `Education`, `Skills`, `Projects`).
- **Text Extraction & Readability:** Flags scanned PDF images, corrupt characters, and text density issues.
- **Work Experience Structure:** Verifies reverse-chronological order and explicit start/end dates.
- **Skills Visibility:** Evaluates dedicated skills sections vs scattered keywords.
- **Special Characters:** Detects non-standard unicode bullets, decorative dividers, and rating bars.

---

## Actionable Recommendations Engine

Improvements are categorized into three distinct priority levels:
- **HIGH PRIORITY:** Missing contact information, scanned PDF formatting, prompt injection flags, passive experience bullets without metrics, and missing core required skills (when a JD is provided).
- **MEDIUM PRIORITY:** Low evidenced-to-claimed skills ratio, missing or overly generic professional summary, missing technical projects, and missing preferred job skills.
- **LOW PRIORITY:** Inconsistent date formatting, non-standard section titles, and decorative formatting symbols.

Every recommendation includes:
1. **Issue:** The identified gap.
2. **Why It Matters:** How it affects recruiters and ATS parsers.
3. **Suggested Action:** Direct guidance on what to modify.
4. **Example:** Clear before/after rewrite labeled as an illustrative example.

---

## Setup & Running the Application

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Virtual environment recommended

### 2. Running the Flask Application
```bash
python app.py
```
Open your browser at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 3. Running the Automated Test Suite
```bash
pytest -q
```
All **157 automated tests** execute and pass in under 35 seconds.
