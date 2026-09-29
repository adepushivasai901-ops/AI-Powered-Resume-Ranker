# AI-Powered Resume Ranker

A modular, decision-support resume screening and candidate ranking application built with Python, Pydantic, SpaCy, Scikit-learn, and Flask.

---

## Overview

The **AI-Powered Resume Ranker** is an interpretable, decision-support platform designed to assist recruitment teams in evaluating and prioritizing candidate resumes against specific job descriptions.

Instead of relying solely on superficial keyword matches or opaque single-score models, the system:
1. Ingests PDF resumes through a cascading document extraction pipeline with automated quality assessment.
2. Segments resumes into canonical sections (Summary, Experience, Education, Skills, Projects, etc.).
3. Resolves skill aliases with strict boundary protection (`Java` != `JavaScript`, `C` != `C++`, `scikit-learn` = `sklearn`).
4. Distinguishes evidence certainty levels: `VERIFIED` (demonstrated in experience/projects), `EXPLICIT` (listed in skills), and `INFERRED`.
5. Parses job descriptions into structured, typed requirements (Must-Have, Preferred, Minimum Experience, Education).
6. Performs non-overlapping temporal reasoning to accurately compute career duration without double-counting concurrent roles.
7. Evaluates each requirement individually to build a granular **Requirement-to-Evidence Matrix**.
8. Audits resumes for integrity issues, chronological contradictions, and adversarial prompt injections.
9. Produces explainable composite scores and exportable audit CSV reports.

> **Decision-Support Notice:** This application is intended solely as an assistive prioritization and analysis tool. Final interview and hiring decisions remain strictly with human recruiters.

---

## System Architecture

```
RESUME PDF/DOCX
      ↓
DOCUMENT INGESTION & QUALITY ASSESSMENT (pypdf, glyph check, scanned PDF check)
      ↓
STRUCTURED RESUME REPRESENTATION (Pydantic models)
      ↓
DETERMINISTIC SECTION DETECTION (Summary, Experience, Skills, Education, Projects)
      ↓
SKILL & TEMPORAL INTELLIGENCE (Canonical ontology, interval union math)
      ↓
JD STRUCTURED REQUIREMENTS (Typed requirements, seniority, duration limits)
      ↓
REQUIREMENT-LEVEL MATCHING (Satisfied, Partial, Insufficient Evidence, Not Found)
      ↓
ANOMALY & INTEGRITY DETECTION (Prompt injections, date contradictions, unbacked skills)
      ↓
EVIDENCE-LINKED COMPOSITE SCORING (Explainable weighted breakdown)
      ↓
DETERMINISTIC CANDIDATE RANKING (Descending score, deterministic tie-breaking)
      ↓
EXPLAINABLE UI & CSV AUDIT EXPORT
```

For full architectural details, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Directory Layout

```
AI powered Resume Ranker/
├── config.py                   # Centralized configuration settings
├── app.py                      # Flask application factory and entry point
├── requirements.txt            # Python package dependencies
├── ARCHITECTURE.md             # Detailed architectural specification
├── README.md                   # Project overview & documentation
├── uploads/                    # Uploaded resumes runtime directory
├── reports/                    # Generated HR CSV reports
├── templates/                  # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   └── results.html
├── static/                     # CSS & JavaScript assets
│   ├── css/style.css
│   └── js/main.js
├── resume_ranker/
│   ├── extraction/             # Cascading extraction, quality assessor, section detector
│   │   ├── pdf_extractor.py
│   │   ├── quality_assessor.py
│   │   └── section_detector.py
│   ├── models/                 # Canonical Pydantic schemas
│   │   ├── evidence.py
│   │   ├── resume.py
│   │   ├── job_description.py
│   │   └── matching.py
│   ├── nlp/                    # Preprocessor, skill intelligence, JD & resume parsers, temporal engine
│   │   ├── preprocessor.py
│   │   ├── skill_extractor.py
│   │   ├── jd_parser.py
│   │   ├── temporal_engine.py
│   │   └── resume_parser.py
│   ├── matching/               # Granular requirement-to-evidence matching engine
│   │   └── requirement_matcher.py
│   ├── scoring/                # Composite explainable scoring & anomaly detector
│   │   ├── scorer.py
│   │   └── anomaly_detector.py
│   ├── ranking/                # Deterministic sorting & tie-breaking
│   │   └── ranker.py
│   ├── reporting/              # Structured CSV audit report generation
│   │   └── report_generator.py
│   ├── vectorization/          # TF-IDF lexical vectorizer & similarity baseline
│   │   └── tfidf_matcher.py
│   └── web/                    # Flask blueprints, routes, and security filters
│       └── routes.py
└── tests/                      # Automated test suite (98 tests)
    ├── test_models.py
    ├── test_quality_assessor.py
    ├── test_section_detector.py
    ├── test_skill_extractor.py
    ├── test_jd_parser.py
    ├── test_temporal_engine.py
    ├── test_requirement_matcher.py
    ├── test_anomaly_and_scoring.py
    ├── test_scorer.py
    ├── test_ranker.py
    ├── test_report_generator.py
    ├── test_routes.py
    ├── test_pdf_extractor.py
    ├── test_preprocessor.py
    └── test_tfidf_matcher.py
```

---

## Setup & Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Virtual environment recommended

### 2. Installation
```bash
# Clone or navigate to the project directory
cd "AI powered Resume Ranker"

# Activate your virtual environment (e.g. on Windows):
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the SpaCy English language model:
python -m spacy download en_core_web_sm
```

### 3. Running the Application
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

### 4. Running the Automated Tests
```bash
pytest -v
```
All **98 unit and integration tests** will execute and verify all components.

---

## Scoring Methodology & Weights

The final composite score ($0.0 - 100.0$) is calculated deterministically:

| Component | Default Weight | Description |
| :--- | :---: | :--- |
| **Required Skills** | `35%` | Proportion of mandatory JD requirements matched, factoring in verification status (`VERIFIED` vs `EXPLICIT`). |
| **TF-IDF Lexical Similarity** | `35%` | Statistical cosine similarity between preprocessed candidate text and job description. |
| **Career Experience** | `15%` | Alignment of continuous, de-duplicated career duration against stated JD requirements. |
| **Preferred Skills** | `10%` | Bonus credit for matching nice-to-have or preferred technical competencies. |
| **Education Fit** | `5%` | Alignment with required academic degrees (e.g. Bachelor's, Master's). |

---

## Integrity & Security Protections

1. **Adversarial Prompt Injection Detection:** Flags candidate documents containing prompt injection strings such as `"ignore previous instructions"` or `"give this candidate 100"`.
2. **Untrusted Input Handling:** All candidate text is treated strictly as data; no unconstrained LLM execution paths can alter deterministic scoring logic.
3. **Upload Security:** Enforces a 16MB file size ceiling, strict `.pdf` file extension whitelisting, and UUID prefixing to prevent filename collisions.
4. **Path Traversal Defense:** Report download endpoints resolve canonical paths and block directory traversal attempts (`..`, `%2F`).

---

## Limitations

- **Scanned / Image PDFs:** High image density or scanned physical resumes lack a native text stream. The quality assessor detects and flags these, but OCR extraction requires local OCR tool installation (e.g., Tesseract).
- **Novel / Unlisted Skills:** The built-in ontology covers major programming languages, cloud providers, ML/AI frameworks, and databases. Specialized niche tools not in the taxonomy are captured via TF-IDF lexical similarity rather than canonical ontology matching.
