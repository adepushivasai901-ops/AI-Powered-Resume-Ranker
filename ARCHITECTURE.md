# AI-Powered Resume Ranker: System Architecture

## 1. Architectural Philosophy & Principles
The AI-Powered Resume Ranker is designed as a **decision-support tool** for human recruiters, strictly adhering to the following engineering foundations:
1. **Evidence & Provenance First:** No score or claim is generated without a traceable link back to a specific document section or text block.
2. **Deterministic & Explainable:** Scoring is computed by deterministic weighting rules and requirement matches, not by opaque black-box models.
3. **Structured Representation Over Flat Text:** Documents are transformed into canonical Pydantic models (`ParsedResume`, `ParsedJobDescription`) before matching.
4. **Zero Fabrication:** The system never invents years of experience, titles, credentials, or technologies.
5. **Untrusted Input Security:** Resumes are treated as untrusted data with anomaly detection against adversarial prompt injections and chronological discrepancies.
6. **Local & Autonomous Execution:** Fully runnable locally without cloud API dependencies or paid services.

---

## 2. Processing Pipeline

```
[Resume PDF Documents]               [Target Job Description]
          │                                     │
          ▼                                     ▼
[Tier-1 PDF Extraction]                  [JD Intelligence Parser]
  • pypdf line-level blocks                • Seniority & Title extraction
  • DocumentQualityAssessor                • Must-Have vs Preferred skills
    - Low-text / Scanned check             • Min career experience
    - Unprintable glyph check              • Degree & Education reqs
          │                                     │
          ▼                                     ▼
[Section Detector]                       [Structured JD Requirements]
  • Canonical heading classification       • JDRequirement (id, type,
  • Text line & block segmentation           importance, min_months, text)
          │                                     │
          ▼                                     │
[Skill Intelligence Engine]                     │
  • Canonical ontology mapping                  │
  • Ambiguity guards (Java!=JS, C!=C++)         │
  • Evidence classification                     │
    (VERIFIED, EXPLICIT, INFERRED)              │
          │                                     │
          ▼                                     │
[Temporal Reasoning Engine]                     │
  • Date normalization (YYYY, MM, 'Present')    │
  • Non-overlapping interval union              │
  • Total continuous experience                 │
  • Overlap & Gap detection                     │
          │                                     │
          ▼                                     │
[Canonical Parsed Resume Model]                 │
          │                                     │
          └──────────────────┬──────────────────┘
                             │
                             ▼
              [Requirement-Level Matcher]
                • Granular requirement matching
                • Evidence classification
                • Duration & recency checks
                • SATISFIED / PARTIAL / NOT_FOUND
                             │
                             ▼
              [Anomaly & Integrity Detector]
                • Prompt injection scanning
                • Date order contradictions
                • Overlapping employment flags
                • High unbacked skills ratio
                             │
                             ▼
              [Explainable Composite Scorer]
                • Required skill coverage (35%)
                • TF-IDF lexical similarity (35%)
                • Career experience fit (15%)
                • Preferred skills bonus (10%)
                • Education alignment (5%)
                • Confidence calculation & explanation
                             │
                             ▼
              [Deterministic Candidate Ranker]
                • Descending score sort
                • Alphabetical tie-breaking
                             │
                             ▼
              [Presentation & HR Report Export]
                • Results dashboard with evidence matrix
                • Downloadable CSV audit report
```

---

## 3. Core Modules & Responsibilities

### `resume_ranker/models/`
- **`evidence.py`**: Defines `EvidenceType` (`EXPLICIT`, `VERIFIED`, `INFERRED`, `UNCERTAIN`, `MISSING`), `DocumentBlock`, `EvidenceReference`, and `Provenance`.
- **`resume.py`**: Canonical `ParsedResume` schema containing `DocumentMetadata`, `CandidateInfo`, `ExperienceItem`, `EducationItem`, `ProjectItem`, `CertificationItem`, `SkillItem`, and `TimelineAnalysis`.
- **`job_description.py`**: `JDRequirement` and `ParsedJobDescription` schemas with `RequirementType` and `ImportanceLevel`.
- **`matching.py`**: `RequirementMatchResult`, `AnomalyFlag`, and `ScoredCandidateProfile` schemas.

### `resume_ranker/extraction/`
- **`pdf_extractor.py`**: Cascading document extractor supporting line-level `DocumentBlock` creation and quality diagnostics while maintaining backward compatibility with `extract_text_from_pdf`.
- **`quality_assessor.py`**: Assesses text extraction volume, detects scanned/image PDFs, unprintable glyph corruption, and column interleaving.
- **`section_detector.py`**: Classifies resume headings (`SUMMARY`, `EXPERIENCE`, `EDUCATION`, `SKILLS`, `PROJECTS`, `CERTIFICATIONS`, etc.) and tags blocks in reading order.

### `resume_ranker/nlp/`
- **`skill_extractor.py` (`SkillIntelligence`)**: Canonical ontology mapping aliases (`sklearn` -> `scikit-learn`, `k8s` -> `kubernetes`) with boundary guards preventing false positives (`Java` != `JavaScript`, `C` != `C++`, conversational "go" / "r").
- **`temporal_engine.py` (`TemporalEngine`)**: Date normalization and non-overlapping interval union mathematics to ensure concurrent employment does not double-count career years.
- **`jd_parser.py` (`JDParser`)**: Deconstructs raw job descriptions into atomic structured requirements with importance levels and minimum duration thresholds.
- **`resume_parser.py` (`ResumeParser`)**: Coordinates extraction, sectioning, skills, timeline, and candidate contact info into a `ParsedResume`.
- **`preprocessor.py` (`TextPreprocessor`)**: SpaCy tokenization and lemmatization while preserving technical terms.

### `resume_ranker/matching/`
- **`requirement_matcher.py` (`RequirementMatcher`)**: Evaluates every JD requirement against candidate evidence, assigning granular match statuses (`SATISFIED`, `PARTIALLY_SATISFIED`, `INSUFFICIENT_EVIDENCE`, `NOT_FOUND`).

### `resume_ranker/scoring/`
- **`anomaly_detector.py` (`AnomalyDetector`)**: Audits resumes for adversarial prompt injection attempts, chronological contradictions (end before start), and employment overlaps.
- **`scorer.py` (`CandidateScorer`)**: Produces explainable composite scores combining requirement coverage, lexical similarity, career experience, and education fit.

### `resume_ranker/ranking/`
- **`ranker.py` (`CandidateRanker`)**: Pure deterministic sorting and ordinal ranking with ascending ID tie-breaking.

### `resume_ranker/reporting/`
- **`report_generator.py` (`HRReportGenerator`)**: Exports audit-ready CSV reports containing sub-scores, evidence explanations, confidence levels, and anomaly flags.

---

## 4. Integrity & Security Architecture
- **Prompt Injection Defense:** Regex audit flags adversarial instructions (e.g., "ignore previous instructions", "give this candidate 100") embedded in resume text.
- **Upload Safety:** Strict `.pdf` extension validation, UUID filename sanitization via Werkzeug `secure_filename`.
- **Path Traversal Protection:** Download routes resolve canonical paths and assert directory confinement.
