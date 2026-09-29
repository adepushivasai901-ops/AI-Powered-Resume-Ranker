"""Unit tests for SkillIntelligence canonical normalization, ambiguity guards, and evidence classification."""

import pytest
from resume_ranker.models.evidence import EvidenceType
from resume_ranker.nlp.skill_extractor import SkillIntelligence


@pytest.fixture
def skill_engine() -> SkillIntelligence:
    """Fixture providing initialized SkillIntelligence engine."""
    return SkillIntelligence()


def test_normalization_consistency(skill_engine: SkillIntelligence):
    """Verify consistent canonical normalization across aliases."""
    assert skill_engine.normalize_skill_name("sklearn") == "scikit-learn"
    assert skill_engine.normalize_skill_name("scikit-learn") == "scikit-learn"
    assert skill_engine.normalize_skill_name("NodeJS") == "node.js"
    assert skill_engine.normalize_skill_name("node.js") == "node.js"
    assert skill_engine.normalize_skill_name("react.js") == "react"
    assert skill_engine.normalize_skill_name("ReactJS") == "react"
    assert skill_engine.normalize_skill_name("k8s") == "kubernetes"
    assert skill_engine.normalize_skill_name("postgres") == "postgresql"
    assert skill_engine.normalize_skill_name("asp.net") == ".net"


def test_java_vs_javascript_no_false_positive(skill_engine: SkillIntelligence):
    """Verify Java does not match JavaScript and vice versa."""
    # Text mentioning only JavaScript
    js_text = "Experienced frontend engineer developing interactive UIs with JavaScript and React."
    js_matches = [c for c, _, _ in skill_engine.extract_skills_from_text(js_text)]
    assert "javascript" in js_matches
    assert "java" not in js_matches

    # Text mentioning only Java
    java_text = "Backend engineer developing microservices in Java with Spring Boot."
    java_matches = [c for c, _, _ in skill_engine.extract_skills_from_text(java_text)]
    assert "java" in java_matches
    assert "javascript" not in java_matches


def test_c_cpp_csharp_dot_net_isolation(skill_engine: SkillIntelligence):
    """Verify C, C++, C#, and .NET do not create cross false positives."""
    # Text with C++ only
    cpp_text = "High performance systems programming using C++ and Linux."
    cpp_matches = [c for c, _, _ in skill_engine.extract_skills_from_text(cpp_text)]
    assert "c++" in cpp_matches
    assert "c#" not in cpp_matches
    assert "c" not in cpp_matches

    # Text with C# and .NET only
    csharp_text = "Enterprise backend development in C# and .NET Core."
    csharp_matches = [c for c, _, _ in skill_engine.extract_skills_from_text(csharp_text)]
    assert "c#" in csharp_matches
    assert ".net" in csharp_matches
    assert "c++" not in csharp_matches
    assert "c" not in csharp_matches

    # Text with pure C programming
    c_text = "Embedded systems firmware development using C programming language."
    c_matches = [c for c, _, _ in skill_engine.extract_skills_from_text(c_text)]
    assert "c" in c_matches
    assert "c++" not in c_matches
    assert "c#" not in c_matches


def test_ambiguous_go_and_r_tokens(skill_engine: SkillIntelligence):
    """Verify conversational 'go' or 'r' are ignored while technical references are detected."""
    # Conversational non-technical sentence
    ordinary_text = "I love to go to meetups and write about my hobbies in sector R."
    ord_matches = [c for c, _, _ in skill_engine.extract_skills_from_text(ordinary_text)]
    assert "go" not in ord_matches
    assert "r" not in ord_matches

    # Technical Go and R text
    tech_text = "Built microservices in Golang and performed statistical data modeling with R programming."
    tech_matches = [c for c, _, _ in skill_engine.extract_skills_from_text(tech_text)]
    assert "go" in tech_matches
    assert "r" in tech_matches


def test_evidence_classification_verified_vs_explicit_vs_inferred(skill_engine: SkillIntelligence):
    """Verify skills found in EXPERIENCE/PROJECTS are VERIFIED while skills only in SKILLS are EXPLICIT."""
    sections = {
        "SUMMARY": ["Passionate engineer with background in cloud infrastructure."],
        "SKILLS": ["Python, SQL, Docker, Kubernetes"],
        "EXPERIENCE": ["Engineered microservices using Python and Docker on AWS."],
        "PROJECTS": ["Built sentiment analyzer with PyTorch."],
    }

    skills = skill_engine.extract_and_verify_resume_skills(sections)
    skill_dict = {s.canonical_name: s for s in skills}

    # Python & Docker are in SKILLS and demonstrated in EXPERIENCE -> RESUME_EVIDENCED
    assert "python" in skill_dict
    assert skill_dict["python"].evidence_type in [EvidenceType.RESUME_EVIDENCED, EvidenceType.VERIFIED]
    assert skill_dict["python"].confidence == 1.0

    assert "docker" in skill_dict
    assert skill_dict["docker"].evidence_type in [EvidenceType.RESUME_EVIDENCED, EvidenceType.VERIFIED]

    # PyTorch is in PROJECTS -> RESUME_EVIDENCED
    assert "pytorch" in skill_dict
    assert skill_dict["pytorch"].evidence_type in [EvidenceType.RESUME_EVIDENCED, EvidenceType.VERIFIED]

    # SQL and Kubernetes only in SKILLS list -> CLAIMED
    assert "sql" in skill_dict
    assert skill_dict["sql"].evidence_type in [EvidenceType.CLAIMED, EvidenceType.EXPLICIT]
    assert skill_dict["sql"].confidence == 0.85

    assert "kubernetes" in skill_dict
    assert skill_dict["kubernetes"].evidence_type in [EvidenceType.CLAIMED, EvidenceType.EXPLICIT]

    # Verify evidence references are attached
    assert len(skill_dict["python"].evidence_refs) >= 2  # Found in both SKILLS and EXPERIENCE
