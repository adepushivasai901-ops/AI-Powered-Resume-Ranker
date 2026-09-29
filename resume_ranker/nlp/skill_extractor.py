"""Skill Intelligence module with canonical normalization, ambiguity guards, and evidence classification."""

import re
from typing import Dict, List, Optional, Set, Tuple

from resume_ranker.models.evidence import DocumentBlock, EvidenceReference, EvidenceType
from resume_ranker.models.resume import SkillItem


class SkillIntelligence:
    """Extracts, canonicalizes, and verifies skills against evidence sections."""

    # Canonical skill taxonomy: canonical_name -> list of aliases / variants
    SKILL_TAXONOMY: Dict[str, List[str]] = {
        # Programming Languages
        "python": ["python", "python3", "py3"],
        "javascript": ["javascript", "js", "ecmascript"],
        "typescript": ["typescript", "ts"],
        "java": ["java"],
        "c++": ["c++", "cpp"],
        "c#": ["c#", "csharp", "c-sharp"],
        "c": ["c"],
        "go": ["golang", "go language"],
        "r": ["r language", "r programming"],
        "ruby": ["ruby"],
        "php": ["php"],
        "scala": ["scala"],
        "rust": ["rust"],
        "kotlin": ["kotlin"],
        "swift": ["swift"],
        # Web & Frameworks
        ".net": [".net", "dotnet", ".net core", "asp.net"],
        "react": ["react", "react.js", "reactjs", "react native"],
        "angular": ["angular", "angularjs"],
        "vue": ["vue", "vue.js", "vuejs"],
        "node.js": ["node.js", "nodejs", "node js"],
        "flask": ["flask"],
        "fastapi": ["fastapi", "fast-api"],
        "django": ["django"],
        "spring": ["spring boot", "spring framework", "spring"],
        "html": ["html", "html5"],
        "css": ["css", "css3"],
        # Data Science & Machine Learning
        "scikit-learn": ["scikit-learn", "sklearn", "scikit"],
        "pytorch": ["pytorch", "torch"],
        "tensorflow": ["tensorflow", "tf"],
        "keras": ["keras"],
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "scipy": ["scipy"],
        "machine learning": ["machine learning", "ml"],
        "deep learning": ["deep learning", "dl"],
        "nlp": ["nlp", "natural language processing"],
        "computer vision": ["computer vision", "cv"],
        "llm": ["llm", "large language models", "generative ai", "genai"],
        "spacy": ["spacy"],
        # Data Engineering & Databases
        "sql": ["sql"],
        "postgresql": ["postgresql", "postgres", "psql"],
        "mysql": ["mysql"],
        "mongodb": ["mongodb", "mongo"],
        "redis": ["redis"],
        "spark": ["apache spark", "spark", "pyspark"],
        "kafka": ["apache kafka", "kafka"],
        "hadoop": ["hadoop"],
        "elasticsearch": ["elasticsearch", "elastic search"],
        # Cloud, DevOps & Tools
        "aws": ["aws", "amazon web services"],
        "gcp": ["gcp", "google cloud platform", "google cloud"],
        "azure": ["microsoft azure", "azure"],
        "docker": ["docker"],
        "kubernetes": ["kubernetes", "k8s"],
        "git": ["git", "github", "gitlab"],
        "ci/cd": ["ci/cd", "ci-cd", "continuous integration", "cicd"],
        "linux": ["linux", "unix", "ubuntu"],
        "rest apis": ["rest apis", "rest api", "restful", "restful api", "rest"],
        "graphql": ["graphql"],
        "microservices": ["microservices", "microservice architecture"],
        # BI & Analytics
        "tableau": ["tableau"],
        "power bi": ["power bi", "powerbi"],
        "excel": ["excel", "ms excel"],
    }

    # Strict token patterns for symbols or easily confused words
    SPECIAL_PATTERNS: Dict[str, str] = {
        "c++": r"(?i)(?:\b|(?<=\s))c\+\+(?:\b|(?=[\s,;.]))",
        "c#": r"(?i)(?:\b|(?<=\s))c\#(?:\b|(?=[\s,;.]))",
        ".net": r"(?i)(?:\b|(?<=\s))\.net(?:\b|(?=[\s,;.]))",
        "ci/cd": r"(?i)(?:\b|(?<=\s))ci/cd(?:\b|(?=[\s,;.]))",
        # Ambiguous single-letter 'c': only match when explicitly associated with programming or tech list
        "c": r"(?i)(?:\b(?:c(?:\s*,\s*c\+\+|\s*/\s*c\+\+|(?:\s+programming|\s+language))\b)|(?:\b(?:languages?|programming|technologies)\s*:[^.\n]*\bc\b))",
        # Ambiguous 'r': only match when contextualized with language/programming/python
        "r": r"(?i)(?:\b(?:r(?:\s+programming|\s+language|\s*[/,]\s*python))\b|(?:\b(?:languages?|tools)\s*:[^.\n]*\br\b))",
        # Ambiguous 'go': only match when accompanied by language, or written as golang
        "go": r"(?i)(?:\bgolang\b|\bgo(?:\s+programming|\s+language|\s*[/,]\s*python)\b|(?:\b(?:languages?|backend)\s*:[^.\n]*\bgo\b))",
        # Java: match Java without matching JavaScript
        "java": r"(?i)\bjava\b(?!script)",
        # JavaScript: match Javascript without being conflated with Java
        "javascript": r"(?i)\bjavascript\b|\bjs\b",
    }

    def __init__(self):
        """Pre-compile regex patterns for fast deterministic scanning."""
        self._compiled_patterns: List[Tuple[str, re.Pattern, str]] = []

        for canonical, aliases in self.SKILL_TAXONOMY.items():
            # If a special boundary pattern exists for canonical skill
            if canonical in self.SPECIAL_PATTERNS:
                pat = re.compile(self.SPECIAL_PATTERNS[canonical])
                self._compiled_patterns.append((canonical, pat, canonical))

            # Compile aliases
            for alias in aliases:
                if alias in self.SPECIAL_PATTERNS:
                    continue
                if len(alias) <= 3:
                    # Short acronyms: strict boundary
                    pattern_str = r"(?i)\b" + re.escape(alias) + r"\b"
                else:
                    pattern_str = r"(?i)\b" + re.escape(alias) + r"\b"
                pat = re.compile(pattern_str)
                self._compiled_patterns.append((canonical, pat, alias))

    def normalize_skill_name(self, raw_name: str) -> str:
        """Map raw skill string or alias to canonical form."""
        if not raw_name or not isinstance(raw_name, str):
            return ""
        cleaned = raw_name.strip().lower()

        # Direct canonical match
        if cleaned in self.SKILL_TAXONOMY:
            return cleaned

        # Alias lookup
        for canonical, aliases in self.SKILL_TAXONOMY.items():
            if cleaned in [a.lower() for a in aliases]:
                return canonical

        return cleaned

    def extract_skills_from_text(
        self,
        text: str,
        section_name: Optional[str] = None,
        block_id: Optional[str] = None,
        page: Optional[int] = None,
    ) -> List[Tuple[str, str, EvidenceReference]]:
        """Scan text and return detected (canonical_name, alias_matched, EvidenceReference) tuples."""
        if not text:
            return []

        results: List[Tuple[str, str, EvidenceReference]] = []
        seen_canonicals: Set[str] = set()

        for canonical, pattern, alias in self._compiled_patterns:
            if canonical in seen_canonicals:
                continue

            match = pattern.search(text)
            if match:
                seen_canonicals.add(canonical)
                # Form snippet
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                snippet = text[start:end].strip()

                ref = EvidenceReference(
                    block_id=block_id,
                    section=section_name,
                    text_snippet=snippet,
                    page=page,
                    evidence_type=EvidenceType.EXPLICIT,
                    confidence=1.0 if len(alias) > 2 else 0.9,
                )
                results.append((canonical, match.group(0), ref))

        return results

    def extract_and_verify_resume_skills(
        self,
        sections: Dict[str, List[str]],
        blocks: Optional[List[DocumentBlock]] = None,
    ) -> List[SkillItem]:
        """Extract canonical skills across segmented resume sections with evidence verification.

        Classification logic:
        - VERIFIED: Mentioned in EXPERIENCE or PROJECTS (demonstrated practical usage)
        - EXPLICIT: Mentioned in SKILLS section (directly claimed in technical competencies)
        - INFERRED: Found only in SUMMARY or inferred from context
        - UNCERTAIN: Ambiguous context or low extraction confidence

        Args:
            sections: Dictionary of section name -> list of lines.
            blocks: Optional list of structured DocumentBlocks.

        Returns:
            List of SkillItem instances populated with evidence references and verification levels.
        """
        # Map: canonical_name -> { "aliases": set(), "refs": list, "sections": set() }
        skill_evidence_map: Dict[str, Dict] = {}

        # 1. Scan by section
        for section_name, lines in sections.items():
            combined_text = "\n".join(lines)
            matches = self.extract_skills_from_text(combined_text, section_name=section_name)

            for canonical, matched_str, ref in matches:
                if canonical not in skill_evidence_map:
                    skill_evidence_map[canonical] = {
                        "aliases": set(),
                        "refs": [],
                        "sections": set(),
                    }
                skill_evidence_map[canonical]["aliases"].add(matched_str)
                skill_evidence_map[canonical]["refs"].append(ref)
                skill_evidence_map[canonical]["sections"].add(section_name)

        # 2. Reconcile evidence types
        skill_items: List[SkillItem] = []
        for canonical, data in skill_evidence_map.items():
            sections_present = data["sections"]
            refs = data["refs"]
            aliases = sorted(list(data["aliases"]))

            if "EXPERIENCE" in sections_present or "PROJECTS" in sections_present:
                evidence_type = EvidenceType.VERIFIED
                confidence = 1.0
            elif "SKILLS" in sections_present:
                evidence_type = EvidenceType.EXPLICIT
                confidence = 0.85
            elif "SUMMARY" in sections_present:
                evidence_type = EvidenceType.INFERRED
                confidence = 0.65
            else:
                evidence_type = EvidenceType.UNCERTAIN
                confidence = 0.50

            # Update evidence_type on each linked reference
            for ref in refs:
                ref.evidence_type = evidence_type

            skill_items.append(
                SkillItem(
                    canonical_name=canonical,
                    aliases_detected=aliases,
                    evidence_type=evidence_type,
                    confidence=confidence,
                    evidence_refs=refs,
                )
            )

        # Sort alphabetically by canonical name for deterministic output
        return sorted(skill_items, key=lambda s: s.canonical_name)
