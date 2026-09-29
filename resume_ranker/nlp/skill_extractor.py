"""Skill Intelligence module with canonical normalization, ambiguity guards, and evidence classification."""

import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from resume_ranker.models.evidence import DocumentBlock, EvidenceReference, EvidenceType
from resume_ranker.models.resume import SkillItem

logger = logging.getLogger(__name__)


class SkillIntelligence:
    """Extracts, canonicalizes, and verifies skills against evidence sections."""

    # Built-in fallback taxonomy if external JSON file is missing
    FALLBACK_TAXONOMY: Dict[str, Dict[str, List[str]]] = {
        "programming_languages": {
            "python": ["python", "python3", "py3"],
            "javascript": ["javascript", "js", "ecmascript"],
            "typescript": ["typescript", "ts"],
            "java": ["java", "core java", "java 8", "java 11", "java 17", "java 21"],
            "c++": ["c++", "cpp"],
            "c#": ["c#", "csharp", "c-sharp"],
            "c": ["c"],
            "go": ["golang", "go language", "go programming"],
            "r": ["r language", "r programming"],
            "ruby": ["ruby", "ruby on rails"],
            "php": ["php"],
            "scala": ["scala"],
            "rust": ["rust"],
            "kotlin": ["kotlin"],
            "swift": ["swift"],
            "dart": ["dart"],
            "perl": ["perl"],
            "shell": ["bash", "shell script", "shell scripting", "zsh", "sh"],
            "matlab": ["matlab"],
        },
        "frameworks_and_libraries": {
            ".net": [".net", "dotnet", ".net core", "asp.net"],
            "react": ["react", "react.js", "reactjs", "react native"],
            "angular": ["angular", "angularjs"],
            "vue": ["vue", "vue.js", "vuejs"],
            "node.js": ["node.js", "nodejs", "node js"],
            "flask": ["flask"],
            "fastapi": ["fastapi", "fast-api"],
            "django": ["django"],
            "spring": ["spring boot", "spring framework", "spring"],
            "express": ["express.js", "expressjs", "express"],
            "html": ["html", "html5"],
            "css": ["css", "css3", "sass", "scss"],
            "tailwind": ["tailwind css", "tailwind"],
            "bootstrap": ["bootstrap"],
            "jquery": ["jquery"],
            "graphql": ["graphql"],
        },
        "data_science_and_ai": {
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
            "huggingface": ["huggingface", "hugging face", "transformers"],
            "data analysis": ["data analysis", "data analytics"],
        },
        "databases_and_data_engineering": {
            "sql": ["sql"],
            "postgresql": ["postgresql", "postgres", "psql"],
            "mysql": ["mysql"],
            "mongodb": ["mongodb", "mongo"],
            "redis": ["redis"],
            "spark": ["apache spark", "spark", "pyspark"],
            "kafka": ["apache kafka", "kafka"],
            "hadoop": ["hadoop"],
            "elasticsearch": ["elasticsearch", "elastic search"],
            "snowflake": ["snowflake"],
            "bigquery": ["bigquery", "big query"],
            "dynamodb": ["dynamodb"],
            "cassandra": ["cassandra"],
        },
        "cloud_and_devops": {
            "aws": ["aws", "amazon web services"],
            "gcp": ["gcp", "google cloud platform", "google cloud"],
            "azure": ["microsoft azure", "azure"],
            "docker": ["docker", "docker container", "dockerized"],
            "kubernetes": ["kubernetes", "k8s"],
            "git": ["git", "github", "gitlab"],
            "ci/cd": ["ci/cd", "ci-cd", "continuous integration", "continuous deployment", "cicd"],
            "linux": ["linux", "unix", "ubuntu"],
            "terraform": ["terraform"],
            "ansible": ["ansible"],
            "rest apis": ["rest apis", "rest api", "restful", "restful api", "rest"],
            "microservices": ["microservices", "microservice architecture"],
        },
        "design_and_creative": {
            "figma": ["figma"],
            "sketch": ["sketch app", "sketch tool", "sketch"],
            "adobe xd": ["adobe xd", "xd"],
            "photoshop": ["photoshop", "adobe photoshop"],
            "illustrator": ["illustrator", "adobe illustrator"],
            "invision": ["invision"],
            "ui/ux design": ["ui/ux design", "ui/ux", "ui design", "ux design", "user experience", "user interface", "user interface design", "user experience design"],
            "user research": ["user research", "ux research", "usability testing", "user testing"],
            "wireframing": ["wireframing", "wireframes", "wireframe"],
            "prototyping": ["prototyping", "prototypes", "prototype"],
            "design systems": ["design systems", "design system", "style guides"],
            "interaction design": ["interaction design", "ixd"],
        },
        "product_and_management": {
            "agile": ["agile", "agile methodology", "agile development", "agile methodologies"],
            "scrum": ["scrum", "scrum master"],
            "kanban": ["kanban"],
            "jira": ["jira", "atlassian jira"],
            "confluence": ["confluence", "atlassian confluence"],
            "product management": ["product management", "product strategy", "product roadmap"],
            "stakeholder management": ["stakeholder management", "stakeholder engagement"],
        },
        "content_marketing_and_communication": {
            "copywriting": ["copywriting", "copy writing", "copywriter"],
            "content strategy": ["content strategy", "content strategist", "content management"],
            "seo": ["seo", "search engine optimization", "seo blogging"],
            "technical writing": ["technical writing", "technical documentation", "tech writing"],
            "proofreading": ["proofreading", "copyediting", "editing"],
            "storytelling": ["storytelling"],
            "a/b testing": ["a/b testing", "ab testing", "split testing"],
        },
        "business_intelligence_and_analytics": {
            "tableau": ["tableau"],
            "power bi": ["power bi", "powerbi"],
            "excel": ["excel", "ms excel", "microsoft excel"],
        },
    }

    # Strict token patterns for symbols or easily confused words
    SPECIAL_PATTERNS: Dict[str, str] = {
        # Java: match Java without matching JavaScript, Java Script, Javascript, Java-Script, or JavaBeans
        "java": r"(?i)\bjava\b(?!(?:[-_\w]|(?:\s*script)))",
        # JavaScript: match Javascript without being conflated with Java (including Java Script, Java-Script)
        "javascript": r"(?i)\b(?:java[-_\s]?script|ecmascript|js)\b",
        # C++
        "c++": r"(?i)(?:(?<=\s)c\+\+|\bc\+\+|^c\+\+)(?=(?:[\s,;.)/]|$))",
        # C#
        "c#": r"(?i)(?:(?<=\s)c\#|\bc\#|^c\#)(?=(?:[\s,;.)/]|$))",
        # .NET: match .NET, dotnet, .net core, asp.net, but NOT email domains like user@example.net
        ".net": r"(?i)(?:(?<=\s)\.net|^\.net)(?=(?:[\s,;.)/]|$))|(?:\b(?:dotnet|\.net\s+core|asp\.net)\b)",
        # Ambiguous single-letter 'c': only match when explicitly associated with programming or tech lists
        "c": r"(?i)(?:\b(?:ansi\s+c|embedded\s+c|c(?:\s*,\s*c\+\+|\s*/\s*c\+\+|\s+and\s+c\+\+|(?:\s+programming|\s+language)))\b)|(?:\b(?:languages?|programming|technologies)\s*:[^.\n]*\bc\b)",
        # Ambiguous 'r': only match when contextualized with language/programming/tools
        "r": r"(?i)(?:\b(?:r(?:\s+programming|\s+language|\s+scripting|\s*[/,]\s*(?:python|sas|spss)))\b|(?:\b(?:languages?|tools|analytics)\s*:[^.\n]*\br\b))",
        # Ambiguous 'go': only match when accompanied by language, or written as golang
        "go": r"(?i)(?:\bgolang\b|\bgo(?:\s+programming|\s+language|\s+developer|\s*[/,]\s*(?:python|rust|c\+\+|java))\b|(?:\b(?:languages?|backend|technologies)\s*:[^.\n]*\bgo\b))",
        # CI/CD:
        "ci/cd": r"(?i)(?:(?<=\s)ci/cd|\bci/cd|^ci/cd)(?=(?:[\s,;.)/]|$))|(?:\b(?:continuous\s+integration|continuous\s+deployment|cicd)\b)",
        # UI/UX design:
        "ui/ux design": r"(?i)\b(?:ui/ux(?:\s+design)?|user\s+experience(?:\s+design)?|user\s+interface(?:\s+design)?)\b",
    }

    # Backward compatibility property for direct access
    @property
    def SKILL_TAXONOMY(self) -> Dict[str, List[str]]:
        """Flattened mapping of canonical skill names to their aliases."""
        return self._flat_taxonomy

    def __init__(self, taxonomy_path: Optional[Union[str, Path]] = None):
        """Initialize SkillIntelligence engine with external or fallback taxonomy.

        Args:
            taxonomy_path: Optional path to external JSON skill taxonomy.
        """
        self.taxonomy_path = self._resolve_taxonomy_path(taxonomy_path)
        self._categories, self._flat_taxonomy, self._category_map = self._load_taxonomy()
        self._compiled_patterns: List[Tuple[str, re.Pattern, str]] = []
        self._compile_patterns()

    def _resolve_taxonomy_path(self, path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Find the most relevant taxonomy JSON file."""
        if path:
            p = Path(path)
            if p.is_file():
                return p

        # Check environment variable
        env_path = os.environ.get("SKILL_TAXONOMY_PATH")
        if env_path and Path(env_path).is_file():
            return Path(env_path)

        # Check standard data/ folder in project
        candidate_paths = [
            Path("data/skills_taxonomy.json"),
            Path(__file__).resolve().parent.parent / "data" / "skills_taxonomy.json",
            Path(__file__).resolve().parent.parent.parent / "data" / "skills_taxonomy.json",
        ]
        for cp in candidate_paths:
            if cp.is_file():
                return cp.resolve()

        return None

    def _load_taxonomy(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, str]]:
        """Load and normalize taxonomy into category dict, flat dict, and reverse category map."""
        categories: Dict[str, List[str]] = {}
        flat_tax: Dict[str, List[str]] = {}
        cat_map: Dict[str, str] = {}

        raw_data = None
        if self.taxonomy_path and self.taxonomy_path.is_file():
            try:
                with open(self.taxonomy_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            except Exception as e:
                logger.warning("Could not read taxonomy from %s: %s; using fallback.", self.taxonomy_path, e)

        if not raw_data:
            raw_data = self.FALLBACK_TAXONOMY

        # Raw data can be structured as {category: {canonical: [aliases]}} or {canonical: [aliases]}
        for cat_or_canon, val in raw_data.items():
            if isinstance(val, dict):
                # Nested by category
                cat_name = cat_or_canon
                categories[cat_name] = []
                for canonical, aliases in val.items():
                    canon_clean = canonical.lower().strip()
                    categories[cat_name].append(canon_clean)
                    flat_tax[canon_clean] = [a.lower().strip() for a in aliases]
                    cat_map[canon_clean] = cat_name
            elif isinstance(val, list):
                # Flat canonical -> aliases
                canon_clean = cat_or_canon.lower().strip()
                flat_tax[canon_clean] = [a.lower().strip() for a in val]
                categories.setdefault("general", []).append(canon_clean)
                cat_map[canon_clean] = "general"

        return categories, flat_tax, cat_map

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for fast deterministic scanning."""
        for canonical, aliases in self._flat_taxonomy.items():
            # If a special boundary pattern exists for canonical skill
            if canonical in self.SPECIAL_PATTERNS:
                pat = re.compile(self.SPECIAL_PATTERNS[canonical])
                self._compiled_patterns.append((canonical, pat, canonical))

            # Compile aliases
            for alias in aliases:
                if alias in self.SPECIAL_PATTERNS:
                    continue

                # Punctuation/symbol handling
                if any(sym in alias for sym in ["+", "#", ".", "/", "-", "&"]):
                    pattern_str = (
                        r"(?i)(?:(?<=\s)"
                        + re.escape(alias)
                        + r"|^"
                        + re.escape(alias)
                        + r"|\b"
                        + re.escape(alias)
                        + r"\b)(?=(?:[\s,;.)/]|$))"
                    )
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
        if cleaned in self._flat_taxonomy:
            return cleaned

        # Alias lookup
        for canonical, aliases in self._flat_taxonomy.items():
            if cleaned in aliases:
                return canonical

        return cleaned

    def get_skill_category(self, canonical_name: str) -> Optional[str]:
        """Return the category name for a canonical skill (e.g. 'design_and_creative')."""
        normalized = self.normalize_skill_name(canonical_name)
        return self._category_map.get(normalized)

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
        - VERIFIED: Evidenced in EXPERIENCE or PROJECTS (practical demonstrated usage in roles/projects)
        - EXPLICIT: Claimed in SKILLS section (listed as a competency without role context)
        - INFERRED: Found only in SUMMARY or inferred from contextual description
        - UNCERTAIN: Ambiguous mention or weak extraction confidence

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
