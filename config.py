"""Configuration settings for the AI-Powered Resume Ranker application."""

import os
from pathlib import Path


class Config:
    """Base configuration settings."""

    # Project directories
    BASE_DIR = Path(__file__).resolve().parent
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    REPORT_FOLDER = BASE_DIR / "reports"
    DATA_FOLDER = BASE_DIR / "data"

    # Security
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

    # Upload restrictions
    ALLOWED_EXTENSIONS = {"pdf"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size

    # NLP & Skills Taxonomy configuration
    SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")
    SKILL_TAXONOMY_PATH = Path(
        os.environ.get(
            "SKILL_TAXONOMY_PATH",
            BASE_DIR / "resume_ranker" / "data" / "skills_taxonomy.json",
        )
    )


class DevelopmentConfig(Config):
    """Development environment configuration."""

    DEBUG = True
    TESTING = False


class TestingConfig(Config):
    """Testing environment configuration."""

    DEBUG = False
    TESTING = True
    UPLOAD_FOLDER = Config.BASE_DIR / "tests" / "test_uploads"
    REPORT_FOLDER = Config.BASE_DIR / "tests" / "test_reports"


class ProductionConfig(Config):
    """Production environment configuration."""

    DEBUG = False
    TESTING = False
