"""Flask application factory and entry point for AI-Powered Resume Ranker."""

import os
from flask import Flask
from config import Config, DevelopmentConfig
from resume_ranker.web.routes import main_bp


def create_app(config_class=DevelopmentConfig) -> Flask:
    """Create and configure Flask application instance.

    Args:
        config_class: Configuration class with environment settings.

    Returns:
        Configured Flask application object.
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.from_object(config_class)

    # Ensure runtime directories exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["REPORT_FOLDER"], exist_ok=True)

    # Register blueprints
    app.register_blueprint(main_bp)

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="127.0.0.1", port=5000, debug=True)
