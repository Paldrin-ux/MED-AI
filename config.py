import os
from datetime import timedelta

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    # Core
    SECRET_KEY = os.environ.get("SECRET_KEY", "medai-dev-secret-change-in-prod")
    DEBUG = False
    TESTING = False

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'medai.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Uploads
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "uploads")
    MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64 MB
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm", "nii", "nii.gz"}

    # Session
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    # AI Model
    MODEL_PATH = os.path.join(BASE_DIR, "app", "ai", "weights")
    INFERENCE_DEVICE = "cpu"  # "cuda" if GPU available

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True

config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
