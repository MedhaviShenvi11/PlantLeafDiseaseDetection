import os

class Config:
    SECRET_KEY = "plantleaf@2026"

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

    MAX_CONTENT_LENGTH = 5 * 1024 * 1024