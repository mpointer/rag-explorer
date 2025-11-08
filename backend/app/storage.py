"""File Storage Configuration"""
import os

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)
