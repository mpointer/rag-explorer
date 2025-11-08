"""Extract text from file paths"""
from pathlib import Path
from .extract_text import extract_text


def extract_text_from_file(file_path: str) -> str:
    """Extract text from a file path"""
    with open(file_path, 'rb') as f:
        file_content = f.read()
    filename = Path(file_path).name
    return extract_text(file_content, filename=filename)
