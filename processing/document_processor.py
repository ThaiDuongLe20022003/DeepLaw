"""
PDF document processing and text extraction functions.
"""

import pdfplumber
import logging
from typing import List, Any

logger = logging.getLogger(__name__)


def extract_all_pages_as_images(file_upload) -> List[Any]:
    """Extract all pages from a PDF file as images."""
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages


def count_tokens(text: str) -> int:
    """Simple token counter"""
    return len(text.split())