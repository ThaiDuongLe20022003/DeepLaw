"""
Configuration package for the application.

This package contains all application settings, constants, and configuration parameters.
"""

from .settings import (
    PERSIST_DIRECTORY,
    METRICS_DIR,
    STREAMLIT_CONFIG,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEARCH_K,
    EVALUATION_CRITERIA,
    RATING_SCALE
)

__all__ = [
    'PERSIST_DIRECTORY',
    'METRICS_DIR', 
    'STREAMLIT_CONFIG',
    'DEFAULT_EMBEDDING_MODEL',
    'DEFAULT_CHUNK_SIZE',
    'DEFAULT_CHUNK_OVERLAP',
    'DEFAULT_SEARCH_K',
    'EVALUATION_CRITERIA',
    'RATING_SCALE'
]