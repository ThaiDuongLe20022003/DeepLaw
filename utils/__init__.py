"""
Utility functions and helpers package.

This package contains common utility functions, helpers, and initialization routines
used across the DeepLaw RAG application.
"""

from .helpers import (
    extract_model_names,
    setup_logging,
    initialize_session_state
)

__all__ = [
    'extract_model_names',
    'setup_logging', 
    'initialize_session_state'
]

# Re-export for easy access
extract_model_names = extract_model_names
setup_logging = setup_logging
initialize_session_state = initialize_session_state