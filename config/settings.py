"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Configuration settings and constants for the DeepLaw RAG application.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os

# Directory paths
PERSIST_DIRECTORY = os.path.join("data", "vectors")
METRICS_DIR = os.path.join("data", "metrics")

# Ensure directories exist
os.makedirs(METRICS_DIR, exist_ok = True)
os.makedirs(PERSIST_DIRECTORY, exist_ok = True)

# Environment variables
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "DeepLaw",
    "page_icon": "🧠",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Model settings
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SEARCH_K = 4

# Evaluation settings
EVALUATION_CRITERIA = [
    "faithfulness",
    "groundedness", 
    "factual_consistency",
    "relevance",
    "completeness",
    "fluency"
]

RATING_SCALE = {
    "Excellent": (9.0, 10.0),
    "Good": (8.0, 8.9),
    "Fair": (6.5, 7.9),
    "Average": (5.0, 6.4),
    "Poor / Weak": (0.0, 4.9)
}