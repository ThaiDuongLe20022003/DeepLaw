
"""
Utility functions and helpers for the DeepLaw RAG application.
"""

import logging
import streamlit as st
from typing import Tuple, Any

from data_models.models import ChatMessage  


logger = logging.getLogger(__name__)

def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.
    """
    logger.info("Extracting model names from models_info")
    try:
        if hasattr(models_info, "models"):
            model_names = tuple(model.model for model in models_info.models)
        else:
            model_names = tuple()
            
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(levelname)s - %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )


def initialize_session_state():
    """Initialize Streamlit session state with ChatMessage objects"""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    # Convert existing dictionary messages to ChatMessage objects if needed
    if st.session_state["messages"] and isinstance(st.session_state["messages"][0], dict):
        converted_messages = []
        for msg in st.session_state["messages"]:
            if isinstance(msg, dict):
                converted_messages.append(ChatMessage(
                    role = msg.get("role", ""),
                    content = msg.get("content", ""),
                    evaluations = msg.get("evaluations", [])
                ))
            else:
                converted_messages.append(msg)
        st.session_state["messages"] = converted_messages
    
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "file_upload" not in st.session_state:
        st.session_state["file_upload"] = None
    if "pdf_pages" not in st.session_state:
        st.session_state["pdf_pages"] = None
    if "current_file_name" not in st.session_state:  
        st.session_state["current_file_name"] = None
    if "file_processed" not in st.session_state:     
        st.session_state["file_processed"] = False
