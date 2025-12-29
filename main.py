"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain with multi-judge evaluation.
"""

import streamlit as st
import ollama
import warnings

# Suppress torch warning and Streamlit thread warnings
warnings.filterwarnings('ignore', category = UserWarning, message = '.*torch.classes.*')
warnings.filterwarnings('ignore', message = '.*missing ScriptRunContext.*')

# Import modular components
from config import STREAMLIT_CONFIG
from evaluation import MetricsCollector
from ui import render_sidebar, render_metrics_summary, render_chat_interface
from ui import render_pdf_uploader, render_pdf_viewer, render_delete_button, handle_pdf_upload
from utils import setup_logging, initialize_session_state, extract_model_names


def main():
    """Main application function"""
    # Setup application
    setup_logging()
    st.set_page_config(**STREAMLIT_CONFIG)
    initialize_session_state()  
    
    # Get available models
    available_models = get_available_models()
    
    # Initialize metrics collector
    if "metrics_collector" not in st.session_state:
        st.session_state["metrics_collector"] = MetricsCollector()
    
    # Create layout
    col1, col2 = st.columns([1.5, 2])
    
    # Render sidebar
    with st.sidebar:
        selected_model, evaluation_enabled, judge_evaluator = render_sidebar(
            available_models, 
            st.session_state.get("selected_model", available_models[0] if available_models else ""),
            st.session_state["metrics_collector"]
        )
        st.session_state["selected_model"] = selected_model
        st.session_state["evaluation_enabled"] = evaluation_enabled
        
        # Store judge_evaluator in session state if it exists
        if judge_evaluator is not None:
            st.session_state["judge_evaluator"] = judge_evaluator
        elif not evaluation_enabled and "judge_evaluator" in st.session_state:
            # Clear evaluator if evaluation is disabled
            del st.session_state["judge_evaluator"]
        
        # Display metrics summary
        render_metrics_summary(st.session_state["metrics_collector"])
    
    # Main content - PDF upload and viewer
    with col1:
        st.subheader("Document Management")
        file_upload = render_pdf_uploader()
        handle_pdf_upload(file_upload)
        render_pdf_viewer()
        render_delete_button()
    
    # Main content - Chat interface
    with col2:
        st.subheader("Legal Document Chat")
        render_chat_interface(
            st.session_state["vector_db"],
            st.session_state["selected_model"],
            st.session_state["evaluation_enabled"],
            st.session_state.get("judge_evaluator"),  # Get from session state
            st.session_state["metrics_collector"]
        )


def get_available_models():
    """Get available Ollama models"""
    try:
        models_info = ollama.list()
        return extract_model_names(models_info)
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        return tuple()


if __name__ == "__main__":
    main()