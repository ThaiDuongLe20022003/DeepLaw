"""
PDF viewer components for the Streamlit application.
"""

import streamlit as st

from processing.document_processor import extract_all_pages_as_images
from processing.vector_db import create_simple_vector_db, delete_vector_db


def render_pdf_uploader():
    """Render PDF file uploader and return the uploaded file"""
    return st.file_uploader(
        "Upload a PDF file ↓", 
        type = "pdf", 
        accept_multiple_files = False,
        key = "pdf_uploader"
    )


def handle_pdf_upload(file_upload):
    """Handle PDF file upload and processing with smart detection"""
    if file_upload is not None:
        # Smart detection: Check if this is a NEW file
        is_new_file = (
            st.session_state["vector_db"] is None or                    # First upload
            st.session_state["current_file_name"] != file_upload.name or  # Different file name
            not st.session_state.get("file_processed", False)          # Not processed yet
        )
        
        if is_new_file:
            with st.spinner(f"Processing new PDF: {file_upload.name}..."):
                try:
                    # Clear previous state if exists
                    if st.session_state["vector_db"] is not None:
                        delete_vector_db(st.session_state["vector_db"])
                    
                    # Create new vector database
                    st.session_state["vector_db"] = create_simple_vector_db(file_upload)
                    
                    # Store file reference and name
                    st.session_state["file_upload"] = file_upload
                    st.session_state["current_file_name"] = file_upload.name
                    
                    # Extract pages as images
                    with st.session_state["file_upload"] as pdf_file:
                        st.session_state["pdf_pages"] = extract_all_pages_as_images(pdf_file)
                    
                    # Mark as processed and clear chat history
                    st.session_state["file_processed"] = True
                    st.session_state["messages"] = []  # Clear chat for new document
                    
                    # Clear metrics for new session
                    if "metrics_collector" in st.session_state:
                        st.session_state.metrics_collector.current_session_metrics = []
                    
                    st.success(f"Successfully processed: {file_upload.name}")
                    st.balloons()
                    
                    # Force rerun to refresh the UI
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.session_state["file_processed"] = False
        else:
            st.info(f"📄 Using previously processed file: {file_upload.name}")
    else:
        # No file uploaded
        if st.session_state.get("current_file_name"):
            st.info(f"Current document: {st.session_state['current_file_name']}")
        else:
            st.info("Please upload a PDF document to begin.")


def render_pdf_viewer():
    """Render PDF viewer with zoom controls"""
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        st.write(f" **Document:** {st.session_state.get('current_file_name', 'Unknown')}")
        
        zoom_level = st.slider(
            "Zoom Level", 
            min_value = 100, 
            max_value = 1000, 
            value = 700, 
            step = 50,
            key = "zoom_slider"
        )

        with st.container(height = 410, border = True):
            for i, page_image in enumerate(st.session_state["pdf_pages"]):
                st.image(page_image, width = zoom_level, caption=f"Page {i+1}")


def render_delete_button():
    """Render delete collection button"""
    delete_collection = st.button(
        "Delete collection & Clear Chat", 
        type = "secondary",
        key = "delete_button",
        help = "Remove current document and start fresh"
    )

    if delete_collection:
        # Clear all document-related state
        delete_vector_db(st.session_state["vector_db"])
        
        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)
        st.session_state.pop("current_file_name", None)
        st.session_state.pop("file_processed", None)
        st.session_state.pop("messages", None)
        
        if "metrics_collector" in st.session_state:
            st.session_state.metrics_collector.current_session_metrics = []
        
        st.success("🧹 All documents and chat history cleared!")
        st.rerun()