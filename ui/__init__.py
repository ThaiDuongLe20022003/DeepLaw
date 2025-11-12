"""
UI components for the Streamlit application interface.

This package contains all Streamlit UI components organized by functionality:
- Sidebar configuration and controls
- Chat interface and message handling  
- PDF viewer and document display
"""

from .sidebar import (
    render_sidebar,
    render_metrics_summary,
    render_saved_metrics_section
)

from .chat_interface import (
    render_chat_interface,
    display_chat_history, 
    handle_user_input,
    update_session_with_metrics
)

from .pdf_viewer import (
    render_pdf_uploader,
    render_pdf_viewer,
    render_delete_button,
    handle_pdf_upload
)

__all__ = [
    # Sidebar functions
    'render_sidebar',
    'render_metrics_summary',
    'render_saved_metrics_section',
    
    # Chat interface functions
    'render_chat_interface',
    'display_chat_history',
    'handle_user_input', 
    'update_session_with_metrics',
    
    # PDF viewer functions
    'render_pdf_uploader',
    'render_pdf_viewer',
    'render_delete_button',
    'handle_pdf_upload'
]

# Organized exports by component type
SIDEBAR_COMPONENTS = [
    'render_sidebar',
    'render_metrics_summary',
    'render_saved_metrics_section'
]

CHAT_COMPONENTS = [
    'render_chat_interface',
    'display_chat_history',
    'handle_user_input',
    'update_session_with_metrics'
]

PDF_COMPONENTS = [
    'render_pdf_uploader', 
    'render_pdf_viewer',
    'render_delete_button',
    'handle_pdf_upload'
]