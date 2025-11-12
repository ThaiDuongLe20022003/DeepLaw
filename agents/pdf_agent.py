"""
PDF Processing Agent - Handles document ingestion and vectorization.
"""

import logging
from typing import Dict, Any
import streamlit as st

from agents.base_agent import BaseAgent
from processing.document_processor import extract_all_pages_as_images
from processing.vector_db import create_simple_vector_db

class PDFProcessingAgent(BaseAgent):
    """Agent responsible for PDF document processing and vector storage"""
    
    def __init__(self):
        super().__init__("pdf_agent", "PDF Processing")
        self.logger = logging.getLogger(__name__)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process PDF file and create vector database"""
        file_upload = data.get("file_upload")
        query = data.get("query", "")
        
        self.logger.info(f"PDF Agent processing file: {file_upload.name}")
        
        try:
            # Create vector database using existing function
            vector_db = create_simple_vector_db(file_upload)
            
            # Extract pages as images using existing function
            with file_upload as pdf_file:
                pdf_pages = extract_all_pages_as_images(pdf_file)
            
            # Update session state (shared memory)
            st.session_state["vector_db"] = vector_db
            st.session_state["file_upload"] = file_upload
            st.session_state["pdf_pages"] = pdf_pages
            
            # Publish results to shared context
            result = {
                "status": "success",
                "document_processed": True,
                "pages_extracted": len(pdf_pages),
                "vector_db_created": True,
                "document_name": file_upload.name
            }
            
            self.update_shared_context("pdf_processing_result", result)
            self.logger.info("PDF processing completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"PDF processing failed: {e}")
            error_result = {
                "status": "error",
                "error": str(e),
                "document_processed": False
            }
            return error_result