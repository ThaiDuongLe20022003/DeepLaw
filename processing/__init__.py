"""
Processing module for document handling and RAG operations.

This module contains functions for PDF processing, vector database management, 
and Retrieval-Augmented Generation (RAG) chain operations.
"""

from .vector_db import (
    create_simple_vector_db,
    get_simple_retriever, 
    delete_vector_db
)

from .document_processor import (
    extract_all_pages_as_images,
    count_tokens
)

from .rag_chain import (
    process_question_simple,
    generate_response_with_metrics
)

__all__ = [
    # Vector database functions
    'create_simple_vector_db',
    'get_simple_retriever', 
    'delete_vector_db',
    
    # Document processing functions
    'extract_all_pages_as_images',
    'count_tokens',
    
    # RAG chain functions
    'process_question_simple',
    'generate_response_with_metrics'
]

# Grouped exports for better organization
VECTOR_DB_FUNCTIONS = [
    'create_simple_vector_db',
    'get_simple_retriever', 
    'delete_vector_db'
]

DOCUMENT_FUNCTIONS = [
    'extract_all_pages_as_images',
    'count_tokens'
]

RAG_FUNCTIONS = [
    'process_question_simple',
    'generate_response_with_metrics'
]