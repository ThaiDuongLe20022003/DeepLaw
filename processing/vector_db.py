"""
Vector database creation and management functions.
"""

import os
import tempfile
import shutil
import logging
from typing import Optional

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from config.settings import PERSIST_DIRECTORY, DEFAULT_EMBEDDING_MODEL, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


logger = logging.getLogger(__name__)


def create_simple_vector_db(file_upload) -> Chroma:
    """Create a simple vector DB without complex embeddings to avoid the meta tensor error"""
    logger.info(f"Creating simple vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    try:
        path = os.path.join(temp_dir, file_upload.name)
        with open(path, "wb") as f:
            f.write(file_upload.getvalue())
        
        # Use PyPDFLoader for simplicity
        loader = PyPDFLoader(path)
        data = loader.load_and_split()
        
        # Simple text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = DEFAULT_CHUNK_SIZE,
            chunk_overlap = DEFAULT_CHUNK_OVERLAP,
            length_function = len
        )
        chunks = text_splitter.split_documents(data)
        logger.info(f"Document split into {len(chunks)} chunks")
        
        # Use a simpler embedding model 
        embeddings = HuggingFaceEmbeddings(
            model_name = DEFAULT_EMBEDDING_MODEL,
            model_kwargs = {'device': 'cpu'},
            encode_kwargs = {'normalize_embeddings': False}  # Simpler configuration
        )
        
        # Create vector store
        vector_db = Chroma.from_documents(
            documents = chunks,
            embedding = embeddings,
            persist_directory = PERSIST_DIRECTORY,
            collection_name = f"pdf_{hash(file_upload.name)}"
        )
        
        logger.info("Simple vector DB created successfully")
        return vector_db
        
    except Exception as e:
        logger.error(f"Error creating vector DB: {e}")
        st.error(f"Error creating vector database: {str(e)}")
        raise
    finally:
        shutil.rmtree(temp_dir)


def get_simple_retriever(vector_db: Chroma):
    """Create a simple retriever without complex query expansion"""
    # Simple retriever configuration
    retriever = vector_db.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": 4}  # Retrieve 4 most similar documents
    )
    return retriever


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """Delete the vector database and clear related session state."""
    logger.info("Deleting vector DB")
    if vector_db is not None:
        try:
            vector_db.delete_collection()
            
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)
            
            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
            logger.error(f"Error deleting collection: {e}")
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")