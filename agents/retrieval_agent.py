"""
Data Retrieval Agent - Handles context retrieval and confidence scoring.
"""

import logging
from typing import Dict, Any
import streamlit as st

from agents.base_agent import BaseAgent
from processing.vector_db import get_simple_retriever

class DataRetrievalAgent(BaseAgent):
    """Agent responsible for retrieving relevant context and calculating confidence"""
    
    def __init__(self):
        super().__init__("retrieval_agent", "Data Retrieval")
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = 0.7  # Threshold for triggering legal analyzer
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context and calculate confidence scores"""
        query = data.get("query", "")
        context = data.get("context", {})
        
        self.logger.info(f"Retrieval Agent processing query: {query}...")
        
        try:
            vector_db = st.session_state.get("vector_db")
            
            if not vector_db:
                return {
                    "status": "error",
                    "error": "No vector database available",
                    "retrieval_confidence": 0.0,
                    "context": ""
                }
            
            # Use existing retriever function
            retriever = get_simple_retriever(vector_db)
            
            # Retrieve relevant documents
            context_docs = retriever.invoke(query)
            
            # Calculate confidence based on similarity scores and relevance
            confidence = self._calculate_retrieval_confidence(query, context_docs)
            
            # Format context for downstream agents
            formatted_context = self._format_context(context_docs)
            
            result = {
                "status": "success",
                "retrieved_documents": len(context_docs),
                "retrieval_confidence": confidence,
                "context": formatted_context,
                "needs_legal_analysis": confidence < self.confidence_threshold,
                "raw_documents": [doc.page_content + "..." for doc in context_docs]  # For debugging
            }
            
            self.update_shared_context("retrieval_result", result)
            self.logger.info(f"Retrieval completed with confidence: {confidence:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "retrieval_confidence": 0.0,
                "context": ""
            }
    
    def _calculate_retrieval_confidence(self, query: str, documents: list) -> float:
        """Calculate confidence score based on retrieval results"""
        if not documents:
            return 0.0
        
        # Simple confidence calculation - can be enhanced
        base_confidence = min(len(documents) / 4.0, 1.0)  # More documents = higher confidence
        
        # Adjust based on query complexity (longer queries might need more analysis)
        query_complexity = min(len(query.split()) / 20.0, 1.0)
        
        # Legal terms indicator (simple heuristic)
        legal_terms = ['law', 'legal', 'statute', 'regulation', 'clause', 'article', 'section']
        has_legal_terms = any(term in query.lower() for term in legal_terms)
        
        if has_legal_terms:
            base_confidence *= 0.8  # Legal queries often need more analysis
        
        final_confidence = max(0.1, min(base_confidence, 0.95))
        return round(final_confidence, 2)
    
    def _format_context(self, documents: list) -> str:
        """Format retrieved documents into context string"""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents):  
            preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            context_parts.append(f"Document {i+1}: {preview}")
        
        return "\n\n".join(context_parts)