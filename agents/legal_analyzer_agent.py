"""
Legal Analyzer Agent - Provides specialized legal reasoning and analysis.
"""

import logging
from typing import Dict, Any
import streamlit as st

from agents.base_agent import BaseAgent
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class LegalAnalyzerAgent(BaseAgent):
    """Agent responsible for specialized legal analysis and reasoning"""
    
    def __init__(self):
        super().__init__("legal_analyzer_agent", "Legal Analyzer")
        self.logger = logging.getLogger(__name__)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform legal analysis on retrieved context"""
        query = data.get("query", "")
        retrieved_context = data.get("retrieved_context", "")
        confidence = data.get("confidence", 0.0)
        
        self.logger.info(f"Legal Analyzer processing query (confidence: {confidence})")
        
        try:
            selected_model = st.session_state.get("selected_model")
            if not selected_model:
                return {
                    "status": "error",
                    "error": "No model selected",
                    "legal_analysis": "",
                    "analysis_provided": False,
                    "confidence_boost": 0.0
                }
            
            llm = ChatOllama(model = selected_model, request_timeout = 120.0)
            
            # Specialized legal analysis prompt
            legal_prompt = ChatPromptTemplate.from_template("""
            You are a specialized legal analyst. Your task is to provide deep legal reasoning 
            and analysis based on the provided context.
            
            USER QUERY: {query}
            
            RETRIEVED LEGAL CONTEXT:
            {context}
            
            ANALYSIS REQUEST:
            Please provide a comprehensive legal analysis that includes:
            1. Identification of key legal principles or statutes mentioned
            2. Analysis of how the context relates to the query
            3. Any potential legal implications or considerations
            4. Gaps in the available information that might need clarification
            
            Provide your analysis in a structured but natural format:
            """)
            
            # Generate legal analysis
            legal_chain = legal_prompt | llm
            legal_analysis = legal_chain.invoke({
                "query": query,
                "context": retrieved_context
            })
            
            result = {
                "status": "success",
                "legal_analysis": legal_analysis.content,
                "analysis_provided": True,
                "confidence_boost": 0.15  # Legal analysis increases confidence
            }
            
            self.update_shared_context("legal_analysis_result", result)
            self.logger.info("Legal analysis completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Legal analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "legal_analysis": "",
                "analysis_provided": False,
                "confidence_boost": 0.0
            }