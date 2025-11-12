"""
Response Generation Agent - Creates final user-facing responses.
"""

import logging
import time
from typing import Dict, Any
import streamlit as st

from agents.base_agent import BaseAgent
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from processing.document_processor import count_tokens

class ResponseGenerationAgent(BaseAgent):
    """Agent responsible for generating final user-facing responses"""
    
    def __init__(self):
        super().__init__("response_agent", "Response Generation")
        self.logger = logging.getLogger(__name__)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final response combining all agent outputs"""
        query = data.get("query", "")
        context = data.get("context", {})
        
        self.logger.info("Response Agent generating final response")
        
        try:
            start_time = time.time()
            
            # Get selected model from session state
            selected_model = st.session_state.get("selected_model")
            if not selected_model:
                return {
                    "status": "error",
                    "error": "No model selected",
                    "final_response": "Please select a model first.",
                    "response_time": 0,
                    "token_count": 0
                }
            
            llm = ChatOllama(model = selected_model, request_timeout = 120.0)
            
            # Build comprehensive context from all agents
            full_context = self._build_comprehensive_context(context)
            
            # Generate response
            response = self._generate_response(query, full_context, llm)
            response_time = time.time() - start_time
            token_count = count_tokens(response)
            
            result = {
                "status": "success",
                "final_response": response,
                "response_time": response_time,
                "token_count": token_count,
                "agents_used": context.get("agents_engaged", []),
                "retrieval_confidence": context.get("confidence_scores", {}).get("retrieval", 0.0)
            }
            
            self.update_shared_context("final_response", result)
            self.logger.info("Response generation completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "final_response": f"I apologize, but I encountered an error: {str(e)}",
                "response_time": 0,
                "token_count": 0
            }
    
    def _build_comprehensive_context(self, context: Dict[str, Any]) -> str:
        """Build comprehensive context from all agent outputs"""
        context_parts = []
        
        # Add retrieved context
        if context.get("context"):
            context_parts.append(f"RETRIEVED DOCUMENT CONTEXT:\n{context['context']}")
        
        # Add legal analysis if available
        legal_analysis = context.get("legal_analysis")
        if legal_analysis:
            context_parts.append(f"LEGAL ANALYSIS:\n{legal_analysis}")
        
        # Add confidence information
        confidence = context.get("confidence_scores", {}).get("retrieval")
        if confidence is not None:
            context_parts.append(f"RETRIEVAL CONFIDENCE: {confidence:.2f}")
        
        return "\n\n".join(context_parts) if context_parts else "No additional context available."
    
    def _generate_response(self, query: str, context: str, llm) -> str:
        """Generate the final response using LLM"""
        prompt_template = ChatPromptTemplate.from_template("""
        You are a professional legal expert assistant. Based on the following information, provide a helpful and accurate response to the user's query.

        CONTEXT INFORMATION:
        {context}

        USER QUERY: {question}

        Please provide a comprehensive, accurate, and helpful answer. If the context doesn't contain sufficient information to answer the question fully, acknowledge this and provide the best answer possible based on the available information.

        ANSWER:
        """)
        
        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({
            "question": query,
            "context": context
        })
        
        return response