"""
Multi-agent system for horizontal collaboration in legal document analysis.
"""

from .base_agent import BaseAgent
from .agent_manager import AgentManager
from .pdf_agent import PDFProcessingAgent
from .retrieval_agent import DataRetrievalAgent
from .legal_analyzer_agent import LegalAnalyzerAgent
from .response_agent import ResponseGenerationAgent
from .qa_agent import QualityAssuranceAgent

__all__ = [
    'BaseAgent',
    'AgentManager',
    'PDFProcessingAgent', 
    'DataRetrievalAgent',
    'LegalAnalyzerAgent',
    'ResponseGenerationAgent',
    'QualityAssuranceAgent'
]