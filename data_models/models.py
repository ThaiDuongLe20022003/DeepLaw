"""
Data models for the DeepLaw RAG application.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class QuantitativeMetrics:
    """Quantitative performance and accuracy metrics"""
    # Accuracy metrics 
    precision: str
    recall: str  
    f1_score: str
    accuracy_method: str  
    
    # Confidence 
    confidence_rate: str
    
    # Latency metrics  
    total_processing_time: str
    retrieval_time: str
    generation_time: str
    evaluation_time: str
    
    # Memory usage 
    ram_usage: str
    gpu_memory_usage: str
    
    # Error rate 
    error_rate: str
    
    # Performance metrics
    tokens_generated: str
    tokens_per_second: str
    
    # Additional context 
    response_length: str
    context_chunks_used: str
    
    # BERTScore info
    bertscore_available: str
    bertscore_initialized: str


@dataclass
class LLMEvaluation:
    """Comprehensive LLM evaluation metrics using LLM-as-a-judge"""
    faithfulness: float
    groundedness: float
    factual_consistency: float
    relevance: float
    completeness: float
    fluency: float
    overall_score: float
    evaluation_notes: str
    judge_model: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "faithfulness": self.faithfulness,
            "groundedness": self.groundedness,
            "factual_consistency": self.factual_consistency,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "fluency": self.fluency,
            "overall_score": self.overall_score,
            "evaluation_notes": self.evaluation_notes,
            "judge_model": self.judge_model
        }


@dataclass
class LLMMetrics:
    """Data class to store LLM performance metrics"""
    timestamp: str
    query: str
    response: str
    context: str
    response_time: float
    token_count: int
    tokens_per_second: float
    model: str
    session_id: str
    evaluations: List[LLMEvaluation]
    quantitative_metrics: Optional[QuantitativeMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp,
            "query": self.query,
            "response": self.response,
            "context": self.context,
            "response_time": self.response_time,
            "token_count": self.token_count,
            "tokens_per_second": self.tokens_per_second,
            "model": self.model,
            "session_id": self.session_id,
            "evaluations": [eval_obj.to_dict() for eval_obj in self.evaluations],
            "quantitative_metrics": self.quantitative_metrics.to_dict() if self.quantitative_metrics else None
        }


@dataclass 
class ChatMessage:
    """Data class for chat messages"""
    role: str
    content: str
    evaluations: List[Dict[str, Any]] = field(default_factory = list)
    quantitative_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility"""
        return {
            "role": self.role,
            "content": self.content,
            "evaluations": self.evaluations,
            "quantitative_metrics": self.quantitative_metrics
        }