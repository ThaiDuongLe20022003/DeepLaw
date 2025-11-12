"""
Main metrics collector class with quantitative metrics support.
"""

import logging
from datetime import datetime  
from typing import Dict, Any, List

from data_models.models import LLMMetrics, LLMEvaluation, QuantitativeMetrics
from evaluation.metrics_storage import MetricsStorage
from evaluation.metrics_analyzer import MetricsAnalyzer
from evaluation.report_generator import ReportGenerator
from evaluation.quantitative_analyzer import QuantitativeAnalyzer
from config.settings import METRICS_DIR  


logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and manages LLM performance metrics with quantitative analysis"""
    
    def __init__(self, metrics_dir: str = None):
        self.metrics_dir = metrics_dir or METRICS_DIR
        self.metrics_storage = MetricsStorage(self.metrics_dir)
        self.metrics_analyzer = MetricsAnalyzer()
        self.quantitative_analyzer = QuantitativeAnalyzer()
        self.current_session_metrics: List[LLMMetrics] = []
        self.total_requests = 0
    
    def record_metrics(self, 
                     query: str, 
                     response: str, 
                     context: str,
                     context_chunks: List[str],
                     response_time: float, 
                     token_count: int, 
                     model: str, 
                     session_id: str,
                     evaluations: List[LLMEvaluation],
                     retrieval_confidence: float = 0.5,
                     start_time: float = None,
                     retrieval_time: float = None,
                     generation_time: float = None,
                     evaluation_time: float = None) -> LLMMetrics:
        """Record metrics with quantitative analysis"""
        
        self.total_requests += 1
        
        # Calculate quantitative metrics (returns dict with string values containing units)
        quantitative_metrics_dict = self.quantitative_analyzer.create_comprehensive_metrics(
            query = query,
            response = response,
            context_chunks = context_chunks,
            retrieval_confidence = retrieval_confidence,
            start_time = start_time or (datetime.now().timestamp() - response_time),
            token_count = token_count,
            retrieval_time = retrieval_time,
            generation_time = generation_time,
            evaluation_time = evaluation_time
        )
        
        # Convert dict to QuantitativeMetrics object (all fields are strings with units)
        quantitative_metrics = QuantitativeMetrics(**quantitative_metrics_dict)
        
        tokens_per_second = token_count / response_time if response_time > 0 else 0
        
        metrics = LLMMetrics(
            timestamp = datetime.now().isoformat(),
            query = query,
            response = response,
            context = context[:1000],
            response_time = response_time,
            token_count = token_count,
            tokens_per_second = tokens_per_second,
            model = model,
            session_id = session_id,
            evaluations = evaluations,
            quantitative_metrics = quantitative_metrics
        )
        
        self.current_session_metrics.append(metrics)
        
        # Auto-save each evaluation to JSON file
        self.metrics_storage.save_single_evaluation(metrics)
        
        return metrics
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics"""
        return self.metrics_analyzer.calculate_session_summary(self.current_session_metrics)
    
    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        summary = self.get_session_summary()
        model_name = self.current_session_metrics[0].model if self.current_session_metrics else "N/A"
        return ReportGenerator.generate_report(summary, model_name)
    
    def list_evaluation_files(self) -> List[str]:
        """List all evaluation files"""
        return self.metrics_storage.list_evaluation_files()
    
    def load_evaluation_file(self, filename: str) -> Dict[str, Any]:
        """Load evaluation data from file"""
        return self.metrics_storage.load_evaluation_file(filename)
    
    def clear_session_metrics(self):
        """Clear current session metrics"""
        self.current_session_metrics = []
        self.total_requests = 0
        logger.info("Session metrics cleared")
    
    def get_quantitative_summary(self) -> Dict[str, Any]:
        """Get summary of quantitative metrics across all sessions"""
        if not self.current_session_metrics:
            return {}
        
        quantitative_data = []
        for metric in self.current_session_metrics:
            if metric.quantitative_metrics:
                quantitative_data.append(metric.quantitative_metrics)
        
        if not quantitative_data:
            return {}
        
        def extract_number(metric_string: str) -> float:
            """Extract numeric value from metric string"""
            try:
                cleaned = ''.join(c for c in metric_string if c.isdigit() or c in '.-')
                return float(cleaned) if cleaned else 0.0
            except (ValueError, IndexError):
                return 0.0
        
        # Calculate averages from extracted numbers
        avg_precision = sum(extract_number(qm.precision) for qm in quantitative_data) / len(quantitative_data)
        avg_recall = sum(extract_number(qm.recall) for qm in quantitative_data) / len(quantitative_data)
        avg_f1 = sum(extract_number(qm.f1_score) for qm in quantitative_data) / len(quantitative_data)
        avg_confidence = sum(extract_number(qm.confidence_rate) for qm in quantitative_data) / len(quantitative_data)
        avg_total_time = sum(extract_number(qm.total_processing_time) for qm in quantitative_data) / len(quantitative_data)
        avg_ram_usage = sum(extract_number(qm.ram_usage) for qm in quantitative_data) / len(quantitative_data)
        avg_gpu_usage = sum(extract_number(qm.gpu_memory_usage) for qm in quantitative_data) / len(quantitative_data)
        avg_error_rate = sum(extract_number(qm.error_rate) for qm in quantitative_data) / len(quantitative_data)
        
        # Format the averages with the same formatting rules as individual metrics
        return {
            "total_conversations": len(quantitative_data),
            "average_precision": f"{avg_precision:.2f}",
            "average_recall": f"{avg_recall:.2f}",
            "average_f1_score": f"{avg_f1:.2f}",
            "average_confidence": f"{avg_confidence:.3f}",
            "average_processing_time": f"{avg_total_time:.2f}s",
            "average_ram_usage": f"{avg_ram_usage:.0f}MB",
            "average_gpu_usage": f"{avg_gpu_usage:.0f}MB",
            "average_error_rate": f"{avg_error_rate:.2f}%",
            "total_requests": self.total_requests
        }