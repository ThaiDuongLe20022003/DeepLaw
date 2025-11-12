"""
Metrics data persistence and storage operations.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

from data_models.models import LLMMetrics
from config.settings import METRICS_DIR


logger = logging.getLogger(__name__)

class MetricsStorage:
    """Handles storage and retrieval of metrics data"""
    
    def __init__(self, metrics_dir: str = METRICS_DIR):
        self.metrics_dir = metrics_dir or METRICS_DIR
        os.makedirs(self.metrics_dir, exist_ok = True)
    
    def save_single_evaluation(self, metric: LLMMetrics) -> str:
        """Save a single evaluation to JSON file"""
        filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(metric.query[:20])}.json"
        filepath = os.path.join(self.metrics_dir, filename)
        
        metric_dict = self._dataclass_to_dict(metric)
        
        try:
            with open(filepath, 'w', encoding = 'utf-8') as f:
                json.dump(metric_dict, f, indent = 2, ensure_ascii = False)
            logger.info(f"Evaluation saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving evaluation: {e}")
            return ""
    
    def save_all_metrics(self, metrics: list, filename: str = None) -> str:
        """Save all metrics to a single JSON file"""
        if not filename:
            filename = f"llm_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.metrics_dir, filename)
        
        metrics_dicts = self._prepare_metrics_for_storage(metrics)
        
        try:
            with open(filepath, 'w', encoding = 'utf-8') as f:
                json.dump(metrics_dicts, f, indent = 2, ensure_ascii = False)
            logger.info(f"All metrics saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving all metrics: {e}")
            return ""
    
    def _dataclass_to_dict(self, obj):
        """Safely convert dataclass to dictionary"""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field in obj.__dataclass_fields__:
                value = getattr(obj, field)
                if isinstance(value, list):
                    result[field] = [self._dataclass_to_dict(item) if hasattr(item, '__dataclass_fields__') else item for item in value]
                else:
                    result[field] = self._dataclass_to_dict(value) if hasattr(value, '__dataclass_fields__') else value
            return result
        else:
            return obj
    
    def _prepare_metrics_for_storage(self, metrics: list) -> list:
        """Prepare metrics data for JSON storage"""
        metrics_dicts = []
        for metric in metrics:
            metric_data = {
                "timestamp": metric.timestamp,
                "query": metric.query,
                "response": metric.response[:2000] + "..." if len(metric.response) > 2000 else metric.response,
                "context_preview": metric.context[:1000] + "..." if len(metric.context) > 1000 else metric.context,
                "response_time": round(metric.response_time, 2),
                "token_count": metric.token_count,
                "tokens_per_second": round(metric.tokens_per_second, 2),
                "model": metric.model,
                "session_id": metric.session_id
            }
            
            if metric.evaluations:
                metric_data["evaluations"] = []
                for eval_obj in metric.evaluations:
                    # Support both object and dict evaluation representations
                    if isinstance(eval_obj, dict):
                        def _g(k, d = eval_obj):
                            return d.get(k)
                    else:
                        def _g(k, o = eval_obj):
                            return getattr(o, k, None)

                    def _safe_float(v, default=0.0):
                        try:
                            return float(v)
                        except Exception:
                            return float(default)

                    eval_data = {
                        "faithfulness": round(_safe_float(_g('faithfulness')), 1),
                        "groundedness": round(_safe_float(_g('groundedness')), 1),
                        "factual_consistency": round(_safe_float(_g('factual_consistency')), 1),
                        "relevance": round(_safe_float(_g('relevance')), 1),
                        "completeness": round(_safe_float(_g('completeness')), 1),
                        "fluency": round(_safe_float(_g('fluency')), 1),
                        "overall_score": round(_safe_float(_g('overall_score')), 1),
                        "evaluation_notes": _g('evaluation_notes') or "",
                        "judge_model": _g('judge_model') or ""
                    }
                    metric_data["evaluations"].append(eval_data)
            
            metrics_dicts.append(metric_data)
        
        return metrics_dicts
    
    def list_evaluation_files(self) -> list:
        """List all evaluation files in metrics directory"""
        evaluation_files = []
        if os.path.exists(self.metrics_dir):
            evaluation_files = [f for f in os.listdir(self.metrics_dir) 
                              if f.endswith('.json') and f.startswith('evaluation_')]
            evaluation_files.sort(reverse = True)
        return evaluation_files
    
    def load_evaluation_file(self, filename: str) -> Dict[str, Any]:
        """Load evaluation data from file"""
        filepath = os.path.join(self.metrics_dir, filename)
        try:
            with open(filepath, 'r', encoding = 'utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading evaluation file {filename}: {e}")
            return {}