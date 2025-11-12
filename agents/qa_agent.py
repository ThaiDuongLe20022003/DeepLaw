"""
Quality Assurance Agent - Performs post-processing evaluation and validation.
"""

import logging
from typing import Dict, Any, List
import streamlit as st

from agents.base_agent import BaseAgent

class QualityAssuranceAgent(BaseAgent):
    """Agent responsible for quality assurance through multi-judge evaluation"""
    
    def __init__(self):
        super().__init__("qa_agent", "Quality Assurance")
        self.logger = logging.getLogger(__name__)
        self.judge_evaluator = None
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform quality assurance evaluation on the generated response"""
        query = data.get("query", "")
        response = data.get("response", "")
        context = data.get("context", "")
        agent_sequence = data.get("agent_sequence", [])
        
        self.logger.info("QA Agent performing post-processing evaluation")
        
        try:
            # Get judge evaluator from session state or initialize
            if not self.judge_evaluator:
                self.judge_evaluator = st.session_state.get("judge_evaluator")
                if not self.judge_evaluator:
                    return {
                        "status": "skipped",
                        "evaluations": [],
                        "reason": "Judge evaluator not available"
                    }
            
            # Perform multi-judge evaluation using existing evaluator
            evaluations = self.judge_evaluator.evaluate_response(query, response, context)
            
            # Convert evaluations to serializable format (be defensive about types)
            def _safe_float(v, default=0.0):
                try:
                    return float(v)
                except Exception:
                    return float(default)

            eval_dicts = []
            numeric_scores = []
            for eval_obj in evaluations:
                # eval_obj may have numeric fields as strings; coerce safely
                faith = _safe_float(getattr(eval_obj, 'faithfulness', None))
                ground = _safe_float(getattr(eval_obj, 'groundedness', None))
                factual = _safe_float(getattr(eval_obj, 'factual_consistency', None))
                relevance = _safe_float(getattr(eval_obj, 'relevance', None))
                completeness = _safe_float(getattr(eval_obj, 'completeness', None))
                fluency = _safe_float(getattr(eval_obj, 'fluency', None))
                overall = _safe_float(getattr(eval_obj, 'overall_score', None))

                eval_dict = {
                    "faithfulness": round(faith, 1),
                    "groundedness": round(ground, 1),
                    "factual_consistency": round(factual, 1),
                    "relevance": round(relevance, 1),
                    "completeness": round(completeness, 1),
                    "fluency": round(fluency, 1),
                    "overall_score": round(overall, 1),
                    "evaluation_notes": getattr(eval_obj, 'evaluation_notes', '') or '',
                    "judge_model": getattr(eval_obj, 'judge_model', '') or ''
                }
                eval_dicts.append(eval_dict)
                numeric_scores.append(overall)
            
            result = {
                "status": "success",
                "evaluations": eval_dicts,
                "total_judges": len(evaluations),
                "average_score": (sum(numeric_scores) / len(numeric_scores)) if numeric_scores else 0,
                "agent_sequence_evaluated": agent_sequence
            }
            
            self.update_shared_context("qa_evaluation", result)
            self.logger.info(f"QA evaluation completed with {len(evaluations)} judges")
            
            return result
            
        except Exception as e:
            self.logger.error(f"QA evaluation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "evaluations": []
            }
    
    def should_trigger_reprocessing(self, evaluations: List[Dict[str, Any]], threshold: float = 6.0) -> bool:
        """Determine if response should be reprocessed due to low quality"""
        if not evaluations:
            return False
        
        avg_score = sum(eval["overall_score"] for eval in evaluations) / len(evaluations)
        return avg_score < threshold