"""
LLM-as-a-judge evaluation system using multiple models with parallel execution.
"""

import re
import json
import concurrent.futures
from typing import Dict, List, Any
from langchain_ollama import ChatOllama

from data_models.models import LLMEvaluation

class LLMJudgeEvaluator:
    """LLM-as-a-judge evaluation system using multiple models with parallel execution"""
    
    def __init__(self, judge_models):
        self.judge_models = judge_models
        self.evaluation_prompt = """You are an expert evaluator of AI responses. Please evaluate the following response based on
        the given context and query.

        QUERY: {query}

        CONTEXT: {context}

        RESPONSE: {response}

        Please evaluate on a scale of 0.0-10.0 for each criterion:

        1. FAITHFULNESS (0.0-10.0): Does the answer rely solely on the provided context without hallucination?
        2. GROUNDEDNESS (0.0-10.0): Can all information be directly traced back to the context?
        3. FACTUAL CONSISTENCY (0.0-10.0): How factually accurate is the response compared to the context?
        4. RELEVANCE (0.0-10.0): How well does the response address the specific query?
        5. COMPLETENESS (0.0-10.0): Does the response cover all important aspects of the query?
        6. FLUENCY (0.0-10.0): Is the response natural, coherent, and well-written?

        Calculate an overall_score (0.0-10.0) as a weighted average:
        - Faithfulness, Groundedness, Factual Consistency: 20% each
        - Relevance: 15%
        - Completeness: 15%
        - Fluency: 10%

        Provide your evaluation in JSON format exactly as follows:
        {{
        "faithfulness": 8.5,
        "groundedness": 9.0,
        "factual_consistency": 9.2,
        "relevance": 8.8,
        "completeness": 7.5,
        "fluency": 9.5,
        "overall_score": 8.7,
        "evaluation_notes": "Brief explanation of scores"
        }}

        Only respond with valid JSON, no other text."""
    
    def _get_rating_category(self, score: float) -> str:
        """Convert overall score to rating category"""
        if score >= 9.0:
            return "Excellent"
        elif score >= 8.0:
            return "Good"
        elif score >= 6.5:
            return "Fair"
        elif score >= 5.0:
            return "Average"
        else:
            return "Poor / Weak"
    
    def evaluate_response(self, query: str, response: str, context: str) -> List[LLMEvaluation]:
        """Evaluate response using multiple LLM judges in PARALLEL"""
        # Prepare evaluation data
        evaluation_data = {
            "query": query,
            "response": response,
            "context": context[:1000]
        }
        
        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers = len(self.judge_models)) as executor:
            # Submit all evaluation tasks to run in parallel
            future_to_model = {
                executor.submit(self._evaluate_single_judge, judge_model, evaluation_data): judge_model
                for judge_model in self.judge_models
            }
            
            # Collect results as they complete
            evaluations = []
            for future in concurrent.futures.as_completed(future_to_model):
                judge_model = future_to_model[future]
                try:
                    # Get result with timeout
                    evaluation = future.result(timeout = 60.0)  
                    evaluations.append(evaluation)
                except Exception as e:
                    print(f"Evaluation error from {judge_model}: {e}")
                    # Use fallback evaluation
                    evaluations.append(LLMEvaluation(
                        faithfulness = 5.0,
                        groundedness = 5.0,
                        factual_consistency = 5.0,
                        relevance = 5.0,
                        completeness = 5.0,
                        fluency = 6.0,
                        overall_score = 5.2,
                        evaluation_notes = f"Poor / Weak: Evaluation failed: {str(e)}",
                        judge_model = judge_model
                    ))
        
        return evaluations
    
    def _evaluate_single_judge(self, judge_model: str, evaluation_data: Dict[str, Any]) -> LLMEvaluation:
        """Evaluate response using a single judge model"""
        try:
            judge_llm = ChatOllama(model = judge_model, request_timeout = 60.0)
            
            prompt = self.evaluation_prompt.format(
                query = evaluation_data["query"],
                context = evaluation_data["context"],
                response = evaluation_data["response"]
            )
            
            evaluation_response = judge_llm.invoke(prompt)
            eval_text = evaluation_response.content.strip()
            eval_data = self._parse_evaluation_response(eval_text)
            
            # Add rating category to evaluation notes
            overall_score = eval_data.get('overall_score', 5.0)
            rating_category = self._get_rating_category(overall_score)
            eval_data['evaluation_notes'] = f"{rating_category}: {eval_data.get('evaluation_notes', '')}"
            eval_data['judge_model'] = judge_model
            
            return LLMEvaluation(**eval_data)
            
        except Exception as e:
            print(f"Single judge evaluation error from {judge_model}: {e}")
            # Return fallback evaluation
            return LLMEvaluation(
                faithfulness = 5.0,
                groundedness = 5.0,
                factual_consistency = 5.0,
                relevance = 5.0,
                completeness = 5.0,
                fluency = 6.0,
                overall_score = 5.2,
                evaluation_notes = f"Poor / Weak: Evaluation failed: {str(e)}",
                judge_model = judge_model
            )
    
    def _parse_evaluation_response(self, text: str) -> Dict[str, Any]:
        """Parse the evaluation response from the judge LLM"""
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                eval_data = json.loads(json_match.group())
                
                # Fix the specific phi3 typo and other common typos
                if 'completenece' in eval_data:
                    eval_data['completeness'] = eval_data.pop('completenece')
                if 'completeneity' in eval_data:
                    eval_data['completeness'] = eval_data.pop('completeneity')
                return eval_data
            except json.JSONDecodeError:
                pass
        
        # Fallback evaluation
        return {
            "faithfulness": 5.0,
            "groundedness": 5.0,
            "factual_consistency": 5.0,
            "relevance": 5.0,
            "completeness": 5.0,
            "fluency": 6.0,
            "overall_score": 5.2,
            "evaluation_notes": "Average: Automatic fallback evaluation"
        }