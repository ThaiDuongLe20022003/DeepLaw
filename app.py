import os
import chainlit as cl
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
import asyncio

# Configure embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Initialize Ollama LLMs
llm = Ollama(model="gemma3:270m", request_timeout=120.0)
judge_llm = Ollama(model="llama3.1:8b", request_timeout=180.0)

# Configuration paths
STORAGE_DIR = "./storage"
METRICS_DIR = "./metrics"

# Ensure directories exist
os.makedirs(METRICS_DIR, exist_ok=True)

@dataclass
class LLMEvaluation:
    """Comprehensive LLM evaluation metrics using LLM-as-a-judge"""
    faithfulness: float  # 0-10: Does the answer rely on the provided context?
    groundedness: float  # 0-10: Can information be traced back to context?
    factual_consistency: float  # 0-10: Factual alignment with context
    relevance: float  # 0-10: Addresses the actual query
    completeness: float  # 0-10: Covers all important aspects
    fluency: float  # 0-10: Natural, coherent, and well-written
    overall_score: float  # 0-10: Overall quality score
    evaluation_notes: str  # Detailed explanation from judge

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
    evaluation: Optional[LLMEvaluation] = None

class LLMJudgeEvaluator:
    """LLM-as-a-judge evaluation system using a more powerful model"""
    
    def __init__(self, judge_llm):
        self.judge_llm = judge_llm
        self.evaluation_prompt = """You are an expert evaluator of AI responses. Please evaluate the following response based on the given context and query.

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
    
    async def evaluate_response(self, query: str, response: str, context: str) -> LLMEvaluation:
        """Evaluate response using LLM-as-a-judge approach"""
        try:
            prompt = self.evaluation_prompt.format(
                query=query,
                context=context[:2000],
                response=response
            )
            
            evaluation_response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.judge_llm.complete(prompt)
            )
            
            eval_text = evaluation_response.text.strip()
            eval_data = self._parse_evaluation_response(eval_text)
            
            # Add rating category to evaluation notes
            overall_score = eval_data.get('overall_score', 5.0)
            rating_category = self._get_rating_category(overall_score)
            eval_data['evaluation_notes'] = f"{rating_category}: {eval_data.get('evaluation_notes', '')}"
            
            return LLMEvaluation(**eval_data)
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return LLMEvaluation(
                faithfulness=5.0,
                groundedness=5.0,
                factual_consistency=5.0,
                relevance=5.0,
                completeness=5.0,
                fluency=6.0,
                overall_score=5.2,
                evaluation_notes=f"Poor / Weak: Evaluation failed: {str(e)}"
            )
    
    def _parse_evaluation_response(self, text: str) -> Dict[str, Any]:
        """Parse the evaluation response from the judge LLM"""
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
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

class MetricsCollector:
    """Collects and manages LLM performance metrics with LLM-as-a-judge evaluation"""
    
    def __init__(self, metrics_dir: str = METRICS_DIR, judge_llm=None):
        self.metrics_dir = metrics_dir
        self.current_session_metrics: List[LLMMetrics] = []
        self.judge_evaluator = LLMJudgeEvaluator(judge_llm) if judge_llm else None
    
    async def record_metrics(self, query: str, response: str, context: str, 
                           response_time: float, token_count: int, 
                           model: str, session_id: str) -> LLMMetrics:
        """Record metrics with LLM-as-a-judge evaluation"""
        tokens_per_second = token_count / response_time if response_time > 0 else 0
        
        evaluation = None
        if self.judge_evaluator:
            evaluation = await self.judge_evaluator.evaluate_response(query, response, context)
        
        metrics = LLMMetrics(
            timestamp=datetime.now().isoformat(),
            query=query,
            response=response,
            context=context[:1000],
            response_time=response_time,
            token_count=token_count,
            tokens_per_second=tokens_per_second,
            model=model,
            session_id=session_id,
            evaluation=evaluation
        )
        
        self.current_session_metrics.append(metrics)
        return metrics
    
    def save_metrics_to_file(self, filename: str = None):
        """Save metrics to JSON file"""
        if not filename:
            filename = f"llm_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.metrics_dir, filename)
        
        metrics_dicts = []
        for metric in self.current_session_metrics:
            metric_data = {
                "timestamp": metric.timestamp,
                "query": metric.query,
                "response": metric.response[:1000] + "..." if len(metric.response) > 1000 else metric.response,
                "context_preview": metric.context[:500] + "..." if len(metric.context) > 500 else metric.context,
                "response_time": round(metric.response_time, 2),
                "token_count": metric.token_count,
                "tokens_per_second": round(metric.tokens_per_second, 2),
                "model": metric.model,
                "session_id": metric.session_id
            }
            
            if metric.evaluation:
                metric_data["evaluation"] = {
                    "faithfulness": round(metric.evaluation.faithfulness, 1),
                    "groundedness": round(metric.evaluation.groundedness, 1),
                    "factual_consistency": round(metric.evaluation.factual_consistency, 1),
                    "relevance": round(metric.evaluation.relevance, 1),
                    "completeness": round(metric.evaluation.completeness, 1),
                    "fluency": round(metric.evaluation.fluency, 1),
                    "overall_score": round(metric.evaluation.overall_score, 1),
                    "evaluation_notes": metric.evaluation.evaluation_notes,
                    "evaluation_method": "LLM-as-a-judge (0.0-10.0 scale)"
                }
            
            metrics_dicts.append(metric_data)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dicts, f, indent=2, ensure_ascii=False)
        
        print(f"Comprehensive metrics saved to {filepath}")
        return filepath
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics"""
        if not self.current_session_metrics:
            return {}
        
        response_times = [m.response_time for m in self.current_session_metrics]
        token_counts = [m.token_count for m in self.current_session_metrics]
        tokens_per_second = [m.tokens_per_second for m in self.current_session_metrics]
        
        evaluations = [m.evaluation for m in self.current_session_metrics if m.evaluation]
        
        summary = {
            "total_interactions": len(self.current_session_metrics),
            "avg_response_time": round(sum(response_times) / len(response_times), 2),
            "min_response_time": round(min(response_times), 2),
            "max_response_time": round(max(response_times), 2),
            "avg_tokens_per_second": round(sum(tokens_per_second) / len(tokens_per_second), 2),
            "total_tokens_generated": sum(token_counts),
            "avg_tokens_per_response": round(sum(token_counts) / len(token_counts), 1),
            "evaluated_responses": len(evaluations)
        }
        
        if evaluations:
            summary.update({
                "avg_faithfulness": round(sum(e.faithfulness for e in evaluations) / len(evaluations), 1),
                "avg_groundedness": round(sum(e.groundedness for e in evaluations) / len(evaluations), 1),
                "avg_factual_consistency": round(sum(e.factual_consistency for e in evaluations) / len(evaluations), 1),
                "avg_relevance": round(sum(e.relevance for e in evaluations) / len(evaluations), 1),
                "avg_completeness": round(sum(e.completeness for e in evaluations) / len(evaluations), 1),
                "avg_fluency": round(sum(e.fluency for e in evaluations) / len(evaluations), 1),
                "avg_overall_score": round(sum(e.overall_score for e in evaluations) / len(evaluations), 1),
            })
            
            # Count rating categories
            rating_counts = {"Excellent": 0, "Good": 0, "Fair": 0, "Average": 0, "Poor / Weak": 0}
            for eval_obj in evaluations:
                score = eval_obj.overall_score
                if score >= 9.0:
                    rating_counts["Excellent"] += 1
                elif score >= 8.0:
                    rating_counts["Good"] += 1
                elif score >= 6.5:
                    rating_counts["Fair"] += 1
                elif score >= 5.0:
                    rating_counts["Average"] += 1
                else:
                    rating_counts["Poor / Weak"] += 1
            
            summary["rating_distribution"] = rating_counts
        
        return summary
    
    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        summary = self.get_session_summary()
        if not summary:
            return "No metrics collected yet."
        
        report = [
            "=== LLM-AS-A-JUDGE EVALUATION REPORT ===",
            f"Session ID: {self.current_session_metrics[0].session_id if self.current_session_metrics else 'N/A'}",
            f"Total Interactions: {summary['total_interactions']}",
            f"Evaluated Responses: {summary['evaluated_responses']}",
            f"Average Response Time: {summary['avg_response_time']}s",
            f"Total Tokens Generated: {summary['total_tokens_generated']}",
            f"Average Throughput: {summary['avg_tokens_per_second']} tokens/s",
        ]
        
        if 'avg_overall_score' in summary:
            report.extend([
                "",
                "=== QUALITY EVALUATION (0.0-10.0 scale) ===",
                f"Overall Quality: {summary['avg_overall_score']}/10.0",
                f"Faithfulness: {summary['avg_faithfulness']}/10.0 (reliance on context)",
                f"Groundedness: {summary['avg_groundedness']}/10.0 (traceability to context)",
                f"Factual Consistency: {summary['avg_factual_consistency']}/10.0 (accuracy vs context)",
                f"Relevance: {summary['avg_relevance']}/10.0 (addresses query)",
                f"Completeness: {summary['avg_completeness']}/10.0 (covers all aspects)",
                f"Fluency: {summary['avg_fluency']}/10.0 (natural language)",
            ])
            
            if 'rating_distribution' in summary:
                report.extend([
                    "",
                "=== RATING DISTRIBUTION ===",
                f"Excellent (9.0-10.0): {summary['rating_distribution']['Excellent']} responses",
                f"Good (8.0-8.9): {summary['rating_distribution']['Good']} responses",
                f"Fair (6.5-7.9): {summary['rating_distribution']['Fair']} responses",
                f"Average (5.0-6.4): {summary['rating_distribution']['Average']} responses",
                f"Poor / Weak (<5.0): {summary['rating_distribution']['Poor / Weak']} responses",
                ])
        
        report.extend([
            "",
            "RATING SCALE:",
            "9.0 – 10.0: Excellent",
            "8.0 – 8.9: Good", 
            "6.5 – 7.9: Fair",
            "5.0 – 6.4: Average",
            "< 5.0: Poor / Weak",
            "",
            f"Model: {self.current_session_metrics[0].model if self.current_session_metrics else 'N/A'}",
            f"Judge Model: {judge_llm.model}",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "========================================="
        ])
        
        return "\n".join(report)

def load_index():
    """Loads the pre-built vector index from storage"""
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    return load_index_from_storage(storage_context)

def create_query_engine(index):
    """Creates an optimized query engine"""
    base_retriever = index.as_retriever(similarity_top_k=6)
    
    retriever = AutoMergingRetriever(
        base_retriever, 
        storage_context=index.storage_context, 
        verbose=False
    )
    
    reranker = SentenceTransformerRerank(
        top_n=3, 
        model="BAAI/bge-reranker-base"
    )
    
    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker],
        streaming=True,
        llm=llm
    )

def count_tokens(text: str) -> int:
    """Simple token counter"""
    return len(text.split())

def extract_context_from_response(response) -> str:
    """Extract context information from response object"""
    try:
        if hasattr(response, 'source_nodes') and response.source_nodes:
            context_texts = []
            for node in response.source_nodes[:3]:
                if hasattr(node, 'node') and hasattr(node.node, 'text'):
                    context_texts.append(node.node.text)
            return "\n".join(context_texts)
        return "Context not available"
    except:
        return "Context extraction failed"

@cl.on_chat_start
async def init_chat():
    """Initializes chat session with LLM-as-a-judge evaluation"""
    index = load_index()
    query_engine = create_query_engine(index)
    chat_memory = ChatMemoryBuffer.from_defaults(token_limit=1500, llm=llm)
    metrics_collector = MetricsCollector(judge_llm=judge_llm)
    
    cl.user_session.set("query_engine", query_engine)
    cl.user_session.set("chat_memory", chat_memory)
    cl.user_session.set("metrics_collector", metrics_collector)
    
    chat_memory.put("System: Hello! How can I help you today?")

@cl.on_chat_resume
async def resume_chat():
    """Handles chat session resumption"""
    await init_chat()

@cl.on_chat_end
async def on_chat_end():
    """Handle chat session end - save comprehensive metrics"""
    metrics_collector = cl.user_session.get("metrics_collector")
    if metrics_collector and metrics_collector.current_session_metrics:
        metrics_collector.save_metrics_to_file()
        print(metrics_collector.generate_report())

@cl.password_auth_callback
def authenticate(username: str, password: str):
    """Simple password-based authentication"""
    valid_users = {"admin": "secret"}
    if username in valid_users and valid_users[username] == password:
        return cl.User(identifier=username)
    return None

@cl.on_message
async def handle_message(message: cl.Message):
    """Processes incoming messages with LLM-as-a-judge evaluation"""
    query_engine = cl.user_session.get("query_engine")
    chat_memory = cl.user_session.get("chat_memory")
    metrics_collector = cl.user_session.get("metrics_collector")
    
    try:
        history_text = chat_memory.get()
        full_prompt = f"{history_text}\nUser: {message.content}"
    except Exception as e:
        full_prompt = f"User: {message.content}"
    
    reply = cl.Message(content="")
    await reply.send()
    
    start_time = time.time()
    response = await asyncio.get_event_loop().run_in_executor(
        None, lambda: query_engine.query(full_prompt)
    )
    
    full_response = ""
    token_count = 0
    for token in response.response_gen:
        full_response += token
        token_count = count_tokens(full_response)
        await reply.stream_token(token)
    await reply.update()
    
    response_time = time.time() - start_time
    
    context = extract_context_from_response(response)
    
    if metrics_collector:
        session_id = cl.user_session.get("id", "unknown_session")
        await metrics_collector.record_metrics(
            query=message.content,
            response=full_response,
            context=context,
            response_time=response_time,
            token_count=token_count,
            model="gemma3:270m",
            session_id=session_id
        )
    
    chat_memory.put(f"User: {message.content}")
    chat_memory.put(f"Assistant: {full_response}")

@cl.action_callback("show_metrics")
async def on_action_show_metrics(action: cl.Action):
    """Action to show current evaluation metrics"""
    metrics_collector = cl.user_session.get("metrics_collector")
    if metrics_collector:
        report = metrics_collector.generate_report()
        await cl.Message(content=f"```\n{report}\n```").send()
    else:
        await cl.Message(content="No evaluation metrics available yet.").send()

@cl.action_callback("save_metrics")
async def on_action_save_metrics(action: cl.Action):
    """Action to save metrics to file"""
    metrics_collector = cl.user_session.get("metrics_collector")
    if metrics_collector and metrics_collector.current_session_metrics:
        filename = metrics_collector.save_metrics_to_file()
        await cl.Message(content=f"Comprehensive evaluation saved to: {filename}").send()
    else:
        await cl.Message(content="No metrics to save.").send()

@cl.action_callback("evaluate_last")
async def on_action_evaluate_last(action: cl.Action):
    """Action to show detailed evaluation of last response"""
    metrics_collector = cl.user_session.get("metrics_collector")
    if metrics_collector and metrics_collector.current_session_metrics:
        last_metric = metrics_collector.current_session_metrics[-1]
        if last_metric.evaluation:
            eval_data = last_metric.evaluation
            evaluation = [
                "=== DETAILED LLM-AS-A-JUDGE EVALUATION ===",
                f"Query: {last_metric.query[:100]}...",
                "",
                "SCORES (0.0-10.0 scale):",
                f"Faithfulness: {eval_data.faithfulness:.1f}/10.0 - Relies on context without hallucination",
                f"Groundedness: {eval_data.groundedness:.1f}/10.0 - Information traceable to context",
                f"Factual Consistency: {eval_data.factual_consistency:.1f}/10.0 - Accurate vs context",
                f"Relevance: {eval_data.relevance:.1f}/10.0 - Addresses the query",
                f"Completeness: {eval_data.completeness:.1f}/10.0 - Covers important aspects",
                f"Fluency: {eval_data.fluency:.1f}/10.0 - Natural and coherent",
                f"Overall Score: {eval_data.overall_score:.1f}/10.0 - Comprehensive quality",
                "",
                "EVALUATION NOTES:",
                eval_data.evaluation_notes,
                "",
                "RATING SCALE:",
                "9.0 – 10.0: Excellent",
                "8.0 – 8.9: Good", 
                "6.5 – 7.9: Fair",
                "5.0 – 6.4: Average",
                "< 5.0: Poor / Weak",
                "=========================================="
            ]
            await cl.Message(content="\n".join(evaluation)).send()
        else:
            await cl.Message(content="No LLM evaluation available for last response.").send()
    else:
        await cl.Message(content="No responses to evaluate yet.").send()

if __name__ == "__main__":
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    print("=== LLM-AS-A-JUDGE EVALUATION SYSTEM ===")
    print(f"Main Model: {llm.model}")
    print(f"Judge Model: {judge_llm.model}")
    print("Evaluation Scale: 0.0 - 10.0")
    print("Rating Categories:")
    print("9.0 – 10.0: Excellent")
    print("8.0 – 8.9: Good") 
    print("6.5 – 7.9: Fair")
    print("5.0 – 6.4: Average")
    print("< 5.0: Poor / Weak")
    print("Available actions: show_metrics, save_metrics, evaluate_last")
    print("========================================")