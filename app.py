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
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Initialize Ollama LLMs - Multiple judge models for diverse evaluation
llm = Ollama(model="gemma3:270m", request_timeout=120.0)

# Multiple judge models for comprehensive evaluation
judge_models = {
    "llama3.1_8b": Ollama(model="llama3.1:8b", request_timeout=180.0),
    "mixtral_8x7b": Ollama(model="mixtral:8x7b", request_timeout=180.0),
    "qwen2_7b": Ollama(model="qwen2:7b", request_timeout=180.0),
    "gemma2_9b": Ollama(model="gemma2:9b", request_timeout=180.0)
}

# Configuration paths
STORAGE_DIR = "./storage"
METRICS_DIR = "./metrics"

# Ensure directories exist
os.makedirs(METRICS_DIR, exist_ok=True)

@dataclass
class JudgeEvaluation:
    """Individual evaluation from a specific judge model"""
    judge_model: str
    faithfulness: float  # 0-10
    groundedness: float  # 0-10
    factual_consistency: float  # 0-10
    relevance: float  # 0-10
    completeness: float  # 0-10
    fluency: float  # 0-10
    overall_score: float  # 0-10
    evaluation_notes: str
    confidence: float  # 0-1: Judge's confidence in evaluation
    rating_category: str  # Excellent, Good, Fair, Average, Poor/Weak

@dataclass
class LLMEvaluation:
    """Complete evaluation from multiple judges without averaging"""
    judges: Dict[str, JudgeEvaluation]  # Detailed evaluations from each judge model
    evaluation_summary: str  # Human-readable summary of all evaluations

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

class MultiJudgeEvaluator:
    """Multi-LLM evaluation system with detailed individual scoring"""
    
    def __init__(self, judge_models: Dict[str, Ollama]):
        self.judge_models = judge_models
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.evaluation_prompt_template = """As an expert AI response evaluator, provide a detailed analysis:

QUERY: {query}

CONTEXT: {context}

RESPONSE: {response}

Evaluate each criterion on 0.0-10.0 scale with detailed explanations:

1. FAITHFULNESS (0-10): How well does the response rely on the provided context without hallucination or fabrication?
2. GROUNDEDNESS (0-10): Can every piece of information be directly traced back to the context?
3. FACTUAL CONSISTENCY (0-10): How factually accurate is the response compared to the context provided?
4. RELEVANCE (0-10): How well does the response address the specific query and user intent?
5. COMPLETENESS (0-10): Does the response cover all important aspects and answer the query thoroughly?
6. FLUENCY (0-10): Is the response natural, coherent, well-structured, and easy to understand?

Calculate an overall_score (0.0-10.0) considering all criteria.

Also provide:
- confidence (0.0-1.0): Your confidence in this evaluation
- detailed_notes: Comprehensive explanation of your scores

Respond ONLY with valid JSON:
{{
  "faithfulness": 8.5,
  "groundedness": 9.0,
  "factual_consistency": 9.2,
  "relevance": 8.8,
  "completeness": 7.5,
  "fluency": 9.5,
  "overall_score": 8.7,
  "confidence": 0.9,
  "detailed_notes": "Comprehensive explanation of each score..."
}}"""
    
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
    
    def _parse_evaluation_response(self, text: str, model_name: str) -> Dict[str, Any]:
        """Parse evaluation response from judge model"""
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group())
                # Validate required fields
                required_fields = ['faithfulness', 'groundedness', 'factual_consistency', 
                                 'relevance', 'completeness', 'fluency', 'overall_score']
                if all(field in eval_data for field in required_fields):
                    return eval_data
        except Exception as e:
            print(f"Parse error from {model_name}: {e}")
        
        # Fallback evaluation
        return {
            "faithfulness": 5.0,
            "groundedness": 5.0,
            "factual_consistency": 5.0,
            "relevance": 5.0,
            "completeness": 5.0,
            "fluency": 6.0,
            "overall_score": 5.2,
            "confidence": 0.5,
            "detailed_notes": f"Fallback evaluation from {model_name} - parsing failed"
        }
    
    async def evaluate_with_judge(self, judge_name: str, judge_llm: Ollama, 
                                 query: str, response: str, context: str) -> JudgeEvaluation:
        """Evaluate response with a specific judge model"""
        try:
            prompt = self.evaluation_prompt_template.format(
                query=query,
                context=context[:1500],
                response=response
            )
            
            # Run evaluation in thread pool
            eval_response = await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: judge_llm.complete(prompt)
            )
            
            eval_data = self._parse_evaluation_response(eval_response.text, judge_name)
            overall_score = eval_data.get('overall_score', 5.2)
            
            return JudgeEvaluation(
                judge_model=judge_name,
                faithfulness=eval_data.get('faithfulness', 5.0),
                groundedness=eval_data.get('groundedness', 5.0),
                factual_consistency=eval_data.get('factual_consistency', 5.0),
                relevance=eval_data.get('relevance', 5.0),
                completeness=eval_data.get('completeness', 5.0),
                fluency=eval_data.get('fluency', 6.0),
                overall_score=overall_score,
                evaluation_notes=eval_data.get('detailed_notes', 'No detailed notes provided'),
                confidence=eval_data.get('confidence', 0.5),
                rating_category=self._get_rating_category(overall_score)
            )
            
        except Exception as e:
            print(f"Evaluation error from {judge_name}: {e}")
            return JudgeEvaluation(
                judge_model=judge_name,
                faithfulness=5.0,
                groundedness=5.0,
                factual_consistency=5.0,
                relevance=5.0,
                completeness=5.0,
                fluency=6.0,
                overall_score=5.2,
                evaluation_notes=f"Evaluation error: {str(e)}",
                confidence=0.3,
                rating_category="Poor / Weak"
            )
    
    def _generate_evaluation_summary(self, judge_evaluations: List[JudgeEvaluation]) -> str:
        """Generate a human-readable summary of all evaluations"""
        if not judge_evaluations:
            return "No evaluations available"
        
        summary_lines = [
            "=== MULTI-JUDGE EVALUATION SUMMARY ===",
            f"Total Judges: {len(judge_evaluations)}",
            ""
        ]
        
        # Add each judge's overall score and rating
        for judge_eval in judge_evaluations:
            summary_lines.append(
                f"🧠 {judge_eval.judge_model}: {judge_eval.overall_score:.1f}/10.0 "
                f"({judge_eval.rating_category}) - Confidence: {judge_eval.confidence:.1f}"
            )
        
        summary_lines.extend([
            "",
            "RATING SCALE:",
            "9.0 – 10.0: Excellent",
            "8.0 – 8.9: Good", 
            "6.5 – 7.9: Fair",
            "5.0 – 6.4: Average",
            "< 5.0: Poor / Weak",
            "======================================"
        ])
        
        return "\n".join(summary_lines)
    
    async def evaluate_response(self, query: str, response: str, context: str) -> LLMEvaluation:
        """Evaluate response using multiple judge models in parallel"""
        evaluation_tasks = []
        
        # Create evaluation tasks for all judge models
        for judge_name, judge_llm in self.judge_models.items():
            task = self.evaluate_with_judge(judge_name, judge_llm, query, response, context)
            evaluation_tasks.append(task)
        
        # Run all evaluations in parallel
        judge_evaluations = await asyncio.gather(*evaluation_tasks)
        
        # Convert to dictionary
        judges_dict = {e.judge_model: e for e in judge_evaluations}
        
        # Generate comprehensive summary
        evaluation_summary = self._generate_evaluation_summary(judge_evaluations)
        
        return LLMEvaluation(
            judges=judges_dict,
            evaluation_summary=evaluation_summary
        )

class MetricsCollector:
    """Collects and manages LLM performance metrics with detailed multi-judge evaluation"""
    
    def __init__(self, metrics_dir: str = METRICS_DIR, judge_models=None):
        self.metrics_dir = metrics_dir
        self.current_session_metrics: List[LLMMetrics] = []
        self.multi_judge_evaluator = MultiJudgeEvaluator(judge_models) if judge_models else None
    
    async def record_metrics(self, query: str, response: str, context: str, 
                           response_time: float, token_count: int, 
                           model: str, session_id: str) -> LLMMetrics:
        """Record metrics with detailed multi-judge evaluation"""
        tokens_per_second = token_count / response_time if response_time > 0 else 0
        
        evaluation = None
        if self.multi_judge_evaluator:
            evaluation = await self.multi_judge_evaluator.evaluate_response(
                query, response, context
            )
        
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
        """Save detailed metrics to JSON file"""
        if not filename:
            filename = f"detailed_judge_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.metrics_dir, filename)
        
        metrics_dicts = []
        for metric in self.current_session_metrics:
            metric_data = {
                "timestamp": metric.timestamp,
                "query": metric.query,
                "response": metric.response[:800] + "..." if len(metric.response) > 800 else metric.response,
                "context_preview": metric.context[:400] + "..." if len(metric.context) > 400 else metric.context,
                "response_time": round(metric.response_time, 2),
                "token_count": metric.token_count,
                "tokens_per_second": round(metric.tokens_per_second, 2),
                "model": metric.model,
                "session_id": metric.session_id
            }
            
            if metric.evaluation:
                # Store detailed evaluations from each judge
                judges_data = {}
                for judge_name, judge_eval in metric.evaluation.judges.items():
                    judges_data[judge_name] = {
                        "scores": {
                            "faithfulness": judge_eval.faithfulness,
                            "groundedness": judge_eval.groundedness,
                            "factual_consistency": judge_eval.factual_consistency,
                            "relevance": judge_eval.relevance,
                            "completeness": judge_eval.completeness,
                            "fluency": judge_eval.fluency,
                            "overall_score": judge_eval.overall_score
                        },
                        "rating_category": judge_eval.rating_category,
                        "confidence": judge_eval.confidence,
                        "evaluation_notes": judge_eval.evaluation_notes
                    }
                
                metric_data["multi_judge_evaluation"] = {
                    "evaluation_summary": metric.evaluation.evaluation_summary,
                    "judges": judges_data,
                    "total_judges": len(metric.evaluation.judges),
                    "evaluation_method": "Detailed multi-judge evaluation without averaging"
                }
            
            metrics_dicts.append(metric_data)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dicts, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed multi-judge metrics saved to {filepath}")
        return filepath
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary without averaging scores"""
        if not self.current_session_metrics:
            return {}
        
        response_times = [m.response_time for m in self.current_session_metrics]
        token_counts = [m.token_count for m in self.current_session_metrics]
        
        evaluations = [m.evaluation for m in self.current_session_metrics if m.evaluation]
        
        summary = {
            "total_interactions": len(self.current_session_metrics),
            "evaluated_responses": len(evaluations),
            "avg_response_time": round(sum(response_times) / len(response_times), 2),
            "total_tokens_generated": sum(token_counts),
            "judge_models": list(judge_models.keys()) if evaluations else []
        }
        
        if evaluations:
            # Count rating categories across all judges and responses
            rating_counts = {"Excellent": 0, "Good": 0, "Fair": 0, "Average": 0, "Poor / Weak": 0}
            total_judge_evaluations = 0
            
            for eval_obj in evaluations:
                for judge_eval in eval_obj.judges.values():
                    rating_counts[judge_eval.rating_category] += 1
                    total_judge_evaluations += 1
            
            summary["rating_distribution"] = {
                category: f"{count} evaluations ({count/total_judge_evaluations*100:.1f}%)"
                for category, count in rating_counts.items()
            }
            summary["total_judge_evaluations"] = total_judge_evaluations
        
        return summary
    
    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        summary = self.get_session_summary()
        if not summary:
            return "No metrics collected yet."
        
        report = [
            "=== DETAILED MULTI-JUDGE EVALUATION REPORT ===",
            f"Total Interactions: {summary['total_interactions']}",
            f"Evaluated Responses: {summary['evaluated_responses']}",
            f"Judge Models: {', '.join(summary['judge_models'])}",
            f"Total Judge Evaluations: {summary.get('total_judge_evaluations', 0)}",
            f"Average Response Time: {summary['avg_response_time']}s",
            f"Total Tokens Generated: {summary['total_tokens_generated']}",
        ]
        
        if 'rating_distribution' in summary:
            report.extend([
                "",
                "=== RATING DISTRIBUTION ACROSS ALL JUDGES ===",
                f"Excellent (9.0-10.0): {summary['rating_distribution']['Excellent']}",
                f"Good (8.0-8.9): {summary['rating_distribution']['Good']}",
                f"Fair (6.5-7.9): {summary['rating_distribution']['Fair']}",
                f"Average (5.0-6.4): {summary['rating_distribution']['Average']}",
                f"Poor / Weak (<5.0): {summary['rating_distribution']['Poor / Weak']}",
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
            "NOTE: Each judge provides independent evaluation without averaging",
            f"Main Model: {llm.model}",
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
    """Initializes chat session with detailed multi-judge evaluation"""
    index = load_index()
    query_engine = create_query_engine(index)
    chat_memory = ChatMemoryBuffer.from_defaults(token_limit=1500, llm=llm)
    metrics_collector = MetricsCollector(judge_models=judge_models)
    
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
    """Handle chat session end - save detailed metrics"""
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
    """Processes incoming messages with detailed multi-judge evaluation"""
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
        await cl.Message(content=f"Detailed multi-judge evaluation saved to: {filename}").send()
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
            
            # Create detailed evaluation message
            evaluation = [
                "=== DETAILED MULTI-JUDGE EVALUATION ===",
                f"Query: {last_metric.query[:100]}...",
                f"Response Preview: {last_metric.response[:200]}...",
                "",
                "SUMMARY:",
                eval_data.evaluation_summary,
                "",
                "=== DETAILED SCORES BY JUDGE ==="
            ]
            
            # Add detailed scores for each judge
            for judge_name, judge_eval in eval_data.judges.items():
                evaluation.extend([
                    f"",
                    f"🧠 JUDGE: {judge_name}",
                    f"   Overall: {judge_eval.overall_score:.1f}/10.0 ({judge_eval.rating_category})",
                    f"   Confidence: {judge_eval.confidence:.1f}",
                    f"   Faithfulness: {judge_eval.faithfulness:.1f}/10.0",
                    f"   Groundedness: {judge_eval.groundedness:.1f}/10.0",
                    f"   Factual Consistency: {judge_eval.factual_consistency:.1f}/10.0",
                    f"   Relevance: {judge_eval.relevance:.1f}/10.0",
                    f"   Completeness: {judge_eval.completeness:.1f}/10.0",
                    f"   Fluency: {judge_eval.fluency:.1f}/10.0",
                    f"   Notes: {judge_eval.evaluation_notes[:150]}...",
                    f"   ─────────────────────────"
                ])
            
            evaluation.extend([
                "",
                "RATING SCALE:",
                "9.0 – 10.0: Excellent",
                "8.0 – 8.9: Good", 
                "6.5 – 7.9: Fair",
                "5.0 – 6.4: Average",
                "< 5.0: Poor / Weak",
                "=========================================="
            ])
            
            # Send as multiple messages if too long
            full_evaluation = "\n".join(evaluation)
            if len(full_evaluation) > 4000:
                # Send summary first
                await cl.Message(content=eval_data.evaluation_summary).send()
                # Send detailed scores
                detailed_part = "\n".join(evaluation[8:])  # Skip the header
                await cl.Message(content=f"```\n{detailed_part}\n```").send()
            else:
                await cl.Message(content=f"```\n{full_evaluation}\n```").send()
                
        else:
            await cl.Message(content="No multi-judge evaluation available for last response.").send()
    else:
        await cl.Message(content="No responses to evaluate yet.").send()

@cl.action_callback("show_judge_details")
async def on_action_show_judge_details(action: cl.Action):
    """Action to show detailed information about each judge model"""
    judge_info = [
        "=== JUDGE MODELS INFORMATION ===",
        "",
        "🧠 llama3.1_8b: Meta's Llama 3.1 8B model - Balanced reasoning and accuracy",
        "   Strengths: General knowledge, reasoning, balanced evaluation",
        "",
        "🧠 mixtral_8x7b: Mixtral 8x7B MoE model - Expert-level evaluation",
        "   Strengths: Technical accuracy, comprehensive analysis",
        "",
        "🧠 qwen2_7b: Qwen2 7B model - Multilingual capability",
        "   Strengths: Language understanding, contextual analysis", 
        "",
        "🧠 gemma2_9b: Google's Gemma2 9B model - Modern evaluation",
        "   Strengths: Latest knowledge, nuanced understanding",
        "",
        "EVALUATION CRITERIA:",
        "• Faithfulness: Reliance on context without hallucination",
        "• Groundedness: Information traceable to context",
        "• Factual Consistency: Accuracy compared to context", 
        "• Relevance: Addresses the specific query",
        "• Completeness: Covers all important aspects",
        "• Fluency: Natural, coherent language",
        "",
        "SCALE: 0.0 - 10.0 (No averaging between judges)",
        "=========================================="
    ]
    
    await cl.Message(content="\n".join(judge_info)).send()

if __name__ == "__main__":
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    print("=== DETAILED MULTI-JUDGE EVALUATION SYSTEM ===")
    print(f"Main Model: {llm.model}")
    print(f"Judge Models: {', '.join(judge_models.keys())}")
    print("Evaluation Scale: 0.0 - 10.0")
    print("Rating Categories: Excellent, Good, Fair, Average, Poor/Weak")
    print("Available actions: show_metrics, save_metrics, evaluate_last, show_judge_details")
    print("========================================")