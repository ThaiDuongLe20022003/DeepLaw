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
from typing import Dict, List, Any
from dataclasses import dataclass
import re
import asyncio
from collections import defaultdict
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss


# Configure embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name = "BAAI/bge-small-en-v1.5")

# Initialize main LLM
llm = Ollama(model = "gemma3:270m", request_timeout = 120.0)

# Initialize multiple judge LLMs
judge_llms = {
    "gpt-oss:20b": Ollama(model = "gpt-oss:20b", request_timeout = 3600.0),
    "llama3.1:8b": Ollama(model = "llama3.1:8b", request_timeout = 3600.0),
    "mistral:7b": Ollama(model = "mistral:7b", request_timeout = 3600.0),
    "command-r7b:7b": Ollama(model = "command-r7b:7b", request_timeout = 3600.0),
    "phi3:3.8b": Ollama(model = "phi3:3.8b", request_timeout = 3600.0),
    "cogito:3b": Ollama(model = "cogito:3b", request_timeout = 3600.0),
    "deepseek-r1:1.5b": Ollama(model = "deepseek-r1:1.5b", request_timeout = 3600.0),
    "falcon3:1b": Ollama(model = "falcon3:1b", request_timeout = 3600.0),
    "qwen3:0.6b": Ollama(model = "qwen3:0.6b", request_timeout = 3600.0),
    #"gemma3:270m": Ollama(model="gemma3:270m", request_timeout=180.0),
}

# Configuration paths
STORAGE_DIR = "./storage"
METRICS_DIR = "./metrics"

# Ensure directories exist
os.makedirs(METRICS_DIR, exist_ok = True)

@dataclass
class JudgeEvaluation:
    """Evaluation from a single judge LLM"""
    judge_model: str
    faithfulness: float
    groundedness: float
    factual_consistency: float
    relevance: float
    completeness: float
    fluency: float
    overall_score: float
    evaluation_notes: str

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
    evaluations: List[JudgeEvaluation]  # Multiple evaluations from different judges

class MultiJudgeEvaluator:
    """Multi-LLM evaluation system using multiple judge models"""
    
    def __init__(self, judge_llms):
        self.judge_llms = judge_llms
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
    
    async def _evaluate_with_judge(self, judge_name: str, judge_llm: Ollama, query: str, response: str, context: str) -> JudgeEvaluation:
        """Evaluate response with a single judge LLM"""
        try:
            prompt = self.evaluation_prompt.format(
                query = query,
                context = context[:2000],
                response = response
            )
            
            evaluation_response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: judge_llm.complete(prompt)
            )
            
            eval_text = evaluation_response.text.strip()
            eval_data = self._parse_evaluation_response(eval_text)
            
            # Add rating category to evaluation notes
            overall_score = eval_data.get('overall_score', 5.0)
            rating_category = self._get_rating_category(overall_score)
            eval_data['evaluation_notes'] = f"{rating_category}: {eval_data.get('evaluation_notes', '')}"
            
            return JudgeEvaluation(judge_model = judge_name, **eval_data)
            
        except Exception as e:
            print(f"Evaluation error with {judge_name}: {e}")
            return JudgeEvaluation(
                judge_model = judge_name,
                faithfulness = 5.0,
                groundedness=5.0,
                factual_consistency=5.0,
                relevance=5.0,
                completeness=5.0,
                fluency=6.0,
                overall_score=5.2,
                evaluation_notes=f"Poor / Weak: Evaluation failed: {str(e)}"
            )
    
    async def evaluate_response(self, query: str, response: str, context: str) -> List[JudgeEvaluation]:
        """Evaluate response using multiple judge LLMs in parallel"""
        evaluation_tasks = []
        
        for judge_name, judge_llm in self.judge_llms.items():
            task = self._evaluate_with_judge(judge_name, judge_llm, query, response, context)
            evaluation_tasks.append(task)
        
        # Run all evaluations in parallel
        evaluations = await asyncio.gather(*evaluation_tasks)
        return evaluations
    
    def _parse_evaluation_response(self, text: str) -> Dict[str, Any]:
        """Parse the evaluation response from a judge LLM"""
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
    """Collects and manages LLM performance metrics with multi-LLM evaluation"""
    
    def __init__(self, metrics_dir: str = METRICS_DIR, judge_llms=None):
        self.metrics_dir = metrics_dir
        self.current_session_metrics: List[LLMMetrics] = []
        self.multi_judge_evaluator = MultiJudgeEvaluator(judge_llms) if judge_llms else None
    
    async def record_metrics(self, query: str, response: str, context: str, 
                           response_time: float, token_count: int, 
                           model: str, session_id: str) -> LLMMetrics:
        """Record metrics with multi-LLM evaluation"""
        tokens_per_second = token_count / response_time if response_time > 0 else 0
        
        evaluations = []
        if self.multi_judge_evaluator:
            evaluations = await self.multi_judge_evaluator.evaluate_response(query, response, context)
        
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
            evaluations=evaluations
        )
        
        self.current_session_metrics.append(metrics)
        return metrics
    
    def save_metrics_to_file(self, filename: str = None):
        """Save metrics to JSON file"""
        if not filename:
            filename = f"multi_judge_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
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
                "session_id": metric.session_id,
                "evaluations": []
            }
            
            for eval_obj in metric.evaluations:
                metric_data["evaluations"].append({
                    "judge_model": eval_obj.judge_model,
                    "faithfulness": round(eval_obj.faithfulness, 1),
                    "groundedness": round(eval_obj.groundedness, 1),
                    "factual_consistency": round(eval_obj.factual_consistency, 1),
                    "relevance": round(eval_obj.relevance, 1),
                    "completeness": round(eval_obj.completeness, 1),
                    "fluency": round(eval_obj.fluency, 1),
                    "overall_score": round(eval_obj.overall_score, 1),
                    "evaluation_notes": eval_obj.evaluation_notes,
                })
            
            metrics_dicts.append(metric_data)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dicts, f, indent=2, ensure_ascii=False)
        
        print(f"Multi-judge metrics saved to {filepath}")
        return filepath
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics"""
        if not self.current_session_metrics:
            return {}
        
        response_times = [m.response_time for m in self.current_session_metrics]
        token_counts = [m.token_count for m in self.current_session_metrics]
        tokens_per_second = [m.tokens_per_second for m in self.current_session_metrics]
        
        # Collect all evaluations
        all_evaluations = []
        for metric in self.current_session_metrics:
            all_evaluations.extend(metric.evaluations)
        
        summary = {
            "total_interactions": len(self.current_session_metrics),
            "avg_response_time": round(sum(response_times) / len(response_times), 2),
            "min_response_time": round(min(response_times), 2),
            "max_response_time": round(max(response_times), 2),
            "avg_tokens_per_second": round(sum(tokens_per_second) / len(tokens_per_second), 2),
            "total_tokens_generated": sum(token_counts),
            "avg_tokens_per_response": round(sum(token_counts) / len(token_counts), 1),
            "total_evaluations": len(all_evaluations),
            "judge_models_used": list(self.multi_judge_evaluator.judge_llms.keys()) if self.multi_judge_evaluator else []
        }
        
        if all_evaluations:
            # Calculate average scores across all evaluations
            summary.update({
                "avg_faithfulness": round(sum(e.faithfulness for e in all_evaluations) / len(all_evaluations), 1),
                "avg_groundedness": round(sum(e.groundedness for e in all_evaluations) / len(all_evaluations), 1),
                "avg_factual_consistency": round(sum(e.factual_consistency for e in all_evaluations) / len(all_evaluations), 1),
                "avg_relevance": round(sum(e.relevance for e in all_evaluations) / len(all_evaluations), 1),
                "avg_completeness": round(sum(e.completeness for e in all_evaluations) / len(all_evaluations), 1),
                "avg_fluency": round(sum(e.fluency for e in all_evaluations) / len(all_evaluations), 1),
                "avg_overall_score": round(sum(e.overall_score for e in all_evaluations) / len(all_evaluations), 1),
            })
            
            # Calculate scores by judge model
            judge_scores = defaultdict(list)
            for eval_obj in all_evaluations:
                judge_scores[eval_obj.judge_model].append(eval_obj.overall_score)
            
            judge_avg_scores = {}
            for judge, scores in judge_scores.items():
                judge_avg_scores[judge] = round(sum(scores) / len(scores), 1)
            
            summary["avg_scores_by_judge"] = judge_avg_scores
            
            # Count rating categories
            rating_counts = {"Excellent": 0, "Good": 0, "Fair": 0, "Average": 0, "Poor / Weak": 0}
            for eval_obj in all_evaluations:
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
            "=== MULTI-LLM EVALUATION REPORT ===",
            f"Session ID: {self.current_session_metrics[0].session_id if self.current_session_metrics else 'N/A'}",
            f"Total Interactions: {summary['total_interactions']}",
            f"Total Evaluations: {summary['total_evaluations']}",
            f"Judge Models: {', '.join(summary['judge_models_used'])}",
            f"Average Response Time: {summary['avg_response_time']}s",
            f"Total Tokens Generated: {summary['total_tokens_generated']}",
            f"Average Throughput: {summary['avg_tokens_per_second']} tokens/s",
        ]
        
        if 'avg_overall_score' in summary:
            report.extend([
                "",
                "=== OVERALL QUALITY EVALUATION (0.0-10.0 scale) ===",
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
                    f"Excellent (9.0-10.0): {summary['rating_distribution']['Excellent']} evaluations",
                    f"Good (8.0-8.9): {summary['rating_distribution']['Good']} evaluations",
                    f"Fair (6.5-7.9): {summary['rating_distribution']['Fair']} evaluations",
                    f"Average (5.0-6.4): {summary['rating_distribution']['Average']} evaluations",
                    f"Poor / Weak (<5.0): {summary['rating_distribution']['Poor / Weak']} evaluations",
                ])
            
            if 'avg_scores_by_judge' in summary:
                report.extend([
                    "",
                    "=== AVERAGE SCORES BY JUDGE ===",
                ])
                for judge, score in summary['avg_scores_by_judge'].items():
                    report.append(f"{judge}: {score}/10.0")
        
        report.extend([
            "",
            "RATING SCALE:",
            "9.0 – 10.0: Excellent",
            "8.0 – 8.9: Good", 
            "6.5 – 7.9: Fair",
            "5.0 – 6.4: Average",
            "< 5.0: Poor / Weak",
            "",
            f"Main Model: {self.current_session_metrics[0].model if self.current_session_metrics else 'N/A'}",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "========================================="
        ])
        
        return "\n".join(report)

def load_index():
    """Loads the index from storage folder with FAISS vectors"""
    try:
        # Check if storage directory exists
        if not os.path.exists("./storage"):
            print("Storage directory not found. Creating new index...")
            from data_loader import create_index
            return create_index()
        
        # Load metadata
        metadata_path = "./storage/index_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print(f"Loading index with {metadata.get('total_nodes', 'unknown')} nodes")
        else:
            print("Metadata file not found, loading with default settings")
        
        # Load FAISS index
        faiss_index_path = "./storage/faiss_index.bin"
        if os.path.exists(faiss_index_path):
            print("Loading FAISS index...")
            faiss_index = faiss.read_index(faiss_index_path)
        else:
            print("FAISS index not found, creating new one...")
            from data_loader import create_index
            return create_index()
        
        # Create vector store
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        # Load storage context from persisted metadata
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir="./storage"  # Load metadata from storage folder
        )
        
        # Load index from storage
        index = load_index_from_storage(storage_context)
        print("Index loaded successfully from storage folder")
        return index
        
    except Exception as e:
        print(f"Error loading index: {e}")
        print("Creating new index...")
        from data_loader import create_index
        return create_index()

def create_query_engine(index):
    """Creates optimized query engine for large datasets"""
    base_retriever = index.as_retriever(
        similarity_top_k=8,
        vector_store_query_mode="default"
    )
    
    retriever = AutoMergingRetriever(
        base_retriever, 
        storage_context=index.storage_context, 
        verbose=False
    )
    
    reranker = SentenceTransformerRerank(
        top_n=4,
        model="BAAI/bge-reranker-base"
    )
    
    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker],
        streaming=True,
        llm=llm,
        response_mode="compact"
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
    """Initializes chat session with multi-LLM evaluation"""
    index = load_index()
    query_engine = create_query_engine(index)
    chat_memory = ChatMemoryBuffer.from_defaults(token_limit=1500, llm=llm)
    metrics_collector = MetricsCollector(judge_llms=judge_llms)
    
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
    """Processes incoming messages with multi-LLM evaluation"""
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
        await cl.Message(content=f"Multi-judge evaluation saved to: {filename}").send()
    else:
        await cl.Message(content="No metrics to save.").send()

@cl.action_callback("evaluate_last")
async def on_action_evaluate_last(action: cl.Action):
    """Action to show detailed evaluation of last response"""
    metrics_collector = cl.user_session.get("metrics_collector")
    if metrics_collector and metrics_collector.current_session_metrics:
        last_metric = metrics_collector.current_session_metrics[-1]
        if last_metric.evaluations:
            evaluation_report = [
                "=== MULTI-LLM EVALUATION DETAILS ===",
                f"Query: {last_metric.query[:100]}...",
                "",
                "EVALUATIONS BY JUDGE MODELS:",
            ]
            
            for eval_obj in last_metric.evaluations:
                evaluation_report.extend([
                    "",
                    f"--- {eval_obj.judge_model} ---",
                    f"Faithfulness: {eval_obj.faithfulness:.1f}/10.0",
                    f"Groundedness: {eval_obj.groundedness:.1f}/10.0",
                    f"Factual Consistency: {eval_obj.factual_consistency:.1f}/10.0",
                    f"Relevance: {eval_obj.relevance:.1f}/10.0",
                    f"Completeness: {eval_obj.completeness:.1f}/10.0",
                    f"Fluency: {eval_obj.fluency:.1f}/10.0",
                    f"Overall Score: {eval_obj.overall_score:.1f}/10.0",
                    f"Evaluation Notes: {eval_obj.evaluation_notes}",
                ])
            
            # Calculate average scores
            avg_faithfulness = sum(e.faithfulness for e in last_metric.evaluations) / len(last_metric.evaluations)
            avg_groundedness = sum(e.groundedness for e in last_metric.evaluations) / len(last_metric.evaluations)
            avg_factual_consistency = sum(e.factual_consistency for e in last_metric.evaluations) / len(last_metric.evaluations)
            avg_relevance = sum(e.relevance for e in last_metric.evaluations) / len(last_metric.evaluations)
            avg_completeness = sum(e.completeness for e in last_metric.evaluations) / len(last_metric.evaluations)
            avg_fluency = sum(e.fluency for e in last_metric.evaluations) / len(last_metric.evaluations)
            avg_overall = sum(e.overall_score for e in last_metric.evaluations) / len(last_metric.evaluations)
            
            evaluation_report.extend([
                "",
                "=== SUMMARY ACROSS ALL JUDGES ===",
                f"Average Faithfulness: {avg_faithfulness:.1f}/10.0",
                f"Average Groundedness: {avg_groundedness:.1f}/10.0",
                f"Average Factual Consistency: {avg_factual_consistency:.1f}/10.0",
                f"Average Relevance: {avg_relevance:.1f}/10.0",
                f"Average Completeness: {avg_completeness:.1f}/10.0",
                f"Average Fluency: {avg_fluency:.1f}/10.0",
                f"Average Overall Score: {avg_overall:.1f}/10.0",
                "",
                "RATING SCALE:",
                "9.0 – 10.0: Excellent",
                "8.0 – 8.9: Good", 
                "6.5 – 7.9: Fair",
                "5.0 – 6.4: Average",
                "< 5.0: Poor / Weak",
                "=========================================="
            ])
            
            await cl.Message(content="\n".join(evaluation_report)).send()
        else:
            await cl.Message(content="No LLM evaluations available for last response.").send()
    else:
        await cl.Message(content="No responses to evaluate yet.").send()

if __name__ == "__main__":
    os.makedirs(METRICS_DIR, exist_ok=True)