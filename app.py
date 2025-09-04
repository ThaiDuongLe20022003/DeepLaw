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
import pandas as pd
from dataclasses import dataclass
import re

# Configure embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Initialize Ollama LLM
llm = Ollama(model="deepseek-r1:1.5b", request_timeout=120.0)

# Configuration paths
STORAGE_DIR = "./storage"
METRICS_DIR = "./metrics"

# Ensure metrics directory exists
os.makedirs(METRICS_DIR, exist_ok=True)

@dataclass
class AccuracyMetrics:
    """Metrics for accuracy evaluation"""
    relevance_score: float  # 0-1: How relevant is the response to the query
    factual_accuracy: float  # 0-1: Factual correctness
    completeness: float  # 0-1: How complete is the answer
    coherence: float  # 0-1: How coherent and well-structured
    overall_accuracy: float  # Combined accuracy score

@dataclass
class LLMMetrics:
    """Data class to store LLM performance metrics"""
    timestamp: str
    query: str
    response: str
    response_time: float
    token_count: int
    tokens_per_second: float
    model: str
    session_id: str
    accuracy: Optional[AccuracyMetrics] = None
    context_relevance: float = 0.0  # How well context was used
    retrieval_quality: float = 0.0  # Quality of retrieved documents

class AccuracyEvaluator:
    """Evaluates accuracy of LLM responses using multiple techniques"""
    
    def __init__(self):
        self.patterns = {
            'uncertainty': re.compile(r'\b(maybe|perhaps|possibly|I think|I believe|probably|likely|unclear)\b', re.IGNORECASE),
            'confident': re.compile(r'\b(certainly|definitely|absolutely|without doubt|clearly)\b', re.IGNORECASE),
            'hedging': re.compile(r'\b(some|many|several|various|often|sometimes)\b', re.IGNORECASE)
        }
    
    def evaluate_response(self, query: str, response: str, context: str = "") -> AccuracyMetrics:
        """Comprehensive accuracy evaluation"""
        # Basic text analysis
        text_quality = self._evaluate_text_quality(response)
        relevance = self._evaluate_relevance(query, response)
        factual_accuracy = self._evaluate_factual_accuracy(response, context)
        completeness = self._evaluate_completeness(query, response)
        coherence = self._evaluate_coherence(response)
        
        # Combined score (weighted average)
        overall_accuracy = (
            relevance * 0.3 +
            factual_accuracy * 0.3 +
            completeness * 0.2 +
            coherence * 0.2
        )
        
        return AccuracyMetrics(
            relevance_score=relevance,
            factual_accuracy=factual_accuracy,
            completeness=completeness,
            coherence=coherence,
            overall_accuracy=overall_accuracy
        )
    
    def _evaluate_text_quality(self, text: str) -> float:
        """Evaluate basic text quality"""
        if len(text.strip()) < 10:
            return 0.3
        
        # Check for common issues
        score = 1.0
        if len(text) > 1000:  # Too verbose
            score *= 0.8
        if self.patterns['uncertainty'].search(text):
            score *= 0.9
        if self.patterns['confident'].search(text):
            score *= 1.1
        
        return min(max(score, 0.1), 1.0)
    
    def _evaluate_relevance(self, query: str, response: str) -> float:
        """Evaluate how relevant the response is to the query"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 0.5
        
        # Jaccard similarity
        intersection = query_words.intersection(response_words)
        union = query_words.union(response_words)
        
        if not union:
            return 0.5
            
        similarity = len(intersection) / len(union)
        return min(max(similarity, 0.1), 1.0)
    
    def _evaluate_factual_accuracy(self, response: str, context: str) -> float:
        """Evaluate factual accuracy (simplified)"""
        # This is a simplified version - in production, use fact-checking APIs
        # or more sophisticated methods
        
        # Check for obvious issues
        issues = [
            "I don't know", "I cannot answer", "no information",
            "not sure", "uncertain", "maybe"
        ]
        
        base_score = 0.7
        for issue in issues:
            if issue.lower() in response.lower():
                base_score *= 0.7
        
        # If response is very short, penalize
        if len(response.split()) < 5:
            base_score *= 0.6
            
        return min(max(base_score, 0.1), 1.0)
    
    def _evaluate_completeness(self, query: str, response: str) -> float:
        """Evaluate how complete the answer is"""
        # Simple heuristic based on response length and content
        query_complexity = len(query.split()) / 5  # Normalize
        response_adequacy = len(response.split()) / 20  # Normalize
        
        score = min(response_adequacy / max(query_complexity, 1), 1.0)
        return min(max(score, 0.2), 1.0)
    
    def _evaluate_coherence(self, response: str) -> float:
        """Evaluate coherence and structure"""
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', response)
        valid_sentences = [s for s in sentences if len(s.strip().split()) > 3]
        
        if not sentences:
            return 0.3
            
        coherence_score = len(valid_sentences) / len(sentences)
        return min(max(coherence_score, 0.3), 1.0)

class MetricsCollector:
    """Collects and manages LLM performance metrics"""
    
    def __init__(self, metrics_dir: str = METRICS_DIR):
        self.metrics_dir = metrics_dir
        self.current_session_metrics: List[LLMMetrics] = []
        self.accuracy_evaluator = AccuracyEvaluator()
        
    def record_metrics(self, query: str, response: str, response_time: float, 
                      token_count: int, model: str, session_id: str, 
                      context: str = "") -> LLMMetrics:
        """Record metrics for a single interaction"""
        tokens_per_second = token_count / response_time if response_time > 0 else 0
        
        # Evaluate accuracy
        accuracy_metrics = self.accuracy_evaluator.evaluate_response(
            query, response, context
        )
        
        metrics = LLMMetrics(
            timestamp=datetime.now().isoformat(),
            query=query,
            response=response,
            response_time=response_time,
            token_count=token_count,
            tokens_per_second=tokens_per_second,
            model=model,
            session_id=session_id,
            accuracy=accuracy_metrics
        )
        
        self.current_session_metrics.append(metrics)
        return metrics
    
    def save_metrics_to_file(self, filename: str = None):
        """Save metrics to JSON file"""
        if not filename:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.metrics_dir, filename)
        
        # Convert metrics to dict for JSON serialization
        metrics_dicts = []
        for metric in self.current_session_metrics:
            metric_data = {
                "timestamp": metric.timestamp,
                "query": metric.query,
                "response": metric.response[:500] + "..." if len(metric.response) > 500 else metric.response,
                "response_time": metric.response_time,
                "token_count": metric.token_count,
                "tokens_per_second": metric.tokens_per_second,
                "model": metric.model,
                "session_id": metric.session_id,
                "context_relevance": metric.context_relevance,
                "retrieval_quality": metric.retrieval_quality
            }
            
            if metric.accuracy:
                metric_data["accuracy"] = {
                    "relevance_score": metric.accuracy.relevance_score,
                    "factual_accuracy": metric.accuracy.factual_accuracy,
                    "completeness": metric.accuracy.completeness,
                    "coherence": metric.accuracy.coherence,
                    "overall_accuracy": metric.accuracy.overall_accuracy
                }
            
            metrics_dicts.append(metric_data)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dicts, f, indent=2)
        
        print(f"Metrics saved to {filepath}")
        return filepath
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for current session"""
        if not self.current_session_metrics:
            return {}
        
        response_times = [m.response_time for m in self.current_session_metrics]
        token_counts = [m.token_count for m in self.current_session_metrics]
        tokens_per_second = [m.tokens_per_second for m in self.current_session_metrics]
        
        # Accuracy statistics
        accuracy_scores = [m.accuracy.overall_accuracy for m in self.current_session_metrics if m.accuracy]
        relevance_scores = [m.accuracy.relevance_score for m in self.current_session_metrics if m.accuracy]
        factual_scores = [m.accuracy.factual_accuracy for m in self.current_session_metrics if m.accuracy]
        
        summary = {
            "total_interactions": len(self.current_session_metrics),
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "avg_tokens_per_second": sum(tokens_per_second) / len(tokens_per_second),
            "total_tokens_generated": sum(token_counts),
            "avg_tokens_per_response": sum(token_counts) / len(token_counts)
        }
        
        if accuracy_scores:
            summary.update({
                "avg_accuracy": sum(accuracy_scores) / len(accuracy_scores),
                "avg_relevance": sum(relevance_scores) / len(relevance_scores),
                "avg_factual_accuracy": sum(factual_scores) / len(factual_scores),
                "min_accuracy": min(accuracy_scores),
                "max_accuracy": max(accuracy_scores)
            })
        
        return summary
    
    def generate_report(self) -> str:
        """Generate a human-readable report"""
        summary = self.get_session_summary()
        if not summary:
            return "No metrics collected yet."
        
        report = [
            "=== LLM Performance Metrics Report ===",
            f"Session ID: {self.current_session_metrics[0].session_id if self.current_session_metrics else 'N/A'}",
            f"Total Interactions: {summary['total_interactions']}",
            f"Total Tokens Generated: {summary['total_tokens_generated']}",
            f"Average Response Time: {summary['avg_response_time']:.2f}s",
            f"Response Time Range: {summary['min_response_time']:.2f}s - {summary['max_response_time']:.2f}s",
            f"Average Tokens/Second: {summary['avg_tokens_per_second']:.2f}",
            f"Average Tokens/Response: {summary['avg_tokens_per_response']:.1f}",
        ]
        
        if 'avg_accuracy' in summary:
            report.extend([
                f"Average Accuracy: {summary['avg_accuracy']:.3f}/1.0",
                f"Accuracy Range: {summary['min_accuracy']:.3f} - {summary['max_accuracy']:.3f}",
                f"Average Relevance: {summary['avg_relevance']:.3f}/1.0",
                f"Average Factual Accuracy: {summary['avg_factual_accuracy']:.3f}/1.0"
            ])
        
        report.extend([
            f"Model: {self.current_session_metrics[0].model if self.current_session_metrics else 'N/A'}",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "======================================"
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

@cl.on_chat_start
async def init_chat():
    """Initializes chat session with memory and metrics"""
    index = load_index()
    query_engine = create_query_engine(index)
    chat_memory = ChatMemoryBuffer.from_defaults(token_limit=1500, llm=llm)
    metrics_collector = MetricsCollector()
    
    cl.user_session.set("query_engine", query_engine)
    cl.user_session.set("chat_memory", chat_memory)
    cl.user_session.set("metrics_collector", metrics_collector)
    cl.user_session.set("session_start_time", time.time())
    

@cl.on_chat_resume
async def resume_chat():
    """Handles chat session resumption"""
    await init_chat()

@cl.on_chat_end
async def on_chat_end():
    """Handle chat session end - save metrics"""
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
    """Processes incoming messages with streaming response and metrics collection"""
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
    response = await cl.make_async(query_engine.query)(full_prompt)
    
    full_response = ""
    token_count = 0
    for token in response.response_gen:
        full_response += token
        token_count = count_tokens(full_response)
        await reply.stream_token(token)
    await reply.update()
    
    response_time = time.time() - start_time
    
    if metrics_collector:
        session_id = cl.user_session.get("id", "unknown_session")
        metrics_collector.record_metrics(
            query=message.content,
            response=full_response,
            response_time=response_time,
            token_count=token_count,
            model="deepseek-r1:1.5b",
            session_id=session_id
        )
    
    chat_memory.put(f"User: {message.content}")
    chat_memory.put(f"Assistant: {full_response}")

@cl.action_callback("show_metrics")
async def on_action_show_metrics(action: cl.Action):
    """Action to show current metrics"""
    metrics_collector = cl.user_session.get("metrics_collector")
    if metrics_collector:
        report = metrics_collector.generate_report()
        await cl.Message(content=f"```\n{report}\n```").send()
    else:
        await cl.Message(content="No metrics available yet.").send()

@cl.action_callback("save_metrics")
async def on_action_save_metrics(action: cl.Action):
    """Action to save metrics to file"""
    metrics_collector = cl.user_session.get("metrics_collector")
    if metrics_collector and metrics_collector.current_session_metrics:
        filename = metrics_collector.save_metrics_to_file()
        await cl.Message(content=f"Metrics saved to: {filename}").send()
    else:
        await cl.Message(content="No metrics to save.").send()

@cl.action_callback("evaluate_accuracy")
async def on_action_evaluate_accuracy(action: cl.Action):
    """Action to evaluate accuracy of last response"""
    metrics_collector = cl.user_session.get("metrics_collector")
    if metrics_collector and metrics_collector.current_session_metrics:
        last_metric = metrics_collector.current_session_metrics[-1]
        if last_metric.accuracy:
            accuracy = last_metric.accuracy
            evaluation = [
                "=== Accuracy Evaluation ===",
                f"Query: {last_metric.query[:100]}...",
                f"Overall Accuracy: {accuracy.overall_accuracy:.3f}/1.0",
                f"Relevance: {accuracy.relevance_score:.3f}/1.0",
                f"Factual Accuracy: {accuracy.factual_accuracy:.3f}/1.0",
                f"Completeness: {accuracy.completeness:.3f}/1.0",
                f"Coherence: {accuracy.coherence:.3f}/1.0",
                "==========================="
            ]
            await cl.Message(content="\n".join(evaluation)).send()
        else:
            await cl.Message(content="No accuracy data available for last response.").send()
    else:
        await cl.Message(content="No responses to evaluate yet.").send()

if __name__ == "__main__":
    os.makedirs(METRICS_DIR, exist_ok=True)
    print("Chat application with accuracy evaluation ready to start...")
    print("Available actions: show_metrics, save_metrics, evaluate_accuracy")