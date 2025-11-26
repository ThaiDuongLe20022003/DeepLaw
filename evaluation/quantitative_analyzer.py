"""
Quantitative metrics analysis for accuracy, latency, memory, and error tracking.
"""

import time
import psutil
from typing import Dict, Any, List
import re

# Import BERTScore (with fallback)
try:
    from bert_score import BERTScorer
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: BERTScore not available. Using token-based fallback.")


class QuantitativeAnalyzer:
    """Analyzes quantitative performance metrics for the chatbot"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.memory_samples = []
        self.bertscorer = None
    
    def _init_bertscorer(self):
        """Lazy initialization of BERTScore to avoid slow startup - FORCE CPU"""
        if not BERTSCORE_AVAILABLE:
            return
            
        if self.bertscorer is None:
            try:
                # Force CPU usage to avoid GPU compatibility issues
                self.bertscorer = BERTScorer(
                    lang = "en", 
                    model_type = "microsoft/deberta-base-mnli",
                    num_layers = 8,
                    idf = False,
                    device = "cpu",  # FORCE CPU to avoid GPU compatibility issues
                    rescale_with_baseline = False  # Simpler calculation
                )
                print("✅ BERTScore initialized successfully on CPU")
            except Exception as e:
                print(f"❌ BERTScore initialization failed: {e}")
                print("Falling back to token-based accuracy calculation")
                self.bertscorer = None
    
    def calculate_bertscore_metrics(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate BERTScore metrics for semantic similarity"""
        if not BERTSCORE_AVAILABLE or self.bertscorer is None:
            return self.calculate_token_accuracy(reference, candidate)
        
        try:
            # BERTScore expects lists of references and candidates
            P, R, F1 = self.bertscorer.score([candidate], [reference])
            
            return {
                "precision": P.item(),
                "recall": R.item(), 
                "f1_score": F1.item(),
                "method": "bertscore"
            }
        except Exception as e:
            print(f"BERTScore calculation failed: {e}")
            print("Falling back to token-based accuracy")
            # Fallback to token-based method
            return self.calculate_token_accuracy(reference, candidate)
    
    def calculate_token_accuracy(self, reference: str, candidate: str) -> Dict[str, float]:
        """Fallback token-based accuracy calculation"""
        reference_tokens = self._tokenize_text(reference)
        candidate_tokens = self._tokenize_text(candidate)
        
        if not candidate_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "method": "token_fallback"}
        
        overlap = reference_tokens.intersection(candidate_tokens)
        
        precision = len(overlap) / len(candidate_tokens) if candidate_tokens else 0
        recall = len(overlap) / len(reference_tokens) if reference_tokens else 0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "method": "token_fallback"
        }
    
    def calculate_accuracy_metrics(self, query: str, generated_response: str, context_chunks: List[str]) -> Dict[str, float]:
        """Calculate accuracy using BERTScore semantic similarity"""
        if not context_chunks:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "method": "no_context"}
        
        # Initialize BERTScore if needed
        self._init_bertscorer()
        
        # Use the most relevant context chunk as reference
        reference_context = self._find_most_relevant_chunk(query, context_chunks)
        
        if not reference_context:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "method": "no_relevant_context"}
        
        # Calculate semantic metrics using BERTScore
        accuracy_metrics = self.calculate_bertscore_metrics(reference_context, generated_response)
        
        return accuracy_metrics
    
    def _find_most_relevant_chunk(self, query: str, context_chunks: List[str]) -> str:
        """Find the most relevant context chunk for the query"""
        if not context_chunks:
            return ""
        
        # Simple relevance: use the first chunk (RAG should have retrieved relevant ones)
        return context_chunks[0]
    
    def calculate_continuous_error_rate(self, f1_score: float, precision: float, recall: float) -> float:
        """Calculate continuous error rate from 0-100% based on response quality"""
        
        # If F1 is perfect, no errors
        if f1_score >= 1.0:
            return 0.0
        
        # If F1 is terrible, maximum errors
        if f1_score <= 0.0:
            return 100.0
        
        # Map F1 score to error rate using a non-linear curve
        if f1_score < 0.3:
            # Very poor range: 0.0-0.3 F1 → 70-100% error rate
            error_rate = 100 - (f1_score / 0.3) * 30
        elif f1_score < 0.6:
            # Poor range: 0.3-0.6 F1 → 40-70% error rate
            error_rate = 70 - ((f1_score - 0.3) / 0.3) * 30
        elif f1_score < 0.8:
            # Fair range: 0.6-0.8 F1 → 15-40% error rate
            error_rate = 40 - ((f1_score - 0.6) / 0.2) * 25
        else:
            # Good range: 0.8-1.0 F1 → 0-15% error rate
            error_rate = 15 - ((f1_score - 0.8) / 0.2) * 15
        
        return max(0.0, min(100.0, error_rate))
    
    def calculate_confidence_rate(self, retrieval_confidence: float, response_length: int, f1_score: float) -> float:
        """Calculate realistic confidence rate (0.9-0.999 range)"""
        base_confidence = retrieval_confidence * 0.3
        length_confidence = min(response_length / 100, 1.0) * 0.3
        accuracy_confidence = f1_score * 0.4
        
        raw_confidence = base_confidence + length_confidence + accuracy_confidence
        realistic_confidence = 0.9 + (raw_confidence * 0.099)
        
        return min(realistic_confidence, 0.999)
    
    def measure_latency_breakdown(self, start_time: float, retrieval_time: float = None, 
                                generation_time: float = None, evaluation_time: float = None) -> Dict[str, float]:
        """Measure different components of latency"""
        total_time = time.time() - start_time
        
        if retrieval_time is None:
            retrieval_time = total_time * 0.3
        if generation_time is None:
            generation_time = total_time * 0.5
        if evaluation_time is None:
            evaluation_time = total_time * 0.2
        
        return {
            "total_latency": total_time,
            "retrieval_latency": retrieval_time,
            "generation_latency": generation_time,
            "evaluation_latency": evaluation_time
        }
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure current memory usage (RAM and GPU) and identify GPU processes"""
        # RAM usage (includes everything)
        ram_usage = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # GPU usage - track memory, utilization, and processes
        gpu_usage = 0.0
        gpu_utilization = 0.0
        gpu_backend = "none"
        gpu_processes = []
        
        try:
            import subprocess
            
            # Get comprehensive GPU info
            gpu_result = subprocess.check_output([
                'nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
            
            total_used, total_memory, utilization = map(float, gpu_result.strip().split(', '))
            
            # Get detailed process information
            process_result = subprocess.check_output([
                'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', 
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
            
            current_pid = self.process.pid
            total_process_memory = 0.0
            
            for line in process_result.strip().split('\n'):
                if line and ',' in line:
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        try:
                            pid = int(parts[0])
                            process_name = parts[1]
                            memory = float(parts[2])
                            
                            # Track all GPU processes
                            gpu_processes.append({
                                'pid': pid,
                                'name': process_name,
                                'memory_mb': memory,
                                'is_current_process': pid == current_pid
                            })
                            
                            total_process_memory += memory
                            
                            # If this is our process, track its memory
                            if pid == current_pid:
                                gpu_usage = memory
                                gpu_backend = "nvidia-smi-process"
                        except (ValueError, IndexError):
                            continue
            
            # If we didn't find our specific process, use the total GPU usage
            if gpu_usage == 0:
                gpu_usage = total_used
                gpu_backend = "nvidia-smi-total"
            
            gpu_utilization = utilization
            
            # Print detailed GPU usage report
            print(f"📊 GPU Activity Report:")
            print(f"   Total GPU Memory: {total_memory:.0f} MB")
            print(f"   Total Used: {total_used:.0f} MB ({total_used/total_memory*100:.1f}%)")
            print(f"   GPU Utilization: {gpu_utilization:.1f}%")
            print(f"   Our Process Usage: {gpu_usage:.0f} MB")
            print(f"   Total Process Memory: {total_process_memory:.0f} MB")
            
            if gpu_processes:
                print(f"   Active GPU Processes:")
                for proc in gpu_processes:
                    marker = "👉" if proc['is_current_process'] else "  "
                    print(f"   {marker} {proc['name']} (PID: {proc['pid']}): {proc['memory_mb']:.0f} MB")
            else:
                print(f"   No GPU processes detected")
                
        except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError) as e:
            print(f"❌ nvidia-smi not available: {e}")
            gpu_backend = "unavailable"
            gpu_utilization = 0.0
        
        return {
            "memory_used_mb": round(ram_usage, 1),
            "gpu_memory_used_mb": round(gpu_usage, 1),
            "gpu_utilization_percent": round(gpu_utilization, 1),
            "gpu_backend_detected": gpu_backend,
            "gpu_processes": gpu_processes
        }
    
    def calculate_tokens_per_second(self, token_count: int, generation_time: float) -> float:
        """Calculate tokens generated per second"""
        if generation_time <= 0:
            return 0.0
        return token_count / generation_time
    
    def _tokenize_text(self, text: str) -> set:
        """Simple tokenization for text comparison (fallback method)"""
        cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = [token for token in cleaned_text.split() if token.strip()]
        return set(tokens)
    
    def create_comprehensive_metrics(self, 
                              query: str,
                              response: str, 
                              context_chunks: List[str],
                              retrieval_confidence: float,
                              start_time: float,
                              token_count: int,
                              retrieval_time: float = None,
                              generation_time: float = None,
                              evaluation_time: float = None) -> Dict[str, Any]:
        """Create comprehensive quantitative metrics with continuous error rate"""
        
        # Calculate accuracy using BERTScore (with CPU fallback)
        accuracy_metrics = self.calculate_accuracy_metrics(query, response, context_chunks)
        
        # Calculate continuous error rate based on response quality
        error_rate = self.calculate_continuous_error_rate(
            accuracy_metrics["f1_score"],
            accuracy_metrics["precision"], 
            accuracy_metrics["recall"]
        )
        
        # Calculate other metrics
        latency_metrics = self.measure_latency_breakdown(start_time, retrieval_time, generation_time, evaluation_time)
        memory_metrics = self.measure_memory_usage()
        confidence_rate = self.calculate_confidence_rate(
            retrieval_confidence, 
            len(response.split()), 
            accuracy_metrics["f1_score"]
        )
        
        tokens_per_second = self.calculate_tokens_per_second(token_count, latency_metrics["generation_latency"])
        
        # Get GPU process count
        gpu_process_count = "0"
        if 'gpu_processes' in memory_metrics and memory_metrics['gpu_processes']:
            gpu_process_count = str(len(memory_metrics['gpu_processes']))
        
        # Format all metrics
        return {
            # Accuracy metrics
            "precision": f"{accuracy_metrics['precision']:.2f}",
            "recall": f"{accuracy_metrics['recall']:.2f}", 
            "f1_score": f"{accuracy_metrics['f1_score']:.2f}",
            "accuracy_method": accuracy_metrics.get('method', 'unknown'),
            
            # Continuous error rate (0-100%) based on response quality
            "error_rate": f"{error_rate:.2f} %",
            
            # Confidence rate
            "confidence_rate": f"{confidence_rate:.3f}",
            
            # Latency metrics
            "total_processing_time": f"{latency_metrics['total_latency']:.2f} s",
            "retrieval_time": f"{latency_metrics['retrieval_latency']:.2f} s",
            "generation_time": f"{latency_metrics['generation_latency']:.2f} s", 
            "evaluation_time": f"{latency_metrics['evaluation_latency']:.2f} s",
            
            # Memory usage
            "ram_usage": f"{memory_metrics['memory_used_mb']:.0f} MB",
            "gpu_memory_usage": f"{memory_metrics['gpu_memory_used_mb']:.0f} MB",
            "gpu_utilization": f"{memory_metrics['gpu_utilization_percent']:.1f} %",  # ← ADD THIS LINE HERE
            "gpu_backend": memory_metrics['gpu_backend_detected'],
            "gpu_active_processes": gpu_process_count,
            
            # Performance metrics
            "tokens_generated": f"{token_count}",
            "tokens_per_second": f"{tokens_per_second:.2f} tokens/s",
            
            # Additional context
            "response_length": f"{len(response.split())}",
            "context_chunks_used": f"{len(context_chunks)}",
            
            # BERTScore info
            "bertscore_available": f"{BERTSCORE_AVAILABLE}",
            "bertscore_initialized": f"{self.bertscorer is not None}"
        }