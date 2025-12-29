"""
Chat interface components for the Streamlit application.
"""

import streamlit as st

from data_models.models import ChatMessage
from processing.rag_chain import generate_response_with_metrics

def render_chat_interface(vector_db, selected_model, evaluation_enabled, judge_evaluator, metrics_collector):
    """Render the main chat interface"""
    message_container = st.container(height = 500, border = True)

    # Display chat history
    display_chat_history(message_container)

    # Chat input and processing
    if prompt := st.chat_input("Enter a prompt here...", key = "chat_input"):
        handle_user_input(prompt, message_container, vector_db, selected_model, 
                         evaluation_enabled, judge_evaluator, metrics_collector)


def display_chat_history(message_container):
    """Display the chat message history"""
    for i, message in enumerate(st.session_state["messages"]):
        avatar = "🤖" if message.role == "assistant" else "😎"
        with message_container.chat_message(message.role, avatar = avatar):
            st.markdown(message.content)
            
            # Show evaluation scores if available
            if message.evaluations:
                def _safe_f(v, key='overall_score'):
                    try:
                        if isinstance(v, dict):
                            return float(v.get(key, 0) or 0)
                        return float(getattr(v, key, 0) or 0)
                    except Exception:
                        return 0.0

                avg_score = sum(_safe_f(ev) for ev in message.evaluations) / len(message.evaluations)
                st.caption(f"Average Evaluation: {avg_score:.1f}/10.0 ({len(message.evaluations)} judges)")
            
            # Show quantitative metrics if available
            if message.quantitative_metrics:
                qm = message.quantitative_metrics
                if isinstance(qm, dict):
                    f1 = qm.get('f1_score', 0)
                    total_latency = qm.get('total_latency', 0)
                    confidence = qm.get('confidence_rate', 0)
                else:
                    f1 = getattr(qm, 'f1_score', 0)
                    total_latency = getattr(qm, 'total_latency', 0)
                    confidence = getattr(qm, 'confidence_rate', 0)
                try:
                    f1f = float(f1)
                except Exception:
                    f1f = 0.0
                try:
                    latf = float(total_latency)
                except Exception:
                    latf = 0.0
                try:
                    conf = float(confidence)
                except Exception:
                    conf = 0.0
                st.caption(f"⚡ F1: {f1f:.2f} | Latency: {latf:.2f}s | Confidence: {conf:.2f}")


def handle_user_input(prompt, message_container, vector_db, selected_model, 
                     evaluation_enabled, judge_evaluator, metrics_collector):
    """Handle user input and generate response"""
    try:
        # Add user message to chat
        st.session_state["messages"].append(ChatMessage(role = "user", content = prompt))
        with message_container.chat_message("user", avatar = "😎"):
            st.markdown(prompt)

        # Process and display assistant response
        with message_container.chat_message("assistant", avatar = "🤖"):
            with st.spinner("Processing your question..."):
                result = generate_response_with_metrics(
                    prompt, vector_db, selected_model, evaluation_enabled, judge_evaluator
                )
                
                st.markdown(result["response"])
                
                # Record metrics and update session state
                if result["success"]:
                    if evaluation_enabled and "evaluations" in result:
                        update_session_with_metrics(prompt, result, selected_model, metrics_collector)
                    else:
                        # Create ChatMessage without evaluations but with basic metrics
                        quantitative_metrics = {
                            "response_time": result["response_time"],
                            "token_count": result["token_count"],
                            "retrieval_confidence": result.get("retrieval_confidence", 0.5)
                        }
                        st.session_state["messages"].append(
                            ChatMessage(
                                role = "assistant", 
                                content = result["response"],
                                quantitative_metrics = quantitative_metrics
                            )
                        )
                else:
                    metrics_collector.record_error()  

    except Exception as e:
        st.error(f"Error: {str(e)}", icon = "⛔️")
        metrics_collector.record_error()  


def update_session_with_metrics(prompt, result, selected_model, metrics_collector):
    """Update session state with metrics and evaluations"""
    # Record metrics with quantitative data
    metrics = metrics_collector.record_metrics(
        query = prompt,
        response = result["response"],
        context = result["context"],
        context_chunks = result.get("context_chunks", []),  
        response_time = result["response_time"],
        token_count = result["token_count"],
        model = selected_model,
        session_id = "streamlit_session",
        evaluations = result["evaluations"],
        retrieval_confidence = result.get("retrieval_confidence", 0.5),  
        start_time = result.get("start_time"),  
        retrieval_time = result.get("retrieval_time"),  
        generation_time = result.get("generation_time"),  
        evaluation_time = result.get("evaluation_time")  
    )
    
    # Helper: safely convert values to float before rounding
    def _safe_float(val, default = 0.0):
        try:
            return float(val)
        except Exception:
            return float(default)

    # Convert evaluations to dictionaries for session state storage
    eval_dicts = []
    for eval_obj in result["evaluations"]:
        # Support both LLMEvaluation objects and plain dicts (some code paths return dicts)
        if isinstance(eval_obj, dict):
            faith = eval_obj.get("faithfulness", 0.0)
            ground = eval_obj.get("groundedness", 0.0)
            factual = eval_obj.get("factual_consistency", 0.0)
            relevance = eval_obj.get("relevance", 0.0)
            completeness = eval_obj.get("completeness", 0.0)
            fluency = eval_obj.get("fluency", 0.0)
            overall = eval_obj.get("overall_score", 0.0)
            notes = eval_obj.get("evaluation_notes", "")
            judge = eval_obj.get("judge_model", "")
            qmetrics = eval_obj.get("quantitative_metrics")
        else:
            # Use getattr to avoid AttributeError if attribute missing
            faith = getattr(eval_obj, "faithfulness", 0.0)
            ground = getattr(eval_obj, "groundedness", 0.0)
            factual = getattr(eval_obj, "factual_consistency", 0.0)
            relevance = getattr(eval_obj, "relevance", 0.0)
            completeness = getattr(eval_obj, "completeness", 0.0)
            fluency = getattr(eval_obj, "fluency", 0.0)
            overall = getattr(eval_obj, "overall_score", 0.0)
            notes = getattr(eval_obj, "evaluation_notes", "")
            judge = getattr(eval_obj, "judge_model", "")
            # quantitative_metrics may be a dataclass, object, or dict — handle both
            qmetrics = getattr(eval_obj, "quantitative_metrics", None)

        eval_dict = {
            "faithfulness": round(_safe_float(faith), 1),
            "groundedness": round(_safe_float(ground), 1),
            "factual_consistency": round(_safe_float(factual), 1),
            "relevance": round(_safe_float(relevance), 1),
            "completeness": round(_safe_float(completeness), 1),
            "fluency": round(_safe_float(fluency), 1),
            "overall_score": round(_safe_float(overall), 1),
            "rating": notes.split(":")[0] if isinstance(notes, str) else "",
            "judge_model": judge
        }

        # Add quantitative metrics if available (handle object or dict)
        if qmetrics:
            if isinstance(qmetrics, dict):
                eval_dict["quantitative_metrics"] = {
                    "precision": round(_safe_float(qmetrics.get("precision", 0)), 3),
                    "recall": round(_safe_float(qmetrics.get("recall", 0)), 3),
                    "f1_score": round(_safe_float(qmetrics.get("f1_score", 0)), 3),
                    "confidence_rate": round(_safe_float(qmetrics.get("confidence_rate", 0)), 3),
                    "total_latency": round(_safe_float(qmetrics.get("total_latency", 0)), 2),
                    "memory_used_mb": round(_safe_float(qmetrics.get("memory_used_mb", 0)), 1),
                    "error_rate": round(_safe_float(qmetrics.get("error_rate", 0)), 2)
                }
            else:
                # Assume object with attributes
                eval_dict["quantitative_metrics"] = {
                    "precision": round(_safe_float(getattr(qmetrics, "precision", 0)), 3),
                    "recall": round(_safe_float(getattr(qmetrics, "recall", 0)), 3),
                    "f1_score": round(_safe_float(getattr(qmetrics, "f1_score", 0)), 3),
                    "confidence_rate": round(_safe_float(getattr(qmetrics, "confidence_rate", 0)), 3),
                    "total_latency": round(_safe_float(getattr(qmetrics, "total_latency", 0)), 2),
                    "memory_used_mb": round(_safe_float(getattr(qmetrics, "memory_used_mb", 0)), 1),
                    "error_rate": round(_safe_float(getattr(qmetrics, "error_rate", 0)), 2)
                }

        eval_dicts.append(eval_dict)

    # Prepare quantitative metrics for chat message
    quantitative_metrics = None
    if metrics.quantitative_metrics:
        # Helper function to safely convert string metrics to float for rounding
        def safe_convert_and_round(metric_value, decimals):
            try:
                # Extract number from string 
                if isinstance(metric_value, str):
                    # Remove non-numeric characters except decimal point
                    cleaned = ''.join(c for c in metric_value if c.isdigit() or c in '.-')
                    number = float(cleaned) if cleaned else 0.0
                else:
                    number = float(metric_value)
                return round(number, decimals)
            except (ValueError, TypeError):
                return 0.0
        
        quantitative_metrics = {
            "precision": safe_convert_and_round(metrics.quantitative_metrics.precision, 3),
            "recall": safe_convert_and_round(metrics.quantitative_metrics.recall, 3),
            "f1_score": safe_convert_and_round(metrics.quantitative_metrics.f1_score, 3),
            "confidence_rate": safe_convert_and_round(metrics.quantitative_metrics.confidence_rate, 3),
            "total_latency": safe_convert_and_round(metrics.quantitative_metrics.total_processing_time, 2),
            "retrieval_latency": safe_convert_and_round(metrics.quantitative_metrics.retrieval_time, 2),
            "generation_latency": safe_convert_and_round(metrics.quantitative_metrics.generation_time, 2),
            "evaluation_latency": safe_convert_and_round(metrics.quantitative_metrics.evaluation_time, 2),
            "memory_used_mb": safe_convert_and_round(metrics.quantitative_metrics.ram_usage, 1),
            "gpu_memory_used_mb": safe_convert_and_round(metrics.quantitative_metrics.gpu_memory_usage, 1),
            "error_rate": safe_convert_and_round(metrics.quantitative_metrics.error_rate, 2),
            "tokens_generated": metrics.quantitative_metrics.tokens_generated, 
            "tokens_per_second": safe_convert_and_round(metrics.quantitative_metrics.tokens_per_second, 1)
        }    
    
    # Create ChatMessage object with evaluations and quantitative metrics
    st.session_state["messages"].append(
        ChatMessage(
            role = "assistant", 
            content = result["response"], 
            evaluations = eval_dicts,
            quantitative_metrics = quantitative_metrics
        )
    )