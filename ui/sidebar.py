"""
Sidebar components and configuration for the Streamlit application.
"""

import os
import json
import streamlit as st

from evaluation.evaluator import LLMJudgeEvaluator
from config.settings import METRICS_DIR


def render_sidebar(available_models, selected_model, metrics_collector):
    """Render the sidebar with configuration options"""
    st.header("⚙️ Configuration")

    # Add document link here
    st.markdown("[📄 Find legal documents to upload here](https://github.com/ThaiDuongLe20022003/Thesis/tree/main/Law%20Data)")
    
    # Model selection for response only
    if available_models:
        selected_model = st.selectbox(
            "Select Response Model", 
            available_models,
            key = "response_model_select",
            help = "Choose the model that will generate responses to your questions"
        )
    else:
        st.error("No Ollama models found. Please make sure Ollama is running.")
        st.stop()
    
    # Get judge models (all models except the selected one)
    judge_models = [model for model in available_models if model != selected_model]
    
    # Display judge models info
    st.header("👨‍⚖️ Judge Models")
    if judge_models:
        st.write(f"**{len(judge_models)} models** will evaluate each response:")
        for model in judge_models:
            st.write(f"• {model}")
    else:
        st.warning("No other models available for evaluation")
    
    # Evaluation toggle
    evaluation_enabled = st.toggle(
        "Enable Multi-Judge Evaluation", 
        value = True,
        key = "eval_toggle"
    )
    
    # Initialize multi-judge evaluator ONLY if needed
    judge_evaluator = None
    if judge_models and evaluation_enabled:
        # Check if we need to create a new evaluator
        current_evaluator = st.session_state.get("judge_evaluator")
        current_judge_models = st.session_state.get("current_judge_models", [])
        
        if current_evaluator is None or current_judge_models != judge_models:
            # Only create new evaluator if models changed or doesn't exist
            judge_evaluator = LLMJudgeEvaluator(judge_models)
            st.session_state["judge_evaluator"] = judge_evaluator
            st.session_state["current_judge_models"] = judge_models
        else:
            # Use existing evaluator
            judge_evaluator = current_evaluator
    else:
        # Clear evaluator if evaluation is disabled
        if "judge_evaluator" in st.session_state:
            del st.session_state["judge_evaluator"]
        if "current_judge_models" in st.session_state:
            del st.session_state["current_judge_models"]
    
    # Metrics actions
    st.header("📊 Evaluation Metrics")
    
    if st.button("Show Evaluation Report"):
        report = metrics_collector.generate_report()
        st.text_area("Evaluation Report", report, height = 300)
    
    if st.button("Save All Metrics to File"):
        if metrics_collector.current_session_metrics:
            filename = metrics_collector.save_all_metrics_to_file()
            if filename:
                st.success(f"All metrics saved to: {filename}")
            else:
                st.error("Failed to save metrics.")
        else:
            st.warning("No metrics to save.")
    
    if st.button("Clear Metrics"):
        metrics_collector.current_session_metrics = []
        st.success("Metrics cleared.")
        
    # Saved metrics files section
    render_saved_metrics_section()
    
    return selected_model, evaluation_enabled, judge_evaluator


def render_saved_metrics_section():
    """Render the saved evaluation files section"""
    st.header("📁 Saved Evaluation Files")
    
    # Show list of evaluation files
    evaluation_files = []
    if os.path.exists(METRICS_DIR):
        evaluation_files = [f for f in os.listdir(METRICS_DIR) if f.endswith('.json') and f.startswith('evaluation_')]
        evaluation_files.sort(reverse = True)  # Newest first
    
    if evaluation_files:
        selected_file = st.selectbox("Select evaluation file", evaluation_files)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("View Selected Evaluation"):
                filepath = os.path.join(METRICS_DIR, selected_file)
                try:
                    with open(filepath, 'r', encoding = 'utf-8') as f:
                        data = json.load(f)
                        st.json(data)
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        with col2:
            if st.button("Download Selected Evaluation"):
                filepath = os.path.join(METRICS_DIR, selected_file)
                try:
                    with open(filepath, 'r', encoding = 'utf-8') as f:
                        data = f.read()
                        st.download_button(
                            label = "Download JSON",
                            data = data,
                            file_name = selected_file,
                            mime = "application/json"
                        )
                except Exception as e:
                    st.error(f"Error reading file: {e}")
    else:
        st.info("No evaluation files found")


def render_metrics_summary(metrics_collector):
    """Render current session metrics summary"""
    if metrics_collector.current_session_metrics:
        st.subheader("📈 Current Session Summary")
        summary = metrics_collector.get_session_summary()
        
        if summary:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Interactions", summary["total_interactions"])
                st.metric("Total Evaluations", summary["total_evaluations"])
            
            with col2:
                st.metric("Avg Response Time", f"{summary['avg_response_time']}s")
                if "avg_overall_score" in summary:
                    st.metric("Overall Quality", f"{summary['avg_overall_score']}/10.0")
            
            with col3:
                st.metric("Total Tokens", summary["total_tokens_generated"])
                st.metric("Avg Tokens/s", f"{summary['avg_tokens_per_second']:.1f}")
                
            # Display rating distribution as a bar chart
            if "rating_distribution" in summary:
                st.subheader("Rating Distribution")
                rating_data = summary["rating_distribution"]
                st.bar_chart(rating_data)