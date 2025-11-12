"""
RAG chain and response generation functions with quantitative metrics.
"""

import time
import logging
from typing import Tuple, List

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from processing.vector_db import get_simple_retriever
from processing.document_processor import count_tokens


logger = logging.getLogger(__name__)


def process_question_simple(question: str, vector_db, selected_model: str) -> Tuple[str, str, List[str], float]:
    """Simple question processing with timing and context chunks"""
    logger.info(f"Simple processing: {question}")
    
    start_time = time.time()
    
    llm = ChatOllama(model = selected_model, request_timeout = 120.0)
    retriever = get_simple_retriever(vector_db)
    
    # Time retrieval
    retrieval_start = time.time()
    context_docs = retriever.invoke(question)
    retrieval_time = time.time() - retrieval_start
    
    # Calculate retrieval confidence (simple version)
    retrieval_confidence = min(len(context_docs) / 4.0, 1.0)
    
    # Simple prompt template
    template = """You are a professional legal expert. 
    
    CONTEXT INFORMATION:
    {context}
    
    QUESTION: {question}
    
    Please provide a helpful answer based on the context above. If you cannot find the answer in the context, say so.
    
    ANSWER:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Time generation
    generation_start = time.time()
    response = chain.invoke(question)
    generation_time = time.time() - generation_start
    
    # Get context for evaluation
    context = "\n\n".join([
        f"Document {i+1}: {doc.page_content[:300]}..."
        for i, doc in enumerate(context_docs[:3])
    ])
    
    # Extract raw chunks for accuracy calculation
    context_chunks = [doc.page_content for doc in context_docs[:3]]
    
    total_time = time.time() - start_time
    
    logger.info(f"Processing complete - Total: {total_time:.2f}s, Retrieval: {retrieval_time:.2f}s, Generation: {generation_time:.2f}s")
    
    return response, context, context_chunks, retrieval_confidence, retrieval_time, generation_time


def generate_response_with_metrics(prompt: str, vector_db, selected_model: str, 
                                 evaluation_enabled: bool, judge_evaluator = None) -> dict:
    """
    Generate response with comprehensive metrics tracking.
    Returns dictionary with response, context, and metrics.
    """
    try:
        if vector_db is not None:
            start_time = time.time()
            
            # Get response with detailed timing information
            response, context, context_chunks, retrieval_confidence, retrieval_time, generation_time = process_question_simple(
                prompt, vector_db, selected_model
            )
            
            total_time = time.time() - start_time
            token_count = count_tokens(response)
            
            # Prepare result dictionary
            result = {
                "response": response,
                "context": context,
                "context_chunks": context_chunks,  
                "response_time": total_time,
                "retrieval_time": retrieval_time,  
                "generation_time": generation_time,  
                "token_count": token_count,
                "retrieval_confidence": retrieval_confidence,  
                "start_time": start_time,  
                "success": True
            }
            
            # Add evaluations if enabled
            if evaluation_enabled and judge_evaluator:
                evaluation_start = time.time()
                evaluations = judge_evaluator.evaluate_response(prompt, response, context)
                evaluation_time = time.time() - evaluation_start
                result["evaluations"] = evaluations
                result["evaluation_time"] = evaluation_time  
            
            return result
        else:
            return {
                "response": "Please upload a PDF file first.",
                "context": "",
                "context_chunks": [],  
                "response_time": 0,
                "token_count": 0,
                "success": False
            }
            
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            "response": f"Error: {str(e)}",
            "context": "",
            "context_chunks": [],  
            "response_time": 0,
            "token_count": 0,
            "success": False
        }