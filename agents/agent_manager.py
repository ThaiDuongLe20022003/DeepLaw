"""
Agent Manager orchestrates the horizontal multi-agent system.
"""

import logging
from typing import Dict, Any, List
import streamlit as st

from agents.base_agent import BaseAgent


class AgentManager:
    """Orchestrates communication and workflow between horizontal agents"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger(__name__)
        self.event_subscribers: Dict[str, List[str]] = {}
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the manager"""
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type})")
    
    def publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish event to subscribed agents"""
        self.logger.info(f"Publishing event: {event_type}")
        
        # Store in session state for shared memory
        if "agent_events" not in st.session_state:
            st.session_state.agent_events = {}
        st.session_state.agent_events[event_type] = data
        
        # Notify subscribed agents
        for agent_id in self.event_subscribers.get(event_type, []):
            if agent_id in self.agents:
                self.logger.info(f"Notifying agent {agent_id} of event {event_type}")
    
    def subscribe_to_event(self, agent_id: str, event_type: str) -> None:
        """Subscribe an agent to specific event types"""
        if event_type not in self.event_subscribers:
            self.event_subscribers[event_type] = []
        self.event_subscribers[event_type].append(agent_id)
        self.logger.info(f"Agent {agent_id} subscribed to {event_type}")
    
    def process_query(self, user_query: str, file_upload = None) -> Dict[str, Any]:
        """Main method to process user query through agent pipeline"""
        self.logger.info(f"Processing query: {user_query}")
        
        # Initialize result structure
        result = {
            "query": user_query,
            "agents_engaged": [],
            "confidence_scores": {},
            "final_response": "",
            "evaluations": []
        }
        
        try:
            # Start with PDF processing if file provided
            if file_upload and "pdf_agent" in self.agents:
                pdf_result = self.agents["pdf_agent"].process({
                    "file_upload": file_upload,
                    "query": user_query
                })
                result["agents_engaged"].append("pdf_agent")
                self.publish_event("pdf_processed", pdf_result)
            
            # Continue with retrieval agent
            if "retrieval_agent" in self.agents:
                retrieval_result = self.agents["retrieval_agent"].process({
                    "query": user_query,
                    "context": result
                })
                result.update(retrieval_result)
                result["agents_engaged"].append("retrieval_agent")
                
                # Confidence-based triggering
                confidence = retrieval_result.get("retrieval_confidence", 0)
                result["confidence_scores"]["retrieval"] = confidence
                
                self.publish_event("retrieval_complete", retrieval_result)
                
                # Trigger legal analyzer based on confidence
                if confidence < 0.7 and "legal_analyzer_agent" in self.agents:
                    legal_result = self.agents["legal_analyzer_agent"].process({
                        "query": user_query,
                        "retrieved_context": retrieval_result.get("context", ""),
                        "confidence": confidence
                    })
                    result.update(legal_result)
                    result["agents_engaged"].append("legal_analyzer_agent")
                    self.publish_event("legal_analysis_complete", legal_result)
            
            # Generate final response
            if "response_agent" in self.agents:
                response_result = self.agents["response_agent"].process({
                    "query": user_query,
                    "context": result
                })
                result.update(response_result)
                result["agents_engaged"].append("response_agent")
                self.publish_event("response_generated", response_result)
            
            # Quality assurance (post-processing)
            if "qa_agent" in self.agents:
                qa_result = self.agents["qa_agent"].process({
                    "query": user_query,
                    "response": result.get("final_response", ""),
                    "context": result.get("context", ""),
                    "agent_sequence": result["agents_engaged"]
                })
                result["evaluations"] = qa_result.get("evaluations", [])
                result["agents_engaged"].append("qa_agent")
            
            self.logger.info(f"Query processing complete. Agents engaged: {result['agents_engaged']}")
            
        except Exception as e:
            self.logger.error(f"Error in agent pipeline: {e}")
            result["error"] = str(e)
            result["final_response"] = f"I apologize, but I encountered an error: {str(e)}"
        
        return result