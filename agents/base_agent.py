"""
Base agent class for all specialized agents in the multi-agent system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

class BaseAgent(ABC):
    """Base class for all agents in the horizontal multi-agent system"""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.logger = logging.getLogger(f"{__name__}.{agent_type}")
        self.shared_context = {}
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method to be implemented by each agent"""
        pass
    
    def update_shared_context(self, key: str, value: Any) -> None:
        """Update shared context for inter-agent communication"""
        self.shared_context[key] = value
        self.logger.info(f"Agent {self.agent_id} updated shared context: {key}")
    
    def get_shared_context(self, key: str, default: Any = None) -> Any:
        """Retrieve value from shared context"""
        return self.shared_context.get(key, default)