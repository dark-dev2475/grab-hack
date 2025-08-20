"""
Base module for all agents in the Grab-X project.
Provides common utilities and base classes.
"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import GOOGLE_API_KEY, LLM_MODEL

# Initialize the LLM
def get_llm():
    """Get a properly configured LLM instance."""
    return ChatGoogleGenerativeAI(model=LLM_MODEL)

# Base agent creation
def create_standard_agent(tools: List[BaseTool], system_message: str):
    """
    Create a standardized agent with consistent configuration.
    
    Args:
        tools: List of tools the agent can use
        system_message: System prompt to guide the agent
        
    Returns:
        A configured ReAct agent
    """
    llm = get_llm()
    return create_react_agent(llm, tools, system_message)

# State analysis
def analyze_user_request(messages: List[Dict], service_type: str) -> Dict[str, Any]:
    """
    Analyze user messages to determine intent and required tools.
    
    Args:
        messages: List of messages in the conversation
        service_type: The type of service (car, food, express)
        
    Returns:
        A dict with state variables based on intent analysis
    """
    # This is a placeholder that would normally use NLP to analyze the request
    # For now, we'll use simple keyword matching
    
    last_message = messages[-1]["content"] if messages else ""
    state_updates = {}
    
    # Car service analysis
    if service_type == "car":
        if "traffic" in last_message.lower():
            state_updates["check_traffic"] = True
        if "route" in last_message.lower() or "alternative" in last_message.lower():
            state_updates["alt_route"] = True
        if "notify" in last_message.lower() or "inform" in last_message.lower():
            state_updates["notify"] = True
        if "flight" in last_message.lower() or "airport" in last_message.lower():
            state_updates["flight"] = True
            
    # Food service analysis
    elif service_type == "food":
        if "overloaded" in last_message.lower() or "wait time" in last_message.lower():
            state_updates["overloaded_restaurant"] = True
        if "packaging" in last_message.lower() or "damaged" in last_message.lower():
            state_updates["packaging_dispute"] = True
            
    # Express service analysis
    elif service_type == "express":
        if "not home" in last_message.lower() or "not present" in last_message.lower():
            state_updates["recipient_not_present"] = True
            
    return state_updates
