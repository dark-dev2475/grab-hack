"""
LLM utilities for the Grab-X project.
Provides functions for initializing and configuring LLMs.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import LLM_MODEL

def initialize_llm(model: str = None):
    """
    Initialize a LangChain LLM with proper configuration.
    
    Args:
        model: The model name to use (default: from settings)
        
    Returns:
        A configured ChatGoogleGenerativeAI instance
    """
    model_name = model or LLM_MODEL
    return ChatGoogleGenerativeAI(model=model_name)

def get_chat_model(temperature: float = 0.7, streaming: bool = False):
    """
    Get a chat model with specific configuration.
    
    Args:
        temperature: Creativity of responses (0.0-1.0)
        streaming: Whether to enable streaming responses
        
    Returns:
        A configured ChatGoogleGenerativeAI instance
    """
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=temperature,
        streaming=streaming
    )
