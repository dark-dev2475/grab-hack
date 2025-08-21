"""
Customer communication module for Grab Food service.
Handles all communications with customers after issues are resolved.
Uses LangGraph with proper state management.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, List, Dict, Any, Optional, Literal, TypedDict, Union
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.types import Command

# Import LLM from utils (when ready)
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

# Enhanced State class for communication
class CommunicationState(MessagesState):
    """State class for customer communication."""
    issue_type: Optional[str] = None
    resolution: Optional[str] = None
    customer_id: Optional[str] = None
    order_id: Optional[str] = None
    communication_stage: str = "initial"  # Tracks the current stage of communication

# Tool definitions
@tool
def send_resolution_notification(customer_id: str, message: str) -> Dict[str, Any]:
    """
    Sends a resolution notification to the customer.
    
    Args:
        customer_id: The ID of the customer
        message: The resolution message to send
        
    Returns:
        Status of the notification
    """
    # In a real implementation, this would connect to a notification service
    return {
        "status": "notification_sent",
        "customer_id": customer_id,
        "message": message,
        "timestamp": "2025-08-22T14:30:45Z"
    }

@tool
def issue_satisfaction_survey(customer_id: str, order_id: str) -> Dict[str, Any]:
    """
    Issues a satisfaction survey to the customer.
    
    Args:
        customer_id: The ID of the customer
        order_id: The ID of the order
        
    Returns:
        Status of the survey issuance
    """
    # In a real implementation, this would trigger a survey
    return {
        "status": "survey_issued",
        "customer_id": customer_id,
        "order_id": order_id,
        "survey_id": "srv-20250822-12345",
        "expiry": "2025-08-29T23:59:59Z"
    }

@tool
def offer_promotional_voucher(customer_id: str, amount: float, expiry_days: int) -> Dict[str, Any]:
    """
    Offers a promotional voucher to the customer as goodwill.
    
    Args:
        customer_id: The ID of the customer
        amount: The voucher amount
        expiry_days: Number of days until voucher expires
        
    Returns:
        Status of the voucher issuance
    """
    # In a real implementation, this would generate a voucher
    return {
        "status": "voucher_issued",
        "customer_id": customer_id,
        "voucher_code": "GRABFOOD25",
        "amount": amount,
        "expiry_date": f"2025-08-{22 + expiry_days}T23:59:59Z"
    }

def communicate(state: MessagesState) -> Command[Literal['grab_food']]:
    """
    Communicates with the customer after an issue has been resolved.
    Handles notifications, satisfaction surveys, and promotional vouchers.
    
    Args:
        state: The current state containing message history
        
    Returns:
        Command to go back to the grab_food agent
    """
    # Create a system message to guide the agent
    system_message = """
    You are a Grab Food customer communication agent.
    Your task is to provide clear, concise, and empathetic communication to customers
    after an issue has been resolved.
    
    Follow these steps:
    1. Send a resolution notification to the customer
    2. Issue a satisfaction survey if appropriate
    3. Offer a promotional voucher if the customer experienced significant inconvenience
    
    Use the available tools to complete these tasks efficiently.
    """
    
    # Get the last message to understand what issue was resolved
    last_message = state["messages"][-1] if state["messages"] else None
    issue_description = last_message.content if last_message else "an issue"
    
    # Simulate a customer ID and order ID (in a real app, these would come from the state)
    customer_id = "cust-12345"
    order_id = "order-67890"
    
    # Send notification about the resolution
    notification_result = send_resolution_notification(
        customer_id=customer_id,
        message=f"Your issue regarding {issue_description} has been resolved. Thank you for your patience."
    )
    
    # Issue a satisfaction survey
    survey_result = issue_satisfaction_survey(
        customer_id=customer_id,
        order_id=order_id
    )
    
    # Offer a promotional voucher for the inconvenience
    voucher_result = offer_promotional_voucher(
        customer_id=customer_id,
        amount=5.00,  # $5 voucher
        expiry_days=7  # Valid for 7 days
    )
    
    # Return to grab_food after handling the communication
    goto = "grab_food"
    return Command(goto=goto, update={"next": goto})