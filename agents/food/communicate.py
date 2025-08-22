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
    """Enhanced communication with personalization"""
    # Create a more detailed system message
    system_message = """
    You are a Grab Food customer communication agent.
    
    Your task is to provide personalized, empathetic communication tailored to:
    1. The specific issue the customer experienced
    2. Their order history and loyalty status
    3. The severity of the inconvenience they faced
    4. Their communication preferences
    
    Generate personalized messages that:
    - Address the customer by name
    - Reference specific details of their issue
    - Acknowledge any inconvenience in a genuine way
    - Provide clear information about resolutions
    - End with a forward-looking, positive note
    
    Use the available tools to complete these tasks efficiently.
    """
    
    # Extract more context from messages
    last_messages = state["messages"][-3:] if len(state["messages"]) >= 3 else state["messages"]
    issue_description = " ".join([m.content for m in last_messages if hasattr(m, 'content')])
    
    # Get customer data (simulated)
    customer_id = "cust-12345"
    order_id = "order-67890"
    customer_name = "Alex"  # Would come from real customer data
    
    # Analyze issue severity from message content (simplified)
    is_severe = "spillage" in issue_description or "delay" in issue_description
    
    # Generate personalized message
    personalized_message = f"Hello {customer_name}, we've resolved your issue regarding {issue_description}. "
    if is_severe:
        personalized_message += "We understand this significantly impacted your experience and sincerely apologize. "
    personalized_message += "Thank you for your patience throughout this process."
    
    # Send tailored notification
    notification_result = send_resolution_notification(
        customer_id=customer_id,
        message=personalized_message
    )
    
    # Offer larger voucher for severe issues
    voucher_amount = 10.00 if is_severe else 5.00
    voucher_result = offer_promotional_voucher(
        customer_id=customer_id,
        amount=voucher_amount,
        expiry_days=7
    )
    
    # Return to grab_food after handling the communication
    goto = "grab_food"
    return Command(goto=goto, update={"next": goto})