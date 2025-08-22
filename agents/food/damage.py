"""
Package damage dispute resolution module for Grab Food service.
Handles situations where there's a dispute about damaged packaging.
Uses LangGraph with proper nodes and edges for workflow control.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, List, Dict, Any, Optional, Literal, TypedDict, Union, cast
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

# Import LLM from utils (when ready)
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

# Enhanced State class
class DisputeState(MessagesState):
    """State class for packaging dispute resolution."""
    evidence: Optional[Dict[str, Any]] = None
    customer_response: Optional[str] = None
    driver_response: Optional[str] = None
    analysis_result: Optional[Dict[str, Any]] = None
    resolution: Optional[str] = None
    dispute_stage: str = "initiate"  # Tracks the current stage of the dispute

# Tool definitions
@tool
def initiate_mediation_flow(dispute_details: str) -> Dict[str, Any]:
    """
    Initiates a mediation flow by pausing the order and opening communication channels.
    
    Args:
        dispute_details: Description of the packaging dispute
        
    Returns:
        Status of the mediation initiation
    """
    return {
        "status": "mediation_initiated",
        "message": "Order paused. Communication portal opened between customer and driver.",
        "dispute_details": dispute_details
    }

@tool
def collect_evidence_from_customer(question: str) -> Dict[str, Any]:
    """
    Collects evidence from the customer regarding the damaged packaging.
    
    Args:
        question: Specific question to ask the customer
        
    Returns:
        Customer's response with evidence
    """
    # In a real implementation, this would integrate with a chat UI
    # For now, we'll simulate a response
    return {
        "response": "The packaging was completely torn when I received it. The food was spilling out of the container.",
        "photos_provided": True,
        "timestamp": "2025-08-21T15:30:45Z"
    }

@tool
def collect_evidence_from_driver(question: str) -> Dict[str, Any]:
    """
    Collects evidence from the driver regarding the damaged packaging.
    
    Args:
        question: Specific question to ask the driver
        
    Returns:
        Driver's response with evidence
    """
    # In a real implementation, this would integrate with a chat UI
    # For now, we'll simulate a response
    return {
        "response": "The packaging was intact when I picked it up. I was careful during transport.",
        "photos_provided": True, 
        "timestamp": "2025-08-21T15:35:12Z"
    }

@tool
def issue_instant_refund(amount: float, reason: str) -> Dict[str, Any]:
    """
    Issues an instant refund to the customer.
    
    Args:
        amount: Refund amount
        reason: Reason for the refund
        
    Returns:
        Confirmation of the refund
    """
    return {
        "action": "refund_issued",
        "amount": amount,
        "reason": reason,
        "transaction_id": "ref-20250821-45678",
        "status": "completed"
    }

@tool
def exonerate_driver(reason: str) -> Dict[str, Any]:
    """
    Exonerates the driver from blame in the dispute.
    
    Args:
        reason: Reason for exonerating the driver
        
    Returns:
        Confirmation of driver exoneration
    """
    return {
        "action": "driver_exonerated",
        "reason": reason,
        "status": "completed"
    }

@tool
def log_merchant_packaging_issue(details: str, severity: str) -> Dict[str, Any]:
    """
    Logs an issue with the merchant's packaging for quality control.
    
    Args:
        details: Details of the packaging issue
        severity: Severity level (low, medium, high)
        
    Returns:
        Confirmation of the logged issue
    """
    return {
        "action": "merchant_packaging_logged",
        "details": details,
        "severity": severity,
        "case_id": "pkg-20250821-78901",
        "status": "under_review"
    }

@tool
def analyze_collected_evidence(customer_evidence: Dict[str, Any], driver_evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced evidence analysis with multiple factors"""
    # Extract evidence details
    customer_photos = customer_evidence.get("photos_provided", False)
    driver_photos = driver_evidence.get("photos_provided", False)
    customer_statement = customer_evidence.get("response", "")
    driver_statement = driver_evidence.get("response", "")
    
    # Analyze timing (how soon after delivery the complaint was made)
    customer_timestamp = customer_evidence.get("timestamp", "")
    delivery_timestamp = "2025-08-21T15:20:00Z"  # In a real implementation, get from order data
    
    # Calculate factors (simplified for this example)
    photo_factor = 0.6 if customer_photos else 0.2
    timing_factor = 0.7  # Higher if complaint was immediate
    statement_consistency = 0.8  # Would analyze text for consistency
    
    # Calculate weighted responsibility
    merchant_score = (photo_factor + timing_factor + statement_consistency) / 3
    
    # Determine responsibility with confidence
    if merchant_score > 0.7:
        responsibility = "merchant"
        confidence = merchant_score
    elif merchant_score < 0.4:
        responsibility = "customer"
        confidence = 1 - merchant_score
    else:
        responsibility = "unclear"
        confidence = 0.5
    
    return {
        "responsibility": responsibility,
        "confidence": confidence,
        "reasoning": "Multi-factor analysis including photo evidence, timing, and statement consistency",
        "timestamp": "2025-08-21T15:45:00Z",
        "factors_considered": {
            "photo_evidence": photo_factor,
            "timing": timing_factor,
            "statement_consistency": statement_consistency
        }
    }

# Define the node functions for the graph
def gather_initial_info(state: DisputeState) -> DisputeState:
    """Node to gather initial information about the dispute."""
    # Create a system message explaining the situation
    state.messages.append(
        SystemMessage(content="""
        You are a Grab Food dispute resolution agent handling a damaged packaging case.
        
        Your task is to:
        1. Understand the details of the dispute
        2. Decide what information to collect from the customer
        
        Use the initiate_mediation_flow tool to begin the process.
        """)
    )
    
    # Create a human message to simulate the customer's initial complaint
    state.messages.append(
        HumanMessage(content="The packaging for my order was damaged, and the food spilled all over the bag. I want a refund.")
    )
    
    # Call the ToolNode with the updated state
    return state

def collect_customer_evidence(state: DisputeState) -> DisputeState:
    """Node to collect evidence from the customer."""
    # Add a system message to guide the agent
    state.messages.append(
        SystemMessage(content="""
        Now you need to collect evidence from the customer about the damaged packaging.
        
        Ask specific questions to understand:
        1. What specific damage was observed
        2. When the damage was noticed
        3. If the customer has photos of the damaged packaging
        
        Use the collect_evidence_from_customer tool.
        """)
    )
    
    # Call the tool to collect evidence
    evidence = collect_evidence_from_customer("Please describe the damage to your packaging and provide any photos if available.")
    
    # Store the evidence in the state
    state.evidence = evidence
    state.customer_response = evidence["response"]
    
    # Add a function message with the evidence
    state.messages.append(
        FunctionMessage(content=str(evidence), name="collect_evidence_from_customer")
    )
    
    return state

def collect_driver_evidence(state: DisputeState) -> DisputeState:
    """Node to collect evidence from the driver."""
    # Add a system message to guide the agent
    state.messages.append(
        SystemMessage(content="""
        Now you need to collect evidence from the driver about the packaging.
        
        Ask specific questions to understand:
        1. The condition of the packaging when picked up
        2. How the food was transported
        3. If the driver has photos from pickup
        
        Use the collect_evidence_from_driver tool.
        """)
    )
    
    # Call the tool to collect evidence
    evidence = collect_evidence_from_driver("Please describe the condition of the packaging when you picked up the order and how you transported it.")
    
    # Store the evidence in the state
    if state.evidence:
        state.evidence.update({"driver_evidence": evidence})
    else:
        state.evidence = {"driver_evidence": evidence}
    
    state.driver_response = evidence["response"]
    
    # Add a function message with the evidence
    state.messages.append(
        FunctionMessage(content=str(evidence), name="collect_evidence_from_driver")
    )
    
    return state

def analyze_dispute_evidence(state: DisputeState) -> DisputeState:
    """Node to analyze the collected evidence and determine responsibility."""
    if not state.evidence or not state.customer_response or not state.driver_response:
        # If evidence is missing, we can't analyze properly
        state.messages.append(
            SystemMessage(content="Insufficient evidence collected. Please gather more information.")
        )
        return state
    
    # Get the customer and driver evidence
    customer_evidence = state.evidence
    driver_evidence = state.evidence.get("driver_evidence", {})
    
    # Analyze the evidence
    analysis_result = analyze_collected_evidence(customer_evidence, driver_evidence)
    
    # Store the analysis result in the state
    state.analysis_result = analysis_result
    
    # Add a message with the analysis results
    state.messages.append(
        FunctionMessage(content=str(analysis_result), name="analyze_collected_evidence")
    )
    
    # Add a system message for next steps
    state.messages.append(
        SystemMessage(content=f"""
        Analysis complete. The evidence suggests that the {analysis_result['responsibility']} 
        is likely responsible for the damaged packaging with a confidence of {analysis_result['confidence']}.
        
        Based on this analysis, determine the appropriate resolution action.
        """)
    )
    
    return state

def resolve_dispute(state: DisputeState) -> DisputeState:
    """Enhanced resolution logic with customer value consideration"""
    if not state.analysis_result:
        return state
        
    responsibility = state.analysis_result.get("responsibility", "unknown")
    confidence = state.analysis_result.get("confidence", 0.5)
    
    # Check customer value (simplified - would use actual data)
    customer_value = "high"  # Would be determined from order history
    
    # Refund logic with customer value consideration
    if responsibility == "merchant" or (responsibility == "unclear" and customer_value == "high"):
        # Full refund for clear merchant fault or high-value customers with unclear fault
        refund_amount = 15.00
        reason = "Full refund issued due to packaging issues"
    elif responsibility == "unclear":
        # Partial refund for unclear responsibility with regular customers
        refund_amount = 7.50
        reason = "Partial refund issued as goodwill gesture"
    else:
        # No refund for clear customer fault
        refund_amount = 0
        reason = "No refund issued as analysis indicates damage occurred after delivery"
        
    # Rest of the function remains similar but uses these new variables
    if responsibility == "merchant":
        # Issue a refund to the customer
        refund_result = issue_instant_refund(
            amount=refund_amount,
            reason=reason
        )
        
        # Exonerate the driver
        driver_result = exonerate_driver(
            reason="Driver not responsible for pre-packaging issues"
        )
        
        # Log the merchant packaging issue
        merchant_result = log_merchant_packaging_issue(
            details="Food packaging damaged causing spillage",
            severity="high"
        )
        
        # Update the state with the resolution
        state.resolution = "customer_refunded_merchant_logged"
        
        # Add messages with the resolution actions
        state.messages.append(
            FunctionMessage(content=str(refund_result), name="issue_instant_refund")
        )
        state.messages.append(
            FunctionMessage(content=str(driver_result), name="exonerate_driver")
        )
        state.messages.append(
            FunctionMessage(content=str(merchant_result), name="log_merchant_packaging_issue")
        )
        
    elif responsibility == "customer":
        # If the customer is responsible, no refund is issued
        state.resolution = "no_refund_issued"
        
        # Add a message explaining the decision
        state.messages.append(
            AIMessage(content="""
            Based on the evidence provided, we cannot issue a refund as the analysis indicates 
            the damage likely occurred after delivery. The driver has been exonerated.
            """)
        )
        
        # Exonerate the driver
        driver_result = exonerate_driver(
            reason="Driver not responsible based on evidence analysis"
        )
        
        state.messages.append(
            FunctionMessage(content=str(driver_result), name="exonerate_driver")
        )
        
    else:
        # If responsibility is unclear, partial refund may be issued
        refund_result = issue_instant_refund(
            amount=refund_amount,
            reason=reason
        )
        
        state.resolution = "partial_refund_issued"
        
        state.messages.append(
            FunctionMessage(content=str(refund_result), name="issue_instant_refund")
        )
    
    # Add a final summary message
    state.messages.append(
        AIMessage(content=f"""
        Dispute resolution complete.
        
        Resolution: {state.resolution}
        
        Thank you for your patience during this process. If you have any further concerns,
        please contact Grab customer support.
        """)
    )
    
    return state

def should_collect_more_evidence(state: DisputeState) -> Union[Literal["collect_driver"], Literal["analyze"]]:
    """Conditional edge to determine if more evidence is needed."""
    # If we have customer evidence but not driver evidence, collect from driver
    if state.customer_response and not state.driver_response:
        return "collect_driver"
    
    # Otherwise, proceed to analysis
    return "analyze"

def determine_next_step(state: DisputeState) -> Union[Literal["resolve"], Literal["human_review"], Literal["collect_more"]]:
    """Conditional edge to determine the next step after analysis."""
    if not state.analysis_result:
        return "collect_more"
    
    confidence = state.analysis_result.get("confidence", 0)
    
    # If confidence is high enough, proceed to resolution
    if confidence >= 0.6:
        return "resolve"
    
    # Otherwise, require human review
    return "human_review"

def human_review_node(state: DisputeState) -> DisputeState:
    """Node for human review of complex cases."""
    state.messages.append(
        SystemMessage(content="""
        This case requires human review due to complexity or low confidence in automated analysis.
        
        A Grab support specialist will review the evidence and make a final determination.
        """)
    )
    
    # In a real implementation, this would integrate with a human review system
    # For now, we'll simulate a human decision
    state.messages.append(
        HumanMessage(content="After reviewing the evidence, I've determined the merchant is responsible. Please issue a full refund.")
    )
    
    # Update the analysis result with the human decision
    if state.analysis_result:
        state.analysis_result["responsibility"] = "merchant"
        state.analysis_result["confidence"] = 1.0  # Human decision has full confidence
        state.analysis_result["human_reviewed"] = True
    else:
        state.analysis_result = {
            "responsibility": "merchant",
            "confidence": 1.0,
            "human_reviewed": True
        }
    
    return state

# The ToolNode for all tools
tools_node = ToolNode(tools=[
    initiate_mediation_flow,
    collect_evidence_from_customer,
    collect_evidence_from_driver,
    analyze_collected_evidence,
    issue_instant_refund,
    exonerate_driver,
    log_merchant_packaging_issue
])

def manage_packaging_dispute(dispute_details: str = "Customer reports damaged packaging with food spillage") -> StateGraph:
    """
    Creates and returns a LangGraph workflow for managing packaging disputes.
    
    Args:
        dispute_details: Description of the packaging dispute
        
    Returns:
        A compiled StateGraph workflow ready to be executed
    """
    # Create the workflow
    workflow = StateGraph(DisputeState)
    
    # Add all the nodes
    workflow.add_node("gather_info", gather_initial_info)
    workflow.add_node("collect_customer", collect_customer_evidence)
    workflow.add_node("collect_driver", collect_driver_evidence)
    workflow.add_node("analyze", analyze_dispute_evidence)
    workflow.add_node("resolve", resolve_dispute)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("tools", tools_node)
    
    # Set the entry point
    workflow.set_entry_point("gather_info")
    
    # Connect the nodes with edges
    workflow.add_edge("gather_info", "tools")
    workflow.add_edge("tools", "collect_customer")
    
    # Conditional edge from customer evidence collection
    workflow.add_conditional_edges(
        "collect_customer",
        should_collect_more_evidence,
        {
            "collect_driver": "collect_driver",
            "analyze": "analyze"
        }
    )
    
    workflow.add_edge("collect_driver", "analyze")
    
    # Conditional edge from analysis
    workflow.add_conditional_edges(
        "analyze",
        determine_next_step,
        {
            "resolve": "resolve",
            "human_review": "human_review",
            "collect_more": "collect_customer"
        }
    )
    
    # Connect human review to resolution
    workflow.add_edge("human_review", "resolve")
    
    # Final resolution leads to END
    workflow.add_edge("resolve", END)
    
    # Compile the workflow
    return workflow.compile()

def run_packaging_dispute_workflow(dispute_details: str = "Customer reports damaged packaging with food spillage") -> Dict[str, Any]:
    """
    Runs the packaging dispute resolution workflow with the given dispute details.
    
    Args:
        dispute_details: Description of the packaging dispute
        
    Returns:
        The final state of the workflow execution
    """
    # Create the workflow
    workflow = manage_packaging_dispute(dispute_details)
    
    # Create a memory saver for checkpointing
    memory = MemorySaver()
    
    # Execute the workflow
    config = {"configurable": {"thread_id": "packaging_dispute_1"}}
    
    for event in workflow.stream({
        "messages": [
            SystemMessage(content=f"You are handling a packaging dispute: {dispute_details}"),
            HumanMessage(content=dispute_details)
        ],
        "dispute_stage": "initiate",
    }, config=config):
        # This would normally display progress or allow for interaction
        # In a real application, you could implement checkpointing or visualization here
        pass
    
    # Get the final state
    result = workflow.get_state(config)
    
    return result

# Example usage:
# if __name__ == "__main__":
#     result = run_packaging_dispute_workflow("My food delivery arrived with torn packaging and food spilled in the bag.")
#     for message in result["messages"]:
#         print(f"{message.type}: {message.content[:100]}...")
    
    # Return to grab_food after handling the dispute
    goto = "grab_food"
    return Command(goto=goto, update={"next": goto})
