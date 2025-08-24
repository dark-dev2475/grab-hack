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
from types import SimpleNamespace

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
    customer_evidence: Optional[Dict[str, Any]] = None
    driver_evidence: Optional[Dict[str, Any]] = None
    analysis_result: Optional[Dict[str, Any]] = None
    resolution: Optional[str] = None
    dispute_stage: str = "initiate"  # Tracks the current stage of the dispute
    actions_taken: List[Dict[str, Any]] = []
    dispute_details: Optional[str] = None
    next: str = "grab_food"
    status: str = "in_progress"

# Tool definitions
@tool
def initiate_mediation_flow(dispute_details: str) -> Dict[str, Any]:
    """
    Initiates a real-time mediation flow by pausing the order completion and opening 
    synchronized communication channels between the customer and driver.
    
    Args:
        dispute_details: Description of the packaging dispute
        
    Returns:
        Status of the mediation initiation
    """
    return {
        "status": "mediation_initiated",
        "message": "Order paused. Real-time communication portal opened between customer and driver.",
        "dispute_details": dispute_details
    }

@tool
def collect_evidence_from_customer(question: str) -> Dict[str, Any]:
    """
    Collects evidence from the customer regarding the damaged packaging.
    Prompts the customer to take photos of the damage and answer specific questions.
    
    Args:
        question: Specific question to ask the customer about the packaging
        
    Returns:
        Customer's response with evidence
    """
    # In a real implementation, this would integrate with a chat UI
    # For now, we'll simulate a response
    responses = {
        "Was the packaging seal intact when you received it?": 
            {"response": "No, the seal was broken and food was leaking out", "photos_provided": True},
        "Did you notice any damage before the driver handed you the order?": 
            {"response": "Yes, I saw liquid leaking from the bag when the driver was approaching", "photos_provided": True},
        "Can you describe the condition of the packaging?": 
            {"response": "The container was torn on one side and the food was spilling out", "photos_provided": True},
        "default": 
            {"response": "The packaging was completely torn when I received it. The food was spilling out of the container.", "photos_provided": True}
    }
    
    result = responses.get(question, responses["default"])
    result["timestamp"] = "2025-08-21T15:30:45Z"
    
    return result

@tool
def collect_evidence_from_driver(question: str) -> Dict[str, Any]:
    """
    Collects evidence from the driver regarding the damaged packaging.
    Prompts the driver to take photos and answer specific questions about the order handling.
    
    Args:
        question: Specific question to ask the driver about the order handling
        
    Returns:
        Driver's response with evidence
    """
    # In a real implementation, this would integrate with a chat UI
    # For now, we'll simulate a response
    responses = {
        "Was the packaging properly sealed when you picked it up from the restaurant?": 
            {"response": "Yes, it was sealed properly with the merchant's sticker", "photos_provided": True},
        "How did you transport the order?": 
            {"response": "In the insulated delivery bag, placed flat on the bottom", "photos_provided": True},
        "Did you notice any damage before reaching the customer?": 
            {"response": "No, the order seemed fine when I took it out of my bag", "photos_provided": True},
        "default": 
            {"response": "The packaging was intact when I picked it up. I was careful during transport.", "photos_provided": True}
    }
    
    result = responses.get(question, responses["default"])
    result["timestamp"] = "2025-08-21T15:35:12Z"
    
    return result

@tool
def issue_instant_refund(amount: float, reason: str) -> Dict[str, Any]:
    """
    Issues an instant refund to the customer for the damaged order.
    Immediately compensates the customer without requiring further review.
    
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
    Ensures the driver's rating is not affected by the packaging issue.
    
    Args:
        reason: Reason for exonerating the driver
        
    Returns:
        Confirmation of driver exoneration
    """
    return {
        "action": "driver_exonerated",
        "reason": reason,
        "status": "completed",
        "driver_protected": True
    }

@tool
def log_merchant_packaging_feedback(details: str, severity: str) -> Dict[str, Any]:
    """
    Logs detailed feedback about the merchant's packaging for quality improvement.
    Sends an evidence-backed report to the merchant to help them improve their packaging.
    
    Args:
        details: Specific details of the packaging issue
        severity: Severity level (low, medium, high)
        
    Returns:
        Confirmation of the logged feedback
    """
    return {
        "action": "merchant_packaging_logged",
        "details": details,
        "severity": severity,
        "case_id": "pkg-20250821-78901",
        "status": "under_review",
        "merchant_notified": True
    }

@tool
def notify_resolution(customer_message: str, driver_message: str) -> Dict[str, Any]:
    """
    Notifies both the customer and driver about the resolution outcome.
    Provides clear communication about the resolution and next steps.
    
    Args:
        customer_message: Message to send to the customer
        driver_message: Message to send to the driver
        
    Returns:
        Confirmation that both parties have been notified
    """
    return {
        "action": "parties_notified",
        "customer_message": customer_message,
        "driver_message": driver_message,
        "timestamp": "2025-08-21T15:50:30Z",
        "status": "completed"
    }

# Node functions for the LangGraph
def initiate_mediation_node(state: dict) -> dict:
    """
    Initiates the mediation process between customer and driver.
    """
    # Convert incoming dictionary to object for easier handling
    state_obj = SimpleNamespace(**state)
    
    # Get dispute details from the state or use a default message
    dispute_details = getattr(state_obj, 'dispute_details', 
                             "Dispute regarding damaged packaging during delivery")
    
    # Call the tool to initiate mediation
    result = initiate_mediation_flow(dispute_details)
    
    # Update state with the result
    if not hasattr(state_obj, 'actions_taken'):
        state_obj.actions_taken = []
    
    state_obj.actions_taken.append({"action": "initiate_mediation", "result": result})
    state_obj.messages.append(FunctionMessage(content=str(result), name="initiate_mediation_flow"))
    state_obj.dispute_stage = "collect_evidence"
    
    # Return only the updated fields
    return {
        "actions_taken": state_obj.actions_taken,
        "messages": state_obj.messages,
        "dispute_stage": state_obj.dispute_stage
    }

def collect_evidence_node(state: dict) -> dict:
    """
    Collects evidence from both the customer and driver regarding the dispute.
    Uses the LLM to generate appropriate questions based on the dispute context.
    """
    # Convert incoming dictionary to object for easier handling
    state_obj = SimpleNamespace(**state)
    
    # Create a system message to guide the LLM
    system_message = """
    You are a Grab Food dispute resolution agent handling a packaging damage situation.
    Your task is to generate appropriate questions to ask both the customer and driver 
    to collect evidence about what happened.
    
    Generate questions that:
    1. Are specific and relevant to packaging damage
    2. Help determine responsibility (merchant, driver, or customer)
    3. Request factual information rather than opinions
    
    Provide exactly one question for the customer and one for the driver.
    """
    
    # Prepare prompt with context from the dispute
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="Generate one specific question each for the customer and driver to collect evidence about the damaged packaging. Format as JSON with 'customer_question' and 'driver_question' keys.")
    ])
    
    # Get questions from LLM
    chain = prompt | llm
    response = chain.invoke({"messages": state_obj.messages})
    
    try:
        # Try to extract the questions from the response
        # This is a simplified approach - in a real implementation, use a proper output parser
        response_content = response.content
        import json
        import re
        
        # Try to extract JSON if wrapped in markdown code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
        if json_match:
            response_content = json_match.group(1)
        
        # Try to parse as JSON
        questions = json.loads(response_content)
        customer_question = questions.get("customer_question", "Can you describe the condition of the packaging when you received it?")
        driver_question = questions.get("driver_question", "Was the packaging properly sealed when you picked it up from the restaurant?")
    except Exception as e:
        # Fallback to default questions if parsing fails
        customer_question = "Can you describe the condition of the packaging when you received it?"
        driver_question = "Was the packaging properly sealed when you picked it up from the restaurant?"
    
    # Collect evidence from both parties
    customer_evidence = collect_evidence_from_customer(customer_question)
    driver_evidence = collect_evidence_from_driver(driver_question)
    
    # Update state with collected evidence
    state_obj.customer_evidence = customer_evidence
    state_obj.driver_evidence = driver_evidence
    state_obj.dispute_stage = "analyze_evidence"
    
    # Add evidence collection to messages
    state_obj.messages.append(FunctionMessage(
        content=f"Customer question: {customer_question}\nCustomer response: {customer_evidence['response']}",
        name="collect_evidence_from_customer"
    ))
    state_obj.messages.append(FunctionMessage(
        content=f"Driver question: {driver_question}\nDriver response: {driver_evidence['response']}",
        name="collect_evidence_from_driver"
    ))
    
    # Add to actions taken
    state_obj.actions_taken.append({
        "action": "collect_evidence", 
        "result": {
            "customer_evidence": customer_evidence,
            "driver_evidence": driver_evidence
        }
    })
    
    # Return only the updated fields
    return {
        "customer_evidence": state_obj.customer_evidence,
        "driver_evidence": state_obj.driver_evidence,
        "messages": state_obj.messages,
        "actions_taken": state_obj.actions_taken,
        "dispute_stage": state_obj.dispute_stage
    }

def analyze_evidence_node(state: dict) -> dict:
    """
    Analyzes the collected evidence using the LLM to determine responsibility.
    Makes an intelligent decision based on the evidence provided by both parties.
    """
    # Convert incoming dictionary to object for easier handling
    state_obj = SimpleNamespace(**state)
    
    # Create a system message to guide the LLM
    system_message = """
    You are a Grab Food dispute resolution agent analyzing evidence in a packaging damage case.
    
    Your task is to:
    1. Review the evidence provided by both the customer and driver
    2. Analyze the statements and timing of reports
    3. Consider the presence of photographic evidence
    4. Determine the most likely responsible party (merchant, driver, or customer)
    5. Assign a confidence level to your determination
    
    Be objective and fair in your analysis. Consider all evidence carefully.
    """
    
    # Prepare prompt with evidence from both parties
    customer_evidence = state_obj.customer_evidence or {}
    driver_evidence = state_obj.driver_evidence or {}
    
    evidence_summary = f"""
    Customer Evidence:
    - Statement: {customer_evidence.get('response', 'No response')}
    - Photos Provided: {customer_evidence.get('photos_provided', False)}
    - Timestamp: {customer_evidence.get('timestamp', 'Unknown')}
    
    Driver Evidence:
    - Statement: {driver_evidence.get('response', 'No response')}
    - Photos Provided: {driver_evidence.get('photos_provided', False)}
    - Timestamp: {driver_evidence.get('timestamp', 'Unknown')}
    """
    
    state_obj.messages.append(HumanMessage(content=f"Analyze the following evidence and determine responsibility:\n{evidence_summary}"))
    
    # Generate the analysis
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    chain = prompt | llm | StrOutputParser()
    analysis_text = chain.invoke({"messages": state_obj.messages})
    
    # Parse responsibility from analysis
    responsibility = "merchant"  # Default to merchant responsibility
    confidence = 0.8  # Default confidence
    
    if "driver" in analysis_text.lower() and "fault" in analysis_text.lower():
        responsibility = "driver"
    elif "customer" in analysis_text.lower() and "fault" in analysis_text.lower():
        responsibility = "customer"
    
    # More sophisticated parsing could be implemented here
    
    # Create analysis result
    analysis_result = {
        "responsibility": responsibility,
        "confidence": confidence,
        "reasoning": analysis_text,
        "timestamp": "2025-08-21T15:45:00Z",
        "evidence_considered": {
            "customer_evidence": customer_evidence,
            "driver_evidence": driver_evidence
        }
    }
    
    # Update state with analysis
    state_obj.analysis_result = analysis_result
    state_obj.dispute_stage = "resolve"
    state_obj.messages.append(AIMessage(content=analysis_text))
    
    # Add to actions taken
    state_obj.actions_taken.append({
        "action": "analyze_evidence",
        "result": {
            "responsibility": responsibility,
            "confidence": confidence
        }
    })
    
    # Return only the updated fields
    return {
        "analysis_result": state_obj.analysis_result,
        "messages": state_obj.messages,
        "actions_taken": state_obj.actions_taken,
        "dispute_stage": state_obj.dispute_stage
    }

def resolve_dispute_node(state: dict) -> dict:
    """
    Resolves the dispute based on the evidence analysis.
    Takes appropriate actions (refund, exoneration, merchant feedback) based on the determined responsibility.
    """
    # Convert incoming dictionary to object for easier handling
    state_obj = SimpleNamespace(**state)
    
    # Get the analysis result
    analysis_result = state_obj.analysis_result or {}
    responsibility = analysis_result.get("responsibility", "unclear")
    
    # Create resolution actions based on responsibility
    if responsibility == "merchant":
        # Issue refund, exonerate driver, log merchant issue
        refund_result = issue_instant_refund(10.00, "Merchant packaging fault confirmed by evidence")
        exonerate_result = exonerate_driver("Packaging issue was from merchant, not driver handling")
        merchant_result = log_merchant_packaging_feedback(
            "Food container was not properly sealed by merchant staff", "high"
        )
        
        resolution = "The evidence indicates this was a merchant packaging issue. Customer has been refunded and driver record protected."
    
    elif responsibility == "driver":
        # Still issue refund, but don't exonerate driver
        refund_result = issue_instant_refund(10.00, "Damage during delivery, customer compensated")
        exonerate_result = {"action": "driver_not_exonerated", "reason": "Evidence indicates mishandling during transport"}
        merchant_result = {"action": "no_merchant_action", "reason": "Packaging was appropriate"}
        
        resolution = "The evidence indicates damage occurred during delivery. Customer has been refunded."
    
    else:  # unclear or customer
        # Partial refund as goodwill
        refund_result = issue_instant_refund(5.00, "Goodwill refund despite unclear responsibility")
        exonerate_result = exonerate_driver("Insufficient evidence to assign responsibility to driver")
        merchant_result = log_merchant_packaging_feedback(
            "Consider improving packaging robustness, though not clearly at fault in this case", "low"
        )
        
        resolution = "The evidence is inconclusive. A partial refund has been issued as goodwill."
    
    # Notify both parties
    customer_message = f"We've reviewed your damaged packaging report. {resolution}"
    driver_message = f"The damaged packaging dispute has been resolved. {exonerate_result.get('reason', '')}"
    
    notification_result = notify_resolution(customer_message, driver_message)
    
    # Update state with resolution
    state_obj.resolution = resolution
    state_obj.dispute_stage = "completed"
    state_obj.status = "resolved"
    
    # Add resolution actions to messages
    state_obj.messages.append(FunctionMessage(content=str(refund_result), name="issue_instant_refund"))
    state_obj.messages.append(FunctionMessage(content=str(exonerate_result), name="exonerate_driver"))
    state_obj.messages.append(FunctionMessage(content=str(merchant_result), name="log_merchant_packaging_feedback"))
    state_obj.messages.append(FunctionMessage(content=str(notification_result), name="notify_resolution"))
    
    # Add final summary
    system_prompt = "You are a Grab Food dispute resolution agent. Summarize the dispute resolution process and outcome."
    state_obj.messages.append(HumanMessage(content="Please provide a final summary of this dispute resolution."))
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"messages": state_obj.messages})
    
    state_obj.messages.append(AIMessage(content=summary))
    
    # Add to actions taken
    state_obj.actions_taken.append({
        "action": "resolve_dispute",
        "result": {
            "resolution": resolution,
            "refund_issued": refund_result.get("action") == "refund_issued",
            "driver_exonerated": exonerate_result.get("action") == "driver_exonerated",
            "merchant_feedback": merchant_result.get("action") == "merchant_packaging_logged"
        }
    })
    
    # Return only the updated fields
    return {
        "resolution": state_obj.resolution,
        "messages": state_obj.messages,
        "actions_taken": state_obj.actions_taken,
        "dispute_stage": state_obj.dispute_stage,
        "status": state_obj.status
    }

def final_state_node(state: dict) -> dict:
    """
    Final processing before returning to parent agent.
    """
    # Convert incoming dictionary to object for easier handling
    state_obj = SimpleNamespace(**state)
    
    # Ensure next is set to grab_food (parent agent)
    state_obj.next = "grab_food"
    
    # Return only the updated fields
    return {
        "next": state_obj.next
    }

# Build the packaging dispute resolution graph
def build_packaging_dispute_graph() -> StateGraph:
    """
    Build the LangGraph for packaging dispute resolution.
    
    Returns:
        A StateGraph instance for the workflow
    """
    # Create the graph
    workflow = StateGraph(DisputeState)
    
    # Add nodes
    workflow.add_node("initiate_mediation", initiate_mediation_node)
    workflow.add_node("collect_evidence", collect_evidence_node)
    workflow.add_node("analyze_evidence", analyze_evidence_node)
    workflow.add_node("resolve_dispute", resolve_dispute_node)
    workflow.add_node("final", final_state_node)
    
    # Set the entry point
    workflow.set_entry_point("initiate_mediation")
    
    # Add edges between nodes
    workflow.add_edge("initiate_mediation", "collect_evidence")
    workflow.add_edge("collect_evidence", "analyze_evidence")
    workflow.add_edge("analyze_evidence", "resolve_dispute")
    
    # Add conditional re-analysis edge (circular workflow for better intelligence)
    # If the confidence is low, go back to collecting more evidence
    def conditional_router(state: DisputeState) -> str:
        """Route based on analysis confidence"""
        analysis = state.analysis_result or {}
        confidence = analysis.get("confidence", 0.0)
        if confidence < 0.6 and state.dispute_stage != "completed":
            return "collect_evidence"
        else:
            return "resolve_dispute"
    
    workflow.add_conditional_edges(
        "analyze_evidence",
        conditional_router,
        {
            "collect_evidence": "collect_evidence",
            "resolve_dispute": "resolve_dispute"
        }
    )
    
    # Add final edge to the parent agent
    workflow.add_edge("resolve_dispute", "final")
    workflow.add_edge("final", END)
    
    return workflow

# Main function for managing packaging disputes
def manage_packaging_dispute(state: DisputeState) -> Command[Literal["grab_food"]]:
    """
    Handle situations where there's a dispute about damaged packaging during delivery.
    Uses a sophisticated LangGraph workflow with proper nodes and edges for intelligent resolution.
    
    Args:
        state: The current state including messages
        
    Returns:
        Command with the next routing destination
    """
    # Extract dispute details from the messages if available
    messages = state.messages
    dispute_details = "Dispute regarding damaged packaging during food delivery"
    
    for message in messages:
        if isinstance(message, HumanMessage) and "packaging" in message.content.lower():
            dispute_details = message.content
    
    # Initialize the state if needed
    if not hasattr(state, "dispute_details") or not state.dispute_details:
        state.dispute_details = dispute_details
    
    # Build the graph
    workflow = build_packaging_dispute_graph()
    
    # Create a memory saver to store the state between runs
    memory = MemorySaver()
    
    # Compile the graph with the memory saver
    app = workflow.compile(checkpointer=memory)
    
    # Run the graph
    result = app.invoke(state)
    
    # Return the command to route to the parent agent
    return Command(
        goto=result.next,
        update={
            "next": result.next,
            "messages": result.messages,
            "customer_evidence": result.customer_evidence,
            "driver_evidence": result.driver_evidence,
            "analysis_result": result.analysis_result,
            "resolution": result.resolution,
            "actions_taken": result.actions_taken,
            "status": result.status
        }
    )

# Function for testing and demonstration purposes
def run_packaging_dispute_workflow(dispute_details: str) -> DisputeState:
    """
    Run the packaging dispute resolution workflow with the given dispute details.
    This function is primarily for testing and demonstration purposes.
    
    Args:
        dispute_details: Details of the packaging dispute
        
    Returns:
        The final state after resolution
    """
    # Initialize state
    initial_state = DisputeState(
        messages=[HumanMessage(content=dispute_details)],
        dispute_details=dispute_details
    )
    
    # Build the graph
    workflow = build_packaging_dispute_graph()
    
    # Create a memory saver
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    return result

# When this module is imported, this will be available
def get_tools():
    """Return the tools from this module."""
    return [
        initiate_mediation_flow, 
        collect_evidence_from_customer,
        collect_evidence_from_driver,
        issue_instant_refund,
        exonerate_driver,
        log_merchant_packaging_feedback,
        notify_resolution
    ]

