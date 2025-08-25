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

def initiate_mediation_flow_node(state: DisputeState) -> DisputeState:
    """
    Initiates the mediation process between customer and driver.
    Uses the initiate_mediation_flow tool to open communication channels.
    
    Args:
        state: Current dispute state
        
    Returns:
        Updated state with mediation initiated
    """
    # Get dispute details from the state or use a default message
    dispute_details = state.dispute_details or "Dispute regarding damaged packaging during delivery"
    
    # Call the tool to initiate mediation
    result = initiate_mediation_flow(dispute_details)
    
    # Record the action in the state
    if not hasattr(state, 'actions_taken') or state.actions_taken is None:
        state.actions_taken = []
    
    state.actions_taken.append({"action": "initiate_mediation", "result": result})
    
    # Add the tool result to messages for context
    state.messages.append(FunctionMessage(
        content=f"Mediation initiated: {result['message']}", 
        name="initiate_mediation_flow"
    ))
    
    # Update the dispute stage
    state.dispute_stage = "collect_evidence"
    
    return state
def collect_evidence_node(state: DisputeState) -> DisputeState:
    """
    Collects evidence from both the customer and driver regarding the damaged package.
    Uses the collect_evidence_from_customer and collect_evidence_from_driver tools.
    
    Args:
        state: Current dispute state
        
    Returns:
        Updated state with collected evidence
    """
    # Create system prompt for the LLM to generate relevant questions
    system_prompt = """
    You are a Grab Food dispute resolution agent collecting evidence about damaged packaging.
    Your task is to generate specific questions that will help determine responsibility.
    
    Generate one question for the customer and one for the driver that will:
    1. Be specific to packaging damage issues
    2. Help identify where the damage occurred
    3. Elicit factual information rather than opinions
    
    Format your response as a JSON object with 'customer_question' and 'driver_question' keys.
    """
    
    # Create a prompt that includes any existing context
    human_message = """
    I need to collect evidence from both the customer and driver regarding damaged packaging.
    Please generate one question for each party that will help determine responsibility.
    """
    
    # Add this message to the state
    if not hasattr(state, 'messages') or state.messages is None:
        state.messages = []
    
    state.messages.append(HumanMessage(content=human_message))
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    # Generate questions using the LLM
    chain = prompt | llm
    response = chain.invoke({"messages": state.messages})
    
    # Parse the questions from the response
    # We'll use a try-except block to handle potential parsing errors
    try:
        # Try to extract the questions from the response
        response_content = response.content
        import json
        import re
        
        # Try to extract JSON if wrapped in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_content, re.DOTALL)
        if json_match:
            response_content = json_match.group(1)
        
        # Remove any leading/trailing whitespace and ensure it's valid JSON
        response_content = response_content.strip()
        if not response_content.startswith('{'):
            # If it doesn't look like JSON, try to find JSON-like content
            json_match = re.search(r'({.*})', response_content, re.DOTALL)
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
        
        # Log the parsing failure
        state.messages.append(FunctionMessage(
            content=f"Failed to parse LLM response: {str(e)}. Using default questions.",
            name="collect_evidence_node"
        ))
    
    # Add the generated/fallback questions to the message history
    state.messages.append(AIMessage(content=f"Generated questions:\nCustomer: {customer_question}\nDriver: {driver_question}"))
    
    # Collect evidence from both parties using the tools
    customer_evidence = collect_evidence_from_customer(customer_question)
    driver_evidence = collect_evidence_from_driver(driver_question)
    
    # Save the evidence in the state
    state.customer_evidence = customer_evidence
    state.driver_evidence = driver_evidence
    
    # Add combined evidence to state.evidence for compatibility
    state.evidence = {
        "customer": customer_evidence,
        "driver": driver_evidence
    }
    
    # Add evidence collection to messages
    state.messages.append(FunctionMessage(
        content=f"Customer question: {customer_question}\nCustomer response: {customer_evidence['response']}",
        name="collect_evidence_from_customer"
    ))
    state.messages.append(FunctionMessage(
        content=f"Driver question: {driver_question}\nDriver response: {driver_evidence['response']}",
        name="collect_evidence_from_driver"
    ))
    
    # Record these actions
    if not hasattr(state, 'actions_taken') or state.actions_taken is None:
        state.actions_taken = []
        
    state.actions_taken.append({
        "action": "collect_evidence",
        "result": {
            "customer_question": customer_question,
            "driver_question": driver_question,
            "customer_evidence_collected": bool(customer_evidence),
            "driver_evidence_collected": bool(driver_evidence)
        }
    })
    
    return state

def evidence_analyzer(state: DisputeState) -> DisputeState:
    """
    Analyzes the collected evidence from both the customer and driver.
    This is the core intelligence function that determines responsibility.
    
    Args:
        state: Current dispute state with collected evidence
        
    Returns:
        Updated state with analysis results
    """
    # Check if we have evidence to analyze
    if not hasattr(state, 'customer_evidence') or not hasattr(state, 'driver_evidence'):
        # We don't have evidence to analyze yet
        state.messages.append(FunctionMessage(
            content="Cannot analyze evidence: Missing customer or driver evidence",
            name="evidence_analyzer"
        ))
        return state
    
    # Extract the evidence
    customer_evidence = state.customer_evidence
    driver_evidence = state.driver_evidence
    
    # Create a detailed system prompt for the LLM
    system_prompt = """
    You are a Grab Food dispute resolution specialist with expertise in analyzing evidence.
    
    Your task is to:
    1. Carefully analyze the evidence provided by both the customer and driver
    2. Consider the timing, consistency, and photographic evidence
    3. Determine the most likely responsible party (merchant, driver, or customer)
    4. Assign a confidence level to your determination (0.0-1.0)
    5. Provide detailed reasoning for your conclusion
    
    Be fair, objective, and thorough in your analysis. Consider all possibilities.
    
    Format your response as a detailed analysis with clear sections for:
    - Evidence Summary
    - Key Factors Considered
    - Responsibility Determination
    - Confidence Level
    - Reasoning
    """
    
    # Create a detailed evidence summary for the LLM
    evidence_summary = f"""
    Please analyze the following evidence regarding damaged packaging:
    
    CUSTOMER EVIDENCE:
    - Statement: "{customer_evidence.get('response', 'No response')}"
    - Photos Provided: {customer_evidence.get('photos_provided', False)}
    - Timestamp: {customer_evidence.get('timestamp', 'Unknown')}
    
    DRIVER EVIDENCE:
    - Statement: "{driver_evidence.get('response', 'No response')}"
    - Photos Provided: {driver_evidence.get('photos_provided', False)}
    - Timestamp: {driver_evidence.get('timestamp', 'Unknown')}
    
    ORDER DETAILS:
    - Time of Pickup: Approximately 20 minutes before delivery
    - Packaging Type: Standard restaurant packaging with seals
    - Delivery Method: Motorcycle delivery with insulated bag
    
    Based on this evidence, determine the most likely responsible party (merchant, driver, or customer),
    assign a confidence level, and provide detailed reasoning.
    """
    
    # Add this message to the state
    state.messages.append(HumanMessage(content=evidence_summary))
    
    # Generate the analysis using the LLM
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    chain = prompt | llm | StrOutputParser()
    analysis_text = chain.invoke({"messages": state.messages})
    
    # Add the analysis to the message history
    state.messages.append(AIMessage(content=analysis_text))
    
    # Now we need to extract the responsibility determination from the analysis
    # We'll use a combination of keyword matching and another LLM call to extract structured data
    
    extraction_prompt = f"""
    Based on the following analysis of packaging damage evidence, extract:
    
    1. The determined responsible party (merchant, driver, or customer)
    2. The confidence level (as a decimal between 0 and 1)
    3. A brief summary of the reasoning (1-2 sentences)
    
    Analysis:
    {analysis_text}
    
    Respond in JSON format with keys: "responsibility", "confidence", "reasoning"
    """
    
    # Extract structured data
    state.messages.append(HumanMessage(content=extraction_prompt))
    
    extraction_chain = ChatPromptTemplate.from_messages([
        SystemMessage(content="You extract structured data from analysis text."),
        HumanMessage(content=extraction_prompt)
    ]) | llm
    
    extraction_response = extraction_chain.invoke({})
    
    # Parse the extracted data
    try:
        import json
        import re
        
        # Try to extract JSON
        response_content = extraction_response.content
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_content, re.DOTALL)
        if json_match:
            response_content = json_match.group(1)
        
        # Clean up the response for JSON parsing
        response_content = response_content.strip()
        if not response_content.startswith('{'):
            json_match = re.search(r'({.*})', response_content, re.DOTALL)
            if json_match:
                response_content = json_match.group(1)
        
        # Parse the JSON
        extracted_data = json.loads(response_content)
        
        # Ensure we have the required fields with defaults if missing
        responsibility = extracted_data.get("responsibility", "merchant")  # Default to merchant
        confidence = float(extracted_data.get("confidence", 0.7))  # Default to 0.7
        reasoning = extracted_data.get("reasoning", "Based on evidence analysis")
        
        # Validate and normalize the responsibility
        if responsibility.lower() not in ["merchant", "driver", "customer"]:
            # If it doesn't match expected values, default to merchant
            responsibility = "merchant"
        
        # Ensure confidence is in range 0-1
        confidence = max(0.0, min(1.0, confidence))
        
    except Exception as e:
        # Fallback values if parsing fails
        responsibility = "merchant"  # Default to merchant responsibility
        confidence = 0.7  # Default confidence
        reasoning = "Analysis suggests merchant packaging was likely at fault."
        
        # Log the failure
        state.messages.append(FunctionMessage(
            content=f"Failed to parse structured data: {str(e)}. Using default values.",
            name="evidence_analyzer"
        ))
    
    # Create the analysis result
    analysis_result = {
        "responsibility": responsibility,
        "confidence": confidence,
        "reasoning": reasoning,
        "timestamp": "2025-08-21T15:45:00Z",
        "evidence_considered": {
            "customer_evidence": customer_evidence,
            "driver_evidence": driver_evidence
        },
        "full_analysis": analysis_text
    }
    
    # Save the analysis result in the state
    state.analysis_result = analysis_result
    
    # Record this action
    if not hasattr(state, 'actions_taken') or state.actions_taken is None:
        state.actions_taken = []
        
    state.actions_taken.append({
        "action": "analyze_evidence",
        "result": {
            "responsibility": responsibility,
            "confidence": confidence
        }
    })
    
    return state
def issue_refund_node(state: DisputeState) -> DisputeState:
    """
    Issues a refund to the customer based on the evidence analysis.
    
    Args:
        state: Current dispute state with analysis results
        
    Returns:
        Updated state with refund issued
    """
    # Check if we have analysis results
    if not hasattr(state, 'analysis_result') or not state.analysis_result:
        # We don't have analysis results yet
        state.messages.append(FunctionMessage(
            content="Cannot issue refund: Missing analysis results",
            name="issue_refund_node"
        ))
        return state
    
    # Extract the analysis results
    analysis_result = state.analysis_result
    responsibility = analysis_result.get("responsibility", "unclear")
    confidence = analysis_result.get("confidence", 0.5)
    
    # Determine refund amount based on responsibility and confidence
    refund_amount = 0.0
    refund_reason = ""
    
    if responsibility == "merchant":
        # Full refund if it's clearly the merchant's fault
        if confidence > 0.8:
            refund_amount = 15.00  # Full order amount (example)
            refund_reason = "Full refund due to clear merchant packaging issue"
        else:
            refund_amount = 10.00  # Partial refund for less clear cases
            refund_reason = "Partial refund for likely merchant packaging issue"
    
    elif responsibility == "driver":
        # Partial refund if it's the driver's fault
        refund_amount = 10.00
        refund_reason = "Partial refund due to handling issues during delivery"
    
    else:  # customer or unclear
        # Small goodwill refund for unclear cases
        refund_amount = 5.00
        refund_reason = "Goodwill partial refund despite unclear responsibility"
    
    # Call the tool to issue the refund
    result = issue_instant_refund(refund_amount, refund_reason)
    
    # Add the result to messages
    state.messages.append(FunctionMessage(
        content=f"Refund issued: ${refund_amount:.2f} - {refund_reason}",
        name="issue_instant_refund"
    ))
    
    # Record this action
    if not hasattr(state, 'actions_taken') or state.actions_taken is None:
        state.actions_taken = []
    
    state.actions_taken.append({
        "action": "issue_refund",
        "result": result
    })
    
    return state

def notify_resolution_node(state: DisputeState) -> DisputeState:
    """
    Notifies both the customer and driver about the resolution outcome.
    
    Args:
        state: Current dispute state with resolution details
        
    Returns:
        Updated state with notifications sent
    """
    # Check if we have all the necessary information
    if not hasattr(state, 'analysis_result') or not state.analysis_result:
        state.messages.append(FunctionMessage(
            content="Cannot notify resolution: Missing analysis results",
            name="notify_resolution_node"
        ))
        return state
    
    # Extract necessary information
    analysis_result = state.analysis_result
    responsibility = analysis_result.get("responsibility", "unclear")
    
    # Check if we've issued a refund
    refund_issued = False
    refund_amount = 0.0
    
    for action in state.actions_taken:
        if action.get("action") == "issue_refund" and action.get("result", {}).get("action") == "refund_issued":
            refund_issued = True
            refund_amount = action.get("result", {}).get("amount", 0.0)
            break
    
    # Create a system prompt for the LLM to generate appropriate messages
    system_prompt = """
    You are a Grab Food customer service specialist crafting resolution messages.
    
    Create two separate messages:
    1. A message for the customer explaining the resolution outcome
    2. A message for the driver explaining the resolution outcome
    
    Both messages should be:
    - Clear and professional
    - Empathetic but factual
    - Specific about the outcome (refund, responsibility, etc.)
    - Under 100 words each
    
    Format your response with clear CUSTOMER MESSAGE: and DRIVER MESSAGE: sections.
    """
    
    # Create context for the message generation
    resolution_context = f"""
    Generate appropriate resolution messages for this packaging damage dispute:
    
    Case Details:
    - Determined Responsibility: {responsibility}
    - Refund Issued: {refund_issued}
    - Refund Amount: ${refund_amount:.2f}
    - Reasoning: {analysis_result.get('reasoning', 'Based on evidence analysis')}
    
    Customer needs to know about the refund and outcome.
    Driver needs to know if they've been exonerated or not.
    """
    
    # Generate messages using the LLM
    state.messages.append(HumanMessage(content=resolution_context))
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=resolution_context)
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({})
    
    # Parse the customer and driver messages from the response
    import re
    
    customer_match = re.search(r'CUSTOMER MESSAGE:(.*?)(?:DRIVER MESSAGE:|$)', response, re.DOTALL)
    driver_match = re.search(r'DRIVER MESSAGE:(.*?)$', response, re.DOTALL)
    
    customer_message = customer_match.group(1).strip() if customer_match else "Thank you for reporting the packaging issue. We've processed a refund and addressed the situation."
    driver_message = driver_match.group(1).strip() if driver_match else "Thank you for your cooperation with the packaging dispute. The issue has been resolved."
    
    # Call the tool to send the notifications
    result = notify_resolution(customer_message, driver_message)
    
    # Add the result to messages
    state.messages.append(FunctionMessage(
        content=f"Resolution notifications sent:\nCustomer: {customer_message}\nDriver: {driver_message}",
        name="notify_resolution"
    ))
    
    # Create a comprehensive resolution summary
    resolution_summary = f"""
    Dispute Resolution Summary:
    - Responsibility: {responsibility}
    - Refund Amount: ${refund_amount:.2f}
    - Customer Notified: Yes
    - Driver Notified: Yes
    - Merchant Feedback Logged: Yes
    """
    
    # Save the resolution in the state
    state.resolution = resolution_summary
    
    # Record this action
    if not hasattr(state, 'actions_taken') or state.actions_taken is None:
        state.actions_taken = []
    
    state.actions_taken.append({
        "action": "notify_resolution",
        "result": result
    })
    
    return state
def log_packaging_feedback(state: DisputeState) -> DisputeState:
    """
    Logs feedback about the merchant's packaging based on evidence and analysis.
    
    Args:
        state: Current dispute state with analysis results
        
    Returns:
        Updated state with merchant feedback logged
    """
    # Check if we have analysis results
    if not hasattr(state, 'analysis_result') or not state.analysis_result:
        # We don't have analysis results yet
        state.messages.append(FunctionMessage(
            content="Cannot log packaging feedback: Missing analysis results",
            name="log_packaging_feedback"
        ))
        return state
    
    # Extract the analysis results
    analysis_result = state.analysis_result
    responsibility = analysis_result.get("responsibility", "unclear")
    confidence = analysis_result.get("confidence", 0.5)
    reasoning = analysis_result.get("reasoning", "")
    
    # Determine feedback details and severity based on responsibility and confidence
    feedback_details = ""
    severity = "low"
    
    if responsibility == "merchant":
        if confidence > 0.8:
            severity = "high"
            feedback_details = f"Severe packaging failure detected: {reasoning} Immediate packaging process review recommended."
        elif confidence > 0.6:
            severity = "medium"
            feedback_details = f"Packaging issues identified: {reasoning} Consider reinforcing packaging for liquid items."
        else:
            severity = "low"
            feedback_details = f"Potential packaging improvements needed: {reasoning} Review packaging for similar orders."
    
    elif responsibility == "driver":
        # Even if it's the driver's fault, we might have packaging suggestions
        severity = "low"
        feedback_details = "Although handling was the primary issue, consider more robust packaging for delivery conditions."
    
    else:  # customer or unclear
        severity = "info"
        feedback_details = "Informational report only. No clear packaging issues identified, but continued monitoring recommended."
    
    # Call the tool to log the feedback
    result = log_merchant_packaging_feedback(feedback_details, severity)
    
    # Add the result to messages
    state.messages.append(FunctionMessage(
        content=f"Merchant packaging feedback logged - Severity: {severity} - Details: {feedback_details}",
        name="log_merchant_packaging_feedback"
    ))
    
    # Record this action
    if not hasattr(state, 'actions_taken') or state.actions_taken is None:
        state.actions_taken = []
    
    state.actions_taken.append({
        "action": "log_packaging_feedback",
        "result": result
    })
    
    return state

def main_analyzer(state: DisputeState) -> DisputeState:
    """
    Main analysis function for this packaging dispute agent.
    
    This is the central orchestrator that:
    1. Analyzes the current state of the dispute
    2. Determines which node to activate next
    3. Processes incoming evidence and analysis results
    4. Makes decisions about refunds, driver exoneration, and merchant feedback
    
    Args:
        state: The current dispute state
        
    Returns:
        Updated state with next actions determined
    """
    # First, check the current dispute stage to determine what to do next
    current_stage = state.dispute_stage
    
    # System prompt for the LLM to analyze the situation
    system_prompt = """
    You are a Grab Food dispute resolution specialist handling a packaging damage situation.
    Your task is to analyze the current state of the dispute and determine appropriate next steps.
    
    Consider:
    1. What evidence has been collected so far
    2. What analysis has been performed on the evidence
    3. What actions have already been taken
    4. What actions need to be taken next
    
    Be thorough, fair, and focused on resolving the dispute efficiently.
    """
    
    # Create context-aware messages for the LLM based on the current stage
    if current_stage == "initiate":
        # We're just starting the dispute resolution process
        human_message = """
        A dispute has been reported regarding damaged packaging during delivery.
        Please analyze the initial situation and determine the first steps in the resolution process.
        """
        
        # Add this message to the state
        state.messages.append(HumanMessage(content=human_message))
        
        # Generate the LLM response
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"messages": state.messages})
        
        # Add the AI's analysis to the message history
        state.messages.append(AIMessage(content=response))
        
        # Update state to move to the next stage
        state.dispute_stage = "collect_evidence"
        
        # Log this action
        if not state.actions_taken:
            state.actions_taken = []
        
        state.actions_taken.append({
            "action": "initial_analysis",
            "result": "Moving to evidence collection"
        })
        
    elif current_stage == "collect_evidence" and state.customer_evidence and state.driver_evidence:
        # We have collected evidence and need to analyze it
        
        # Summarize the evidence for the LLM
        customer_evidence = state.customer_evidence
        driver_evidence = state.driver_evidence
        
        evidence_summary = f"""
        Evidence has been collected from both parties:
        
        Customer Evidence:
        - Statement: {customer_evidence.get('response', 'No response')}
        - Photos Provided: {customer_evidence.get('photos_provided', False)}
        - Timestamp: {customer_evidence.get('timestamp', 'Unknown')}
        
        Driver Evidence:
        - Statement: {driver_evidence.get('response', 'No response')}
        - Photos Provided: {driver_evidence.get('photos_provided', False)}
        - Timestamp: {driver_evidence.get('timestamp', 'Unknown')}
        
        Please analyze this evidence and determine the next steps for resolving this dispute.
        """
        
        # Add this message to the state
        state.messages.append(HumanMessage(content=evidence_summary))
        
        # Generate the LLM response
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"messages": state.messages})
        
        # Add the AI's analysis to the message history
        state.messages.append(AIMessage(content=response))
        
        # Update state to move to the next stage
        state.dispute_stage = "analyze_evidence"
        
        # Log this action
        state.actions_taken.append({
            "action": "evidence_review",
            "result": "Moving to evidence analysis"
        })
        
    elif current_stage == "analyze_evidence" and state.analysis_result:
        # We have analysis results and need to determine resolution actions
        
        # Extract the responsibility determination from the analysis
        analysis_result = state.analysis_result
        responsibility = analysis_result.get("responsibility", "unclear")
        confidence = analysis_result.get("confidence", 0.0)
        
        # Create a summary for the LLM
        analysis_summary = f"""
        The evidence has been analyzed with the following results:
        
        Determined Responsibility: {responsibility}
        Confidence Level: {confidence}
        Reasoning: {analysis_result.get('reasoning', 'No reasoning provided')}
        
        Please determine the appropriate resolution actions based on this analysis:
        1. Should a refund be issued to the customer? If so, how much?
        2. Should the driver be exonerated?
        3. What feedback should be sent to the merchant?
        4. What messages should be sent to both parties?
        """
        
        # Add this message to the state
        state.messages.append(HumanMessage(content=analysis_summary))
        
        # Generate the LLM response
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"messages": state.messages})
        
        # Add the AI's analysis to the message history
        state.messages.append(AIMessage(content=response))
        
        # Update state to move to resolution actions
        state.dispute_stage = "resolve"
        
        # Log this action
        state.actions_taken.append({
            "action": "resolution_determination",
            "result": f"Responsibility determined as {responsibility} with {confidence} confidence"
        })
    
    elif current_stage == "resolve" and state.resolution:
        # We have completed the resolution process
        
        # Create a summary for the LLM
        resolution_summary = f"""
        The dispute has been resolved with the following outcome:
        
        Resolution: {state.resolution}
        
        Actions Taken:
        {', '.join([action['action'] for action in state.actions_taken])}
        
        Please provide a final summary of this dispute resolution for our records.
        """
        
        # Add this message to the state
        state.messages.append(HumanMessage(content=resolution_summary))
        
        # Generate the LLM response
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"messages": state.messages})
        
        # Add the AI's analysis to the message history
        state.messages.append(AIMessage(content=response))
        
        # Update state to completed
        state.dispute_stage = "completed"
        state.status = "resolved"
        state.next = "grab_food"  # Return to parent agent
        
        # Log this action
        state.actions_taken.append({
            "action": "final_summary",
            "result": "Dispute resolution completed"
        })
    
    # Return the updated state
    return state



def router(state: DisputeState) -> str:
    """
    Routes to the appropriate node based on the current dispute stage.
    
    Args:
        state: Current dispute state
        
    Returns:
        Name of the next node to execute
    """
    # Get the current dispute stage
    current_stage = state.dispute_stage
    
    # Print the current stage for debugging
    print(f"\n🧭 ROUTER DECISION MAKING")
    print(f"📊 Current dispute stage: {current_stage}")
    
    # Route based on the current stage
    next_node = "main_analyzer"  # Default
    
    if current_stage == "initiate":
        next_node = "initiate_mediation_flow"
        print("⏩ Routing to: initiate_mediation_flow (Need to start the mediation process)")
    
    elif current_stage == "collect_evidence":
        next_node = "collect_evidence"
        print("⏩ Routing to: collect_evidence (Need to gather evidence from both parties)")
    
    elif current_stage == "analyze_evidence":
        next_node = "evidence_analyzer"
        print("⏩ Routing to: evidence_analyzer (Need to analyze collected evidence)")
    
    elif current_stage == "resolve":
        # Check what actions we've already taken
        actions = [action.get("action") for action in state.actions_taken]
        
        if "issue_refund" not in actions:
            next_node = "issue_refund"
            print("⏩ Routing to: issue_refund (Need to issue refund to customer)")
        
        elif "log_packaging_feedback" not in actions:
            next_node = "log_packaging_feedback"
            print("⏩ Routing to: log_packaging_feedback (Need to log feedback for the merchant)")
        
        elif "notify_resolution" not in actions:
            next_node = "notify_resolution"
            print("⏩ Routing to: notify_resolution (Need to notify both parties of resolution)")
        
        else:
            # All actions have been taken, go back to main_analyzer for final summary
            next_node = "main_analyzer"
            print("⏩ Routing to: main_analyzer (All resolution actions completed, need final summary)")
    
    elif current_stage == "completed":
        # We're done, return END
        next_node = END
        print("⏩ Routing to: END (Dispute resolution completed)")
    
    else:
        # Default to main_analyzer if we're not sure
        print(f"⏩ Routing to: main_analyzer (Default routing for unknown stage: {current_stage})")
    
    return next_node

def manage_packaging_dispute(state: DisputeState) -> Command[Literal["grab_food"]]:
    """
    Main function for managing packaging disputes.
    Creates and runs the workflow for packaging dispute resolution.
    
    Args:
        state: Initial state including messages about the dispute
        
    Returns:
        Command with the next routing destination
    """
    # Initialize the state if needed
    if not hasattr(state, 'dispute_stage') or not state.dispute_stage:
        state.dispute_stage = "initiate"
    
    if not hasattr(state, 'actions_taken') or not state.actions_taken:
        state.actions_taken = []
    
    # Extract dispute details from the messages if available
    dispute_details = "Dispute regarding damaged packaging during food delivery"
    
    for message in state.messages:
        if hasattr(message, 'content') and isinstance(message.content, str) and "packaging" in message.content.lower():
            dispute_details = message.content
            break
    
    if not hasattr(state, 'dispute_details') or not state.dispute_details:
        state.dispute_details = dispute_details
    
    # Print info when called directly from CLI
    print(f"\n🔄 MANAGING PACKAGING DISPUTE")
    print(f"📝 Dispute details: {dispute_details}")
    print("-"*80)
    
    # Create the workflow
    workflow = StateGraph(DisputeState)
    
    # Define callbacks for node transitions (for CLI output)
    def on_node_enter(node_name: str, state: DisputeState):
        """Print information when entering a node"""
        print(f"\n🔄 ENTERING NODE: {node_name.upper()}")
        print(f"📊 Current dispute stage: {state.dispute_stage}")
        print("-"*80)
    
    def on_node_exit(node_name: str, state: DisputeState, result: DisputeState):
        """Print information when exiting a node"""
        print(f"\n✅ EXITING NODE: {node_name.upper()}")
        
        # Print the latest message if available
        if state.messages and len(state.messages) > 0:
            latest_message = state.messages[-1]
            print(f"📨 Latest message ({latest_message.type}):")
            
            # Format content based on message type
            if hasattr(latest_message, 'content'):
                content = latest_message.content
                if len(content) > 200:
                    print(f"  {content[:197]}...")
                else:
                    print(f"  {content}")
        
        # Print stage transition if it changed
        if state.dispute_stage != result.dispute_stage:
            print(f"🔄 Stage transition: {state.dispute_stage} -> {result.dispute_stage}")
        
        print("-"*80)
    
    # Add nodes with callbacks for verbose output
    workflow.add_node("main_analyzer", main_analyzer, 
                     on_enter=lambda state: on_node_enter("main_analyzer", state),
                     on_exit=lambda state, result: on_node_exit("main_analyzer", state, result))
    
    workflow.add_node("initiate_mediation_flow", initiate_mediation_flow_node,
                     on_enter=lambda state: on_node_enter("initiate_mediation_flow", state),
                     on_exit=lambda state, result: on_node_exit("initiate_mediation_flow", state, result))
    
    workflow.add_node("collect_evidence", collect_evidence_node,
                     on_enter=lambda state: on_node_enter("collect_evidence", state),
                     on_exit=lambda state, result: on_node_exit("collect_evidence", state, result))
    
    workflow.add_node("evidence_analyzer", evidence_analyzer,
                     on_enter=lambda state: on_node_enter("evidence_analyzer", state),
                     on_exit=lambda state, result: on_node_exit("evidence_analyzer", state, result))
    
    workflow.add_node("issue_refund", issue_refund_node,
                     on_enter=lambda state: on_node_enter("issue_refund", state),
                     on_exit=lambda state, result: on_node_exit("issue_refund", state, result))
    
    workflow.add_node("log_packaging_feedback", log_packaging_feedback,
                     on_enter=lambda state: on_node_enter("log_packaging_feedback", state),
                     on_exit=lambda state, result: on_node_exit("log_packaging_feedback", state, result))
    
    workflow.add_node("notify_resolution", notify_resolution_node,
                     on_enter=lambda state: on_node_enter("notify_resolution", state),
                     on_exit=lambda state, result: on_node_exit("notify_resolution", state, result))
    
    # Add conditional edges from main_analyzer based on the router
    workflow.add_conditional_edges(
        "main_analyzer",
        router,
        {
            "initiate_mediation_flow": "initiate_mediation_flow",
            "collect_evidence": "collect_evidence",
            "evidence_analyzer": "evidence_analyzer",
            "issue_refund": "issue_refund",
            "log_packaging_feedback": "log_packaging_feedback",
            "notify_resolution": "notify_resolution",
            END: END
        },
        on_enter=lambda state: print(f"\n🔀 ROUTER: Determining next node from dispute stage '{state.dispute_stage}'")
    )
    
    # Add edges from other nodes back to main_analyzer
    workflow.add_edge("initiate_mediation_flow", "main_analyzer")
    workflow.add_edge("collect_evidence", "main_analyzer")
    workflow.add_edge("evidence_analyzer", "main_analyzer")
    workflow.add_edge("issue_refund", "main_analyzer")
    workflow.add_edge("log_packaging_feedback", "main_analyzer")
    workflow.add_edge("notify_resolution", "main_analyzer")
    
    # Set the entry point
    workflow.set_entry_point("main_analyzer")
    
    # Create a memory saver for state persistence
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    
    # Print workflow start
    print("\n🚀 STARTING WORKFLOW EXECUTION\n")
    print("-"*80)
    
    # Run the workflow
    result = app.invoke(state)
    
    # Print workflow completion
    print("\n✨ WORKFLOW EXECUTION COMPLETED")
    print(f"📊 Final dispute stage: {result.dispute_stage}")
    print(f"📊 Status: {result.status}")
    print("-"*80)
    
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
    
    # Print the initial state
    print("\n🔄 INITIALIZING PACKAGING DISPUTE RESOLUTION WORKFLOW")
    print(f"📝 Initial dispute details: {dispute_details}")
    print("\n" + "="*80 + "\n")
    
    # Create the workflow
    workflow = StateGraph(DisputeState)
    
    # Define a callback function to print the state transitions
    def on_node_enter(node_name: str, state: DisputeState):
        """Print information when entering a node"""
        print(f"\n🔄 ENTERING NODE: {node_name.upper()}")
        print(f"📊 Current dispute stage: {state.dispute_stage}")
        print("-"*80)
    
    def on_node_exit(node_name: str, state: DisputeState, result: DisputeState):
        """Print information when exiting a node"""
        print(f"\n✅ EXITING NODE: {node_name.upper()}")
        
        # Print the latest message if available
        if state.messages and len(state.messages) > 0:
            latest_message = state.messages[-1]
            print(f"📨 Latest message ({latest_message.type}):")
            
            # Format content based on message type
            if hasattr(latest_message, 'content'):
                content = latest_message.content
                if len(content) > 200:
                    print(f"  {content[:197]}...")
                else:
                    print(f"  {content}")
        
        # Print any new actions taken
        if result.actions_taken and len(result.actions_taken) > 0:
            if not state.actions_taken or len(result.actions_taken) > len(state.actions_taken):
                latest_action = result.actions_taken[-1]
                print(f"🛠️ Action taken: {latest_action.get('action', 'unknown')}")
                
                # Print result details if available
                if 'result' in latest_action:
                    result_info = latest_action['result']
                    if isinstance(result_info, dict):
                        for key, value in result_info.items():
                            if key not in ['status', 'message', 'dispute_details']:
                                print(f"  - {key}: {value}")
                    else:
                        print(f"  - {result_info}")
        
        # Print stage transition if it changed
        if state.dispute_stage != result.dispute_stage:
            print(f"🔄 Stage transition: {state.dispute_stage} -> {result.dispute_stage}")
        
        print("-"*80)
    
    # Add nodes with callbacks
    workflow.add_node("main_analyzer", main_analyzer, 
                     on_enter=lambda state: on_node_enter("main_analyzer", state),
                     on_exit=lambda state, result: on_node_exit("main_analyzer", state, result))
    
    workflow.add_node("initiate_mediation_flow", initiate_mediation_flow_node,
                     on_enter=lambda state: on_node_enter("initiate_mediation_flow", state),
                     on_exit=lambda state, result: on_node_exit("initiate_mediation_flow", state, result))
    
    workflow.add_node("collect_evidence", collect_evidence_node,
                     on_enter=lambda state: on_node_enter("collect_evidence", state),
                     on_exit=lambda state, result: on_node_exit("collect_evidence", state, result))
    
    workflow.add_node("evidence_analyzer", evidence_analyzer,
                     on_enter=lambda state: on_node_enter("evidence_analyzer", state),
                     on_exit=lambda state, result: on_node_exit("evidence_analyzer", state, result))
    
    workflow.add_node("issue_refund", issue_refund_node,
                     on_enter=lambda state: on_node_enter("issue_refund", state),
                     on_exit=lambda state, result: on_node_exit("issue_refund", state, result))
    
    workflow.add_node("log_packaging_feedback", log_packaging_feedback,
                     on_enter=lambda state: on_node_enter("log_packaging_feedback", state),
                     on_exit=lambda state, result: on_node_exit("log_packaging_feedback", state, result))
    
    workflow.add_node("notify_resolution", notify_resolution_node,
                     on_enter=lambda state: on_node_enter("notify_resolution", state),
                     on_exit=lambda state, result: on_node_exit("notify_resolution", state, result))
    
    # Add conditional edges from main_analyzer based on the router
    workflow.add_conditional_edges(
        "main_analyzer",
        router,
        {
            "initiate_mediation_flow": "initiate_mediation_flow",
            "collect_evidence": "collect_evidence",
            "evidence_analyzer": "evidence_analyzer",
            "issue_refund": "issue_refund",
            "log_packaging_feedback": "log_packaging_feedback",
            "notify_resolution": "notify_resolution",
            END: END
        },
        on_enter=lambda state: print(f"\n🔀 ROUTER: Determining next node from dispute stage '{state.dispute_stage}'")
    )
    
    # Add edges from other nodes back to main_analyzer
    workflow.add_edge("initiate_mediation_flow", "main_analyzer")
    workflow.add_edge("collect_evidence", "main_analyzer")
    workflow.add_edge("evidence_analyzer", "main_analyzer")
    workflow.add_edge("issue_refund", "main_analyzer")
    workflow.add_edge("log_packaging_feedback", "main_analyzer")
    workflow.add_edge("notify_resolution", "main_analyzer")
    
    # Set the entry point
    workflow.set_entry_point("main_analyzer")
    
    # Create a memory saver
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    
    # Print workflow start
    print("\n🚀 STARTING WORKFLOW EXECUTION\n")
    print("-"*80)
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print workflow completion
    print("\n✨ WORKFLOW EXECUTION COMPLETED")
    print(f"📊 Final dispute stage: {result.dispute_stage}")
    print(f"📊 Status: {result.status}")
    
    if result.resolution:
        print(f"\n📝 Resolution: {result.resolution}")
    
    if result.analysis_result:
        print("\n📊 Analysis Result:")
        print(f"  - Responsibility: {result.analysis_result.get('responsibility', 'Unknown')}")
        print(f"  - Confidence: {result.analysis_result.get('confidence', 0)}")
        if 'reasoning' in result.analysis_result:
            reasoning = result.analysis_result['reasoning']
            if len(reasoning) > 200:
                print(f"  - Reasoning: {reasoning[:197]}...")
            else:
                print(f"  - Reasoning: {reasoning}")
    
    print("\n" + "="*80 + "\n")
    
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
