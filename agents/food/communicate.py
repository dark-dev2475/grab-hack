"""
Communication module for Grab Food service. the grabfood agent will use this function or say agent to communicate  thaat issue has been addressed to the addministartor and customer and asking for feedbakc from the custmer.
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

def communication_analyzer(state: CommunicationState) -> CommunicationState:
    """
    Main analysis function for the communication agent.
    Analyzes the current state and determines next actions for customer communication.
    
    Args:
        state: Current communication state
        
    Returns:
        Updated state with next actions determined
    """
    # Get the current communication stage
    current_stage = state.communication_stage
    
    # System prompt for the LLM
    system_prompt = """
    You are a Grab Food customer communication specialist.
    
    Your task is to provide personalized, empathetic communication tailored to:
    1. The specific issue the customer experienced
    2. The resolution that was implemented
    3. The severity of the inconvenience they faced
    
    Generate personalized messages that:
    - Address the customer by name when available
    - Reference specific details of their issue
    - Acknowledge any inconvenience in a genuine way
    - Provide clear information about resolutions
    - End with a forward-looking, positive note and request for feedback
    
    Be thorough, empathetic, and focused on rebuilding customer trust.
    """
    
    # Extract context from messages
    issue_description = "a service issue"
    
    for message in state.messages:
        if hasattr(message, 'content') and isinstance(message.content, str):
            if "issue" in message.content.lower() or "problem" in message.content.lower():
                issue_description = message.content
                break
    
    # Create human message based on current stage
    if current_stage == "initial":
        human_message = f"""
        We need to communicate with a customer about a resolved issue: {issue_description}
        
        Please analyze this situation and draft an appropriate communication plan.
        We need to inform both the customer and administrator that the issue has been addressed,
        and we should request feedback from the customer.
        """
        
        # Add this message to the state
        state.messages.append(HumanMessage(content=human_message))
        
        # Generate the LLM response using the prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"messages": state.messages})
        
        # Add the AI's analysis to the message history
        state.messages.append(AIMessage(content=response))
        
        # Update state to move to the next stage
        state.communication_stage = "send_notification"
        
        # Print CLI output to show chain of thought
        print("\n🧠 COMMUNICATION ANALYSIS")
        print("-" * 80)
        print(response[:200] + "..." if len(response) > 200 else response)
        print("-" * 80)
        
    elif current_stage == "send_notification":
        # We're ready to send the notification
        # Extract customer data (simulated in real implementation this would come from the database)
        customer_id = state.customer_id or "cust-12345"
        customer_name = "Valued Customer"  # Would come from real customer data
        
        # Create message for sending notification
        human_message = f"""
        Now that we've analyzed the situation, please draft a personalized message to send to the customer.
        The message should inform them that their issue ({issue_description}) has been resolved,
        and should ask for their feedback on our resolution process.
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
        
        # Add the AI's message to the history
        state.messages.append(AIMessage(content=response))
        
        # Update state to move to the survey stage
        state.communication_stage = "issue_survey"
        
        # Print CLI output to show chain of thought
        print("\n✉️ CUSTOMER NOTIFICATION DRAFTED")
        print("-" * 80)
        print(response[:200] + "..." if len(response) > 200 else response)
        print("-" * 80)
        
    elif current_stage == "issue_survey":
        # We're ready to issue a satisfaction survey
        human_message = """
        Please recommend if we should issue a satisfaction survey to the customer
        and if we should offer a promotional voucher based on the severity of their issue.
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
        
        # Add the AI's recommendation to the history
        state.messages.append(AIMessage(content=response))
        
        # Update state to move to the voucher stage
        state.communication_stage = "offer_voucher"
        
        # Print CLI output to show chain of thought
        print("\n📋 SURVEY RECOMMENDATION")
        print("-" * 80)
        print(response[:200] + "..." if len(response) > 200 else response)
        print("-" * 80)
        
    elif current_stage == "offer_voucher":
        # We're ready to consider offering a voucher
        human_message = """
        Based on all our interactions, please provide a final recommendation 
        on whether to offer a promotional voucher to the customer.
        If yes, recommend an appropriate amount based on the severity of the issue.
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
        
        # Add the AI's recommendation to the history
        state.messages.append(AIMessage(content=response))
        
        # Update state to completed
        state.communication_stage = "completed"
        
        # Print CLI output to show chain of thought
        print("\n🎁 VOUCHER RECOMMENDATION")
        print("-" * 80)
        print(response[:200] + "..." if len(response) > 200 else response)
        print("-" * 80)
    
    # Return the updated state
    return state

def send_notification_node(state: CommunicationState) -> CommunicationState:
    """
    Sends a notification to the customer about the resolution.
    
    Args:
        state: Current communication state
        
    Returns:
        Updated state with notification sent
    """
    # Extract the most recent AI message for the notification content
    notification_message = "Your issue has been resolved. Thank you for your patience."
    
    for i in range(len(state.messages) - 1, -1, -1):
        if isinstance(state.messages[i], AIMessage):
            notification_message = state.messages[i].content
            break
    
    # Get customer ID from state or use default
    customer_id = state.customer_id or "cust-12345"
    
    # Call the tool to send the notification
    result = send_resolution_notification(customer_id, notification_message)
    
    # Add the result to messages
    state.messages.append(FunctionMessage(
        content=f"Notification sent to customer {customer_id}",
        name="send_resolution_notification"
    ))
    
    # Print CLI output
    print("\n✉️ NOTIFICATION SENT")
    print("-" * 80)
    print(f"To: Customer {customer_id}")
    print(f"Message: {notification_message[:100]}..." if len(notification_message) > 100 else notification_message)
    print("-" * 80)
    
    return state

def issue_survey_node(state: CommunicationState) -> CommunicationState:
    """
    Issues a satisfaction survey to the customer.
    
    Args:
        state: Current communication state
        
    Returns:
        Updated state with survey issued
    """
    # Get customer and order IDs from state or use defaults
    customer_id = state.customer_id or "cust-12345"
    order_id = state.order_id or "order-67890"
    
    # Call the tool to issue the survey
    result = issue_satisfaction_survey(customer_id, order_id)
    
    # Add the result to messages
    state.messages.append(FunctionMessage(
        content=f"Satisfaction survey issued to customer {customer_id} for order {order_id}",
        name="issue_satisfaction_survey"
    ))
    
    # Print CLI output
    print("\n📋 SURVEY ISSUED")
    print("-" * 80)
    print(f"Survey ID: {result['survey_id']}")
    print(f"Expiry: {result['expiry']}")
    print("-" * 80)
    
    return state

def offer_voucher_node(state: CommunicationState) -> CommunicationState:
    """
    Offers a promotional voucher to the customer as goodwill.
    
    Args:
        state: Current communication state
        
    Returns:
        Updated state with voucher offered
    """
    # Get customer ID from state or use default
    customer_id = state.customer_id or "cust-12345"
    
    # Determine voucher amount based on issue severity
    # This would be more sophisticated in a real implementation
    issue_description = ""
    for message in state.messages:
        if hasattr(message, 'content') and isinstance(message.content, str):
            if "issue" in message.content.lower() or "problem" in message.content.lower():
                issue_description = message.content
                break
    
    is_severe = "spill" in issue_description.lower() or "delay" in issue_description.lower()
    voucher_amount = 10.00 if is_severe else 5.00
    
    # Call the tool to offer the voucher
    result = offer_promotional_voucher(customer_id, voucher_amount, 7)
    
    # Add the result to messages
    state.messages.append(FunctionMessage(
        content=f"Promotional voucher of ${voucher_amount} offered to customer {customer_id}",
        name="offer_promotional_voucher"
    ))
    
    # Print CLI output
    print("\n🎁 VOUCHER OFFERED")
    print("-" * 80)
    print(f"Voucher code: {result['voucher_code']}")
    print(f"Amount: ${voucher_amount}")
    print(f"Expiry: {result['expiry_date']}")
    print("-" * 80)
    
    return state

def router(state: CommunicationState) -> str:
    """
    Routes to the appropriate node based on the current communication stage.
    
    Args:
        state: Current communication state
        
    Returns:
        Name of the next node to execute
    """
    # Get the current communication stage
    current_stage = state.communication_stage
    
    # Print routing decision
    print("\n🧭 COMMUNICATION ROUTER")
    print(f"Current stage: {current_stage}")
    
    # Route based on the current stage
    if current_stage == "send_notification":
        print("Routing to: send_notification_node")
        return "send_notification_node"
    elif current_stage == "issue_survey":
        print("Routing to: issue_survey_node")
        return "issue_survey_node"
    elif current_stage == "offer_voucher":
        print("Routing to: offer_voucher_node")
        return "offer_voucher_node"
    elif current_stage == "completed":
        print("Communication flow completed. Returning to main orchestrator.")
        return END
    else:
        # Default to analyzer for any other stage
        print("Routing to: communication_analyzer")
        return "communication_analyzer"

def communicate(state: MessagesState) -> Command[Literal['grab_food']]:
    """
    Main function for customer communication.
    Creates and runs the workflow for personalized customer communication.
    
    Args:
        state: Initial state including messages about the issue
        
    Returns:
        Command with the next routing destination
    """
    # Create a proper CommunicationState from the MessagesState
    comm_state = CommunicationState(
        messages=state.messages,
        communication_stage="initial"
    )
    
    # Extract issue details from messages if available
    for message in state.messages:
        if hasattr(message, 'content') and isinstance(message.content, str):
            if "issue" in message.content.lower() or "problem" in message.content.lower():
                comm_state.issue_type = message.content
                break
    
    # Print workflow start
    print("\n🚀 STARTING COMMUNICATION WORKFLOW")
    print("-" * 80)
    
    # Create the workflow graph
    workflow = StateGraph(CommunicationState)
    
    # Add nodes to the graph
    workflow.add_node("communication_analyzer", communication_analyzer)
    workflow.add_node("send_notification_node", send_notification_node)
    workflow.add_node("issue_survey_node", issue_survey_node)
    workflow.add_node("offer_voucher_node", offer_voucher_node)
    
    # Add conditional edges from the analyzer based on the router
    workflow.add_conditional_edges(
        "communication_analyzer",
        router,
        {
            "send_notification_node": "send_notification_node",
            "issue_survey_node": "issue_survey_node",
            "offer_voucher_node": "offer_voucher_node",
            END: END
        }
    )
    
    # Add edges from other nodes back to the analyzer
    workflow.add_edge("send_notification_node", "communication_analyzer")
    workflow.add_edge("issue_survey_node", "communication_analyzer")
    workflow.add_edge("offer_voucher_node", "communication_analyzer")
    
    # Set the entry point
    workflow.set_entry_point("communication_analyzer")
    
    # Compile the workflow
    app = workflow.compile()
    
    # Execute the workflow
    result = app.invoke(comm_state)
    
    # Print workflow completion
    print("\n✅ COMMUNICATION WORKFLOW COMPLETED")
    print(f"Final stage: {result.communication_stage}")
    print("-" * 80)
    
    # Return to the grab_food orchestrator
    goto = "grab_food"
    return Command(goto=goto, update={"next": goto})