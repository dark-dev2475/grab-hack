"""
Main orchestrator for Grab Food agents.
This module orchestrates between different agents for managing restaurant overload,
packaging disputes, and customer communication in the Grab Food service.
This agent will be the main orchestrator for all food-related issues. in the grab app. it will be triggered if any food related issue happpens int the app.
it wiill go to the manage_oveload_restaurant_agent  or damage_packaging_dispute and address the issue , after the issue isaddressed it will goo to the communicate.py and wwill communicate about the situationis handled and then it will finish and comes to end. 
"""

import os
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

from typing import Annotated, List, Dict, Any, Optional, Literal, TypedDict, Union
from typing_extensions import TypedDict
from langchain_core.tools import tool

from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.types import Command

# Import LLM
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

# Import agent modules
from agents.food.communicate import communicate
from agents.food.damage import manage_packaging_dispute
from agents.food.overload import manage_overloaded_restaurant

# Define available members/agents
members = ["manage_overloaded_restaurant", "damaged_packaging_dispute", "communicate"]
options = members + ["FINISH"]

# Define state class
class State(MessagesState):
    """State for the Grab Food orchestrator."""
    next: str = ""
    issue_type: str = ""  # Classify the type of issue
    issue_details: Optional[Dict[str, Any]] = None  # Store details about the issue
    workflow_stage: str = "analyze"  # Track the workflow stage
    current_agent: Optional[str] = None  # Track which agent is currently active
    resolved_issues: List[str] = []  # Track issues that have been resolved
    actions_taken: List[Dict[str, Any]] = []  # Track actions taken by agents

# Router class for determining the next agent to call
class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["manage_overloaded_restaurant", "damaged_packaging_dispute", "communicate", 'FINISH']
    issue_type: str  # Classification of the issue
    confidence: float  # Confidence in the routing decision

# Define system prompt
system_prompt = f"""
You are a supervisor agent of the Grab Food section, tasked with orchestrating between the following specialized agents: {members}.

Your responsibilities:
1. Analyze food-related issues from customer queries
2. Route to the appropriate specialized agent:
   - manage_overloaded_restaurant: For delays, capacity issues, or high order volume
   - damaged_packaging_dispute: For damaged food, spills, or packaging integrity issues
   - communicate: For customer communication after issues are resolved

3. Track the workflow through these stages:
   - First, identify and address the core issue with either the restaurant overload or packaging dispute agent
   - After the issue is resolved, route to the communication agent to inform relevant parties
   - Then finish the workflow

Provide a precise classification of the issue type and determine the most appropriate agent to handle it.
"""

# Define the grab_food orchestrator function
def grab_food(state: State) -> Command[Literal["manage_overloaded_restaurant", "damaged_packaging_dispute", "communicate", '__end__']]:
    """
    Main orchestrator function for Grab Food service.
    Routes to the appropriate agent based on the issue type and current state.
    
    Args:
        state: The current state containing message history and routing information
        
    Returns:
        Command to route to the next agent or end the workflow
    """
    # Print current state for CLI visibility
    print("\n🚀 GRAB FOOD ORCHESTRATOR")
    print("-" * 80)
    
    # Get the current workflow stage
    workflow_stage = state.workflow_stage
    print(f"Current workflow stage: {workflow_stage}")
    
    # Check if we're in a specific workflow stage
    if workflow_stage == "communication" or state.next == "communicate":
        # After issues are resolved, we always go to communication
        print("Routing to communication agent to inform customer of resolution")
        return Command(goto="communicate", update={
            "next": "communicate", 
            "current_agent": "communicate",
            "workflow_stage": "communication"
        })
    
    elif workflow_stage == "resolved" or state.next == "FINISH":
        # If all issues are resolved and communication is complete, finish
        print("All issues resolved and communication complete. Finishing workflow.")
        return Command(goto=END, update={
            "next": "FINISH",
            "workflow_stage": "complete"
        })
    
    elif state.next in ["manage_overloaded_restaurant", "damaged_packaging_dispute"]:
        # If we already know where to go (from a previous analysis), go there
        goto = state.next
        print(f"Continuing with previously determined agent: {goto}")
        return Command(goto=goto, update={
            "current_agent": goto,
            "workflow_stage": "addressing_issue"
        })
    
    # If we need to analyze the issue, use the LLM to determine routing
    print("Analyzing issue to determine appropriate agent...")
    
    # Prepare messages for the router
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add the conversation history
    for message in state.messages:
        if hasattr(message, "type") and hasattr(message, "content"):
            role = "system" if message.type == "system" else \
                  "user" if message.type == "human" else \
                  "assistant" if message.type == "ai" else \
                  "function" if message.type == "function" else "user"
            
            content = message.content
            name = getattr(message, "name", None)
            
            msg_dict = {"role": role, "content": content}
            if name and role == "function":
                msg_dict["name"] = name
            
            messages.append(msg_dict)
    
    # Add context based on resolved issues
    if state.resolved_issues:
        resolved_issues_str = ", ".join(state.resolved_issues)
        messages.append({
            "role": "user", 
            "content": f"The following issues have already been resolved: {resolved_issues_str}. What should be done next?"
        })
    
    # Get routing decision from the LLM
    response = llm.with_structured_output(Router).invoke(messages)
    
    # Extract routing information
    goto = response["next"]
    issue_type = response.get("issue_type", "Unknown issue")
    confidence = response.get("confidence", 0.0)
    
    # Print the routing decision
    print(f"Issue classified as: {issue_type}")
    print(f"Routing to: {goto} (confidence: {confidence:.2f})")
    
    # Record this decision in the actions taken
    if not hasattr(state, 'actions_taken') or state.actions_taken is None:
        state.actions_taken = []
    
    state.actions_taken.append({
        "action": "routing_decision",
        "issue_type": issue_type,
        "agent": goto,
        "confidence": confidence
    })
    
    # Handle FINISH case
    if goto == "FINISH":
        print("No further action needed. Finishing workflow.")
        return Command(goto=END, update={
            "next": "FINISH",
            "issue_type": issue_type,
            "workflow_stage": "complete"
        })
    
    # If we're routing to the communication agent, update workflow stage
    if goto == "communicate":
        workflow_stage = "communication"
    else:
        workflow_stage = "addressing_issue"
    
    # Update the state with the next agent to call and issue information
    return Command(goto=goto, update={
        "next": goto,
        "issue_type": issue_type,
        "current_agent": goto,
        "workflow_stage": workflow_stage
    })

# Create the workflow graph

# Create the workflow graph
def create_grab_food_graph():
    """
    Creates and returns the Grab Food orchestration graph.
    
    Returns:
        A compiled StateGraph workflow ready to be executed
    """
    print("\n🔄 Creating Grab Food workflow graph")
    
    # Create the workflow
    graph = StateGraph(State)
    
    # Add all nodes to the graph
    graph.add_node("grab_food", grab_food)
    graph.add_node("manage_overloaded_restaurant", manage_overloaded_restaurant)
    graph.add_node("damaged_packaging_dispute", manage_packaging_dispute) 
    graph.add_node("communicate", communicate)
    
    # Set the entry point
    graph.set_entry_point("grab_food")
    
    # Add return edges - after each worker agent finishes, control returns to the grab_food router
    print("📊 Setting up workflow connections:")
    print("  - Restaurant overload agent → Orchestrator")
    graph.add_edge("manage_overloaded_restaurant", "grab_food")
    
    print("  - Packaging dispute agent → Orchestrator")
    graph.add_edge("damaged_packaging_dispute", "grab_food")
    
    print("  - Communication agent → Orchestrator")
    graph.add_edge("communicate", "grab_food")
    
    # Handle the FINISH node
    graph.add_node("FINISH", lambda state: state)
    graph.add_edge("FINISH", END)
    
    print("✅ Workflow graph created successfully")
    
    # Compile the workflow
    return graph.compile()

# Function to run the Grab Food workflow
def run_grab_food_workflow(query: str) -> Dict[str, Any]:
    """
    Runs the Grab Food workflow with the given query.
    
    Args:
        query: The user's query or issue description
        
    Returns:
        The final state of the workflow execution
    """
    # Print workflow initiation for CLI visibility
    print("\n" + "=" * 80)
    print(f"🍔 GRAB FOOD WORKFLOW INITIATED")
    print(f"Query: {query}")
    print("=" * 80)
    
    # Create the workflow
    workflow = create_grab_food_graph()
    
    # Prepare the initial state
    initial_state = State(
        messages=[
            SystemMessage(content="You are handling a Grab Food issue."),
            HumanMessage(content=query)
        ],
        next="",
        workflow_stage="analyze",
        issue_type="",
        issue_details={},
        current_agent=None,
        resolved_issues=[],
        actions_taken=[]
    )
    
    # Set up config for state tracking
    config = {"configurable": {"thread_id": f"grab_food_{hash(query) % 10000}"}}
    
    # Execute the workflow with event streaming to show progress
    print("\n📋 WORKFLOW EXECUTION PROGRESS:")
    print("-" * 80)
    
    # Stream execution events
    for event in workflow.stream(initial_state, config=config):
        if event.get("type") == "node":
            node_name = event.get("name", "unknown")
            print(f"⏩ Executing node: {node_name}")
    
    # Get the final state
    result = workflow.get_state(config)
    
    # Print summary of actions taken
    print("\n📊 WORKFLOW SUMMARY:")
    print("-" * 80)
    
    if hasattr(result, 'actions_taken') and result.actions_taken:
        print("Actions taken:")
        for i, action in enumerate(result.actions_taken, 1):
            print(f"  {i}. {action.get('action', 'unknown action')}")
            for key, value in action.items():
                if key != 'action':
                    print(f"     - {key}: {value}")
    
    # Print workflow completion
    print("\n" + "=" * 80)
    print("✅ GRAB FOOD WORKFLOW COMPLETED")
    print(f"Final stage: {getattr(result, 'workflow_stage', 'unknown')}")
    print("=" * 80)
    
    return result

# Create the main application for export
app = create_grab_food_graph()

# Main function for direct execution
if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Grab Food Issue Resolution System")
    parser.add_argument("query", nargs="?", default="The restaurant is taking too long to prepare my order", 
                      help="The food-related issue to resolve")
    parser.add_argument("--verbose", "-v", action="store_true", 
                      help="Show detailed output including all messages")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the workflow
    result = run_grab_food_workflow(args.query)
    
    # Print result based on verbosity
    if args.verbose:
        print("\n📝 FULL MESSAGE HISTORY:")
        print("-" * 80)
        for i, message in enumerate(result["messages"], 1):
            msg_type = message.type.upper() if hasattr(message, 'type') else "UNKNOWN"
            content = message.content if hasattr(message, 'content') else "No content"
            name = f" ({message.name})" if hasattr(message, 'name') and message.name else ""
            
            print(f"{i}. {msg_type}{name}: {content[:150]}")
            if len(content) > 150:
                print("   ...")
            print()
    else:
        # Print just the last AI response
        print("\n🤖 FINAL RESPONSE:")
        print("-" * 80)
        
        ai_messages = [m for m in result["messages"] if hasattr(m, 'type') and m.type == "ai"]
        if ai_messages:
            final_message = ai_messages[-1]
            print(final_message.content)
        else:
            print("No AI response found in the message history.")
    
    print("\n" + "=" * 80)
    print("Thank you for using the Grab Food Issue Resolution System!")
    print("=" * 80)