"""
Main orchestrator for Grab Food agents.
This module orchestrates between different agents for managing restaurant overload,
packaging disputes, and customer communication in the Grab Food service.
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

# Router class for determining the next agent to call
class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["manage_overloaded_restaurant", "damaged_packaging_dispute", "communicate", 'FINISH']

# Define system prompt
system_prompt = f"""
You are a supervisor agent of grab-food section, tasked with managing a orchestration between the following agents: {members}. 
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
    # Prepare messages for the router
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    
    # Add context based on the current next state
    if state.next == "manage_overloaded_restaurant":
        messages.append({"role": "user", "content": "The restaurant is overloaded, please manage the situation."})
    elif state.next == "damaged_packaging_dispute":
        messages.append({"role": "user", "content": "There is a dispute regarding damaged packaging."})
    elif state.next == "communicate":
        messages.append({"role": "user", "content": "Please communicate with the customer."})
    else:
        messages.append({"role": "user", "content": "No further action is needed."})
    
    # Get routing decision from the LLM
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    
    # Handle FINISH case
    if goto == "FINISH":
        goto = END
    
    # Update the state with the next agent to call
    return Command(goto=goto, update={"next": goto})

# Create the workflow graph

# Create the workflow graph
def create_grab_food_graph():
    """
    Creates and returns the Grab Food orchestration graph.
    
    Returns:
        A compiled StateGraph workflow ready to be executed
    """
    # Create the workflow
    graph = StateGraph(State)
    
    # Add all nodes to the graph
    graph.add_node("grab_food", grab_food)
    graph.add_node("manage_overloaded_restaurant", manage_overloaded_restaurant)
    # Note: I'm correcting the name to match your code's Router
    graph.add_node("damaged_packaging_dispute", manage_packaging_dispute) 
    graph.add_node("communicate", communicate)
    
    # Set the entry point
    graph.set_entry_point("grab_food")
    
    # *** THE FIX: ADD RETURN EDGES HERE ***
    # After each worker agent finishes, control returns to the grab_food router.
    graph.add_edge("manage_overloaded_restaurant", "grab_food")
    graph.add_edge("damaged_packaging_dispute", "grab_food")
    graph.add_edge("communicate", "grab_food")
    graph.add_node("FINISH", lambda state: state)
    graph.add_edge("FINISH", END)

    
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
    # Create the workflow
    workflow = create_grab_food_graph()
    
    # Execute the workflow
    config = {"configurable": {"thread_id": "grab_food_1"}}
    
    initial_state = State(
        messages=[
            SystemMessage(content="You are handling a Grab Food issue."),
            HumanMessage(content=query)
        ],
        next=""
    )

    
    for event in workflow.stream(initial_state, config=config):
      pass
    
    # Get the final state
    result = workflow.get_state(config)
    
    return result

# Create the main application for export
app = create_grab_food_graph()

# Example usage:
# if __name__ == "__main__":
#     result = run_grab_food_workflow("The restaurant is taking too long to prepare my order.")
#     for message in result["messages"]:
#         print(f"{message.type}: {message.content[:100]}...")