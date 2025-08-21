"""
Restaurant overload management module for Grab Food service.
Handles situations where a restaurant is overloaded with orders.
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

# State definition
class RestaurantState(MessagesState):
    """State for restaurant overload management workflow."""
    prep_time: int = 40  # Default to 40 minutes if not set
    cuisine: str = "generic"
    actions_taken: List[Dict[str, Any]] = []
    human_decisions: Dict[str, Any] = {}
    current_step: str = "analyze"
    status: str = "in_progress"
    last_response: str = ""
    next: str = "grab_food"

# Tool definitions
@tool
def notify_customer(delay_minutes: int) -> str:
    """
    Notify the customer about long wait times and offer a voucher for the inconvenience. 
    
    Args:
        delay_minutes: The expected delay in minutes
        
    Returns:
        A message confirming the customer has been notified
    """
    return f"Customer notified about {delay_minutes} minute delay. A voucher has been offered for the inconvenience."

@tool
def reroute_driver(delay_minutes: int) -> str:
    """
    Re-route the driver to a short, nearby delivery while food is being prepared.
    
    Args:
        delay_minutes: The expected delay in minutes
        
    Returns:
        A message confirming the driver has been rerouted
    """
    return f"Driver has been rerouted to a nearby delivery that can be completed within {delay_minutes} minutes while food is being prepared."

@tool
def suggest_alternatives(delay_minutes: int) -> str:
    """
    Find a similar restaurant nearby with a shorter wait time and propose it to the customer.
    
    Args:
        delay_minutes: The expected delay in minutes
        
    Returns:
        A message with alternative restaurant suggestions
    """
    return f"Customer has been provided with 3 similar restaurant alternatives with wait times under {delay_minutes//2} minutes."

# Mock function to simulate human interaction
def get_human_decision(state: RestaurantState) -> RestaurantState:
    """
    Simulate a human decision about offering alternative restaurants.
    In a real implementation, this would be a UI interaction.
    
    Args:
        state: The current state
        
    Returns:
        Updated state with human decision
    """
    # Simulate human input based on delay
    prep_time = state.prep_time
    want_alternatives = prep_time > 30
    
    # In a real implementation:
    # want_alternatives = input(f"Delay is {prep_time} minutes. Suggest alternative restaurants? (y/n): ").lower().startswith("y")
    
    state.human_decisions["want_alternatives"] = want_alternatives
    if want_alternatives:
        state.current_step = "suggest_alternatives"
        message = "Customer would like to see alternative restaurants"
    else:
        state.current_step = "complete"
        message = "Customer is willing to wait for their order"
    
    state.messages.append(FunctionMessage(content=message, name="get_human_decision"))
    return state

# Node functions for the LangGraph
def analyze_situation(state: RestaurantState) -> RestaurantState:
    """
    Analyze the restaurant overload situation and plan next steps.
    
    Args:
        state: The current state
        
    Returns:
        Updated state with analysis
    """
    # Create a situation message
    situation_message = f"The restaurant is overloaded with a {state.prep_time}-minute prep time for {state.cuisine} food."
    state.messages.append(HumanMessage(content=situation_message))
    
    # System message explaining the agent's role
    system_message = """
    You are a Grab Food order management agent handling an overloaded restaurant situation.
    
    Analyze the current situation with the overloaded restaurant and determine the severity of the delay.
    Based on your analysis, explain what steps should be taken.
    """
    
    # Generate analysis using LLM
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"messages": state.messages})
    
    # Add the analysis to the state
    state.messages.append(AIMessage(content=response))
    state.last_response = response
    state.current_step = "notify"
    
    return state

def notify_customer_node(state: RestaurantState) -> RestaurantState:
    """
    Execute the notify_customer tool and update state.
    
    Args:
        state: The current state
        
    Returns:
        Updated state with notification result
    """
    prep_time = state.prep_time
    result = notify_customer(prep_time)
    
    # Update state with the action taken
    state.actions_taken.append({"action": "notify_customer", "result": result})
    state.messages.append(FunctionMessage(content=result, name="notify_customer"))
    state.current_step = "reroute"
    
    return state

def reroute_driver_node(state: RestaurantState) -> RestaurantState:
    """
    Execute the reroute_driver tool and update state.
    
    Args:
        state: The current state
        
    Returns:
        Updated state with reroute result
    """
    prep_time = state.prep_time
    result = reroute_driver(prep_time)
    
    # Update state with the action taken
    state.actions_taken.append({"action": "reroute_driver", "result": result})
    state.messages.append(FunctionMessage(content=result, name="reroute_driver"))
    state.current_step = "ask_about_alternatives"
    
    return state

def suggest_alternatives_node(state: RestaurantState) -> RestaurantState:
    """
    Execute the suggest_alternatives tool and update state.
    
    Args:
        state: The current state
        
    Returns:
        Updated state with alternative suggestions
    """
    # Only execute if human wanted alternatives
    if state.human_decisions.get("want_alternatives", False):
        prep_time = state.prep_time
        result = suggest_alternatives(prep_time)
        
        # Update state with the action taken
        state.actions_taken.append({"action": "suggest_alternatives", "result": result})
        state.messages.append(FunctionMessage(content=result, name="suggest_alternatives"))
    
    state.current_step = "complete"
    return state

def complete_process(state: RestaurantState) -> RestaurantState:
    """
    Summarize the actions taken and complete the process.
    
    Args:
        state: The current state
        
    Returns:
        Updated state with summary and completion status
    """
    # Create a summary message
    summary_prompt = f"""
    Summarize the actions taken to handle the overloaded restaurant situation:
    - Delay: {state.prep_time} minutes
    - Cuisine: {state.cuisine}
    - Actions: {state.actions_taken}
    
    What was the outcome of this situation?
    """
    
    # Generate summary using LLM
    state.messages.append(HumanMessage(content=summary_prompt))
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a Grab Food order management agent summarizing your actions."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"messages": state.messages})
    
    # Add the summary to the state
    state.messages.append(AIMessage(content=response))
    state.last_response = response
    state.status = "addressed"
    
    return state

# Router function to determine next node
def router(state: RestaurantState) -> Union[str, Literal["END"]]:
    """
    Route to the next step in the workflow based on current_step.
    
    Args:
        state: The current state
        
    Returns:
        Next node to execute or END
    """
    if state.current_step == "analyze":
        return "analyze_situation"
    elif state.current_step == "notify":
        return "notify_customer"
    elif state.current_step == "reroute":
        return "reroute_driver"
    elif state.current_step == "ask_about_alternatives":
        return "get_human_decision"
    elif state.current_step == "suggest_alternatives":
        return "suggest_alternatives"
    elif state.current_step == "complete":
        return "complete_process"
    else:
        return "END"

# Define final_state to be called when the graph completes
def final_state(state: RestaurantState) -> RestaurantState:
    """
    Final processing before returning to parent agent.
    
    Args:
        state: The current state
        
    Returns:
        Final state to return to parent
    """
    # Ensure next is set to grab_food (parent agent)
    state.next = "grab_food"
    return state

# Build the LangGraph
def build_restaurant_overload_graph() -> StateGraph:
    """
    Build the LangGraph for managing restaurant overload.
    
    Returns:
        A StateGraph instance for the workflow
    """
    # Create the graph
    workflow = StateGraph(RestaurantState)
    
    # Add nodes
    workflow.add_node("analyze_situation", analyze_situation)
    workflow.add_node("notify_customer", notify_customer_node)
    workflow.add_node("reroute_driver", reroute_driver_node)
    workflow.add_node("get_human_decision", get_human_decision)
    workflow.add_node("suggest_alternatives", suggest_alternatives_node)
    workflow.add_node("complete_process", complete_process)
    
    # Add edges
    workflow.add_edge("analyze_situation", "notify_customer")
    workflow.add_edge("notify_customer", "reroute_driver")
    workflow.add_edge("reroute_driver", "get_human_decision")
    workflow.add_conditional_edges(
        "get_human_decision",
        router,
        {
            "suggest_alternatives": "suggest_alternatives",
            "complete_process": "complete_process",
        }
    )
    workflow.add_edge("suggest_alternatives", "complete_process")
    workflow.add_edge("complete_process", END)
    
    # Set the entry point
    workflow.set_entry_point("analyze_situation")
    
    # Add a final state processor
    workflow.add_node("final", final_state)
    workflow.add_edge(END, "final")
    
    return workflow

# Main function for managing overloaded restaurants
def manage_overloaded_restaurant(state: RestaurantState) -> Command[Literal["grab_food", "FINISH"]]:
    """
    Handle situations where a restaurant is overloaded with orders, causing long wait times.
    Uses LangGraph with proper nodes and edges for workflow control.
    
    Args:
        state: The current state including messages
        
    Returns:
        Command with the next routing destination
    """
    # Build the graph
    workflow = build_restaurant_overload_graph()
    
    # Create a memory saver to store the state between runs
    memory = MemorySaver()
    
    # Compile the graph with the memory saver
    app = workflow.compile(checkpointer=memory)
    
    # Initialize the state if needed
    if not hasattr(state, "prep_time"):
        state.prep_time = 40
    if not hasattr(state, "cuisine"):
        state.cuisine = "generic"
    if not hasattr(state, "actions_taken"):
        state.actions_taken = []
    if not hasattr(state, "human_decisions"):
        state.human_decisions = {}
    if not hasattr(state, "current_step"):
        state.current_step = "analyze"
    
    # Run the graph
    result = app.invoke(state)
    
    # Extract the final state
    final_state = result
    
    # Return the command to route to the parent agent
    return Command(
        goto=final_state.next,
        update={
            "next": final_state.next,
            "current_step": final_state.current_step,
            "actions_taken": final_state.actions_taken,
            "human_decisions": final_state.human_decisions,
            "last_response": final_state.last_response,
            "status": final_state.status
        }
    )

# When this module is imported, this will be available
def get_tools():
    """Return the tools from this module."""
    return [notify_customer, reroute_driver, suggest_alternatives]
