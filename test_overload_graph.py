"""
Test script for the restaurant overload management workflow.
Demonstrates how to use the LangGraph implementation.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from agents.food.overload import (
    build_restaurant_overload_graph,
    RestaurantState,
    notify_customer,
    reroute_driver,
    suggest_alternatives
)

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

def main():
    """Run a test of the restaurant overload management workflow."""
    print("Testing Restaurant Overload Management Workflow")
    print("-" * 60)
    
    # Create an initial state
    initial_state = RestaurantState(
        messages=[
            HumanMessage(content="I have an order for a restaurant that's currently overloaded.")
        ],
        prep_time=45,  # 45 minute delay
        cuisine="Italian",
        current_step="analyze"
    )
    
    # Build the graph
    workflow = build_restaurant_overload_graph()
    
    # Create a memory saver
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    
    # Run the workflow
    print("Running the workflow...")
    result = app.invoke(initial_state)
    
    # Print the results
    print("\nWorkflow Results:")
    print("-" * 60)
    print(f"Status: {result.status}")
    print(f"Last Step: {result.current_step}")
    
    print("\nActions Taken:")
    for action in result.actions_taken:
        print(f"- {action['action']}: {action['result']}")
    
    print("\nHuman Decisions:")
    for key, value in result.human_decisions.items():
        print(f"- {key}: {value}")
    
    print("\nFinal Response:")
    print(result.last_response)

if __name__ == "__main__":
    main()
