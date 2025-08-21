# Restaurant Overload Management with LangGraph

## Overview

This module implements a restaurant overload management workflow using LangGraph's state-driven approach with proper nodes and edges. It handles situations where a restaurant is overloaded with orders, causing long wait times for customers.

## Architecture

The implementation uses LangGraph's `StateGraph` to define a workflow with the following components:

1. **State Class**: `RestaurantState` extends `MessagesState` and includes additional fields specific to restaurant overload management.

2. **Node Functions**: Each step in the workflow is implemented as a separate function:
   - `analyze_situation`: Analyzes the restaurant overload situation
   - `notify_customer_node`: Notifies customers about delays
   - `reroute_driver_node`: Reroutes drivers to other deliveries during wait time
   - `get_human_decision`: Gets (or simulates) human input about alternatives
   - `suggest_alternatives_node`: Suggests alternative restaurants
   - `complete_process`: Summarizes actions and completes the process

3. **Router Function**: The `router` function determines the next node based on the current step.

4. **Graph Construction**: The `build_restaurant_overload_graph` function builds a `StateGraph` with appropriate nodes and edges.

5. **Tools**: The module provides tools for notification, rerouting, and suggesting alternatives.

## Workflow

The workflow follows these steps:

1. **Analyze**: Assess the severity of the restaurant overload
2. **Notify**: Inform the customer about delays and offer a voucher
3. **Reroute**: Find alternative tasks for the driver during the wait
4. **Human Decision**: Ask if the customer wants to see alternative restaurants
5. **Suggest Alternatives**: If desired, suggest alternative restaurants
6. **Complete**: Summarize actions and complete the process

## Human-in-the-Loop

The workflow includes a human decision point where the customer is asked if they want to see alternative restaurants. In the current implementation, this is simulated based on the delay time, but it can be replaced with a real UI interaction in production.

## Usage

To use this module, import the `manage_overloaded_restaurant` function and call it with a `RestaurantState` object:

```python
from agents.food.overload import manage_overloaded_restaurant, RestaurantState
from langchain_core.messages import HumanMessage

# Create initial state
state = RestaurantState(
    messages=[HumanMessage(content="Restaurant is overloaded")],
    prep_time=45,
    cuisine="Italian"
)

# Run the workflow
result = manage_overloaded_restaurant(state)
```

Alternatively, you can use the graph directly:

```python
from agents.food.overload import build_restaurant_overload_graph, RestaurantState
from langgraph.checkpoint.memory import MemorySaver

# Build graph
workflow = build_restaurant_overload_graph()
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Run workflow
result = app.invoke(initial_state)
```

## Benefits of LangGraph Implementation

1. **Explicit State Transitions**: Each step in the workflow is explicitly defined
2. **Visual Representation**: The graph can be visualized to understand the workflow
3. **Modularity**: Each step is implemented as a separate function
4. **Reusability**: Components can be reused in other workflows
5. **Maintainability**: Easier to maintain and modify
6. **Testability**: Each component can be tested separately
7. **Human-in-the-Loop**: Built-in support for human intervention

## Testing

A test script (`test_overload_graph.py`) is provided to demonstrate how to use the LangGraph implementation.
