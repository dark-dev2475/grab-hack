"""
Schema module for state management in the Grab-X project.
Defines the structure of state objects used by different agents.
"""

from typing import List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict
from langgraph.graph import MessagesState

# Base state class extended from MessagesState
class State(MessagesState):
    """Base state class for all agents."""
    next: str  # Next agent to route to

# Router classes for different services
class CarRouter(TypedDict):
    """Router for Grab Car service."""
    next: Literal["handle_traffic_dispute", "airport_pickup_assistance", "communicate", "FINISH"]

class FoodRouter(TypedDict):
    """Router for Grab Food service."""
    next: Literal["manage_overloaded_restaurant", "damaged_packaging_dispute", "communicate", "FINISH"]

class ExpressRouter(TypedDict):
    """Router for Grab Express service."""
    next: Literal["handle_recipient_not_present", "communicate", "FINISH"]

# Extended state classes for specific needs
class CarState(State):
    """Extended state for car service with traffic information."""
    check_traffic: bool = False
    alt_route: bool = False
    notify: bool = False
    flight: bool = False
    
class FoodState(State):
    """Extended state for food service with restaurant and dispute information."""
    overloaded_restaurant: bool = False
    packaging_dispute: bool = False
    
class ExpressState(State):
    """Extended state for express service with delivery information."""
    recipient_not_present: bool = False
    needs_locker: bool = False
