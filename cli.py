#!/usr/bin/env python
"""
CLI tool for testing Grab Food agents.
This CLI allows you to test different agents and scenarios for the Grab Food service.
"""

import os
import sys
import argparse
from typing import Dict, Any, List
import json
from datetime import datetime

# Import LangGraph
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

# Import agent modules
from agents.food.food import run_grab_food_workflow
# Note: Other modules imported directly in their functions to avoid circular imports


def print_color(text: str, color: str = "white") -> None:
    """Print text in color."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")


def format_message(message: Dict[str, Any]) -> str:
    """Format a message for display."""
    if message.get("type") == "system":
        return f"SYSTEM: {message.get('content')}"
    elif message.get("type") == "human":
        return f"USER: {message.get('content')}"
    elif message.get("type") == "ai":
        return f"AI: {message.get('content')}"
    elif message.get("type") == "function":
        return f"FUNCTION ({message.get('name', 'unknown')}): {message.get('content')}"
    else:
        return f"{message.get('type', 'UNKNOWN').upper()}: {message.get('content')}"


def display_messages(messages: List[Dict[str, Any]]) -> None:
    """Display a list of messages with appropriate formatting."""
    for message in messages:
        if message.get("type") == "system":
            print_color(format_message(message), "blue")
        elif message.get("type") == "human":
            print_color(format_message(message), "green")
        elif message.get("type") == "ai":
            print_color(format_message(message), "yellow")
        elif message.get("type") == "function":
            print_color(format_message(message), "cyan")
        else:
            print_color(format_message(message), "white")
        print("-" * 80)


def run_orchestrator(query: str, verbose: bool = False) -> None:
    """Run the main Grab Food orchestrator with a query."""
    print_color("\n🚀 Running Grab Food Orchestrator", "magenta")
    print_color(f"Query: {query}", "yellow")
    
    try:
        result = run_grab_food_workflow(query)
        
        if verbose:
            # Display full message history
            print_color("\n📋 Full Message History:", "blue")
            messages = [
                {
                    "type": msg.type,
                    "content": msg.content,
                    "name": getattr(msg, "name", None)
                }
                for msg in result.get("messages", [])
            ]
            display_messages(messages)
        else:
            # Just display the final result
            print_color("\n🏁 Final Result:", "green")
            last_messages = result.get("messages", [])[-3:]
            messages = [
                {
                    "type": msg.type,
                    "content": msg.content,
                    "name": getattr(msg, "name", None)
                }
                for msg in last_messages
            ]
            display_messages(messages)
            
        print_color("\n✅ Orchestrator execution completed", "green")
        
    except Exception as e:
        print_color(f"\n❌ Error running orchestrator: {str(e)}", "red")
        if verbose:
            import traceback
            print_color(traceback.format_exc(), "red")


def run_restaurant_overload(delay_minutes: int = 40, cuisine: str = "Italian", verbose: bool = False) -> None:
    """Run the restaurant overload scenario directly."""
    print_color("\n🚀 Running Restaurant Overload Scenario", "magenta")
    print_color(f"Delay: {delay_minutes} minutes, Cuisine: {cuisine}", "yellow")
    
    try:
        # Create a simulated state with the restaurant information
        initial_state = MessagesState(
            messages=[
                SystemMessage(content=f"Restaurant is overloaded with a {delay_minutes}-minute prep time for {cuisine} food."),
                HumanMessage(content=f"The {cuisine} restaurant is taking {delay_minutes} minutes to prepare orders. Handle this situation.")
            ],
            prep_time=delay_minutes,
            cuisine=cuisine,
            current_step="analyze"
        )
        
        # Import the function here to avoid circular imports
        from agents.food.overload import manage_overloaded_restaurant
        result = manage_overloaded_restaurant(initial_state)
        
        if verbose:
            # Display detailed results
            print_color("\n📋 Actions taken:", "blue")
            for action in result.updates.get("actions_taken", []):
                print_color(f"  - {action.get('action')}: {action.get('result')}", "cyan")
            
            print_color("\n📝 Last response:", "yellow")
            print_color(result.updates.get("last_response", "No response"), "white")
        else:
            # Display summary
            print_color("\n🏁 Scenario completed", "green")
            print_color(f"Status: {result.updates.get('status', 'unknown')}", "yellow")
            print_color(f"Next step: {result.goto}", "yellow")
        
        print_color("\n✅ Restaurant overload handling completed", "green")
        
    except Exception as e:
        print_color(f"\n❌ Error running restaurant overload: {str(e)}", "red")
        if verbose:
            import traceback
            print_color(traceback.format_exc(), "red")


def run_packaging_dispute(dispute_details: str = "Packaging was damaged and food spilled", verbose: bool = False) -> None:
    """Run the packaging dispute scenario directly."""
    print_color("\n🚀 Running Packaging Dispute Scenario", "magenta")
    print_color(f"Dispute: {dispute_details}", "yellow")
    
    try:
        # Create a simulated state with the dispute details
        initial_state = MessagesState(
            messages=[
                SystemMessage(content=f"You are handling a packaging dispute: {dispute_details}"),
                HumanMessage(content=dispute_details)
            ],
            dispute_stage="initiate"
        )
        
        # Import the function here to avoid circular imports
        from agents.food.damage import manage_packaging_dispute
        result = manage_packaging_dispute(initial_state)
        
        if verbose:
            # Display dispute resolution details
            print_color("\n📊 Dispute Resolution:", "blue")
            print_color(f"Next step: {result.goto}", "yellow")
            print_color(f"Updates: {result.updates}", "cyan")
        else:
            # Just display the resolution
            print_color("\n🏁 Dispute Resolution:", "green")
            print_color(f"Next step: {result.goto}", "yellow")
        
        print_color("\n✅ Packaging dispute handling completed", "green")
        
    except Exception as e:
        print_color(f"\n❌ Error running packaging dispute: {str(e)}", "red")
        if verbose:
            import traceback
            print_color(traceback.format_exc(), "red")


def run_communication(issue_description: str = "order delay", verbose: bool = False) -> None:
    """Run the customer communication scenario directly."""
    print_color("\n🚀 Running Customer Communication", "magenta")
    print_color(f"Issue: {issue_description}", "yellow")
    
    try:
        # Create a simulated state with the issue description
        initial_state = MessagesState(
            messages=[
                SystemMessage(content="You are handling customer communication after resolving an issue."),
                HumanMessage(content=f"Please communicate with the customer about their {issue_description} that has been resolved.")
            ]
        )
        
        # Import the function here to avoid circular imports
        from agents.food.communicate import communicate
        result = communicate(initial_state)
        
        if verbose:
            # Display all communication details
            print_color("\n📋 Communication Details:", "blue")
            print_color(f"Next step: {result.goto}", "yellow")
            print_color(f"Updates: {result.updates}", "cyan")
        else:
            # Display a summary
            print_color("\n🏁 Communication completed", "green")
            print_color(f"Next step: {result.goto}", "yellow")
        
        print_color("\n✅ Customer communication completed", "green")
        
    except Exception as e:
        print_color(f"\n❌ Error running communication: {str(e)}", "red")
        if verbose:
            import traceback
            print_color(traceback.format_exc(), "red")


def interactive_mode() -> None:
    """Run the CLI in interactive mode."""
    print_color("\n🤖 Grab Food Agents Interactive CLI", "magenta")
    print_color("=" * 80, "magenta")
    print_color("Type 'exit' or 'quit' to end the session", "yellow")
    
    while True:
        print_color("\nChoose a scenario:", "green")
        print_color("1. Full Orchestrator (routes to appropriate agent)")
        print_color("2. Restaurant Overload")
        print_color("3. Packaging Dispute")
        print_color("4. Customer Communication")
        print_color("5. Exit")
        
        choice = input("\nEnter choice (1-5): ")
        
        if choice == "5" or choice.lower() in ["exit", "quit"]:
            print_color("\nExiting interactive mode. Goodbye!", "magenta")
            break
            
        verbose = input("Show verbose output? (y/n): ").lower().startswith("y")
        
        if choice == "1":
            query = input("\nEnter your query for the orchestrator: ")
            run_orchestrator(query, verbose)
            
        elif choice == "2":
            try:
                delay = int(input("\nEnter delay in minutes (default 40): ") or "40")
                cuisine = input("Enter cuisine type (default Italian): ") or "Italian"
                run_restaurant_overload(delay, cuisine, verbose)
            except ValueError:
                print_color("Please enter a valid number for delay", "red")
                
        elif choice == "3":
            dispute = input("\nEnter dispute details: ") or "Packaging was damaged and food spilled"
            run_packaging_dispute(dispute, verbose)
            
        elif choice == "4":
            issue = input("\nEnter issue description: ") or "order delay"
            run_communication(issue, verbose)
            
        else:
            print_color("Invalid choice. Please try again.", "red")


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="CLI for testing Grab Food agents")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Orchestrator command
    orch_parser = subparsers.add_parser("orchestrator", help="Run the main orchestrator")
    orch_parser.add_argument("query", help="Query to send to the orchestrator")
    orch_parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output")
    
    # Restaurant overload command
    rest_parser = subparsers.add_parser("restaurant", help="Run restaurant overload scenario")
    rest_parser.add_argument("-d", "--delay", type=int, default=40, help="Delay time in minutes")
    rest_parser.add_argument("-c", "--cuisine", default="Italian", help="Cuisine type")
    rest_parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output")
    
    # Packaging dispute command
    pkg_parser = subparsers.add_parser("dispute", help="Run packaging dispute scenario")
    pkg_parser.add_argument("-d", "--details", default="Packaging was damaged and food spilled", 
                          help="Details of the dispute")
    pkg_parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output")
    
    # Communication command
    comm_parser = subparsers.add_parser("communicate", help="Run customer communication scenario")
    comm_parser.add_argument("-i", "--issue", default="order delay", help="Issue description")
    comm_parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output")
    
    # Interactive mode command
    subparsers.add_parser("interactive", help="Run in interactive mode")
    
    # Parse args
    args = parser.parse_args()
    
    # If no arguments were provided, default to interactive mode
    if not args.command:
        interactive_mode()
        return
    
    # Execute the appropriate command
    if args.command == "orchestrator":
        run_orchestrator(args.query, args.verbose)
    elif args.command == "restaurant":
        run_restaurant_overload(args.delay, args.cuisine, args.verbose)
    elif args.command == "dispute":
        run_packaging_dispute(args.details, args.verbose)
    elif args.command == "communicate":
        run_communication(args.issue, args.verbose)
    elif args.command == "interactive":
        interactive_mode()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
