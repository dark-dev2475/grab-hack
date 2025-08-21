"""
Test script for the packaging dispute resolution workflow.
This demonstrates how to use the LangGraph implementation for resolving
disputes related to damaged packaging in Grab Food deliveries.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path to import the modules
sys.path.append(str(Path(__file__).parent.parent))

from agents.food.damage import run_packaging_dispute_workflow

def main():
    """Run a test of the packaging dispute resolution workflow."""
    print("=== Grab Food - Packaging Dispute Resolution Test ===\n")
    
    # Define a test dispute scenario
    dispute_details = "My food delivery arrived with torn packaging and the food was spilled all over the bag. I want a refund and to make sure this doesn't happen again."
    
    print(f"Dispute details: {dispute_details}\n")
    print("Starting workflow...\n")
    
    # Run the workflow
    result = run_packaging_dispute_workflow(dispute_details)
    
    print("=== Workflow Result ===\n")
    print("Messages exchanged during dispute resolution:\n")
    
    # Print the messages from the workflow execution
    for i, message in enumerate(result["messages"]):
        print(f"Message {i+1} ({message.type}):")
        # Truncate long messages for readability
        content = message.content
        if len(content) > 150:
            content = content[:147] + "..."
        print(f"  {content}")
        print()
    
    # Print the final resolution
    print("=== Final Resolution ===")
    resolution = result.get("resolution", "No resolution recorded")
    print(f"Resolution: {resolution}\n")
    
    # Print evidence and analysis summary if available
    if result.get("evidence"):
        print("=== Evidence Summary ===")
        print(f"Customer evidence: {result.get('customer_response', 'None')}")
        print(f"Driver evidence: {result.get('driver_response', 'None')}\n")
    
    if result.get("analysis_result"):
        print("=== Analysis Result ===")
        analysis = result.get("analysis_result")
        print(f"Responsibility: {analysis.get('responsibility', 'Unknown')}")
        print(f"Confidence: {analysis.get('confidence', 0)}")
        if analysis.get("human_reviewed"):
            print("Note: This case was reviewed by a human specialist")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
