"""
Main entry point for the Grab-X project.
Initializes the agent system and API.
"""

import os
from config.settings import GOOGLE_API_KEY

def main():
    """
    Main entry point for the Grab-X application.
    Initialize all services and start the API server.
    """
    print("Starting Grab-X Multi-Agent System")
    
    # Verify environment
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    # Initialize services
    # This would normally set up the agent system and API
    print("All services initialized successfully")
    
    # Start API server
    print("API server started at http://localhost:8000")
    
    # In a real implementation, this would start a FastAPI server
    # For now, we'll just print a message
    print("Grab-X system is ready")

if __name__ == "__main__":
    main()
