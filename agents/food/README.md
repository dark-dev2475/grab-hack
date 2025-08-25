# Grab Food Intelligent Agent System

## Overview

This is an advanced intelligent agent system for Grab Food services, built using LangGraph and LangChain with Google's Gemini 1.5 Pro LLM. The system provides a sophisticated orchestration framework for handling various food-related issues in the Grab app ecosystem.

## System Architecture

The system follows a hierarchical agent architecture with a main orchestrator and specialized agents:

```
Grab Food Orchestrator
├── Restaurant Overload Agent
├── Packaging Dispute Agent
└── Customer Communication Agent
```

### Key Components

1. **Main Orchestrator (`food.py`)**
   - Central routing and decision-making hub
   - Analyzes user queries to determine the issue type
   - Routes to appropriate specialized agents
   - Ensures proper workflow progression
   - Maintains state across agent transitions

2. **Restaurant Overload Agent (`overload.py`)**
   - Handles delays and capacity issues at restaurants
   - Implements circular workflow for situation re-analysis
   - Offers dynamic customer compensation based on delay severity
   - Provides driver rerouting capabilities
   - Suggests alternative restaurants when needed

3. **Packaging Dispute Agent (`damage.py`)**
   - Mediates disputes between customers and drivers regarding packaging damage
   - Collects and analyzes evidence from both parties
   - Determines responsibility with confidence levels
   - Issues appropriate refunds
   - Logs packaging feedback for merchant improvement
   - Exonerates drivers when appropriate

4. **Customer Communication Agent (`communicate.py`)**
   - Handles all customer-facing communication after issue resolution
   - Sends personalized resolution notifications
   - Issues satisfaction surveys for feedback collection
   - Offers promotional vouchers based on issue severity
   - Ensures proper closure of the customer experience

## Intelligent Features

### 1. Sophisticated Workflow Management

The system uses LangGraph's directed graph capabilities to implement advanced workflows:

- **Dynamic routing** based on issue analysis
- **Circular workflows** that can revisit previous steps when needed
- **Proper state management** across agent transitions
- **Comprehensive logging** of actions and decisions

### 2. Reasoning and Decision Making

- **Chain-of-thought reasoning** visible in CLI output
- **Evidence-based analysis** for packaging disputes
- **Confidence levels** in responsibility determinations
- **Severity-based compensation** decisions

### 3. Personalized Customer Experience

- **Tailored communication** based on issue type and severity
- **Dynamic voucher amounts** based on impact to customer
- **Personalized follow-up** after resolution
- **Feedback collection** integrated into the workflow

## Usage Examples

### Example 1: Restaurant Overload

```bash
python -m cli dispute "The restaurant is taking 45 minutes to prepare my pizza order"
```

**System Response:**
```
🚀 GRAB FOOD WORKFLOW INITIATED
Query: The restaurant is taking 45 minutes to prepare my pizza order
=====================================================================

🧭 GRAB FOOD ORCHESTRATOR
Issue classified as: Restaurant delay
Routing to: manage_overloaded_restaurant (confidence: 0.95)

⏩ Executing node: manage_overloaded_restaurant
🔄 ANALYZING RESTAURANT OVERLOAD
Prep time: 45 minutes, Cuisine: Italian (Pizza)

✅ CUSTOMER NOTIFICATION SENT
Customer notified about 45 minute delay. A $5.00 voucher has been offered.

✅ DRIVER REROUTED
Driver has been rerouted to a nearby delivery that can be completed in 30 minutes.

⏩ Executing node: grab_food
🧭 GRAB FOOD ORCHESTRATOR
Routing to communication agent to inform customer of resolution

⏩ Executing node: communicate
🚀 STARTING COMMUNICATION WORKFLOW
📝 PERSONALIZED CUSTOMER MESSAGE DRAFTED
Hello there, we wanted to follow up regarding the delay with your pizza order...

⏩ Executing node: grab_food
🧭 GRAB FOOD ORCHESTRATOR
All issues resolved and communication complete. Finishing workflow.

✅ GRAB FOOD WORKFLOW COMPLETED
Final stage: complete
```

### Example 2: Packaging Dispute

```bash
python -m cli dispute "My food was spilled during delivery because the packaging was damaged"
```

**System Response:**
```
🚀 GRAB FOOD WORKFLOW INITIATED
Query: My food was spilled during delivery because the packaging was damaged
=====================================================================

🧭 GRAB FOOD ORCHESTRATOR
Issue classified as: Packaging damage dispute
Routing to: damaged_packaging_dispute (confidence: 0.97)

⏩ Executing node: damaged_packaging_dispute
🔄 MANAGING PACKAGING DISPUTE
📝 Dispute details: My food was spilled during delivery because the packaging was damaged

✅ MEDIATION INITIATED
Order paused. Real-time communication portal opened between customer and driver.

📸 EVIDENCE COLLECTED FROM CUSTOMER
Response: The container was torn on one side and the food was spilling out
Photos provided: Yes

📸 EVIDENCE COLLECTED FROM DRIVER
Response: The packaging was intact when I picked it up. I was careful during transport.
Photos provided: Yes

🧠 EVIDENCE ANALYSIS COMPLETED
Responsibility: merchant (82% confidence)
Reasoning: The evidence suggests that the packaging was initially defective...

💰 REFUND ISSUED
Amount: $15.00
Reason: Full refund due to clear merchant packaging issue

📋 MERCHANT FEEDBACK LOGGED
Severity: high
Details: Severe packaging failure detected: Immediate packaging process review recommended.

⏩ Executing node: grab_food
🧭 GRAB FOOD ORCHESTRATOR
Routing to communication agent to inform customer of resolution

⏩ Executing node: communicate
🚀 STARTING COMMUNICATION WORKFLOW
...

✅ GRAB FOOD WORKFLOW COMPLETED
Final stage: complete
```

## Technical Implementation

### State Management

The system uses sophisticated state management to track:

- Current workflow stage
- Issue type and details
- Actions taken by each agent
- Evidence collected
- Analysis results
- Resolution status

### Extensibility

The modular architecture allows for easy extension:

- New specialized agents can be added to the orchestrator
- Additional tools can be integrated into existing agents
- New analysis capabilities can be incorporated
- Alternative LLMs can be substituted

## Deployment

The system is designed to be run directly via CLI or integrated into a larger service architecture:

```python
# Direct CLI usage
python -m cli orchestrator "My food delivery issue query"

# API integration
from agents.food.food import run_grab_food_workflow
result = run_grab_food_workflow("Customer query here")
```

## Future Enhancements

Potential areas for system expansion:

1. **Real-time order tracking integration**
2. **Driver performance analytics**
3. **Restaurant quality monitoring**
4. **Predictive delay estimation**
5. **Multi-order coordination**
6. **Dietary preference handling**

## Conclusion

This intelligent agent system represents a sophisticated approach to food delivery issue resolution, demonstrating the power of LangGraph for orchestrating complex workflows between specialized agents. The combination of structured reasoning, personalized communication, and detailed analysis creates a comprehensive solution for Grab Food services.
