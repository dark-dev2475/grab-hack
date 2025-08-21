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
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.types import Command

# Import LLM
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")