"""
ChatLib - A library for building chat applications with LLMs
"""

from .convo import Conversation
from .llm import LLM
from .models import Message, Role
from .tools import Tool
from .vectors import VectorStore

__version__ = "0.1.0"
__all__ = ["Conversation", "LLM", "Message", "Role", "Tool", "VectorStore"]
