"""
LLM model implementations and utilities for ChatLib.

This module provides:
- Base Model class with validation utilities
- Implementations for various LLM providers (OpenAI, Anthropic, Mistral)
- Message transformation utilities
- Cost tracking for different models
"""

from dotenv import load_dotenv
import json
from types import FunctionType
from itertools import groupby
import os
from openai import OpenAI
from anthropic import Anthropic
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Load environment variables from this directory
load_dotenv()

# Default temperature for all models
TEMPERATURE = 0.25

def transform_message(m):
    """
    Transform a message tuple into a dictionary format.
    
    Args:
        m: Either a tuple of (role, content) or a dictionary
        
    Returns:
        dict: Message in dictionary format with 'role' and 'content' keys
    """
    if isinstance(m, (tuple, list)):
        assert len(m) == 2
        return {"role": m[0], "content": m[1]}
    return m

def transform_messages(ms):
    """
    Transform a list of messages into the standard format.
    
    Args:
        ms: List of messages (can be tuples or dictionaries)
        
    Returns:
        list: List of messages in dictionary format
    """
    if not ms:
        return []
    return [transform_message(m) for m in ms]

class Model:
    """
    Base class for LLM model implementations.
    
    This class provides common functionality for all LLM models including:
    - Message validation and type conversion
    - Cost tracking
    - Response formatting
    """
    
    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return self.name if hasattr(self, "name") else super().__repr__()
    
    def execute(self, messages):
        """
        Execute the model with the given messages.
        
        Args:
            messages: List of messages to process
            
        Returns:
            Result: The model's response with token usage information
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError
    
    _type = type
    def valid(self, messages, type='json', iters=None, **kwargs):
        """
        Get a validated response from the model.
        
        Args:
            messages: List of messages to process
            type: Expected response type ('json', 'int', 'float', 'bool', 'list', 'str' or custom function)
            iters: Maximum number of attempts to get valid response
            **kwargs: Additional arguments to pass to execute()
            
        Returns:
            The validated response in the requested type
            
        Raises:
            ValueError: If unable to get valid response after max attempts
        """
        if iters is None:
            iters = 3
        
        for _ in range(iters):
            resp = self.execute(messages, **kwargs)

            if resp.lower().strip() == 'none':
                return None

            if self._type(type) == str:
                if type == 'json':
                    resp = resp.strip()
                    try:
                        if resp[:7] == '```json':
                            resp = resp[7:-3].strip()
                        resp = json.loads(resp)
                        return resp
                    except ValueError:
                        messages.append(('assistant', resp))
                        messages.append(('user', 'Respond with a valid JSON object. Always use double quotes.'))
                        continue

                elif type == 'int':
                    try:
                        resp = resp.strip()
                        resp = int(resp)
                        return resp
                    except ValueError:
                        messages.append(('assistant', resp))
                        messages.append(('user', 'Please respond now with just the number, nothing else.'))
                        continue

                elif type == 'float':
                    try:
                        resp = resp.strip()
                        resp = float(resp)
                        return resp
                    except ValueError:
                        messages.append(('assistant', resp))
                        messages.append(('user', 'Please respond now with just the number, nothing else.'))
                        continue

                elif type == 'bool':
                    trues = ['yes', 'true', '1']
                    falses = ['no', 'false', '0']
                    resp = resp.strip('.,!? ').lower()
                    if resp in trues:
                        return True
                    elif resp in falses:
                        return False
                    else:
                        messages.append(('assistant', resp))
                        messages.append(('user', 'Please respond now with either "yes" or "no".'))
                        continue

                elif type == 'list':
                    resp = resp.strip().split('\n')
                    resp = [r.strip("+- ") for r in resp]
                    resp = [r for r in resp if r]
                    return resp

                elif type == 'str':
                    return resp
                
                else:
                    raise ValueError('Invalid type. Choose one of: json, int, float, str')
                
            elif isinstance(type, FunctionType):
                print(resp)
                try:
                    return type(resp)
                except ValueError as e:
                    message = e.args[0]
                    print('FAIL', resp, message)
                    messages.append(('assistant', resp))
                    messages.append(('user', message))
                    continue

        raise ValueError('Could not get a valid response')
    
class Result:
    """
    Container for LLM response results.
    
    Attributes:
        content (str): The response content
        prompt_tokens (int): Number of tokens in the prompt
        completion_tokens (int): Number of tokens in the completion
    """
    
    def __init__(self, content, prompt_tokens, completion_tokens):
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

class Mistral(Model):
    """
    Mistral AI model implementation.
    
    Costs are in cents per 1M tokens (input/output).
    """
    
    costs = {
        "open-mistral-7b": (25, 25),
        "open-mixtral-8x7b": (70, 70),
        "mistral-small-2402": (100, 300),
        "mistral-large-2402": (400, 1200),
    }

    def __init__(self, model="open-mistral-7b"):
        """
        Initialize a Mistral model.
        
        Args:
            model: The model name to use
        """
        self.model = model
        self.name = f"mistral<{model}>"
        self.cost = Mistral.costs[model]

    def execute(self, messages):
        """
        Execute the Mistral model.
        
        Args:
            messages: List of messages to process
            
        Returns:
            Result: The model's response with token usage information
        """
        messages = transform_messages(messages)
        messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]

        client = MistralClient()

        chat_completion = client.chat(
            messages=messages,
            model=self.model,
            max_tokens=None,
            temperature=TEMPERATURE,
        )

        return Result(
            content=chat_completion.choices[0].message.content,
            prompt_tokens=chat_completion.usage.prompt_tokens,
            completion_tokens=chat_completion.usage.completion_tokens
        )

class OpenAIChat(Model):
    """
    OpenAI model implementation.
    
    Costs are in cents per 1M tokens (input/output).
    """
    
    costs = {
        "gpt-4o": (500, 1500),
        "gpt-4-turbo": (1000, 3000),
        "gpt-4": (3000, 6000),
        "gpt-3.5-turbo": (50, 150),
        "gpt-3.5-turbo-instruct": (50, 150),
        "gpt-4o-mini": (60, 15)
    }

    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize an OpenAI model.
        
        Args:
            model: The model name to use
        """
        self.model = model
        self.name = f"openai<{model}>"
        self.cost = OpenAIChat.costs[model]

    def execute(self, messages):
        """
        Execute the OpenAI model.
        
        Args:
            messages: List of messages to process
            
        Returns:
            Result: The model's response with token usage information
        """
        client = OpenAI()
        messages = transform_messages(messages)

        result = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=None,
            temperature=TEMPERATURE,
        )

        return Result(
            content=result.choices[0].message.content,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens
        )

class AnthropicChat(Model):
    """
    Anthropic Claude model implementation.
    
    Costs are in cents per 1M tokens (input/output).
    """
    
    costs = {
        "claude-3-haiku-20240307": (25, 125),
        "claude-3-sonnet-20240229": (300, 1500),
        "claude-3-opus-20240229": (1500, 7500),
    }

    def __init__(self, model="claude-3-haiku-20240307"):
        """
        Initialize an Anthropic model.
        
        Args:
            model: The model name to use
        """
        self.model = model
        self.name = f"anthropic<{model}>"
        self.cost = AnthropicChat.costs[model]

    def execute(self, messages):
        """
        Execute the Anthropic model.
        
        Args:
            messages: List of messages to process
            
        Returns:
            Result: The model's response with token usage information
        """
        messages = transform_messages(messages)

        # Combine consecutive messages from the same role
        new_messages = []
        for k, g in groupby(messages, key=lambda x: x["role"]):
            new_messages.append({
                "role": k,
                "content": "\n\n".join(x["content"] for x in g)
            })

        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model=self.model,
            messages=new_messages,
            max_tokens=4096,
            temperature=TEMPERATURE,
        )

        content = "\n\n".join(x.text for x in response.content)
        return Result(
            content=content,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens
        )

class Groq(Model):

    costs = { # cents per 1M tokens, input/output
        "gemma-7b-it": (10, 10),
        "mixtral-8x7b-32768": (27, 27),
        "llama3-70b-8192": (64, 80),
        "llama3-8b-8192": (10, 10)
    }

    def __init__(self, model="llama3-8b-8192"):
        self.model = model
        self.name = f"groq<{model}>"
        self.cost = Groq.costs[model]

    def execute(self, messages):
        from groq import Groq
        import os

        messages = transform_messages(messages)

        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_tokens=None,
            temperature=TEMPERATURE,
        )

        return Result(
            content=chat_completion.choices[0].message.content,
            prompt_tokens=chat_completion.usage.prompt_tokens,
            completion_tokens=chat_completion.usage.completion_tokens
        )
    
models = [
    X(x) 
    for X in [Groq, OpenAIChat, AnthropicChat, Mistral]
    for x in X.costs.keys()
]

models = sorted(models, key=lambda x: sum(x.cost))

name_to_model = {m.name: m for m in models}

if __name__ == "__main__":
    for m in models:
        print(m, m.cost)