"""
Tool management and OpenAI Assistant integration for ChatLib.

This module provides:
- Parameter schema definition for tools
- OpenAI Assistant integration with tool support
- Thread management for conversations
- Tool execution and result handling
"""

import time
import json
from datetime import datetime

class Parameters:
    """
    Schema definition for tool parameters.
    
    This class helps define the expected parameters for tools in a format
    compatible with OpenAI's function calling API.
    """
    
    def __init__(self, type="object"):
        """
        Initialize a parameter schema.
        
        Args:
            type: The type of the parameter object (default: "object")
        """
        self.properties = {}
        self.required = []
        self.type = type

    def add_property(self, name, type, description=None, required=False):
        """
        Add a property to the parameter schema.
        
        Args:
            name: Property name
            type: Property type
            description: Property description
            required: Whether the property is required
            
        Returns:
            self: The Parameters instance for method chaining
        """
        self.properties[name] = {
            "type": type,
            "description": description,
        }

        if required:
            self.required.append(name)

        return self

    def json(self):
        """
        Convert the schema to JSON format.
        
        Returns:
            dict: The schema in JSON format
        """
        return {
            "type": self.type,
            "properties": self.properties,
            "required": self.required,
        }

class Thread:
    """
    Manages a conversation thread with an OpenAI Assistant.
    
    This class provides methods for:
    - Creating and managing conversation threads
    - Sending messages
    - Creating assistants
    - Retrieving message history
    """
    
    def __init__(self, client):
        """
        Initialize a new thread.
        
        Args:
            client: OpenAI client instance
        """
        self.client = client
        self.thread = self.client.beta.threads.create()
        self.id = self.thread.id
        self.run_id = None
        self.messages = []

    def create_assistant(self, name, instructions, model="gpt-3.5-turbo"):
        """
        Create a new assistant in this thread.
        
        Args:
            name: Assistant name
            instructions: Assistant instructions
            model: Model to use
            
        Returns:
            Assistant: The created assistant instance
        """
        return Assistant(
            thread=self,
            name=name,
            instructions=instructions,
            model=model,
        )

    def user_say(self, query):
        """
        Send a user message.
        
        Args:
            query: The message content
            
        Returns:
            Message: The created message
        """
        return self.say("user", query)
    
    def system_say(self, query):
        """
        Send a system message.
        
        Args:
            query: The message content
            
        Returns:
            Message: The created message
        """
        return self.say("system", query)
    
    def say(self, role, query):
        """
        Send a message with the specified role.
        
        Args:
            role: Message role ("user" or "system")
            query: The message content
            
        Returns:
            Message: The created message
        """
        self.messages.append(
            {
                "role": role,
                "content": query,
                "when": datetime.now(),
            }
        )

        return self.client.beta.threads.messages.create(
            thread_id=self.id,
            role=role,
            content=query,
        )
    
    def print_last_message(self):
        """Print the most recent message in the thread."""
        messages = self.client.beta.threads.messages.list(thread_id=self.id)
        for msg in messages:
            print(f"{msg.role}: {msg.content[0].text.value}")
            break

class Assistant:
    """
    Manages an OpenAI Assistant with tool support.
    
    This class provides:
    - Assistant creation and initialization
    - Tool registration and execution
    - Message handling
    - Run management
    """
    
    def __init__(self, thread, name, instructions, model="gpt-3.5-turbo"):
        """
        Initialize a new assistant.
        
        Args:
            thread: Thread instance to use
            name: Assistant name
            instructions: Assistant instructions
            model: Model to use
        """
        self.name = name
        self.instructions = instructions
        self.client = thread.client
        
        self.tools = []
        self.tool_fns = {}

        self.model = model
        self.thread = thread

        self.assistant = None
        self.id = None

    def initialize(self):
        """
        Create the assistant in OpenAI.
        
        This must be called before using the assistant.
        """
        self.assistant = self.client.beta.assistants.create(
            name=self.name,
            instructions=self.instructions,
            tools=self._dump_tools(),
            model=self.model,
        )
        self.id = self.assistant.id

    def add_tool(self, *args, name=None, description=None, parameters=None, function=None, **kwargs):
        """
        Add a tool to the assistant.
        
        Args:
            *args: Either a single tool dictionary or tool components
            name: Tool name (if not provided in args)
            description: Tool description (if not provided in args)
            parameters: Tool parameters (if not provided in args)
            function: Function to execute when tool is called
            **kwargs: Additional tool attributes
        """
        if len(args) == 1:
            tool = args[0]
        else:
            tool = {
                "name": name,
                "description": description,
                "parameters": parameters,
            }

        self.tools.append(tool)
        self.tool_fns[name] = function

    def _dump_tools(self):
        """
        Convert tools to OpenAI API format.
        
        Returns:
            list: List of tools in OpenAI format
        """
        return [
            {
                "type": "function",
                "function": tool,
            }
            for tool in self.tools
        ]

    def complete(self):
        """
        Complete the current conversation.
        
        This will:
        1. Create a run
        2. Wait for completion
        3. Handle any required tool calls
        4. Return the final response
        
        Returns:
            str: The assistant's final response
        """
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
        )

        run = self._wait_for_run_completion(run.id)

        while run.status == 'requires_action':
            run = self._submit_tool_outputs(run.id, run.required_action.submit_tool_outputs.tool_calls)
            run = self._wait_for_run_completion(run.id)

        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        for msg in messages:
            self.thread.messages.append(
                {
                    "role": msg.role,
                    "content": msg.content[0].text.value,
                    "when": datetime.now(),
                }
            )
            
            return msg.content[0].text.value

    def _wait_for_run_completion(self, run_id):
        """
        Wait for a run to complete.
        
        Args:
            run_id: The run ID to wait for
            
        Returns:
            Run: The completed run
        """
        while True:
            time.sleep(1)
            run = self.client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=run_id)
            if run.status in ['completed', 'failed', 'requires_action']:
                return run
            
    def _submit_tool_outputs(self, run_id, tools_to_call):
        """
        Submit tool outputs for a run.
        
        Args:
            run_id: The run ID
            tools_to_call: List of tools that need outputs
            
        Returns:
            Run: The updated run
        """
        tool_output_array = []
        for tool in tools_to_call:
            output = None
            tool_call_id = tool.id
            function_name = tool.function.name
            function_args = tool.function.arguments

            print('Running tool', function_name, 'with args', function_args)
            fn = self.tool_fns[function_name]
            output = fn(**json.loads(function_args))

            if output:
                tool_output_array.append({"tool_call_id": tool_call_id, "output": output})

        return self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread.id,
            run_id=run_id,
            tool_outputs=tool_output_array
        )