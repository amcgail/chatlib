# ChatLib

A powerful Python library for building sophisticated chat applications with Large Language Models (LLMs). ChatLib provides a comprehensive toolkit for managing conversations, integrating multiple LLM providers, implementing vector search, and managing tools and assistants.

## Features

- **Conversation Management**
  - Persistent conversation storage in MongoDB
  - Actor-based participant management
  - Message history tracking and formatting
  - Conversation slicing for partial history views
  - Cost tracking for LLM calls

- **LLM Integration**
  - Support for multiple LLM providers:
    - OpenAI (GPT-4, GPT-3.5-turbo)
    - Anthropic (Claude)
    - Mistral AI
    - Groq
  - Message validation and type conversion
  - Response formatting (JSON, YAML, int, float, bool, list, str)
  - Cost tracking per model

- **Vector Search**
  - Text embedding generation
  - Vector similarity search using Pinecone
  - MongoDB integration for object storage
  - Semantic search capabilities

- **Tool Management**
  - OpenAI Assistant integration
  - Tool parameter validation
  - Thread management
  - Tool execution handling

## Installation

You can install ChatLib directly from GitHub:

```bash
pip install git+https://github.com/amcgail/chatlib
```

## Environment Setup

ChatLib requires several environment variables to be set up. Create a `.env` file in your project root with the following variables:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key

# MongoDB Configuration
MONGO_URI=your_mongodb_connection_string
MONGO_DB=your_database_name

# Pinecone Configuration (for vector storage)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
```

### MongoDB Usage

ChatLib uses MongoDB to store:
- Conversation history and metadata
- Actor information
- Tool execution results
- Vector embeddings and references
- LLM call logs and costs

To use MongoDB with ChatLib:
1. Set up a MongoDB instance (local or cloud)
2. Create a database
3. Set the `MONGO_URI` to your MongoDB connection string
4. Set the `MONGO_DB` to your database name

Example MongoDB connection string:
```
mongodb+srv://username:password@cluster.mongodb.net/dbname?retryWrites=true&w=majority
```

## Quick Start

### Basic Conversation

```python
from chatlib import Conversation, Actor, Send

# Create a conversation
conversation = Conversation()

# Add a user message
conversation.say("user", "Hello! How can you help me today?")

# Get AI response using OpenAI
response = Send([
    ("user", "Hello! How can you help me today?")
], model="gpt-4-turbo")
```

### Using Actors

```python
from chatlib import Actor, Conversation

# Create an actor
user = Actor(can_leave=True)
user.save()

# Create a conversation with the actor
conversation = Conversation()
conversation.say(user, "Hello!")
```

### Vector Search

```python
from chatlib import Embedding

# Create an embedding
embedding = Embedding(text="Your text to embed")

# Search for similar content
results = embedding.search(
    pinecone_namespace="your_namespace",
    k=10,
    cutoff=0.4
)
```

### Tool Integration

```python
from chatlib import Thread, Assistant, Parameters

# Create a thread
thread = Thread(client)

# Create an assistant with tools
assistant = thread.create_assistant(
    name="My Assistant",
    instructions="You are a helpful assistant",
    model="gpt-4-turbo"
)

# Add a tool
assistant.add_tool(
    name="search",
    description="Search for information",
    parameters=Parameters()
        .add_property("query", "string", "Search query", required=True)
        .add_property("limit", "integer", "Maximum results", required=False),
    function=search_function
)
```

## Advanced Features

### Conversation Slicing

```python
from chatlib import ConversationSlice

# Load a slice of conversation history
conversation = ConversationSlice.load(
    conversation_id,
    message_end=10  # Load up to 10th message
)
```

### Response Validation

```python
from chatlib import SendValid

# Get a validated JSON response
response = SendValid(
    messages,
    type='json',
    iters=3
)

# Get a validated list response
response = SendValid(
    messages,
    type='list'
)
```

## Documentation

For detailed documentation and examples, please visit the [documentation](https://github.com/amcgail/chatlib/wiki).

## License

This project is licensed under the MIT License - see the LICENSE file for details. 