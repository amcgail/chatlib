# ChatLib

A Python library for building chat applications with Large Language Models (LLMs).

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

# Pinecone Configuration (if using vector storage)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
```

### MongoDB Usage

ChatLib uses MongoDB to store:
- Conversation history
- Message metadata
- Tool execution results
- Vector embeddings (if enabled)

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

```python
from chatlib import Conversation, LLM, Message, Role

# Initialize an LLM
llm = LLM()

# Create a conversation
conversation = Conversation(llm)

# Add a user message
conversation.add_message(Message(role=Role.USER, content="Hello!"))

# Get the AI's response
response = conversation.get_response()
print(response.content)
```

## Features

- Conversation management with context
- Support for multiple LLM providers
- Vector storage for semantic search
- Tool integration for enhanced capabilities
- Message history and context management
- Persistent storage with MongoDB
- Environment-based configuration

## Documentation

For detailed documentation and examples, please visit the [documentation](https://github.com/amcgail/chatlib/wiki).

## License

This project is licensed under the MIT License - see the LICENSE file for details. 