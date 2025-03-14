"""
Pytest configuration and fixtures for ChatLib tests.
"""

import os
import pytest
from unittest.mock import Mock
from pymongo import MongoClient

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """
    Set up a test database for the test session.
    
    This fixture:
    1. Sets MongoDB environment variables to point to test database
    2. Creates fresh collections for each test
    3. Cleans up the test database after tests complete
    """
    # Use default MongoDB URI if not specified
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
    
    # Always use a test database
    os.environ['MONGO_URI'] = mongo_uri
    os.environ['MONGO_DB'] = 'chatlib_test'
    
    # Create client and drop test database to start fresh
    client = MongoClient(mongo_uri)
    client.drop_database('chatlib_test')
    
    yield
    
    # Clean up after all tests
    client.drop_database('chatlib_test')
    client.close()

@pytest.fixture(autouse=True)
def clean_collections():
    """Clean all collections before each test."""
    client = MongoClient(os.environ['MONGO_URI'])
    db = client[os.environ['MONGO_DB']]
    
    # Clear all collections
    for collection in ['actors', 'messages', 'convos', 'LLM_calls']:
        db[collection].delete_many({})
    
    yield
    
    client.close()

@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    response = Mock()
    response.choices = [Mock(message=Mock(content="Test response"))]
    response.usage = Mock(prompt_tokens=10, completion_tokens=20)
    return response

@pytest.fixture
def sample_messages():
    """Sample message formats for testing."""
    return {
        'string': "Hello",
        'tuples': [
            ("user", "Hello"),
            ("assistant", "Hi there"),
        ],
        'dicts': [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
    } 