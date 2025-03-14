"""
Tests for ChatLib functionality.
"""

import pytest
from unittest.mock import Mock, patch
from chatlib.utils import transform_messages, validate_response, ValidError
from chatlib import Send, SendValid

def test_transform_messages_string():
    """Test transforming a string message."""
    result = transform_messages("Hello")
    assert result == [{"role": "user", "content": "Hello"}]

def test_transform_messages_tuples():
    """Test transforming tuple messages."""
    messages = [
        ("user", "Hello"),
        ("assistant", "Hi there"),
    ]
    result = transform_messages(messages)
    assert result == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

def test_transform_messages_dicts():
    """Test that dict messages are returned as-is."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = transform_messages(messages)
    assert result == messages

def test_validate_response_json():
    """Test JSON response validation."""
    # Test valid JSON
    result = validate_response('{"key": "value"}', 'json')
    assert result == {"key": "value"}
    
    # Test JSON with code block
    result = validate_response('```json\n{"key": "value"}\n```', 'json')
    assert result == {"key": "value"}
    
    # Test invalid JSON
    with pytest.raises(ValidError):
        validate_response("not json", 'json')

def test_validate_response_yaml():
    """Test YAML response validation."""
    # Test valid YAML
    result = validate_response('key: value', 'yaml')
    assert result == {"key": "value"}
    
    # Test YAML with code block
    result = validate_response('```yaml\nkey: value\n```', 'yaml')
    assert result == {"key": "value"}
    
    # Test invalid YAML
    with pytest.raises(ValidError):
        validate_response("{invalid: yaml:", 'yaml')

def test_validate_response_numeric():
    """Test numeric response validation."""
    # Test integer
    assert validate_response("42", 'int') == 42
    with pytest.raises(ValidError):
        validate_response("not a number", 'int')
    
    # Test float
    assert validate_response("3.14", 'float') == 3.14
    with pytest.raises(ValidError):
        validate_response("not a float", 'float')

def test_validate_response_bool():
    """Test boolean response validation."""
    # Test true values
    assert validate_response("yes", 'bool') is True
    assert validate_response("true", 'bool') is True
    assert validate_response("1", 'bool') is True
    
    # Test false values
    assert validate_response("no", 'bool') is False
    assert validate_response("false", 'bool') is False
    assert validate_response("0", 'bool') is False
    
    # Test invalid
    with pytest.raises(ValidError):
        validate_response("maybe", 'bool')

def test_validate_response_list():
    """Test list response validation."""
    input_text = """
    - First item
    + Second item
    - Third item
    """
    result = validate_response(input_text, 'list')
    assert result == ["First item", "Second item", "Third item"]

def test_validate_response_custom():
    """Test custom validation function."""
    def validate_even(text):
        num = int(text)
        if num % 2 != 0:
            raise ValueError("Number must be even")
        return num
    
    assert validate_response("42", validate_even) == 42
    with pytest.raises(ValidError):
        validate_response("43", validate_even)

@patch('chatlib.llm.db')
def test_send(mock_db):
    """Test Send function with mocked dependencies."""
    # Mock OpenAI response
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
    
    mock_db.openai.chat.completions.create.return_value = mock_response
    
    result = Send("Test message")
    
    # Verify response
    assert result == "Test response"
    
    # Verify OpenAI was called correctly
    mock_db.openai.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test message"}],
        temperature=0.2,
    )
    
    # Verify cost tracking
    mock_db.mongo['LLM_calls'].insert_one.assert_called_once()
    call_args = mock_db.mongo['LLM_calls'].insert_one.call_args[0][0]
    assert call_args['input'] == 10
    assert call_args['output'] == 20
    assert call_args['model'] == "gpt-4o-mini"

@patch('chatlib.llm.Send')
def test_send_valid(mock_send):
    """Test SendValid function with mocked Send."""
    # Test successful validation
    mock_send.return_value = '{"key": "value"}'
    result = SendValid("Test message", type='json')
    assert result == {"key": "value"}
    assert mock_send.call_count == 1
    
    # Reset mock for retry test
    mock_send.reset_mock()
    mock_send.side_effect = ['invalid json', '{"key": "value"}']
    result = SendValid("Test message", type='json')
    assert result == {"key": "value"}
    assert mock_send.call_count == 2
    
    # Test max retries exceeded
    mock_send.reset_mock()
    mock_send.side_effect = ['invalid json'] * 3
    with pytest.raises(ValueError) as exc:
        SendValid("Test message", type='json')
    assert str(exc.value) == "Could not get a valid response after maximum attempts"
    assert mock_send.call_count == 3 