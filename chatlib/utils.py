"""
Utility functions for message handling and validation in ChatLib.
"""

import json
import yaml
import re
from types import FunctionType

def transform_messages(messages):
    """
    Transform messages into the standard format for LLM APIs.
    
    Args:
        messages: Either a string, list of tuples, or list of dictionaries
        
    Returns:
        list: List of messages in dictionary format with 'role' and 'content' keys
    """
    if isinstance(messages, str):
        return [
            {"role": "user", "content": messages}
        ]
        
    # Convert list of tuples to list of dictionaries
    if isinstance(messages[0], tuple):
        return [
            {"role": role, "content": content}
            for role, content in messages
        ]
    
    return messages

class ValidError(Exception):
    """Exception raised when validation fails."""
    pass

def validate_response(resp, type):
    """
    Validate and convert an LLM response to the specified type.
    
    Args:
        resp: The response string from the LLM
        type: The target type ('json', 'yaml', 'int', 'float', 'bool', 'list', 'str' or custom function)
        
    Returns:
        The validated and converted response
        
    Raises:
        ValidError: If validation fails
        ValueError: If type is invalid
    """
    if resp.lower().strip('. ') == 'none':
        return None

    if isinstance(type, str):
        resp = resp.strip()
        
        if type == 'json':
            try:
                if resp[:7] == '```json':
                    resp = resp[7:-3].strip()
                return json.loads(resp)
            except ValueError:
                raise ValidError('Invalid JSON response')

        elif type == 'yaml':
            try:
                if resp[:7] == '```yaml':
                    resp = resp[7:-3].strip()
                return yaml.safe_load(resp)
            except yaml.YAMLError:
                raise ValidError('Invalid YAML response')

        elif type == 'int':
            try:
                return int(resp)
            except ValueError:
                raise ValidError('Invalid integer response')

        elif type == 'float':
            try:
                return float(resp)
            except ValueError:
                raise ValidError('Invalid float response')

        elif type == 'bool':
            trues = ['yes', 'true', '1']
            falses = ['no', 'false', '0']
            resp = resp.strip('.,!? ').lower()
            if resp in trues:
                return True
            elif resp in falses:
                return False
            else:
                raise ValidError('Invalid boolean response')

        elif type == 'list':
            resp = re.split(r'\n+', resp)
            resp = [r.strip("+- ") for r in resp]
            return [r for r in resp if r]

        elif type == 'str':
            return resp
        
        else:
            raise ValueError('Invalid type. Choose one of: json, yaml, int, float, list, bool, str')
        
    elif isinstance(type, FunctionType):
        try:
            return type(resp)
        except ValueError as e:
            raise ValidError(str(e))

    raise ValueError('Invalid type specification') 