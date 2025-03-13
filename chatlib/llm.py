"""
LLM interaction utilities for ChatLib.

This module provides:
- Message transformation utilities
- Direct LLM interaction functions
- Response validation and type conversion
- Cost tracking and logging
"""

from .common import *

import json
import yaml
import logging
import re
from types import FunctionType

logger = logging.getLogger(__name__)

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


def Send(messages, temperature=0.2, model="gpt-4o-mini", group=None):
    """
    Send messages to the LLM and get a response.
    
    Args:
        messages: Messages to send to the LLM
        temperature: Temperature parameter for response generation
        model: The model to use
        group: Optional group identifier for cost tracking
        
    Returns:
        str: The LLM's response content
        
    Note:
        Costs are tracked in MongoDB under the 'LLM_calls' collection
    """
    #model="gpt-3.5-turbo"
    #model="gpt-4-turbo"

    messages = transform_messages(messages)

    if not messages:
        return None
    
    response = db.openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    inp_tok, out_tok = response.usage.prompt_tokens, response.usage.completion_tokens
    pricing = {
        'gpt-4o-mini': (0.15, 0.60),
        'gpt-4o': (2.5, 10.00),
        'gpt-4': (30, 60),
        'gpt-4-turbo': (10, 30),
    }
    total_cost = (pricing[model][0] * inp_tok + pricing[model][1] * out_tok) / 1e6

    db.mongo['LLM_calls'].insert_one({
        'input': inp_tok,
        'output': out_tok,
        'group': group,
        'cost': total_cost,
        'model': model
    })

    return response.choices[0].message.content
    
class ValidError(Exception):
    """Exception raised when validation fails."""
    pass

_type = type
def SendValid(ms, type='json', iters=3, **kwargs):
    """
    Send messages to the LLM and get a validated response.
    
    Args:
        ms: Messages to send to the LLM
        type: Expected response type ('json', 'yaml', 'int', 'float', 'bool', 'list', 'str' or custom function)
        iters: Maximum number of attempts to get valid response
        **kwargs: Additional arguments to pass to Send()
        
    Returns:
        The validated response in the requested type
        
    Raises:
        ValueError: If unable to get valid response after max attempts
    """
    ms = transform_messages(ms)    
    
    for _ in range(iters):
        resp = Send(ms, **kwargs)

        if resp.lower().strip('. ') == 'none':
            return None

        if isinstance(type, str):
            if type == 'json':
                resp = resp.strip()
                try:
                    if resp[:7] == '```json':
                        resp = resp[7:-3].strip()
                    resp = json.loads(resp)
                    return resp
                except ValueError:
                    ms += [
                        {'role':'assistant', 'content': resp},
                        {'role':'user', 'content': 'Respond with a valid JSON object. Always use double quotes.'}
                    ]
                    continue

            elif type == 'yaml':
                resp = resp.strip()
                try:
                    if resp[:7] == '```yaml':
                        resp = resp[7:-3].strip()
                    resp = yaml.safe_load(resp)
                    return resp
                except yaml.YAMLError:
                    ms += [
                        {'role':'assistant', 'content': resp},
                        {'role':'user', 'content': 'Respond with a valid YAML object.'}
                    ]
                    continue

            elif type == 'int':
                try:
                    resp = resp.strip()
                    resp = int(resp)
                    return resp
                except ValueError:
                    ms += [
                        {'role':'assistant', 'content': resp},
                        {'role':'user', 'content': 'Please respond now with just the number, nothing else.'}
                    ]
                    continue

            elif type == 'float':
                try:
                    resp = resp.strip()
                    resp = float(resp)
                    return resp
                except ValueError:
                    ms += [
                        {'role':'assistant', 'content': resp},
                        {'role':'user', 'content': 'Please respond now with just the number, nothing else.'}
                    ]
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
                    ms += [
                        {'role':'assistant', 'content': resp},
                        {'role':'user', 'content': 'Please respond with either "yes" or "no".'}
                    ]
                    continue

            elif type == 'list':
                resp = re.split(r'\n+', resp)
                resp = [r.strip("+- ") for r in resp]
                resp = [r for r in resp if r]
                return resp

            elif type == 'str':
                return resp
            
            else:
                raise ValueError('Invalid type. Choose one of: json, yaml, int, float, list, bool, str')
            
        elif isinstance(type, FunctionType):
            print(resp)
            try:
                return type(resp)
            except ValueError as e:
                message = e.args[0]
                print('FAIL', resp, message)
                ms.append(('assistant', resp))
                ms.append(('user', message))
                continue

    raise ValueError('Could not get a valid response')