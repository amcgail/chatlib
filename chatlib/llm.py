"""
LLM interaction utilities for ChatLib.

This module provides:
- Direct LLM interaction functions
- Response validation with type conversion
- Cost tracking and logging
"""

import logging
from .utils import transform_messages, validate_response, ValidError
from .db import db

logger = logging.getLogger(__name__)

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
    """
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
        
        try:
            return validate_response(resp, type)
        except ValidError as e:
            ms += [
                {'role':'assistant', 'content': resp},
                {'role':'user', 'content': str(e)}
            ]
            continue

    raise ValueError('Could not get a valid response after maximum attempts')