from .common import *

from .common import openai_client, mongo
import json
import yaml
import logging

logger = logging.getLogger(__name__)

def transform_messages(messages):
    if type(messages) == str:
        return [
            {"role": "user", "content": messages}
        ]
        
    # Convert list of tuples to list of dictionaries
    if type(messages[0]) == tuple:
        return [
            {"role": role, "content": content}
            for role, content in messages
        ]
    
    return messages


def Send(messages, temperature=0.2, model="gpt-4o-mini", group=None):
    #model="gpt-3.5-turbo"
    #model="gpt-4-turbo"

    messages = transform_messages(messages)

    if not len(messages):
        return None
    
    response = openai_client.chat.completions.create(
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

    mongo['LLM_calls'].insert_one({
        'input': inp_tok,
        'output': out_tok,
        'group': group,
        'cost': total_cost,
        'model': model
    })

    return response.choices[0].message.content
    
class ValidError(Exception):
    pass

_type = type
def SendValid(ms, type='json', iters=3, **kwargs):
    from types import FunctionType

    ms = transform_messages(ms)    
    
    for iters in range(iters):
        resp = Send(ms, **kwargs)

        if resp.lower().strip('. ') == 'none':
            return None

        if _type(type) == str:
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
                raise ValueError('Invalid type. Choose one of: json, int, float, list, bool, str')
            
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