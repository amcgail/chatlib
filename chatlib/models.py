from dotenv import load_dotenv

# load from this directory
load_dotenv()

TEMPERATURE = 0.25

def transform_message(m):
    if type(m) in {tuple, list}:
        assert len(m) == 2
        return {"role": m[0], "content": m[1]}
    return m

def transform_messages(ms):
    if not len(ms): return []
    return [transform_message(m) for m in ms]


class Model:
    def __repr__(self) -> str:
        return self.name if hasattr(self, "name") else super().__repr__()
    
    def execute(self, messages):
        raise NotImplementedError
    
    _type = type
    def valid(self, messages, type='json', iters=None, **kwargs):
        from types import FunctionType
        import json

        if iters is None:
            iters = 3
        
        for iters in range(iters):
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
    def __init__(self, content, prompt_tokens, completion_tokens):
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

class Mistral(Model):
    costs = {  # cents per 1M tokens, input/output
        "open-mistral-7b": (25, 25),
        "open-mixtral-8x7b": (70, 70),
        "mistral-small-2402": (100, 300),
        "mistral-large-2402": (400, 1200),
    }

    def __init__(self, model="open-mistral-7b"):
        self.model = model
        self.name = f"mistral<{model}>"
        self.cost = Mistral.costs[model]

    def execute(self, messages):

        from mistralai.client import MistralClient
        from mistralai.models.chat_completion import ChatMessage

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
    costs = { # cents per 1M tokens, input/output
        "gpt-4o": (500, 1500),
        "gpt-4-turbo": (1000, 3000),
        "gpt-4": (3000, 6000),
        "gpt-3.5-turbo": (50, 150),
        "gpt-3.5-turbo-instruct": (50, 150),
        "gpt-4o-mini": (60, 15)
    }

    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.name = f"openai<{model}>"
        self.cost = OpenAIChat.costs[model]

    def execute(self, messages):
        from openai import OpenAI
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
    costs = {  # cents per 1M tokens, input/output
        "claude-3-haiku-20240307": (25, 125),
        "claude-3-sonnet-20240229": (300, 1500),
        "claude-3-opus-20240229": (1500, 7500),
    }

    def __init__(self, model="claude-3-haiku-20240307"):
        self.model = model
        self.name = f"anthropic<{model}>"
        self.cost = AnthropicChat.costs[model]

    def execute(self, messages):
        import os
        from anthropic import Anthropic

        messages = transform_messages(messages)

        # for Anthropic, you can't have multiple user: messages in a row
        from itertools import groupby
        new_messages = []
        for k, g in groupby(messages, key=lambda x: x["role"]):
            new_messages.append({
                "role": k,
                "content": "\n\n".join(x["content"] for x in g)
            })

        # Initialize the client using an API key from environment variables
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        # Create and send the message request
        response = client.messages.create(
            model=self.model,
            messages=new_messages,
            max_tokens=4096,
            temperature=TEMPERATURE,
        )

        # Return the content of the response
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