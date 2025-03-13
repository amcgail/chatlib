"""
ChatLib - A library for building chat applications with LLMs.
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in ['Send', 'SendValid', 'transform_messages', 'ValidError']:
        from .llm import Send, SendValid, transform_messages, ValidError
        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['Send', 'SendValid', 'transform_messages', 'ValidError']
