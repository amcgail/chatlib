import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Prevent actual imports during documentation generation
class MockModule:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return MockModule()

    def __call__(self, *args, **kwargs):
        return MockModule()

    def __getitem__(self, key):
        return MockModule()

# Mock environment variables
os.environ['MONGO_URI'] = 'mongodb://localhost:27017'
os.environ['MONGO_DB'] = 'chatlib'
os.environ['OPENAI_API_KEY'] = 'mock-key'
os.environ['PINECONE_API_KEY'] = 'mock-key'
os.environ['PINECONE_INDEX_NAME'] = 'mock-index'

# Replace external modules with our mock
for mod_name in ['pymongo', 'openai', 'pinecone', 'anthropic', 'mistralai']:
    sys.modules[mod_name] = MockModule()

project = 'ChatLib'
copyright = '2024, Alec McGail'
author = 'Alec McGail'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_typehints = 'description'
add_module_names = False 