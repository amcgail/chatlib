"""
Database access module for ChatLib.

This module provides a clean interface for accessing various databases
with lazy loading to prevent initialization during documentation generation.
"""

from openai import OpenAI
from pymongo import MongoClient
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException
import os

class Database:
    """Singleton class for database access."""
    
    _instance = None
    _mongo = None
    _openai = None
    _pinecone = None
    _pinecone_index = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
        return cls._instance
    
    @property
    def mongo(self):
        """Lazy load MongoDB client."""
        if self._mongo is None:
            self._mongo = MongoClient(os.getenv('MONGO_URI'))[os.getenv('MONGO_DB')]
        return self._mongo
    
    @property
    def openai(self):
        """Lazy load OpenAI client."""
        if self._openai is None:
            self._openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        return self._openai
    
    @property
    def pinecone(self):
        """Lazy load Pinecone client."""
        if self._pinecone is None:
            self._pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        return self._pinecone
    
    @property
    def pinecone_index(self):
        """Lazy load Pinecone index."""
        if self._pinecone_index is None:
            try:
                self._pinecone_index = self.pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))
            except NotFoundException:
                self._pinecone_index = self._create_pinecone_index()
        return self._pinecone_index
    
    def _create_pinecone_index(self):
        """Create a new Pinecone index."""
        self.pinecone.create_index(
            name=os.getenv("PINECONE_INDEX_NAME"),
            dimension=1536, 
            metric="cosine", 
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            ) 
        )
        return self.pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))
    
    def recreate_pinecone_index(self):
        """Recreate the Pinecone index."""
        try:
            self.pinecone.delete_index(os.getenv("PINECONE_INDEX_NAME"))
        except NotFoundException:
            pass
        self._pinecone_index = self._create_pinecone_index()
        return self._pinecone_index

# Create singleton instance
db = Database()
