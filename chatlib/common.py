"""
Common utilities and shared functionality for ChatLib.

This module provides core functionality including:
- MongoDB connection and data mapping
- OpenAI client initialization
- Logging configuration
"""

import os
import re
import json
import logging
from datetime import datetime as dt
from pathlib import Path
from logging import getLogger
from bson.objectid import ObjectId
from .db import db

logging.basicConfig()
logger = getLogger(__name__)

class MongoMapper:
    """
    A base class for MongoDB document mapping.
    
    This class provides a dictionary-like interface for MongoDB documents with automatic
    persistence. It handles ObjectId conversion and provides methods for common MongoDB operations.
    
    Attributes:
        id (ObjectId): The MongoDB document ID
        _info (dict): The document data
        db_name (str): The name of the MongoDB collection (must be set by subclasses)
    """
    
    def __init__(self, _id=None, _info=None, **kwargs):
        """
        Initialize a MongoMapper instance.
        
        Args:
            _id: Either a string/ObjectId or a dictionary containing document data
            _info: Dictionary containing document data
            **kwargs: Additional document fields
        """
        if _id is not None:
            if isinstance(_id, (str, ObjectId)):
                self.id = ObjectId(_id)
                self._info = db.mongo[self.db_name].find_one({'_id': self.id})
            elif isinstance(_id, dict):
                self._info = _id
            else:
                raise ValueError("Argument must be an id, or a dictionary")
        elif _info is not None:
            self._info = _info
        else:
            self._info = kwargs

    @classmethod
    def find(cls, query):
        """
        Find all documents matching the query.
        
        Args:
            query: Either a string/ObjectId or a dictionary query
            
        Returns:
            list: List of MongoMapper instances
        """
        if isinstance(query, str):
            query = {'_id': ObjectId(query)}
        elif isinstance(query, ObjectId):
            query = {'_id': query}
            
        return [cls(info) for info in db.mongo[cls.db_name].find(query)]
    
    @classmethod
    def find_one(cls, query):
        """
        Find a single document matching the query.
        
        Args:
            query: Either a string/ObjectId or a dictionary query
            
        Returns:
            MongoMapper: A single MongoMapper instance or None if not found
        """
        if isinstance(query, str):
            query = {'_id': ObjectId(query)}
        elif isinstance(query, ObjectId):
            query = {'_id': query}

        info = db.mongo[cls.db_name].find_one(query)
        if info is None:
            return None
        return cls(info)

    def __contains__(self, key):
        """Check if a key exists in the document."""
        return key in self._info
        
    def __getitem__(self, key):
        """Get a value from the document."""
        return self._info[key]
    
    def __setitem__(self, key, value):
        """
        Set a value in the document and persist to MongoDB.
        
        Args:
            key: The field name
            value: The value to set
        """
        self._info[key] = value

        if hasattr(self, 'id'):
            db.mongo[self.db_name].update_one(
                {'_id': self.id},
                {'$set': {key: value}}
            )
    
    def get(self, key, *args, **kwargs):
        """
        Get a value from the document with a default fallback.
        
        Args:
            key: The field name
            *args: Positional arguments for default value
            **kwargs: Keyword arguments, including 'default' for fallback value
            
        Returns:
            The value if found, or the default value if specified
            
        Raises:
            ValueError: If key not found and no default specified
        """
        if key in self._info:
            return self._info[key]
        
        if 'default' in kwargs:
            return kwargs['default']
        
        if args:
            return args[0]
        
        raise ValueError(f'Key not found in <{self.__class__}>')

    def save(self):
        """
        Save the document to MongoDB.
        
        Returns:
            self: The MongoMapper instance
        """
        if not hasattr(self, 'id'):
            self.id = db.mongo[self.db_name].insert_one(self._info).inserted_id
        else:
            db.mongo[self.db_name].update_one(
                {'_id': self.id},
                {'$set': self._info}
            )
        return self

    def push(self, k, v):
        """
        Push values to an array field in MongoDB.
        
        Args:
            k: The field name
            v: The value(s) to push (can be single value or list)
        """
        if k not in self._info or not isinstance(self._info[k], list):
            self._info[k] = []

        if not isinstance(v, list):
            v = [v]

        self._info[k] += v

        db.mongo[self.db_name].update_one(
            {'_id': self.id},
            {'$push': {k: {'$each': v}}}
        )