"""
Conversation management for ChatLib.

This module provides:
- Conversation tracking and persistence
- Actor management for conversation participants
- Message history and formatting
- Conversation slicing for partial history views
"""

from .common import *
from . import db, Send

from datetime import datetime as dt
from bson import ObjectId
from logging import getLogger

mongo = db.mongo

logger = getLogger(__name__)

class ConversationEnd(Exception):
    """Exception raised when a conversation should end."""
    pass

class Actor:
    """
    Represents a participant in a conversation.
    
    This class provides persistence and attribute management for conversation actors.
    Actors are stored in MongoDB and can be loaded/saved with their state.
    """
    
    def __init__(self, can_leave=True, **kwargs):
        """
        Initialize an Actor.
        
        Args:
            can_leave: Whether the actor can leave the conversation
            **kwargs: Additional actor attributes
        """
        self.can_leave = can_leave
        self._id = kwargs.get('_id')
        self._data = kwargs

    def __setitem__(self, name, value):
        """
        Set an attribute and persist to MongoDB.
        
        Args:
            name: Attribute name
            value: Attribute value
        """
        self._data[name] = value
        if self._id is not None:
            mongo.actors.update_one({'_id': self._id}, {'$set': {name: value}})

    def __contains__(self, name):
        """Check if an attribute exists."""
        return name in self._data

    def __getitem__(self, name):
        """
        Get an attribute value.
        
        Note: This assumes the data doesn't change while the script runs.
        """
        return self._data[name]

    def save(self):
        """
        Save the actor to MongoDB.
        
        Returns:
            ObjectId: The actor's MongoDB ID
        """
        if not self._id:
            dct = dict(self._data)
            dct['type'] = self.__class__.__name__
            insert = mongo.actors.insert_one(dct)
            self._id = insert.inserted_id
        else:
            mongo.actors.update_one({'_id': self._id}, {'$set': self._data})
        return self._id

    @classmethod
    def load(cls, _id, can_leave=True):
        """
        Load an actor from MongoDB.
        
        Args:
            _id: The actor's MongoDB ID (string or ObjectId)
            can_leave: Whether the actor can leave the conversation
            
        Returns:
            Actor: The loaded actor instance, or None if not found
        """
        if isinstance(_id, str):
            _id = ObjectId(_id)

        _find = mongo.actors.find_one({'_id': _id})
        if not _find:
            return None
        
        # Get the appropriate class based on stored type
        if 'type' in _find:
            cls = globals()[_find['type']]
        
        A = cls(can_leave=can_leave)
        A._data = _find
        A._id = _id
        return A

class Conversation:
    """
    Manages a conversation with message history and persistence.
    
    This class provides:
    - Message storage and retrieval
    - Conversation state management
    - Cost tracking
    - Message formatting
    """
    
    def __init__(self):
        """Initialize a new conversation."""
        self.m = []
        self._id = None

    def total_cost(self):
        """
        Calculate the total cost of LLM calls in this conversation.
        
        Returns:
            float: Total cost in dollars
        """
        calls = mongo.LLM_calls.find({'group': self._id})
        return sum(x['cost'] for x in calls)

    @classmethod
    def load(cls, _id):
        """
        Load a conversation from MongoDB.
        
        Args:
            _id: The conversation's MongoDB ID (string or ObjectId)
            
        Returns:
            Conversation: The loaded conversation instance, or None if not found
        """
        if isinstance(_id, str):
            _id = ObjectId(_id)

        _find = mongo.convos.find_one({'_id': _id})
        if not _find:
            return None
        
        C = cls()
        C.m = sorted(
            mongo.messages.find({'convo': _id}),
            key=lambda x: x['when_server']
        )
        C._id = _id
        return C
    
    def save(self):        
        """
        Save the conversation to MongoDB.
        
        Returns:
            str: The conversation's MongoDB ID as a string
        """
        self._id = mongo.convos.insert_one({
            'finished': False
        }).inserted_id
        return str(self._id)
    
    def __setitem__(self, name, value):
        """Update a conversation attribute in MongoDB."""
        mongo.convos.update_one({'_id': self._id}, {'$set': {name: value}})

    def __getitem__(self, name):
        """Get a conversation attribute from MongoDB."""
        _info = mongo.convos.find_one({'_id': self._id})
        return _info[name] if name in _info else None
    
    def __contains__(self, name):
        """Check if a conversation attribute exists in MongoDB."""
        return name in mongo.convos.find_one({'_id': self._id})

    def delete(self):
        """Delete the conversation and all its messages from MongoDB."""
        mongo.convos.delete_one({'_id': self._id})
        mongo.messages.delete_many({'convo': self._id})
    
    def say(self, role, message, **kwargs):
        """
        Add a message to the conversation.
        
        Args:
            role: The role of the message sender
            message: The message content
            **kwargs: Additional message attributes
            
        Returns:
            dict: The saved message object
        """
        if self._id is None:
            self.save()
            
        obj = {
            'convo': self._id,
            'role': role,
            'message': message,
            'when_server': dt.now(),
            **kwargs
        }
        
        result = mongo.messages.insert_one(obj)
        obj['_id'] = str(result.inserted_id)
        obj['convo'] = str(obj['convo'])

        self.m.append(obj)
        return obj

    def format_convo(self, start=None, end=None, numbered=False):
        """
        Format the conversation history as a string.
        
        Args:
            start: Starting message index (inclusive)
            end: Ending message index (exclusive)
            numbered: Whether to include message numbers
            
        Returns:
            str: Formatted conversation history
        """
        if start is None:
            start = 0
        if end is None:
            end = len(self.m)

        ms_to_format = self.m[start:end]

        if not ms_to_format:
            return '(no conversation yet)'
        
        if len(ms_to_format) <= 1:
            hist = '(start of conversation)'
        else:
            if numbered:
                hist = '\n'.join([f'{mi+1:d} = {x["role"]}: {x["message"]}' for mi, x in enumerate(ms_to_format[:-1])])
            else:
                hist = '\n'.join([f'{x["role"]}: {x["message"]}' for x in ms_to_format[:-1]])

        lastm = ms_to_format[-1]

        if numbered:
            return f'{hist}\n\nMost Recent Message:\n{len(ms_to_format):d} = {lastm["role"]}: {lastm["message"]}'
        else:
            return f'{hist}\n\nMost Recent Message:\n{lastm["role"]}: {lastm["message"]}'

class ConversationSlice(Conversation):
    """
    A view of a conversation up to a specific message.
    
    This class provides functionality to load and view partial conversation history.
    """
    
    @classmethod
    def load(cls, _id, message_end=None):
        """
        Load a slice of a conversation.
        
        Args:
            _id: The conversation's MongoDB ID
            message_end: Either a message index or message ID to end the slice at
            
        Returns:
            ConversationSlice: A conversation view up to the specified message
        """
        C = super().load(_id)

        if message_end is None:
            message_end = len(C.m)
        elif isinstance(message_end, int):
            if message_end < 0:
                message_end = len(C.m) + message_end
            C.message_end = message_end
        elif isinstance(message_end, (str, ObjectId)):
            message_end = str(message_end)
            mi = [i for i, m in enumerate(C.m) if str(m['_id']) == message_end][0]
            C.message_end = mi + 1
            
        C.m = C.m[:C.message_end]
        return C

__all__ = ['Actor', 'Conversation', 'ConversationEnd', 'ConversationSlice']
