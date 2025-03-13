import os
from openai import OpenAI
from bson.objectid import ObjectId
from datetime import datetime as dt
#from groq import Groq

import re
import logging
from logging import getLogger
from pathlib import Path
import json

logging.basicConfig()
logger = getLogger(__name__)

from pymongo import MongoClient



mongo = MongoClient(os.getenv('MONGO_URI'))[os.getenv('MONGO_DB')]

class MongoMapper:
    def __init__(self, _id=None, _info=None, **kwargs):
        if _id is not None:
            if type(_id) == str or type(_id) == ObjectId:

                self.id = ObjectId(_id)
                self._info = mongo[self.db_name].find_one({'_id': self.id})

            elif type(_id) == dict:

                self._info = _id

            else:
                raise ValueError("Argument must be an id, or a dictionary")
            
        elif _info is not None:
            self._info = _info
        else:
            self._info = kwargs

    @classmethod
    def find(cls, query):
        if type(query) == str:
            query = {'_id': ObjectId(query)}
        elif type(query) == ObjectId:
            query = {'_id': query}
            
        return [cls(info) for info in mongo[cls.db_name].find(query)]
    
    @classmethod
    def find_one(cls, query):
        if type(query) == str:
            query = {'_id': ObjectId(query)}
        elif type(query) == ObjectId:
            query = {'_id': query}

        info = mongo[cls.db_name].find_one(query)
        if info is None:
            return None
        return cls(info)

    def __contains__(self, key):
        return key in self._info
        
    def __getitem__(self, key):
        return self._info[key]
    
    def __setitem__(self, key, value):
        self._info[key] = value

        # update it in MongoDB if it's already saved
        if hasattr(self, 'id'):
            mongo[self.db_name].update_one(
                {'_id': self.id},
                {'$set': {key: value}}
            )
    
    def get(self, key, *args, **kwargs):
        if key in self._info:
            return self._info[key]
        
        if 'default' in kwargs:
            return kwargs['default']
        
        if len(args):
            return args[0]
        
        raise ValueError(f'Key not found in <{self.__class__}>')

    def save(self):
        if not hasattr(self, 'id'):
            self.id = mongo[self.db_name].insert_one(self._info).inserted_id
        else:
            mongo.jobs.update_one(
                {'_id': self.id},
                {'$set': self._info}
            )

        return self

    def push(self, k, v):
        if k not in self._info or type(self._info[k]) != list:
            self._info[k] = []

        if type(v) != list:
            v = [v]

        self._info[k] += v

        mongo[self.db_name].update_one(
            {'_id': self.id},
            {'$push': {k: {'$each': v}}}
        )

openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))