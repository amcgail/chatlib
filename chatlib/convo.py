from .common import *
from .llm import Send

logger = getLogger(__name__)

class ConversationEnd(Exception):
    pass

class Actor:
    def __init__(self, can_leave=True, **kwargs):
        self.can_leave = can_leave

        self._id = None if '_id' not in kwargs else kwargs['_id']
        self._data = kwargs

    def __setitem__(self, name, value):
        self._data[name] = value

        if self._id is not None:
            mongo.actors.update_one({'_id': self._id}, {'$set': {name: value}})

    def __contains__(self, name):
        return name in self._data

    def __getitem__(self, name):
        # assumes that no one edits it while this script runs... pretty good I think for now
        return self._data[name]

    def save(self):
        if not self._id:
            dct = dict(self._data)
            dct['type'] = self.__class__.__name__

            # put this actor into mongodb
            insert = mongo.actors.insert_one(dct)
            self._id = insert.inserted_id

        else:
            mongo.actors.update_one({'_id': self._id}, {'$set': self._data})

        return self._id

    @classmethod
    def load(cls, _id, can_leave=True):
        from bson import ObjectId

        if type(_id) == str:
            _id = ObjectId(_id)

        _find = mongo.actors.find_one({'_id': _id})

        if not _find:
            return None
        
        # get the appropriate class
        if 'type' in _find:
            cls = globals()[_find['type']]
        
        A = cls(can_leave=can_leave)

        # populate it
        A._data = _find
        A._id = _id

        return A

class Conversation:
    def __init__(self):
        self.m = []
        self._id = None

    def total_cost(self):
        calls = mongo.LLM_calls.find({
            'group': self._id
        })

        return sum(x['cost'] for x in calls)

    @classmethod
    def load(cls, _id):
        from bson import ObjectId

        if type(_id) == str:
            _id = ObjectId(_id)

        _find = mongo.convos.find_one({'_id': _id})

        if not _find:
            return None
        
        C = cls()

        # populate it
        C.m = sorted(
            mongo.messages.find({'convo': _id}),
            key=lambda x: x['when_server']
        )

        C._id = _id

        return C
    
    def save(self):        
        # otherwise, insert it
        self._id = mongo.convos.insert_one({
            'finished': False
        }).inserted_id

        return str(self._id)
    
    def __setitem__(self, name, value):
        # update it in mongo
        mongo.convos.update_one({'_id': self._id}, {'$set': {name: value}})

    def __getitem__(self, name):
        # get it from mongo
        _info = mongo.convos.find_one({'_id': self._id})
        return _info[name] if name in _info else None
    
    def __contains__(self, name):
        return name in mongo.convos.find_one({'_id': self._id})

    def delete(self):
        mongo.convos.delete_one({'_id': self._id})
        mongo.messages.delete_many({'convo': self._id})
    
    def say(self, role, message, **kwargs):
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
        if start is None:
            start = 0
        if end is None:
            end = len(self.m)

        ms_to_format = self.m[start:end]

        if not len(ms_to_format):
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

    @classmethod
    def load(cls, _id, message_end=None):
        C = super().load(_id)

        if message_end is None:
            message_end = len(C.m)

        elif type(message_end) == int:
            if message_end < 0:
                message_end = len(C.m) + message_end

            C.message_end = message_end

        elif type(message_end) == str or type(message_end) == ObjectId:
            message_end = str(message_end)
            mi = [i for i, m in enumerate(C.m) if str(m['_id']) == message_end][0]
            C.message_end = mi + 1
            
        # and now we slice the message
        C.m = C.m[:C.message_end]

        return C

__all__ = ['Actor', 'Conversation', 'ConversationEnd', 'ConversationSlice']
