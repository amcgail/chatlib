from .common import *

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException

# load vector db
logger.info("Vector DB: %s", os.getenv("PINECONE_INDEX_NAME"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


def create():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536, 
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        ) 
    )

    index = pc.Index(PINECONE_INDEX_NAME)
    return index

try:
    index = pc.Index(PINECONE_INDEX_NAME)

except NotFoundException:
    index = create()

def recreate():
    try:
        pc.delete_index(PINECONE_INDEX_NAME)
    except NotFoundException:
        pass

    return create()

"""
I want to be able to search and get MongoDB objects in return.
I don't think I need to cache all search embeddings in the database...
"""

def clean_text(text):
    import re 

    # first, remove any punctuation at the end
    text = re.sub(r'[^\w\s]$', '', text)

    # strip, and lower
    text = text.strip().lower()

    return text

def _embed(text):
    text = clean_text(text)
    emb = openai_client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
    return emb

class Embedding:

    def __init__(self, text=None, vector=None):
        self.vector = None

        if text:
            self.text = text
            self.vector = _embed(text)

        elif vector:
            self.vector = vector

    def search(self, pinecone_namespace, k=10, filter=None, cutoff=0.4):
        if not self.vector:
            raise ValueError("No vector to search with.")

        results = index.query(
            namespace=pinecone_namespace,
            vector=self.vector,
            filter=filter,
            top_k=k
        )

        # get them from the embeddings database
        objs = [
            (r.id, mongo.embeddings.find_one({"_id": ObjectId(r.id) }))
            for r in results.matches
            if r.score > cutoff
        ]
        
        lost_ids = [x[0] for x in objs if not x[1]]
        objs = [x for x in objs if x[1]] # filter for data loss

        objs = [
            (rid, mongo[x['table']].find_one({"_id": ObjectId(x['obj_id'])}))
            for rid, x in objs
        ]

        lost_ids += [x[0] for x in objs if not x[1]]
        objs = [x[1] for x in objs if x[1]] # filter for data loss

        if len(lost_ids):
            logger.warning(f"Lost {len(lost_ids)} objects in search. Scrubbing from pinecone.")
            index.delete(lost_ids, namespace=pinecone_namespace)

        return objs
    
    def store(self, table, pinecone_namespace=None, id=None, info=None, metadata=None):
        if pinecone_namespace is None:
            pinecone_namespace = table

        if info is not None:
            _find = mongo[table].find_one(info)
            if not _find:
                _ins = mongo[table].insert_one(info)
                id = _ins.inserted_id
            else:
                id = _find['_id']

        if metadata is None:
            metadata = {}

        obj = {
            "vector": self.vector,
            "table": table,
            "obj_id": id,
        }

        _ins = mongo.embeddings.insert_one(obj)
        _id = _ins.inserted_id

        # now store in the vector db
        vectors = [
            {
                "id": str(_id), 
                "values": self.vector,
                "metadata": metadata
            }
        ]

        index.upsert(
            namespace=pinecone_namespace,
            vectors=vectors
        )

        return _id