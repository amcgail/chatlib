"""
Vector storage and search functionality for ChatLib.

This module provides:
- Pinecone vector database integration
- Text embedding generation
- Vector similarity search
- MongoDB integration for object storage
"""

from .common import *
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException
import re
from bson import ObjectId

# Initialize Pinecone client
logger.info("Vector DB: %s", os.getenv("PINECONE_INDEX_NAME"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

def create():
    """
    Create a new Pinecone index.
    
    Returns:
        Index: The created Pinecone index
    """
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536, 
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        ) 
    )
    return pc.Index(PINECONE_INDEX_NAME)

# Initialize or create index
try:
    index = pc.Index(PINECONE_INDEX_NAME)
except NotFoundException:
    index = create()

def recreate():
    """
    Recreate the Pinecone index.
    
    This will delete the existing index if it exists and create a new one.
    
    Returns:
        Index: The newly created Pinecone index
    """
    try:
        pc.delete_index(PINECONE_INDEX_NAME)
    except NotFoundException:
        pass
    return create()

def clean_text(text):
    """
    Clean text for embedding generation.
    
    Args:
        text: Text to clean
        
    Returns:
        str: Cleaned text
    """
    # Remove punctuation at the end
    text = re.sub(r'[^\w\s]$', '', text)
    # Strip whitespace and convert to lowercase
    return text.strip().lower()

def _embed(text):
    """
    Generate embeddings for text using OpenAI's API.
    
    Args:
        text: Text to embed
        
    Returns:
        list: The text embedding vector
    """
    text = clean_text(text)
    emb = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    ).data[0].embedding
    return emb

class Embedding:
    """
    Manages text embeddings and vector search operations.
    
    This class provides:
    - Text embedding generation
    - Vector similarity search
    - Embedding storage in Pinecone and MongoDB
    """
    
    def __init__(self, text=None, vector=None):
        """
        Initialize an embedding.
        
        Args:
            text: Text to generate embeddings for
            vector: Pre-computed embedding vector
        """
        self.vector = None

        if text:
            self.text = text
            self.vector = _embed(text)
        elif vector:
            self.vector = vector

    def search(self, pinecone_namespace, k=10, filter=None, cutoff=0.4):
        """
        Search for similar objects using vector similarity.
        
        Args:
            pinecone_namespace: Pinecone namespace to search in
            k: Number of results to return
            filter: Additional filter criteria
            cutoff: Minimum similarity score threshold
            
        Returns:
            list: List of matching objects from MongoDB
            
        Raises:
            ValueError: If no vector is available for search
        """
        if not self.vector:
            raise ValueError("No vector to search with.")

        results = index.query(
            namespace=pinecone_namespace,
            vector=self.vector,
            filter=filter,
            top_k=k
        )

        # Get objects from MongoDB
        objs = [
            (r.id, mongo.embeddings.find_one({"_id": ObjectId(r.id)}))
            for r in results.matches
            if r.score > cutoff
        ]
        
        # Track and handle lost objects
        lost_ids = [x[0] for x in objs if not x[1]]
        objs = [x for x in objs if x[1]]  # Filter for data loss

        # Get the actual objects from their respective tables
        objs = [
            (rid, mongo[x['table']].find_one({"_id": ObjectId(x['obj_id'])}))
            for rid, x in objs
        ]

        lost_ids += [x[0] for x in objs if not x[1]]
        objs = [x[1] for x in objs if x[1]]  # Filter for data loss

        # Clean up lost objects from Pinecone
        if lost_ids:
            logger.warning(f"Lost {len(lost_ids)} objects in search. Scrubbing from pinecone.")
            index.delete(lost_ids, namespace=pinecone_namespace)

        return objs
    
    def store(self, table, pinecone_namespace=None, id=None, info=None, metadata=None):
        """
        Store an embedding in both Pinecone and MongoDB.
        
        Args:
            table: MongoDB table name
            pinecone_namespace: Pinecone namespace to store in
            id: Optional MongoDB document ID
            info: Optional document info to store in MongoDB
            metadata: Optional metadata to store with the embedding
            
        Returns:
            ObjectId: The MongoDB ID of the stored embedding
        """
        if pinecone_namespace is None:
            pinecone_namespace = table

        # Create or find the MongoDB document
        if info is not None:
            _find = mongo[table].find_one(info)
            if not _find:
                _ins = mongo[table].insert_one(info)
                id = _ins.inserted_id
            else:
                id = _find['_id']

        if metadata is None:
            metadata = {}

        # Store embedding reference in MongoDB
        obj = {
            "vector": self.vector,
            "table": table,
            "obj_id": id,
        }
        _ins = mongo.embeddings.insert_one(obj)
        _id = _ins.inserted_id

        # Store vector in Pinecone
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