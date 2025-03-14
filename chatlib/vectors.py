"""
Vector storage and search functionality for ChatLib.

This module provides:
- Pinecone vector database integration
- Text embedding generation
- Vector similarity search
- MongoDB integration for object storage
"""

from .common import *
from .db import db
import re
from bson import ObjectId

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
    emb = db.openai.embeddings.create(
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

        results = db.pinecone_index.query(
            namespace=pinecone_namespace,
            vector=self.vector,
            filter=filter,
            top_k=k
        )

        # Get objects from MongoDB
        objs = [
            (r.id, db.mongo.embeddings.find_one({"_id": ObjectId(r.id)}))
            for r in results.matches
            if r.score > cutoff
        ]
        
        # Track and handle lost objects
        lost_ids = [x[0] for x in objs if not x[1]]
        objs = [x for x in objs if x[1]]  # Filter for data loss

        # Get the actual objects from their respective tables
        objs = [
            (rid, db.mongo[x['table']].find_one({"_id": ObjectId(x['obj_id'])}))
            for rid, x in objs
        ]

        lost_ids += [x[0] for x in objs if not x[1]]
        objs = [x[1] for x in objs if x[1]]  # Filter for data loss

        # Clean up lost objects from Pinecone
        if lost_ids:
            logger.warning(f"Lost {len(lost_ids)} objects in search. Scrubbing from pinecone.")
            db.pinecone_index.delete(lost_ids, namespace=pinecone_namespace)

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
            _find = db.mongo[table].find_one(info)
            if not _find:
                _ins = db.mongo[table].insert_one(info)
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
        _ins = db.mongo.embeddings.insert_one(obj)
        _id = _ins.inserted_id

        # Store vector in Pinecone
        vectors = [
            {
                "id": str(_id), 
                "values": self.vector,
                "metadata": metadata
            }
        ]
        db.pinecone_index.upsert(
            namespace=pinecone_namespace,
            vectors=vectors
        )

        return _id