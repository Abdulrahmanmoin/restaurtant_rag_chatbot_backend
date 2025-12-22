"""
MongoDB Vector Store Module for RAG Integration
This module handles embedding generation and vector search using MongoDB Atlas.
"""

import os
import re
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class VectorStore:
    """Handles vector embeddings and similarity search with MongoDB Atlas."""
    
    def __init__(self):
        """Initialize the vector store with MongoDB connection and embedding model."""
        self.mongodb_uri = os.getenv("MONGODB_URI")
        self.db_name = os.getenv("MONGODB_DB_NAME", "restaurant_chatbot")
        self.collection_name = os.getenv("MONGODB_COLLECTION_NAME", "kb_embeddings")
        
        if not self.mongodb_uri:
            raise ValueError("MONGODB_URI not found in environment variables. Please check your .env file.")
        
        # Initialize MongoDB client
        import certifi
        self.client = MongoClient(self.mongodb_uri, tls=True, tlsCAFile=certifi.where())
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        
        # Initialize the embedding model (all-MiniLM-L6-v2 is fast and efficient)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
        print("Embedding model loaded successfully!")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for a given text."""
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def create_vector_search_index(self):
        """Create a vector search index on the collection."""
        try:
            # Check if index already exists
            existing_indexes = list(self.collection.list_search_indexes())
            index_names = [idx.get('name') for idx in existing_indexes]
            
            if 'vector_index' in index_names:
                print("Vector search index already exists.")
                return
            
            # Create vector search index
            search_index_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": self.embedding_dimension,
                            "similarity": "cosine"
                        },
                        {
                            "type": "filter",
                            "path": "category"
                        }
                    ]
                },
                name="vector_index",
                type="vectorSearch"
            )
            
            self.collection.create_search_index(search_index_model)
            print("Vector search index created successfully!")
            print("Note: Index may take a few minutes to become active on MongoDB Atlas.")
            
        except Exception as e:
            print(f"Error creating vector search index: {e}")
            print("You may need to create the index manually in MongoDB Atlas.")
            print("See: https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/")
    
    def parse_rag_file(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse the RAG Markdown file into document chunks.
        Supports standard Markdown headers (# and ##).
        """
        documents = []
        
        lines = content.split('\n')
        current_category = "General"
        current_subcategory = ""
        buffer = []
        
        for line in lines:
            stripped = line.strip()
            
            # Main Category (# Header)
            if stripped.startswith('# '):
                # Flush previous buffer
                if buffer:
                    self._add_document(documents, current_category, current_subcategory, buffer)
                    buffer = []
                
                current_category = stripped.lstrip('#').strip()
                current_subcategory = "" # Reset sub and buffer
                
            # Subcategory (## Header)
            elif stripped.startswith('## '):
                # Flush previous buffer
                if buffer:
                    self._add_document(documents, current_category, current_subcategory, buffer)
                    buffer = []
                
                current_subcategory = stripped.lstrip('#').strip()
                
            # Content
            else:
                if stripped:
                    buffer.append(stripped)
                    
        # Flush last buffer
        if buffer:
            self._add_document(documents, current_category, current_subcategory, buffer)
            
        return documents

    def _add_document(self, documents, category, subcategory, lines):
        """Helper to create and append a document."""
        text = "\n".join(lines)
        if not text.strip():
            return
            
        # Create a document
        doc = {
            "category": category,
            "subcategory": subcategory,
            "text": text,
            "metadata": {
                "source": "RAG.md"
            }
        }
        documents.append(doc)
    
    def populate_vector_store(self, rag_path: str = "RAG.md"):
        """Load RAG text file and populate the vector store with embeddings."""
        print(f"Loading knowledge base from {rag_path}...")
        
        if not os.path.exists(rag_path):
             # Try .txt if .md not found
            if os.path.exists("RAG.txt"):
                 rag_path = "RAG.txt"
                 print(f"Found RAG.txt instead, using it.")
            else:
                 raise FileNotFoundError(f"Could not find {rag_path} or RAG.txt")

        with open(rag_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Prepare documents
        documents = self.parse_rag_file(content)
        print(f"Prepared {len(documents)} documents from file.")
        
        # Clear existing documents
        self.collection.delete_many({})
        print("Cleared existing documents.")
        
        # Generate embeddings and insert documents
        print("Generating embeddings and inserting documents...")
        docs_to_insert = []
        
        for i, doc in enumerate(documents):
            embedding = self.generate_embedding(doc['text'])
            doc['embedding'] = embedding
            docs_to_insert.append(doc)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(documents)} documents...")
        
        # Bulk insert
        if docs_to_insert:
            self.collection.insert_many(docs_to_insert)
            print(f"Successfully inserted {len(docs_to_insert)} documents into MongoDB.")
        
        # Create vector search index
        self.create_vector_search_index()
        
        return len(docs_to_insert)
    
    def search(self, query: str, top_k: int = 5, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search using MongoDB Atlas Vector Search.
        
        Args:
            query: The search query text
            top_k: Number of results to return
            category_filter: Optional category to filter by
        
        Returns:
            List of matching documents with their similarity scores
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Build the vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "category": 1,
                    "subcategory": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        # Add category filter if specified
        if category_filter:
            pipeline[0]["$vectorSearch"]["filter"] = {"category": category_filter}
        
        try:
            results = list(self.collection.aggregate(pipeline))
            return results
        except Exception as e:
            error_msg = str(e)
            print(f"Vector search error: {e}")
            
            if "SSL" in error_msg or "Timeout" in error_msg:
                print("\n⚠️  POSSIBLE CAUSE: MongoDB Atlas IP Whitelist Issue")
                print("1. Go to MongoDB Atlas -> Network Access")
                print("2. Add IP Address -> Allow Access from Anywhere (0.0.0.0/0)")
                print("   (Hugging Face Spaces uses dynamic IPs, so explicit IP whitelisting won't work)")
            
            # Fallback to simple text search if vector search fails
            return self._fallback_search(query, top_k)
    
    def _fallback_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback text search when vector search is not available."""
        # Simple regex-based search
        query_words = query.lower().split()
        regex_pattern = "|".join(query_words)
        
        results = list(self.collection.find(
            {"text": {"$regex": regex_pattern, "$options": "i"}},
            {"_id": 0, "text": 1, "category": 1, "subcategory": 1, "metadata": 1}
        ).limit(top_k))
        
        # Add a placeholder score
        for r in results:
            r['score'] = 0.5
        
        return results
    
    def get_rag_context(self, query: str, top_k: int = 5, min_score: float = 0.3, category_filter: Optional[str] = None) -> str:
        """
        Get relevant context for RAG based on user query.
        
        Args:
            query: User's question/query
            top_k: Maximum number of relevant documents to retrieve
            min_score: Minimum similarity score threshold
            category_filter: Optional category to filter results
        
        Returns:
            Formatted context string for the LLM
        """
        results = self.search(query, top_k=top_k, category_filter=category_filter)
        
        if not results:
            return ""
        
        # Filter by minimum score and format context
        relevant_docs = [r for r in results if r.get('score', 0) >= min_score]
        
        if not relevant_docs:
            return ""
        
        context_parts = []
        context_parts.append("=== RELEVANT INFORMATION ===")
        
        for i, doc in enumerate(relevant_docs, 1):
            category = doc.get('category', 'general').upper()
            subcategory = doc.get('subcategory', '')
            text = doc.get('text', '')
            score = doc.get('score', 0)
            
            header = f"[{category}"
            if subcategory:
                header += f" - {subcategory}"
            header += f"] (relevance: {score:.2f})"
            
            context_parts.append(header)
            context_parts.append(text)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()


# Utility function for populating the database
def setup_vector_store():
    """Setup and populate the vector store from RAG.md."""
    print("=" * 50)
    print("MongoDB Vector Store Setup")
    print("=" * 50)
    
    try:
        vs = VectorStore()
        count = vs.populate_vector_store("RAG.md")
        print(f"\n✅ Setup complete! {count} documents indexed.")
        print("\nNote: If using MongoDB Atlas, the vector search index may take")
        print("a few minutes to become active after creation.")
        vs.close()
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        raise


if __name__ == "__main__":
    setup_vector_store()
