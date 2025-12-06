"""
Retriever Module - Handles document retrieval using API or open-weight models
"""

import os
import numpy as np
from typing import List, Tuple
from openai import OpenAI
import pickle


class BaseRetriever:
    """Base class for all retrievers"""
    
    def __init__(self, corpus_dir: str):
        self.corpus_dir = corpus_dir
        self.documents = []
        self.doc_names = []
        self._load_corpus()
    
    def _load_corpus(self):
        """Load all documents from corpus directory"""
        for filename in os.listdir(self.corpus_dir):
            filepath = os.path.join(self.corpus_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.documents.append(content)
                    self.doc_names.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        print(f"✓ Loaded {len(self.documents)} documents from corpus")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str]]:
        """Retrieve top-k relevant documents for query"""
        raise NotImplementedError


class AzureRetriever(BaseRetriever):
    """API-based retriever using Azure embeddings"""
    
    def __init__(self, corpus_dir: str, api_key: str, base_url: str):
        super().__init__(corpus_dir)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "azure/text-embedding-3-small"
        self.embeddings_cache = "embeddings_azure.pkl"
        
        # Generate or load embeddings
        if os.path.exists(self.embeddings_cache):
            print(f"Loading cached Azure embeddings...")
            with open(self.embeddings_cache, 'rb') as f:
                self.doc_embeddings = pickle.load(f)
        else:
            print("Generating Azure embeddings for corpus...")
            self.doc_embeddings = self._embed_documents()
            with open(self.embeddings_cache, 'wb') as f:
                pickle.dump(self.doc_embeddings, f)
        
        print(f"✓ Azure retriever ready with {len(self.doc_embeddings)} document embeddings")
    
    def _embed_documents(self) -> np.ndarray:
        """Generate embeddings for all documents"""
        embeddings = []
        for i, doc in enumerate(self.documents):
            try:
                response = self.client.embeddings.create(
                    input=doc[:8000],  # Truncate long docs
                    model=self.model
                )
                embeddings.append(response.data[0].embedding)
                if (i + 1) % 10 == 0:
                    print(f"  Embedded {i+1}/{len(self.documents)} documents")
            except Exception as e:
                print(f"Error embedding doc {i}: {e}")
                embeddings.append(np.zeros(1536))  # Default embedding size
        
        return np.array(embeddings)
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query"""
        try:
            response = self.client.embeddings.create(
                input=query,
                model=self.model
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error embedding query: {e}")
            return np.zeros(1536)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str]]:
        """Retrieve top-k documents using cosine similarity"""
        query_emb = self._embed_query(query)
        
        # Compute cosine similarity
        similarities = np.dot(self.doc_embeddings, query_emb) / (
            np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return (doc_content, doc_name) pairs
        results = [
            (self.documents[idx], self.doc_names[idx])
            for idx in top_indices
        ]
        
        return results


class LocalRetriever(BaseRetriever):
    """Open-weight retriever using sentence-transformers"""
    
    def __init__(self, corpus_dir: str):
        super().__init__(corpus_dir)
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except ImportError:
            raise ImportError("Please install: pip install sentence-transformers")
        
        self.embeddings_cache = "embeddings_local.pkl"
        
        # Generate or load embeddings
        if os.path.exists(self.embeddings_cache):
            print(f"Loading cached local embeddings...")
            with open(self.embeddings_cache, 'rb') as f:
                self.doc_embeddings = pickle.load(f)
        else:
            print("Generating local embeddings for corpus...")
            self.doc_embeddings = self._embed_documents()
            with open(self.embeddings_cache, 'wb') as f:
                pickle.dump(self.doc_embeddings, f)
        
        print(f"✓ Local retriever ready with {len(self.doc_embeddings)} document embeddings")
    
    def _embed_documents(self) -> np.ndarray:
        """Generate embeddings for all documents"""
        # Truncate long documents
        truncated_docs = [doc[:5000] for doc in self.documents]
        embeddings = self.model.encode(truncated_docs, show_progress_bar=True)
        return np.array(embeddings)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str]]:
        """Retrieve top-k documents using cosine similarity"""
        query_emb = self.model.encode([query])[0]
        
        # Compute cosine similarity
        similarities = np.dot(self.doc_embeddings, query_emb) / (
            np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return (doc_content, doc_name) pairs
        results = [
            (self.documents[idx], self.doc_names[idx])
            for idx in top_indices
        ]
        
        return results


def get_retriever(retriever_type: str, corpus_dir: str, api_key: str = None, base_url: str = None):
    """Factory function to get appropriate retriever"""
    if retriever_type.lower() == "azure":
        if not api_key or not base_url:
            raise ValueError("API key and base URL required for Azure retriever")
        return AzureRetriever(corpus_dir, api_key, base_url)
    
    elif retriever_type.lower() == "local":
        return LocalRetriever(corpus_dir)
    
    elif retriever_type.lower() == "none":
        return None
    
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
