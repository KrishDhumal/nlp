"""
Query Service — High-performance vector retrieval for RAG AI chat.

Loads sentence-transformers embedding model at the module level and queries
the Pinecone vector index "legal-auditor" with namespace and metadata filtering.
Includes automatic fallback to local vector store if Pinecone is unreachable.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from app.core.logging import logger

# 1. Module-level embedding model initialization (all-MiniLM-L6-v2, 384 dimensions)
logger.info("[QueryService] Initializing sentence-transformers model 'all-MiniLM-L6-v2'...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Pinecone Index Connection
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-auditor")
PINECONE_HOST = os.getenv("PINECONE_HOST", "")

_pinecone_index = None

def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None and PINECONE_API_KEY:
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            if PINECONE_HOST:
                _pinecone_index = pc.Index(name=PINECONE_INDEX_NAME, host=PINECONE_HOST)
            else:
                _pinecone_index = pc.Index(name=PINECONE_INDEX_NAME)
            logger.info(f"[QueryService] Connected to Pinecone index '{PINECONE_INDEX_NAME}'")
        except Exception as e:
            logger.warning(f"[QueryService] Failed to connect to Pinecone index: {e}")
            _pinecone_index = None
    return _pinecone_index


def query_document_chunks(query: str, doc_id: str, user_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Embeds the user query and searches Pinecone for document chunks.
    
    Args:
        query: Natural language query from the user.
        doc_id: Unique document or audit ID.
        user_id: Authenticated user ID (Pinecone namespace: user_{user_id}).
        top_k: Number of top matches to retrieve (default: 5).

    Returns:
        List of dicts: [
            {
                "text": str,
                "page_number": int,
                "clause_type": str,
                "score": float
            },
            ...
        ]
        Chunks with similarity score < 0.25 are ignored.
    """
    if not query or not query.strip() or not doc_id:
        return []

    # 1. Embed the query vector
    query_vector = embedding_model.encode(query.strip()).tolist()
    results: List[Dict[str, Any]] = []

    namespace = f"user_{user_id}"
    index = get_pinecone_index()

    if index is not None:
        try:
            # Query Pinecone with strict namespace isolation and doc_id filter
            response = index.query(
                namespace=namespace,
                vector=query_vector,
                top_k=top_k,
                filter={"doc_id": {"$eq": doc_id}},
                include_metadata=True
            )

            matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])
            for match in matches:
                score = float(getattr(match, "score", 0.0) if hasattr(match, "score") else match.get("score", 0.0))
                
                # Ignore scores < 0.25
                if score < 0.25:
                    continue

                meta = getattr(match, "metadata", {}) if hasattr(match, "metadata") else match.get("metadata", {})
                if not meta:
                    continue

                text = meta.get("raw_text") or meta.get("text", "")
                page_num = meta.get("page_number") or meta.get("page_start", 1)
                try:
                    page_number = int(page_num)
                except (ValueError, TypeError):
                    page_number = 1

                clause_type = meta.get("clause_type") or meta.get("section_title") or "General"

                results.append({
                    "text": text,
                    "page_number": page_number,
                    "clause_type": clause_type,
                    "score": round(score, 4)
                })

            if results:
                return results

        except Exception as e:
            logger.warning(f"[QueryService] Pinecone query failed ({e}) — falling back to vector store service.")

    # 2. Local fallback if Pinecone returned no results or had an error
    try:
        from app.services.ai.vector_store import vector_store_service
        raw_docs = vector_store_service.search_similar(query=query, user_id=user_id, audit_id=doc_id, k=top_k)
        for doc in raw_docs:
            meta = doc.metadata or {}
            text = doc.page_content or meta.get("raw_text", "")
            page_num = meta.get("page_number", 1)
            try:
                page_number = int(page_num)
            except (ValueError, TypeError):
                page_number = 1

            clause_type = meta.get("clause_type") or meta.get("section_title") or "General"
            results.append({
                "text": text,
                "page_number": page_number,
                "clause_type": clause_type,
                "score": 0.85
            })
    except Exception as fallback_err:
        logger.error(f"[QueryService] Fallback retrieval error: {fallback_err}")

    return results


class QueryService:
    """Wrapper class providing backward compatibility for existing routes."""
    def __init__(self):
        self.embedding_model = embedding_model

    def retrieve_relevant_chunks(
        self,
        question: str,
        user_id: str,
        audit_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        return query_document_chunks(query=question, doc_id=audit_id, user_id=user_id, top_k=top_k)


query_service = QueryService()
