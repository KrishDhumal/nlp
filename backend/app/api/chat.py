"""
Chat API Router — POST /api/chat/query and POST /api/chat

Provides grounded legal document Q&A using the Member 3 RAG engine.
Integrates with Member 1's JWT authentication dependency with safe fallback.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import jwt

from app.core.config import settings
from app.core.database import get_database
from app.core.logging import logger
from app.services.ai.chat_service import generate_grounded_response

router = APIRouter()
security = HTTPBearer(auto_error=False)


# 1. Request & Response Schemas
class ChatQueryRequest(BaseModel):
    # Supports both new { doc_id, query } and existing { audit_id, question } payloads
    doc_id: Optional[str] = Field(default=None, description="Document or Audit ID to query against")
    audit_id: Optional[str] = Field(default=None, description="Alias for doc_id from legacy frontend")
    query: Optional[str] = Field(default=None, description="User's natural language question")
    question: Optional[str] = Field(default=None, description="Alias for query from legacy frontend")


# 2. Auth resolution with graceful fallback
async def resolve_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """
    Extracts the authenticated user_id using Member 1's JWT mechanism.
    If the token is absent or invalid, provides a fallback guest ID so development
    and testing continue uninterrupted.
    """
    if credentials and credentials.credentials:
        token = credentials.credentials
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            email: Optional[str] = payload.get("sub")
            if email:
                db = get_database()
                user = await db["users"].find_one({"email": email})
                if user and "_id" in user:
                    return str(user["_id"])
        except Exception as e:
            logger.debug(f"[ChatAPI] JWT validation skipped or failed: {e}")

    # Fallback user identifier if auth is not yet wired or running in test mode
    return "default_user"


# 3. Main RAG Query Handler
async def handle_chat_query(
    payload: ChatQueryRequest,
    user_id: str
) -> Dict[str, Any]:
    # Resolve document ID and question text from either key convention
    doc_id = payload.doc_id or payload.audit_id
    query_text = payload.query or payload.question

    if not doc_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document ID ('doc_id' or 'audit_id') is required."
        )

    if not query_text or not query_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string ('query' or 'question') cannot be empty."
        )

    try:
        # Execute grounded response generation
        response = await generate_grounded_response(
            query=query_text.strip(),
            doc_id=str(doc_id).strip(),
            user_id=user_id
        )
        return response
    except Exception as e:
        logger.error(f"[ChatAPI] Error executing grounded response: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate a grounded response. Please try again."
        )


# 4. Endpoints:
# In main.py, router is mounted with prefix="/api", so:
# - /chat/query resolves to /api/chat/query
# - /chat resolves to /api/chat
# - /query is also added as an alias
@router.post("/chat/query")
@router.post("/query")
async def chat_query_endpoint(
    payload: ChatQueryRequest,
    user_id: str = Depends(resolve_user_id)
):
    """
    RAG Chat endpoint matching Member 3 specification: POST /api/chat/query
    Accepts: { doc_id: str, query: str }
    """
    return await handle_chat_query(payload, user_id)


@router.post("/chat")
@router.post("")
@router.post("/")
async def chat_legacy_endpoint(
    payload: ChatQueryRequest,
    user_id: str = Depends(resolve_user_id)
):
    """
    Backwards-compatible endpoint for existing ChatPanel.jsx calls: POST /api/chat
    Accepts: { audit_id: str, question: str } or { doc_id: str, query: str }
    """
    return await handle_chat_query(payload, user_id)
