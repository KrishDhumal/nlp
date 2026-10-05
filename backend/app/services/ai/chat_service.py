"""
Chat Service — Grounded RAG Document Intelligence with Google Gemini 2.5 Flash.

Utilizes the google-genai SDK to generate structured, strictly grounded legal responses
with page-number and clause-type citations. Prohibits hallucination and outside knowledge.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from app.core.logging import logger
from app.services.ai.query_service import query_document_chunks

# 1. Exact Pydantic models aligning with frontend SourceCitation.jsx & ChatPanel.jsx
class Citation(BaseModel):
    page_number: int = Field(
        default=1,
        description="The exact 1-indexed document page number where the excerpt was found."
    )
    clause_type: str = Field(
        default="General",
        description="The legal clause category or topic classification (e.g., Termination, Liability, Indemnification)."
    )
    verbatim_quote: str = Field(
        description="The exact word-for-word excerpt quoted from the source text supporting the claim."
    )
    section: Optional[str] = Field(
        default=None,
        description="Section title or label (e.g. 'Section 5.2' or clause category) displayed in badge."
    )
    text: Optional[str] = Field(
        default=None,
        description="Text content for frontend popover display (mirrors verbatim_quote)."
    )


class ChatResponse(BaseModel):
    answer: str = Field(
        description="Authoritative, professional legal answer derived strictly from the document excerpts."
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of exact citations supporting the answer."
    )
    sources: Optional[List[Citation]] = Field(
        default=None,
        description="List of source citations matching frontend ChatPanel and SourceCitation prop expectations."
    )


# 2. Strict Legal System Prompt
SYSTEM_INSTRUCTION = """You are an elite Legal Intelligence Auditor AI.
Your sole purpose is to answer the user's inquiry regarding the provided legal document excerpts.

STRICT CONSTRAINTS & COMPLIANCE RULES:
1. Grounding: Answer the question ONLY and EXCLUSIVELY using the facts directly stated in the provided DOCUMENT EXCERPTS.
2. Anti-Hallucination: Do NOT assume, infer, extrapolate, or bring in outside legal knowledge, precedents, or assumptions not explicitly present in the text.
3. Insufficient Context: If the answer is not contained within the provided excerpts, say:
   "I don't have enough information in this document to answer that question."
4. Citations: Every substantive point made MUST cite the exact page_number, clause_type, and a verbatim_quote from the excerpts.
5. Verbatim Quote: The verbatim_quote must be an EXACT substring of the provided text.
6. Tone: Objective, precise, and professional.
"""


# 3. Google GenAI Client Initialization
def _get_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Neither GEMINI_API_KEY nor GOOGLE_API_KEY is configured in the environment.")
    return genai.Client(api_key=api_key)


async def generate_grounded_response(
    query: str,
    doc_id: str,
    user_id: str
) -> Dict[str, Any]:
    """
    RAG generation pipeline:
    1. Retrieves relevant document chunks via query_document_chunks.
    2. If no chunks found (or low similarity), returns a fallback response without LLM call.
    3. Calls gemini-2.5-flash with temperature=0.0 and ChatResponse JSON schema.
    4. Returns clean, structured response aligning with frontend requirements.
    """
    # Step 1: Retrieve context chunks from Pinecone / vector store
    chunks = query_document_chunks(query=query, doc_id=doc_id, user_id=user_id, top_k=5)

    # Step 2: Empty retrieval fallback — return immediately without LLM invocation
    if not chunks:
        logger.info(f"[ChatService] No matching chunks found for query '{query}' in doc {doc_id}")
        empty_answer = "I don't have enough information in this document to answer that question."
        return {
            "answer": empty_answer,
            "citations": [],
            "sources": []
        }

    # Step 3: Build formatted excerpts
    context_blocks = []
    for i, c in enumerate(chunks, 1):
        page = c.get("page_number", 1)
        clause = c.get("clause_type", "General")
        snippet = c.get("text", "").strip()
        context_blocks.append(f"[Excerpt {i} | Page {page} | Clause: {clause}]\n{snippet}")

    full_context = "\n\n---\n\n".join(context_blocks)
    user_prompt = f"DOCUMENT EXCERPTS:\n{full_context}\n\nUSER QUESTION:\n{query}"

    # Step 4: Invoke Gemini with resilient model fallback and structured output schema
    try:
        client = _get_genai_client()
        candidate_models = [
            "gemini-flash-lite-latest",
            "gemini-3.6-flash",
            "gemini-3.7-flash",
            "gemini-flash-latest",
            "gemini-2.5-flash",
        ]
        parsed: Optional[ChatResponse] = None
        for model_name in candidate_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=ChatResponse
                    )
                )
                if response and response.parsed:
                    parsed = response.parsed
                    break
            except Exception as model_err:
                logger.warning(f"[ChatService] Model {model_name} failed: {model_err}")
                continue

        if parsed:
            result_citations = []
            for item in parsed.citations:
                quote = item.verbatim_quote or item.text or ""
                cit = {
                    "page_number": item.page_number,
                    "clause_type": item.clause_type,
                    "verbatim_quote": quote,
                    "section": item.section or item.clause_type or f"Page {item.page_number}",
                    "text": quote
                }
                result_citations.append(cit)

            return {
                "answer": parsed.answer,
                "citations": result_citations,
                "sources": result_citations
            }

        # Fallback if parsed schema is somehow not returned
        return {
            "answer": response.text or "I processed your document but could not format the output.",
            "citations": [],
            "sources": []
        }

    except Exception as e:
        logger.error(f"[ChatService] Gemini generation error: {e}")
        # Graceful fallback on API error
        return {
            "answer": "An error occurred while generating a grounded response from the AI model. Please try again.",
            "citations": [],
            "sources": []
        }


class ChatService:
    """Wrapper class providing backward compatibility for existing controllers."""
    def __init__(self):
        pass

    async def generate_grounded_response(self, query: str, doc_id: str, user_id: str) -> Dict[str, Any]:
        return await generate_grounded_response(query=query, doc_id=doc_id, user_id=user_id)

    async def generate_answer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        res = await generate_grounded_response(query=question, doc_id="default", user_id="default")
        return res.get("answer", "")


chat_service = ChatService()
