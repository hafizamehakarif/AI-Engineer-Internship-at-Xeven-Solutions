"""
RAG API ENDPOINT - AI Question Answering
=========================================
Learn: RAG (Retrieval-Augmented Generation), async, startup events, error handling
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
import asyncio

# 1. Create FastAPI app
app = FastAPI(
    title="RAG AI API",
    description="Ask questions about vision therapy & AI",
    version="1.0.0"
)

# ───────────────────────────────────────────────
# MOCK RAG SYSTEM (for learning)
# Real RAG uses: LangChain + embeddings + vector DB + LLM
# ───────────────────────────────────────────────
class MockRAGSystem:
    """Simple RAG system for demonstration"""
    
    def __init__(self):
        # Your knowledge base (add your vision therapy research here!)
        self.documents = [
            {
                "content": "Vision therapy helps improve eye-brain coordination through exercises",
                "source": "Vision Therapy Research 2025"
            },
            {
                "content": "Neural plasticity allows the brain to rewire and recover after injury",
                "source": "Neuroscience Journal"
            },
            {
                "content": "AI can assist in diagnosing vision disorders using computer vision",
                "source": "AI in Medicine 2025"
            },
            {
                "content": "Small Language Models (SLM) are efficient for specialized tasks like vision therapy",
                "source": "ML Research Paper"
            },
            {
                "content": "Rehabilitation therapy uses neural semantic matching to track progress",
                "source": "Rehabilitation Studies"
            }
        ]
    
    async def retrieve(self, query: str) -> List[dict]:
        """
        Step 1: RETRIEVE - Find relevant documents
        Real RAG: uses embeddings + cosine similarity + vector DB (FAISS, ChromaDB)
        """
        query_lower = query.lower()
        results = []
        
        for doc in self.documents:
            # Simple keyword matching (real: embeddings + similarity)
            doc_words = doc["content"].lower().split()
            if any(word in query_lower for word in doc_words[:10]):
                results.append(doc)
        
        return results[:3]  # Return top 3 results
    
    async def generate(self, query: str, context: List[dict]) -> tuple[str, float]:
        """
        Step 2: GENERATE - Create answer from context
        Real RAG: calls LLM (OpenAI, HuggingFace, your custom SLM)
        """
        if not context:
            return "I couldn't find relevant information for your question.", 0.0
        
        # Build context text
        context_text = "\n\n".join([f"📖 {doc['source']}: {doc['content']}" for doc in context])
        
        # Mock answer (real: LLM generates natural response)
        answer = f"""
╔══════════════════════════════════════════╗
║  🤖 AI Answer (Powered by RAG)          ║
╚══════════════════════════════════════════╝

Question: {query}

📚 Context from my knowledge base:
{context_text}

💡 Answer:
Based on the research above, vision therapy and AI work together to help 
people with vision disorders. The neural plasticity in the brain allows 
recovery through targeted exercises. Small Language Models can efficiently 
process this medical knowledge for personalized therapy plans.

(This is a mock answer. In production, an LLM like GPT or your custom SLM 
would generate a more natural response.)
"""
        
        confidence = 0.78  # Mock confidence score
        return answer, confidence

# Global RAG system (loaded once at startup)
rag_system: Optional[MockRAGSystem] = None

# ───────────────────────────────────────────────
# REQUEST/RESPONSE MODELS (Pydantic)
# ───────────────────────────────────────────────
class AskRequest(BaseModel):
    """What we expect when user asks a question"""
    question: str

class AskResponse(BaseModel):
    """What we return after answering"""
    answer: str
    sources: List[str]
    confidence: float

# ───────────────────────────────────────────────
# STARTUP EVENT: Load RAG once
# ───────────────────────────────────────────────
@app.on_event("startup")
async def load_rag_system():
    """
    Load RAG system when server starts
    IMPORTANT: Load once at startup, NOT per request (much faster!)
    """
    print("🚀 Starting RAG AI API...")
    print("📚 Loading knowledge base...")
    
    # Simulate loading (real: load models, connect to DB)
    await asyncio.sleep(1)
    
    rag_system = MockRAGSystem()
    
    print("✅ RAG system loaded successfully!")
    print(f"📖 Knowledge base has {len(rag_system.documents)} documents:")
    for doc in rag_system.documents:
        print(f"   • {doc['source']}")
    print("\n🎯 Try asking: 'What is vision therapy?' or 'How does AI help diagnosis?'")

# ───────────────────────────────────────────────
# HEALTH CHECK
# ───────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Check if API is running"""
    return {
        "status": "healthy",
        "api": "RAG AI API",
        "message": "✅ Ask questions at /ask endpoint"
    }

# ───────────────────────────────────────────────
# MAIN ENDPOINT: POST /ask (RAG)
# ───────────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    RAG Endpoint: Ask a question → Get AI answer
    
    Process:
    1. RETRIEVE relevant documents from knowledge base
    2. GENERATE answer using LLM
    3. Return answer + sources + confidence score
    """
    
    # Error handling: 400 for empty question
    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="❌ Question cannot be empty. Please provide a valid question."
        )
    
    question = request.question
    
    try:
        print(f"\n🤖 Processing: '{question}'")
        
        # Step 1: RETRIEVE
        retrieved_docs = await rag_system.retrieve(question)
        
        # Error handling: 404 for no results
        if not retrieved_docs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="❌ No relevant documents found. Try asking about: vision therapy, neural plasticity, AI diagnosis, SLM, or rehabilitation"
            )
        
        # Step 2: GENERATE
        answer, confidence = await rag_system.generate(question, retrieved_docs)
        
        # Step 3: Return response
        response = AskResponse(
            answer=answer,
            sources=[doc["source"] for doc in retrieved_docs],
            confidence=confidence
        )
        
        print(f"✅ Answer generated (confidence: {confidence:.2f})")
        return response
        
    except Exception as e:
        # Error handling: 500 for LLM errors
        print(f"❌ Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"❌ Error generating answer: {str(e)}"
        )

# ───────────────────────────────────────────────
# RUN THE SERVER
# ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_api:app", host="127.0.0.1", port=8001, reload=True)