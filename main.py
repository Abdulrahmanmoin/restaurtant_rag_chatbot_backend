from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from chatbot import ChatBotService
import uvicorn
from contextlib import asynccontextmanager

# Define Pydantic models for request and response
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    response: str
    history: List[Message]

# Global bot instance
bot = ChatBotService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    Initializes the chatbot model and resources.
    """
    print("Initializing ChatBot Service...")
    try:
        bot.initialize()
        print("ChatBot Service Initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize ChatBot Service: {e}")
        # We might want to re-raise or handle this depending on how critical it is.
        # For now, we'll let the app start but calls might fail.
    
    yield
    
    # Cleanup
    print("Shutting down ChatBot Service...")
    if bot.vector_store:
        bot.vector_store.close()
    print("ChatBot Service Shutdown complete.")

app = FastAPI(
    title="Restaurant Chatbot API",
    description="API for the Pizza Restaurant Chatbot with RAG capabilities.",
    version="1.0.0",
    lifespan=lifespan
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint to interact with the bot.
    Accepts a user message and optional conversation history.
    Returns the bot's response and the updated conversation history.
    """
    try:
        # Convert Pydantic models to list of dicts for the chatbot service
        conversation_history = [msg.model_dump() for msg in request.history] if request.history else []
        
        # If history is empty, initialize with system prompt
        if not conversation_history:
            # We need to make sure system prompt is loaded
            if not bot.full_system_prompt:
                # Fallback if somehow not initialized correctly yet
                 bot.full_system_prompt = "You are a helpful assistant."
            
            conversation_history = [{"role": "system", "content": bot.full_system_prompt}]
        
        # Generate response
        # generate_response yields tokens but also updates conversation_history in place with the User message
        response_generator = bot.generate_response(request.message, conversation_history)
        
        full_response = ""
        for token in response_generator:
            full_response += token
            
        # Append assistant response to history
        conversation_history.append({"role": "assistant", "content": full_response})
        
        # Convert back to Pydantic models
        updated_history = [Message(**msg) for msg in conversation_history]
        
        return ChatResponse(
            response=full_response,
            history=updated_history
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "rag_enabled": bot.vector_store is not None}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))  # Default to 7860 for HF Spaces
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
