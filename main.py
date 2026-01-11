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
    print("ChatBot Service Shutdown complete.")

app = FastAPI(
    title="Restaurant Chatbot API",
    description="API for the Pizza Restaurant Chatbot with keyword-based KB.json.",
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
        # Convert Pydantic models to dicts
        input_history = [msg.model_dump() for msg in request.history] if request.history else []
        
        # Check for Summary Mode (if we have a system message in history that isn't the persona)
        # Note: Frontend history generally doesn't contain the 'system' persona unless we sent it.
        # Our Logic: If history has a role='system', it's a summary we sent previously.
        is_summary_mode = any(msg['role'] == 'system' for msg in input_history)
        
        full_response = ""
        new_history_objs = []
        
        if is_summary_mode:
            # --- SUMMARY MODE ---
            current_summary = next((m['content'] for m in input_history if m['role'] == 'system'), "")
            
            # Prepare generation context
            generation_history = [{"role": "system", "content": bot.full_system_prompt}]
            if current_summary:
                generation_history.append({"role": "system", "content": f"Previous Conversation Summary: {current_summary}"})
            
            # Generate Response
            response_generator = bot.generate_response(request.message, generation_history)
            for token in response_generator:
                full_response += token
            # Update Summary
            new_interaction = f"User: {request.message}\nAssistant: {full_response}"
            new_summary = bot.summarize_conversation(current_summary, new_interaction)
            
            new_history_objs = [Message(role="system", content=new_summary)]
            
        else:   
            # --- RAW MODE ---
            # Check if we should switch to summary mode
            # Threshold: 4 messages (2 user turns + 2 assistant responses)
            if len(input_history) >= 4:
                # --- SWITCH TO SUMMARY MODE ---
                print("Switching to Summary Mode...")
                
                # 1. Summarize existing raw history
                # We concat the raw messages to form a "previous interaction" block
                raw_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in input_history])
                initial_summary = bot.summarize_conversation("", raw_text)
                
                # 2. Generate Response using this summary
                generation_history = [{"role": "system", "content": bot.full_system_prompt}]
                generation_history.append({"role": "system", "content": f"Previous Conversation Summary: {initial_summary}"})
                
                response_generator = bot.generate_response(request.message, generation_history)
                for token in response_generator:
                    full_response += token
                
                # 3. Create Final Summary
                new_interaction = f"User: {request.message}\nAssistant: {full_response}"
                final_summary = bot.summarize_conversation(initial_summary, new_interaction)
                
                new_history_objs = [Message(role="system", content=final_summary)]
                
            else:
                # --- STAY IN RAW MODE ---
                
                # Prepare generation context: System Prompt + Raw History
                generation_history = [{"role": "system", "content": bot.full_system_prompt}]
                generation_history.extend(input_history)
                
                # Generate Response
                # Note: generate_response appends the new user message to conversation_history internally
                response_generator = bot.generate_response(request.message, generation_history)
                for token in response_generator:
                    full_response += token
                
                # Return updated raw history
                # We need to take the input history + new user msg + new asst msg
                new_history_objs = [Message(**m) for m in input_history]
                new_history_objs.append(Message(role="user", content=request.message))
                new_history_objs.append(Message(role="assistant", content=full_response))

        return ChatResponse(
            response=full_response,
            history=new_history_objs
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))  # Default to 7860 for HF Spaces
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
