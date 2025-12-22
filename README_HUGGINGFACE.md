# 🍕 Restaurant Chatbot - Hugging Face Spaces Deployment

## Quick Start

### 1. Create a New Space on Hugging Face

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Choose **"Docker"** as the SDK
4. Select **"Blank"** template
5. Set visibility (Public/Private)

### 2. Upload Files to Your Space

Upload the following files to your Space repository:

```
├── Dockerfile          # Docker configuration
├── main.py             # FastAPI entry point (app instance)
├── chatbot.py          # Chatbot logic
├── vector_store.py     # RAG/Vector store logic
├── requirements.txt    # Python dependencies
├── systemPrompt.txt    # System prompt configuration
├── KB.json             # Knowledge base
├── RAG.md              # RAG content
└── .env (optional)     # Environment variables
```

### 3. Configure Secrets (Recommended)

Instead of using `.env` file, configure secrets in your HF Space:

1. Go to your Space → **Settings** → **Repository secrets**
2. Add the following secrets:

| Secret Name | Description |
|-------------|-------------|
| `MONGODB_URI` | Your MongoDB Atlas connection string |
| `MONGODB_DB_NAME` | Database name (default: `restaurant_chatbot`) |
| `MONGODB_COLLECTION_NAME` | Collection name (default: `kb_embeddings`) |

### 4. Deploy

Once files are uploaded, Hugging Face will automatically build and deploy your Space.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Hugging Face Spaces                     │
├─────────────────────────────────────────────────────────┤
│  Docker Container                                        │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Uvicorn Server (Port 7860)                     │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │  FastAPI Application (app.py)           │    │    │
│  │  │  ┌─────────────────────────────────┐    │    │    │
│  │  │  │  ChatBotService (chatbot.py)   │    │    │    │
│  │  │  │  - Qwen2.5-0.5B-Instruct       │    │    │    │
│  │  │  │  - Loaded once at startup       │    │    │    │
│  │  │  └─────────────────────────────────┘    │    │    │
│  │  │  ┌─────────────────────────────────┐    │    │    │
│  │  │  │  VectorStore (vector_store.py) │    │    │    │
│  │  │  │  - MongoDB Atlas Vector Search │    │    │    │
│  │  │  │  - Sentence Transformers       │    │    │    │
│  │  │  └─────────────────────────────────┘    │    │    │
│  │  └─────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message and API info |
| `/health` | GET | Health check (model status, RAG status) |
| `/chat` | POST | Chat with the bot |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

### POST /chat

**Request:**
```json
{
  "message": "What pizzas do you have?",
  "history": []
}
```

**Response:**
```json
{
  "response": "We have a variety of delicious pizzas...",
  "history": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "What pizzas do you have?"},
    {"role": "assistant", "content": "We have a variety of delicious pizzas..."}
  ]
}
```

---

## CPU-Only Optimization

This deployment is optimized for HF Spaces **free tier (CPU-only)**:

1. **Single Worker**: Avoids memory issues
2. **CPU-Only PyTorch**: Smaller image, no CUDA overhead
3. **Model Caching**: Models loaded once at startup
4. **Efficient Model**: Qwen2.5-0.5B-Instruct (small but capable)

---

## Troubleshooting

### Space is stuck on "Building"
- Check the build logs for errors
- Ensure all required files are uploaded
- Verify `requirements.txt` syntax

### "Out of Memory" errors
- This is normal for first load (model download)
- The container should recover after restart
- Consider upgrading to a paid tier for more memory

### Chat returns 503 error
- Check `/health` endpoint
- Model may still be loading (wait 1-2 minutes)
- Check Space logs for initialization errors

### MongoDB connection fails
- Verify `MONGODB_URI` secret is set correctly
- Ensure IP is whitelisted in MongoDB Atlas (use `0.0.0.0/0` for Spaces)
- Check if MongoDB Atlas cluster is running

---

## Local Testing

```bash
# Build the Docker image
docker build -t restaurant-chatbot .

# Run locally
docker run -p 7860:7860 --env-file .env restaurant-chatbot

# Test the API
curl http://localhost:7860/health
```

---

## License

MIT License
