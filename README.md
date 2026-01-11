# Restaurant Chatbot Backend

This project implements a restaurant chatbot using the OpenRouter API for LLM functionality and a keyword-based knowledge base.

## Features
- **OpenRouter Integration:** Uses high-quality LLMs via OpenRouter.
- **Knowledge Base:** Keyword-based retrieval from `KB.json` to provide accurate restaurant information.
- **FastAPI:** Modern, fast (high-performance) web framework for building APIs.
- **Streaming Support:** Real-time token streaming for a better user experience.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables:**
    Create a `.env` file with your OpenRouter API key:
    ```
    OPENROUTER_API_KEY=your_key_here
    ```

3.  **Run the Chatbot (CLI):**
    ```bash
    python chatbot.py
    ```

4.  **Run the API Server:**
    ```bash
    python main.py
    ```

## How it works
- The chatbot uses a keyword-based system to find relevant information in `KB.json`.
- It prefixes the user's query with this context and sends it to an LLM via OpenRouter.
- The LLM acts as "Daniel Siddiqui", a friendly waiter, using the provided context to answer customer questions.
