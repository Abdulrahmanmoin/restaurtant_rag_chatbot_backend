# Simple LLM Chatbot

This project implements a simple chatbot using the Hugging Face `transformers` library and Microsoft's `DialoGPT-medium` model.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Chatbot:**
    ```bash
    python chatbot.py
    ```

## How it works

-   It uses `microsoft/DialoGPT-medium`, a model trained on Reddit conversations, making it good for casual chit-chat.
-   The script maintains a chat history context so the bot remembers previous turns in the conversation.
