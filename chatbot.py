from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import json
from threading import Thread
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG vector store
try:
    from vector_store import VectorStore
    RAG_ENABLED = True
except ImportError:
    RAG_ENABLED = False
    print("Warning: vector_store module not found. Running without RAG.")


def load_system_prompt():
    """Load system prompt from file"""
    with open("systemPrompt.txt", "r", encoding="utf-8") as f:
        content = f.read()
        content = content.strip().strip('"""').strip()
        return content


def load_knowledge_base():
    """Load knowledge base from JSON file"""
    with open("KB.json", "r", encoding="utf-8") as f:
        return json.load(f)


def get_kb_context(query, kb):
    """Retrieve relevant KB information based on query intents (fallback method)."""
    query = query.lower()
    context = []
    
    # 1. Restaurant Info & Services
    if any(x in query for x in ['restaurant', 'location', 'address', 'time', 'open', 'close', 'contact', 'service', 'delivery', 'dine', 'takeaway']):
        r = kb.get('restaurant', {})
        context.append(f"Restaurant: {r.get('name')} ({r.get('country')})")
        context.append(f"Services: {', '.join(r.get('services', []))}")
        
    # 2. Payment
    if any(x in query for x in ['pay', 'pay', 'card', 'cash', 'money', 'wallet']):
        context.append(f"Payment Methods: {', '.join(kb.get('payment_methods', []))}")
        
    # 3. Menu Categories
    menu = kb.get('menu', {})
    
    # Generic 'menu' query - show categories
    if 'menu' in query:
        cats = list(menu.keys())
        context.append(f"Available Menu Categories: {', '.join(cats)}")
        
    # Specific categories
    # Pizza
    if 'pizza' in query:
        context.append("=== PIZZA MENU ===")
        for subcat, items in menu.get('Pizza', {}).items():
            context.append(f"[{subcat}]")
            for item in items:
                context.append(f"- {item['name']}: {item['description']}")
                
    # Other categories mapping
    cat_keywords = {
        'appetizer': 'Appetizers & Starters',
        'starter': 'Appetizers & Starters',
        'wing': 'Chicken Wings',
        'calzone': 'Calzones',
        'pasta': 'Pastas',
        'kid': 'Kids Meal',
        'dessert': 'Desserts',
        'cake': 'Desserts',
        'drink': 'Beverages & Sides',
        'beverage': 'Beverages & Sides',
        'side': 'Beverages & Sides'
    }
    
    for kw, cat_key in cat_keywords.items():
        if kw in query:
            items = menu.get(cat_key, [])
            context.append(f"\n[{cat_key}]")
            if isinstance(items, list):
                if items and isinstance(items[0], dict):
                    for item in items:
                        context.append(f"- {item['name']}: {item['description']}")
                else:
                    for item in items:
                        context.append(f"- {item}")
                        
    # 4. Deals
    if any(x in query for x in ['deal', 'offer', 'promo', 'discount', 'price', 'cost']):
        context.append("\n=== DEALS & OFFERS ===")
        deals = kb.get('deals', {})
        for cat, items in deals.items():
            context.append(f"[{cat}]")
            for item in items:
                if isinstance(item, dict):
                    name = item.get('name', '')
                    desc = item.get('description', '')
                    if name: context.append(f"- {name}: {desc}")
                    else: context.append(f"- {desc}")
                else: # fallback
                    context.append(f"- {item}")

    # Log found context for debugging (optional)
    if context:
        print(f"\n[System: Found {len(context)} relevant sections]")
        
    return "\n".join(context)


def get_rag_intent(query):
    """Determine RAG search intent/category based on query keywords."""
    query = query.lower()
    
    # Map keywords to RAG.md categories
    # Note: Category names must EXACTLY match the # Headers in RAG.md
    
    if any(x in query for x in ['deal', 'offer', 'promo', 'discount']):
        return "DEALS & COMBOS"
    
    if any(x in query for x in ['pizza', 'flavor', 'crust', 'topping', 'menu']):
        return "🍕 Pizza Menu"
        
    if any(x in query for x in ['starter', 'appetizer', 'wing', 'garlic', 'potato', 'bite', 'roll']):
        return "APPETIZERS & STARTERS"
        
    if any(x in query for x in ['restaurant', 'location', 'address', 'time', 'open', 'close', 'contact', 'service', 'dine', 'takeaway', 'pay', 'method']):
        return "🍕 Pizza Alchemy"
        
    if any(x in query for x in ['order', 'deliver', 'online', 'website', 'app']):
        return "🍕 Pizza Alchemy"
        
    return None

class ChatBotService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.vector_store = None
        self.system_prompt = None
        self.kb = None
        self.restaurant_name = "Pizza Restaurant"
        self.full_system_prompt = ""

    def initialize(self):
        print("Loading Qwen2.5-0.5B-Instruct...")
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Loading model {model_name}...", flush=True)
        try:
            if torch.cuda.is_available():
                print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                print("No GPU found - using CPU", flush=True)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                
            print("Model loaded successfully.", flush=True)
        except Exception as e:
            print(f"Error loading model: {e}", flush=True)
            raise e
        
        # Load config
        try:
            self.system_prompt = load_system_prompt()
            self.kb = load_knowledge_base()
            self.restaurant_name = self.kb.get("restaurant", {}).get("name", "Pizza Restaurant")
            
            self.full_system_prompt = f"""{self.system_prompt}
            
            You are Daniel Siddiqui, the friendly waiter at {self.restaurant_name}.
            You have access to the restaurant's knowledge base.
            Relevant information will be provided to you based on the customer's query.
            Use this information to assist the customer.
            Be concise and friendly.
            """
            
            print("✅ Config loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading config files: {e}")
            raise e
            
        # Initialize RAG
        if RAG_ENABLED:
            try:
                mongodb_uri = os.getenv("MONGODB_URI")
                if mongodb_uri and "<" not in mongodb_uri:
                    print("Initializing RAG with MongoDB Vector Search...")
                    self.vector_store = VectorStore()
                    print("✅ RAG enabled with semantic search!")
                else:
                    print("⚠️  MongoDB URI not configured. Using keyword-based retrieval.")
            except Exception as e:
                print(f"⚠️  Could not initialize RAG: {e}")

    def generate_response(self, user_input, conversation_history=None):
        if conversation_history is None:
            conversation_history = [{"role": "system", "content": self.full_system_prompt}]
        
        # Context Retrieval
        kb_context = ""
        if self.vector_store:
            rag_filter = get_rag_intent(user_input)
            if rag_filter:
                print(f"[System: Detected Intent -> {rag_filter}]")
            kb_context = self.vector_store.get_rag_context(user_input, top_k=5, min_score=0.3, category_filter=rag_filter)
        
        if not kb_context:
            kb_context = get_kb_context(user_input, self.kb)
            
        full_input = user_input
        if kb_context:
            full_input = f"Context from Knowledge Base:\n{kb_context}\n\nCustomer: {user_input}"
            
        conversation_history.append({"role": "user", "content": full_input})
        
        prompt = self.tokenizer.apply_chat_template(conversation_history, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        response_text = ""
        for new_text in streamer:
            response_text += new_text
            yield new_text
            
        thread.join()
        
        # We need to return the pure assistant response to be added to history by the caller
        # The caller is responsible for maintaining history state
        return response_text

def chat():
    bot = ChatBotService()
    bot.initialize()
    
    print("\n" + "="*50)
    print(f"🍕 {bot.restaurant_name} Chatbot")
    if bot.vector_store:
        print("🔍 RAG Mode: Semantic Search (MongoDB Vector DB)")
    else:
        print("📝 Mode: Keyword-based Retrieval")
    print("="*50)
    print("Type 'quit' to exit.\n")
    
    messages = [{"role": "system", "content": bot.full_system_prompt}]
    
    while True:
        try:
            user_input = input(">> Customer: ")
        except EOFError:
            break
        
        if user_input.lower() in ["quit", "exit"]:
            print(f"Daniel Siddiqui: Thanks for visiting {bot.restaurant_name}! See you soon! 🍕")
            if bot.vector_store:
                bot.vector_store.close()
            break
            
        print("Daniel Siddiqui: ", end="", flush=True)
        
        # Generate and stream response
        generator = bot.generate_response(user_input, messages)
        full_response = ""
        for token in generator:
            print(token, end="", flush=True)
            full_response += token
        print()
        
        # Add assistant response to history (user input is added within generate_response but that's local scope if passed by value, 
        # actually specialized handling needed because generate_response appends to passed list? 
        # Wait, lists are mutable. But I constructed a new prompt inside.
        # Let's check generate_response logic: `conversation_history.append(...)`. Yes, it modifies the list.
        # But we also need to append the ASSISTANT'S response.
        messages.append({"role": "assistant", "content": full_response})
        
        if len(messages) > 7:
            messages = [messages[0]] + messages[-6:]

if __name__ == "__main__":
    chat()
