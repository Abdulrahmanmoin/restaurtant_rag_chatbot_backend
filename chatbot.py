import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def load_system_prompt():
    """Load system prompt from file"""
    if not os.path.exists("systemPrompt.txt"):
        return "You are a helpful assistant."
    with open("systemPrompt.txt", "r", encoding="utf-8") as f:
        content = f.read()
        content = content.strip().strip('"""').strip()
        return content

def load_knowledge_base():
    """Load knowledge base from JSON file"""
    if not os.path.exists("KB.json"):
        return {}
    with open("KB.json", "r", encoding="utf-8") as f:
        return json.load(f)

def get_kb_context(query, kb, conversation_history=None):
    """Retrieve relevant KB information based on query intents (fallback method)."""
    query = query.lower()
    
    # Extract context from conversation history if available
    history_text = ""
    if conversation_history:
        for msg in conversation_history:
            if msg.get('role') in ['user', 'assistant', 'system']:
                history_text += " " + msg.get('content', '').lower()
    
    # Combine query with history for better context matching
    combined_query = query + " " + history_text
    context = []
    
    # 1. Restaurant Info & Services
    if any(x in combined_query for x in ['restaurant', 'location', 'address', 'time', 'open', 'close', 'contact', 'service', 'delivery', 'dine', 'takeaway']):
        r = kb.get('restaurant', {})
        context.append(f"Restaurant: {r.get('name')} ({r.get('country')})")
        context.append(f"Services: {', '.join(r.get('services', []))}")
        
    # 2. Payment
    if any(x in combined_query for x in ['pay', 'card', 'cash', 'money', 'wallet']):
        context.append(f"Payment Methods: {', '.join(kb.get('payment_methods', []))}")
        
    # 3. Menu Categories
    menu = kb.get('menu', {})
    
    # Generic 'menu' query - show categories
    if 'menu' in combined_query:
        cats = list(menu.keys())
        context.append(f"Available Menu Categories: {', '.join(cats)}")
        
    # Specific categories
    # Pizza - check both current query and history
    if any(x in combined_query for x in ['pizza', 'ingredient', 'topping', 'king crust', 'contain', 'made of', 'include']):
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
        if kw in combined_query:
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
    if any(x in combined_query for x in ['deal', 'offer', 'promo', 'discount', 'price', 'cost']):
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

    return "\n".join(context)

class ChatBotService:
    def __init__(self):
        self.client = None
        self.system_prompt = None
        self.kb = None
        self.restaurant_name = "Alchemy Pizza"
        self.full_system_prompt = ""
        # self.model = "qwen/qwen3-4b:free" # Current valid Qwen3 free endpoint (Jan 2026)
        # self.model = "Qwen/Qwen2.5-0.5B-Instruct" # Current valid Qwen3 free endpoint (Jan 2026)
        self.model = "models/gemini-2.5-flash" # Latest Gemini model

    def load_summary_prompt(self):
        """Load summary indication prompt from file"""
        if not os.path.exists("summaryPrompt.txt"):
            return "You are a helpful assistant that summarizes conversation traces."
        with open("summaryPrompt.txt", "r", encoding="utf-8") as f:
            return f.read().strip()

    def initialize(self):
        print("Initializing Gemini Client...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Warning: GEMINI_API_KEY not found in environment.")
        
        self.client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
        )
        
        # Load config
        try:
            self.system_prompt = load_system_prompt()
            self.summary_prompt = self.load_summary_prompt()
            self.kb = load_knowledge_base()
            self.restaurant_name = self.kb.get("restaurant", {}).get("name", "Alchemy Pizza")
            
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

    def generate_response(self, user_input, conversation_history=None):
        if conversation_history is None:
            conversation_history = [{"role": "system", "content": self.full_system_prompt}]
        
        # Context Retrieval - pass conversation history for better context
        kb_context = get_kb_context(user_input, self.kb, conversation_history)
            
        full_input = user_input
        if kb_context:
            full_input = f"Context from Knowledge Base:\n{kb_context}\n\nCustomer: {user_input}"
            
        conversation_history.append({"role": "user", "content": full_input})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation_history,
                stream=True,
                max_tokens=500, # Explicitly limit output tokens
            )
            
            response_text = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response_text += content
                    yield content
            
            return response_text
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            yield f"I'm sorry, I'm having trouble connecting right now. Error: {str(e)}"
            return f"Error: {str(e)}"

    def summarize_conversation(self, last_summary, new_interaction):
        """Summarize the conversation to keep context small."""
        
        prompt = f"""
        Previous Summary:
        {last_summary}
        
        New Interaction:
        {new_interaction}
        
        Request: Based on the system prompt instructions, create a new summary.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.summary_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250
            )
            content = response.choices[0].message.content
            if content:
                content = content.strip()
                print(f"Generated Summary: {content[:50]}...") # Log start of summary
                return content
            else:
                print("Warning: Empty summary received from LLM.")
                # Fallback: maintain last summary + concise indication of new interaction
                return f"{last_summary}\n[New Interaction]: {new_interaction[:100]}...".strip()
                
        except Exception as e:
            print(f"Error generating summary: {e}")
            # Fallback: maintain last summary + concise indication of new interaction
            # IMPORTANT: Do not simply append the entire interaction loop if the model fails, 
            # otherwise we hit rate limits with massive payloads.
            return f"{last_summary}\n[Unsummarized Interaction]".strip()

def chat():
    bot = ChatBotService()
    bot.initialize()
    
    print("\n" + "="*50)
    print(f"🍕 {bot.restaurant_name} Chatbot (Gemini)")
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
            break
            
        print("Daniel Siddiqui: ", end="", flush=True)
        
        # Generate and stream response
        generator = bot.generate_response(user_input, messages)
        full_response = ""
        for token in generator:
            print(token, end="", flush=True)
            full_response += token
        print()
        
        # Add assistant response to history
        messages.append({"role": "assistant", "content": full_response})
        
        if len(messages) > 10: # Increased history limit as API can handle it better
            messages = [messages[0]] + messages[-9:]

if __name__ == "__main__":
    chat()
