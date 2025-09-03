import random
import requests

# --------- Random Responses ----------
def random_greeting():
    return random.choice([
        "👋 Hey there! How are you today?",
        "Hiya! 😃 What’s up?",
        "Hello friend, nice to see you again!",
        "🌟 Hey hey! Hope you’re doing awesome!",
        "Hi 🙌 I was waiting for you!",
        "Hey 👋 ready to chat?",
        "Hello 👩‍⚕️ How can I help with your health today?",
    ])

def random_goodbye():
    return random.choice([
        "Goodbye! 👋 Stay healthy!",
        "See you later, friend! Take care 💚",
        "👋 Bye-bye! Don’t forget to drink water!",
        "Catch you soon 🚀 Stay safe!",
        "Bye for now 🧑‍⚕️ Your health matters!",
    ])

def random_fallback():
    return random.choice([
        "🤖 I’m still learning, but I’ll get better soon!",
        "Hmm, I don’t fully understand — try rephrasing? 🧐",
        "Can you explain that differently? 🧠",
        "I might not know that yet, but I’ll improve 💪",
        "Interesting! I’ll remember this for later 📘",
    ])

# --------- API Helpers ----------
def search_duckduckgo(query: str) -> str:
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        res = requests.get(url, timeout=5).json()
        if res.get("AbstractText"):
            return f"🔎 DuckDuckGo: {res['AbstractText']}"
        return "I couldn’t find anything useful on DuckDuckGo 🤔"
    except Exception:
        return "❌ DuckDuckGo search failed."

def search_wikipedia(query: str) -> str:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
        res = requests.get(url, timeout=5).json()
        if "extract" in res:
            return f"📘 Wikipedia: {res['extract']}"
        return "No Wikipedia article found 😕"
    except Exception:
        return "❌ Wikipedia search failed."

def search_openfda(drug: str) -> str:
    try:
        url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug}&limit=1"
        res = requests.get(url, timeout=5).json()
        if "results" in res:
            data = res["results"][0]
            info = data.get("indications_and_usage", ["No usage info available"])[0]
            return f"💊 OpenFDA: {drug} — {info}"
        return f"No FDA data found for {drug}"
    except Exception:
        return "❌ OpenFDA search failed."

# --------- BrainTY Main Function ----------
def chatbot_response(message: str) -> str:
    msg = message.lower()

    # Greetings
    if any(word in msg for word in ["hello", "hi", "hey", "good morning", "good evening"]):
        return random_greeting()

    # Goodbye
    elif any(word in msg for word in ["bye", "goodbye", "see you", "later", "cya"]):
        return random_goodbye()

    # Gratitude
    elif "thank" in msg:
        return random.choice([
            "🙏 You’re most welcome!",
            "Happy to help 💙",
            "Always here for you 👩‍⚕️",
            "No problem at all 🚀",
            "It’s my pleasure 🤖",
        ])

    # Small talk
    elif "how are you" in msg:
        return random.choice([
            "I’m doing great, thanks for asking! 🤖",
            "I feel fantastic 🌟 What about you?",
            "Running at full speed ⚡",
            "I’m good and ready to help 👨‍⚕️",
        ])

    # Medicine search with OpenFDA
    elif msg.startswith("drug "):
        drug_name = msg.replace("drug ", "").strip()
        return search_openfda(drug_name)

    # Wikipedia search
    elif msg.startswith("wiki "):
        topic = msg.replace("wiki ", "").strip()
        return search_wikipedia(topic)

    # DuckDuckGo search
    elif msg.startswith("search "):
        query = msg.replace("search ", "").strip()
        return search_duckduckgo(query)

    # Fallback
    else:
        return random_fallback()