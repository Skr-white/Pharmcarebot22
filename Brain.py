import os
import random
import requests
from datetime import datetime

# ===========================
# ENVIRONMENT VARIABLES
# ===========================
HF_KEY = os.getenv("HF_API_KEY")
OWM_KEY = os.getenv("WEATHER_API_KEY")
HF_HEADERS = {"Authorization": f"Bearer {HF_KEY}"}

# ===========================
# RANDOM SMALL TALK
# ===========================
def random_greeting():
    return random.choice([
        "👋 Hey there, superstar! How are you today?",
        "🌟 Hiya! 😃 What’s on your mind?",
        "Hello friend 🫂 Always nice to see you again!",
        "💫 Hey hey! Hope your day’s shining bright!",
        "🙌 Hi! I was waiting for you! What’s new?",
        "Hey 👋 Ready for some smart talk?",
        "👩‍⚕️ Hello, how can I help with your health or knowledge today?",
    ])

def random_goodbye():
    return random.choice([
        "👋 Goodbye, legend! Stay strong & healthy 💚",
        "✨ See you later! Don’t forget to smile 🙂",
        "🚀 Bye-bye! Keep pushing forward!",
        "💧 Don’t forget to drink water, friend!",
        "🧑‍⚕️ Take care! Your health matters to me 💊",
    ])

def random_fallback():
    return random.choice([
        "🤖 I didn’t fully catch that… could you rephrase it? 🧐",
        "Hmm 🤔 interesting… can you explain it another way?",
        "I’m still learning 🧠 but I’ll get sharper every day 💪",
        "Ooooh that’s a tough one 😅 but I’ll note it down 📘",
        "✨ Try a command like `wiki sun`, `drug ibuprofen`, or `weather Lagos`",
    ])

# ===========================
# FUN EXTRAS (Jokes & Facts)
# ===========================
def get_joke():
    jokes = [
        "😂 Why don’t scientists trust atoms? Because they make up everything!",
        "🤣 Why did the math book look sad? Because it had too many problems.",
        "😅 I’m reading a book on anti-gravity… it’s impossible to put down!",
        "😜 Why did the scarecrow win an award? Because he was outstanding in his field!",
    ]
    return random.choice(jokes)

def get_fun_fact():
    facts = [
        "🌍 Did you know? Honey never spoils. Archaeologists found 3,000-year-old honey still edible!",
        "🧠 Your brain generates about 20 watts of electricity — enough to power a light bulb 💡",
        "💧 Water can boil and freeze at the same time (called the triple point).",
        "🚀 A day on Venus is longer than a year on Venus!",
    ]
    return random.choice(facts)

# ===========================
# HUGGING FACE NLP
# ===========================
def hf_query(model: str, text: str):
    try:
        url = f"https://api-inference.huggingface.co/models/{model}"
        res = requests.post(url, headers=HF_HEADERS, json={"inputs": text}, timeout=20)
        if res.status_code == 200:
            return res.json()
        return {"error": res.text}
    except Exception as e:
        return {"error": str(e)}

def summarize_text(text: str):
    out = hf_query("facebook/bart-large-cnn", text)
    if isinstance(out, list) and "summary_text" in out[0]:
        return f"📝 **Summary:** {out[0]['summary_text']}"
    return "⚠️ Sorry, I couldn’t summarize that."

def expand_text(text: str):
    out = hf_query("gpt2", text + " -> explain in detail")
    if isinstance(out, list) and "generated_text" in out[0]:
        return f"📖 **Expanded:** {out[0]['generated_text']}"
    return "⚠️ Expansion failed."

def shorten_text(text: str):
    out = hf_query("facebook/bart-large-cnn", text)
    if isinstance(out, list) and "summary_text" in out[0]:
        return f"✂️ **Shortened:** {out[0]['summary_text']}"
    return "⚠️ Shortening failed."

def paraphrase_text(text: str):
    out = hf_query("Vamsi/T5_Paraphrase_Paws", text)
    if isinstance(out, list) and "generated_text" in out[0]:
        return f"🔄 **Paraphrased:** {out[0]['generated_text']}"
    return "⚠️ Paraphrasing failed."

# ===========================
# DRUG INFORMATION (multi-API)
# ===========================
def get_drug_info(drug: str):
    apis = []
    try:
        url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug}&limit=1"
        res = requests.get(url, timeout=5).json()
        if "results" in res:
            apis.append("💊 OpenFDA: " + res["results"][0].get("indications_and_usage", ["No info"])[0])
    except:
        pass
    try:
        url = f"https://rxnav.nlm.nih.gov/REST/drugs.json?name={drug}"
        res = requests.get(url, timeout=5).json()
        if "drugGroup" in res:
            apis.append("💊 RxNav: Found entry for this drug ✅")
    except:
        pass
    try:
        url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/drugnames.json?drug_name={drug}"
        res = requests.get(url, timeout=5).json()
        if "data" in res:
            apis.append("💊 DailyMed: Registered medicine 🧾")
    except:
        pass
    try:
        url = f"https://rximage.nlm.nih.gov/api/rximage/1/rxbase?name={drug}"
        res = requests.get(url, timeout=5).json()
        if "nlmRxImages" in res:
            apis.append("💊 RxImage: Has pill images 📸")
    except:
        pass
    if apis:
        return "\n".join(apis)
    return f"❌ Sorry, no info found for *{drug}*."

# ===========================
# SEARCH & WIKI (multi-API)
# ===========================
def search_wikipedia(query: str):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
        res = requests.get(url, timeout=5).json()
        if "extract" in res:
            return f"📘 Wikipedia: {res['extract']}"
    except:
        pass
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        res = requests.get(url, timeout=5).json()
        if res.get("AbstractText"):
            return f"🔎 DuckDuckGo: {res['AbstractText']}"
    except:
        pass
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{query}"
        res = requests.get(url, timeout=5).json()
        if isinstance(res, list):
            return f"📖 Dictionary: {res[0]['meanings'][0]['definitions'][0]['definition']}"
    except:
        pass
    return f"❌ No results for '{query}'"

# ===========================
# WEATHER
# ===========================
def get_weather_info(city: str):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OWM_KEY}&units=metric"
        res = requests.get(url, timeout=5).json()
        if "main" in res:
            temp = res["main"]["temp"]
            desc = res["weather"][0]["description"]
            return f"🌦️ Weather in {city.title()}: {temp}°C, {desc}"
    except:
        pass
    return f"❌ Couldn’t fetch weather for {city}."

# ===========================
# NEWS
# ===========================
def get_news():
    try:
        url = "https://gnews.io/api/v4/top-headlines?lang=en&token=demo"
        res = requests.get(url, timeout=5).json()
        if "articles" in res:
            title = res["articles"][0]["title"]
            link = res["articles"][0]["url"]
            return f"📰 News: {title}\n🔗 {link}"
    except:
        pass
    return "❌ Couldn’t fetch news."

# ===========================
# DATE & TIME
# ===========================
def get_datetime():
    now = datetime.now()
    return f"🕒 Current Date & Time: {now.strftime('%A, %d %B %Y | %I:%M %p')}"

# ===========================
# HELP MENU
# ===========================
def get_help():
    return (
        "📖 **PharmaCare Bot Commands**\n\n"
        "💊 *Drug Info:*\n"
        "`drug ibuprofen`, `drug paracetamol`\n\n"
        "📘 *Knowledge:*\n"
        "`wiki diabetes`, `define stress`, `search healthy diet`\n\n"
        "🌦️ *Weather:*\n"
        "`weather Lagos`, `weather New York`\n\n"
        "📰 *News:*\n"
        "`news health`, `news technology`\n\n"
        "🧠 *Smart NLP:*\n"
        "`summarize <text>`, `expand <text>`, `shorten <text>`, `paraphrase <text>`\n\n"
        "🎉 *Fun Extras:*\n"
        "`joke`, `fact`\n\n"
        "⚡ Just chat with me naturally too!"
    )

# ===========================
# MAIN ROUTER
# ===========================
def chatbot_response(message: str) -> str:
    msg = message.lower().strip()

    # Greetings / Goodbye
    if any(word in msg for word in ["hello", "hi", "hey", "good morning", "good evening"]):
        return random_greeting()
    elif any(word in msg for word in ["bye", "goodbye", "see you", "later"]):
        return random_goodbye()
    elif "thank" in msg:
        return "🙏 You’re most welcome! Anytime ✨"

    # Fun
    elif "joke" in msg:
        return get_joke()
    elif "fact" in msg:
        return get_fun_fact()

    # NLP
    elif msg.startswith("summarize "):
        return summarize_text(msg.replace("summarize ", "").strip())
    elif msg.startswith("expand "):
        return expand_text(msg.replace("expand ", "").strip())
    elif msg.startswith("shorten "):
        return shorten_text(msg.replace("shorten ", "").strip())
    elif msg.startswith("paraphrase "):
        return paraphrase_text(msg.replace("paraphrase ", "").strip())

    # Knowledge
    elif msg.startswith("drug "):
        return get_drug_info(msg.replace("drug ", "").strip())
    elif msg.startswith("wiki "):
        return search_wikipedia(msg.replace("wiki ", "").strip())
    elif msg.startswith("weather "):
        return get_weather_info(msg.replace("weather ", "").strip())
    elif "news" in msg:
        return get_news()
    elif "time" in msg or "date" in msg:
        return get_datetime()
    elif "help" in msg or msg == "/help":
        return get_help()

    # Fallback
    return random_fallback()










