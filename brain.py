# brain.py (with caching added)
import os, re, random, requests, threading, time
from datetime import datetime
from urllib.parse import quote_plus

# ---- ENV KEYS ----
HF_KEY  = os.getenv("HF_API_KEY")
OWM_KEY = os.getenv("WEATHER_API_KEY")
HF_HEADERS = {"Authorization": f"Bearer {HF_KEY}"} if HF_KEY else {}

# ---- Cache helper (TTL-based) ----
class TTLCache:
    def __init__(self, ttl=300, maxsize=256):
        self.ttl = ttl
        self.maxsize = maxsize
        self.cache = {}

    def get(self, key):
        val = self.cache.get(key)
        if not val: return None
        expire, result = val
        if time.time() > expire:
            del self.cache[key]
            return None
        return result

    def set(self, key, result):
        if len(self.cache) >= self.maxsize:
            self.cache.pop(next(iter(self.cache)))  # drop oldest
        self.cache[key] = (time.time() + self.ttl, result)

_api_cache = TTLCache(ttl=300, maxsize=300)

# ---- Helpers ----
def _http_get(url, headers=None, timeout=8):
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        return r if r.ok else None
    except:
        return None

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

# ---- Small talk ----
def random_greeting():
    return random.choice([
        "👋 Hey there, how’s your day?",
        "😃 Hi! Ready to chat?",
        "🙌 Hello! Always happy to see you.",
        "🌟 Hey hey! What’s new with you?",
    ])

def random_goodbye():
    return random.choice([
        "👋 Goodbye! Stay safe and hydrated 💧",
        "✨ Catch you later, take care!",
        "🚀 Bye-bye, keep pushing forward!",
    ])

def random_fallback():
    return random.choice([
        "🤔 I didn’t quite get that. Try `/help`.",
        "🧠 Hmmm, can you rephrase?",
        "I’m learning every day! Maybe try another way.",
    ])

# ---- Fun APIs ----
def get_joke():
    key = "joke"
    cached = _api_cache.get(key)
    if cached: return cached
    r = _http_get("https://official-joke-api.appspot.com/random_joke")
    if r:
        j = r.json()
        result = f"🤣 {j['setup']} — {j['punchline']}"
    else:
        result = "🤣 No jokes right now."
    _api_cache.set(key, result)
    return result

def get_cat_fact():
    key = "catfact"
    cached = _api_cache.get(key)
    if cached: return cached
    r = _http_get("https://catfact.ninja/fact")
    result = f"🐱 {r.json().get('fact')}" if r else "🐱 Cat is sleeping."
    _api_cache.set(key, result)
    return result

def get_activity():
    key = "activity"
    cached = _api_cache.get(key)
    if cached: return cached
    r = _http_get("https://www.boredapi.com/api/activity")
    result = f"🎯 {r.json().get('activity')}" if r else "🎯 Can’t think of anything."
    _api_cache.set(key, result)
    return result

def get_dog():
    r = _http_get("https://dog.ceo/api/breeds/image/random")
    return r.json().get("message") if r else "🐶 Dog not found."

def number_fact(num="random"):
    key = f"num:{num}"
    cached = _api_cache.get(key)
    if cached: return cached
    r = _http_get(f"http://numbersapi.com/{quote_plus(num)}/trivia?json")
    result = f"🔢 {r.json().get('text')}" if r else "No number fact."
    _api_cache.set(key, result)
    return result

# ---- HuggingFace NLP ----
def _hf(model, text):
    if not HF_KEY: return {"error":"HF key missing"}
    try:
        r = requests.post(f"https://api-inference.huggingface.co/models/{model}",
                          headers=HF_HEADERS, json={"inputs": text}, timeout=25)
        return r.json() if r.ok else {"error":r.text}
    except Exception as e:
        return {"error": str(e)}

def summarize_text(text):
    out = _hf("facebook/bart-large-cnn", text)
    return f"📝 {out[0]['summary_text']}" if isinstance(out,list) and out else "⚠️ Summarize failed."

def expand_text(text):
    out = _hf("gpt2", text+" -> explain in detail")
    return f"📖 {out[0]['generated_text']}" if isinstance(out,list) and out else "⚠️ Expand failed."

def paraphrase_text(text):
    out = _hf("Vamsi/T5_Paraphrase_Paws", text)
    return f"🔄 {out[0]['generated_text']}" if isinstance(out,list) and out else "⚠️ Paraphrase failed."

# ---- Drug Info ----
def openfda_drug(drug: str):
    r = _http_get(f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{quote_plus(drug)}&limit=1")
    if r and "results" in r.json():
        d = r.json()["results"][0]
        return f"OpenFDA: {d.get('indications_and_usage',['No info'])[0]}"
    return None

def rxnav_drug(drug: str):
    r = _http_get(f"https://rxnav.nlm.nih.gov/REST/drugs.json?name={quote_plus(drug)}")
    if r and r.json().get("drugGroup",{}).get("conceptGroup"):
        return f"RxNav: Found matches for {drug}."
    return None

def dailymed_drug(drug: str):
    r = _http_get(f"https://dailymed.nlm.nih.gov/dailymed/services/v2/drugnames.json?drug_name={quote_plus(drug)}")
    if r and r.json().get("data"):
        return f"DailyMed: Registered medicine."
    return None

def get_drug_info(drug: str):
    key = f"drug:{drug.lower()}"
    cached = _api_cache.get(key)
    if cached: return cached
    for fn in (lambda: openfda_drug(drug), lambda: rxnav_drug(drug), lambda: dailymed_drug(drug)):
        r = fn()
        if r:
            _api_cache.set(key, r)
            return r
    result = f"❌ Sorry, no info found for {drug}."
    _api_cache.set(key, result)
    return result

# ---- Wiki/Search ----
def search_wikipedia(query: str):
    key = f"wiki:{query.lower()}"
    cached = _api_cache.get(key)
    if cached: return cached
    query=_clean(query)
    if not query: return "📘 Give me a topic."
    r=_http_get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(query)}")
    if r and "extract" in r.json():
        result = f"📘 {r.json()['extract']}"
        _api_cache.set(key, result)
        return result
    r=_http_get(f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json")
    if r and r.json().get("AbstractText"):
        result = f"🔎 {r.json()['AbstractText']}"
        _api_cache.set(key, result)
        return result
    result = f"❌ No info on {query}"
    _api_cache.set(key, result)
    return result

# ---- Weather ----
def openweather_city(city: str):
    if not OWM_KEY: return None
    try:
        r=_http_get(f"http://api.openweathermap.org/data/2.5/weather?q={quote_plus(city)}&appid={OWM_KEY}&units=metric")
        if r and "main" in r.json():
            j=r.json(); return f"🌦 {city.title()}: {j['main']['temp']}°C, {j['weather'][0]['description']}"
    except: return None
    return None

def open_meteo_city(city: str):
    g = _http_get(f"https://geocoding-api.open-meteo.com/v1/search?name={quote_plus(city)}&count=1")
    if not g or not g.json().get('results'): return None
    loc = g.json()['results'][0]
    lat, lon = loc['latitude'], loc['longitude']
    w = _http_get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true")
    if w and w.json().get("current_weather"):
        cw = w.json()["current_weather"]
        return f"🌦 {loc['name']}: {cw['temperature']}°C, wind {cw['windspeed']} m/s"
    return None

def get_weather_info(city: str):
    key = f"weather:{city.lower()}"
    cached = _api_cache.get(key)
    if cached: return cached
    for fn in (lambda: openweather_city(city), lambda: open_meteo_city(city)):
        resp = fn()
        if resp:
            _api_cache.set(key, resp)
            return resp
    result = f"❌ Couldn't fetch weather for {city}."
    _api_cache.set(key, result)
    return result

# ---- News ----
def get_news(topic: str = None):
    key = f"news:{(topic or '').lower()}"
    cached = _api_cache.get(key)
    if cached: return cached
    if topic:
        r = _http_get(f"https://gnews.io/api/v4/search?q={quote_plus(topic)}&token=demo&lang=en")
    else:
        r = _http_get("https://gnews.io/api/v4/top-headlines?token=demo&lang=en")
    if r and r.json().get("articles"):
        a = r.json()["articles"][0]
        result = f"📰 {a.get('title')}\n{a.get('url')}"
    else:
        result = "❌ No news."
    _api_cache.set(key, result)
    return result

# ---- Main Brain ----
def chatbot_response(msg:str)->str:
    text=_clean(msg).lower()
    if not text: return "Say something so I can help 😊"

    # greetings
    if any(w in text for w in ["hello","hi","hey"]): return random_greeting()
    if any(w in text for w in ["bye","goodbye"]): return random_goodbye()
    if "how are you" in text: return "I’m doing great 🤖 thanks for asking!"

    # fun
    if "joke" in text: return get_joke()
    if "cat fact" in text: return get_cat_fact()
    if "activity" in text or "bored" in text: return get_activity()
    if "dog" in text: return get_dog()
    if "number" in text: return number_fact(text.split()[-1])

    # nlp
    if "summarize" in text: return summarize_text(msg)
    if "expand" in text: return expand_text(msg)
    if "paraphrase" in text or "rephrase" in text: return paraphrase_text(msg)
    if "shorten" in text: return summarize_text(msg)

    # info
    if "drug" in text or "tablet" in text or "medicine" in text: 
        return get_drug_info(msg.split()[-1])
    if "weather" in text or "temperature" in text or "forecast" in text: 
        return get_weather_info(msg.split()[-1])
    if "news" in text or "headline" in text: 
        return get_news()
    if "wiki" in text or "what is" in text or "tell me about" in text: 
        return search_wikipedia(msg.split()[-1])

    return random_fallback()