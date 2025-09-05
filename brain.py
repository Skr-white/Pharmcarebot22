""" pharmacare_bot_with_api_selector.py

Updated bot core that:

Integrates many no-auth public APIs (only calls one best API per intent)

Uses a small priority/heuristic-based selector so we do NOT query every API

Adds a Telegram "typing" (continuous three-dots) indicator context manager that runs in a daemon thread while the bot is processing a message

Keeps existing HF (Hugging Face) helpers but guarded when HF_KEY absent


HOW TO USE

Set environment variables (optional where noted):

TELEGRAM_TOKEN    -> your Telegram bot token (optional unless you use typing/send)

HF_API_KEY        -> Hugging Face API key (optional for HF features)

OPENWEATHER_KEY   -> OpenWeatherMap API key (optional)


Typical integration in your webhook/poller: from pharmacare_bot_with_api_selector import process_message process_message(chat_id, incoming_text)


This file purposely tries a single API per user intent (the first working one). """

import os import requests import random import threading import time from datetime import datetime

========== ENV / KEYS ===========

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # optional (required if using Telegram send/typing helpers) HF_KEY = os.getenv("HF_API_KEY")              # optional (required for HF summarization/expand) OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")

HF_HEADERS = {"Authorization": f"Bearer {HF_KEY}"} if HF_KEY else {} DEFAULT_TIMEOUT = 8

========== UTILITIES ===========

def _get(url, params=None, headers=None, timeout=DEFAULT_TIMEOUT): try: h = headers or {} # Some public APIs dislike empty User-Agent; provide a mild one if "User-Agent" not in h: h["User-Agent"] = "PharmaCareBot/1.0 (+https://example.com)" r = requests.get(url, params=params, headers=h, timeout=timeout) r.raise_for_status() return r.json() if r.text and r.headers.get("content-type", "").startswith("application/json") else r.text except Exception as e: # keep silent on failures, return None return None


def get_app_overview():
    return (
        "👋 <b>Welcome to PharmaCare Bot!</b>\n\n"
        "I’m your smart assistant for quick health info, knowledge, fun facts, and more.\n\n"
        "Here’s what I can do:\n\n"
        "💊 <b>Drug Info</b>\n"
        "Ask about any medicine: <code>drug ibuprofen</code>\n\n"
        "📘 <b>Knowledge & Search</b>\n"
        "Look up Wikipedia, dictionary, or quick answers.\n"
        "Example: <code>wiki diabetes</code> or <code>search blockchain</code>\n\n"
        "🌦️ <b>Weather</b>\n"
        "Check live weather: <code>weather Lagos</code>\n\n"
        "📰 <b>News</b>\n"
        "Get the latest headlines: <code>news</code>\n\n"
        "🧠 <b>Smart NLP</b>\n"
        "Summarize text (needs HF key): <code>summarize climate change article</code>\n\n"
        "🎉 <b>Fun</b>\n"
        "Get a <code>joke</code> or a <code>fact</code>\n\n"
        "🕒 <b>Date & Time</b>\n"
        "Ask me for the current time/date.\n\n"
        "👤 <b>Name Guess</b>\n"
        "Predict age, gender, country: <code>guess John</code>\n\n"
        "🏛️ <b>Universities</b>\n"
        "Find universities: <code>universities in Canada</code>\n\n"
        "🏠 <b>Zip Lookup</b>\n"
        "Get US city/state from zip: <code>zip 90210</code>\n\n"
        "👥 <b>Random User</b>\n"
        "Get a random profile: <code>random user</code>\n\n"
        "🎵 <b>Music</b>\n"
        "Search for artists: <code>artist Beyonce</code>\n\n"
        "🍎 <b>Food</b>\n"
        "Search OpenFoodFacts: <code>food chocolate</code>\n\n"
        "🌍 <b>Countries</b>\n"
        "Look up country info: <code>country Japan</code>\n\n"
        "⚡ <b>How to Use</b>\n"
        "Type naturally or use these commands. I’ll respond right away 🚀\n\n"
        "— Your companion, <b>PharmaCare Bot</b> 🤖💚"
    )
========== TYPING INDICATOR (Telegram) ==========

class TypingIndicator: """Context manager that sends repeated sendChatAction("typing") calls to Telegram

Usage:
    with TypingIndicator(telegram_token, chat_id):
        # long processing here
        resp = chatbot_response(text)

Notes:
- TELEGRAM_TOKEN must be set.
- This starts a daemon thread that repeatedly posts the typing action every `interval` seconds.
- Too aggressive intervals may hit rate limits — default 3s is reasonable.
"""
def __init__(self, token, chat_id, interval=3.0):
    self.token = token
    self.chat_id = chat_id
    self.interval = interval
    self._stop = threading.Event()
    self._thread = None

def _worker(self):
    url = f"https://api.telegram.org/bot{self.token}/sendChatAction"
    payload = {"chat_id": self.chat_id, "action": "typing"}
    while not self._stop.is_set():
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception:
            # ignore network errors here; the bot still works without typing action
            pass
        # sleep but wake early if stopped
        self._stop.wait(self.interval)

def __enter__(self):
    if not self.token or not self.chat_id:
        # nothing to do
        return self
    self._thread = threading.Thread(target=self._worker, daemon=True)
    self._thread.start()
    return self

def __exit__(self, exc_type, exc, tb):
    if self._thread is None:
        return
    self._stop.set()
    # give the thread a moment to finish
    self._thread.join(timeout=1.0)

========== HUGGING FACE HELPERS (guarded) ==========

def hf_query(model: str, text: str): if not HF_KEY: return {"error": "HF key missing"} try: url = f"https://api-inference.huggingface.co/models/{model}" res = requests.post(url, headers=HF_HEADERS, json={"inputs": text}, timeout=10) res.raise_for_status() return res.json() except Exception as e: return {"error": str(e)}

def summarize_text(text: str): out = hf_query("facebook/bart-large-cnn", text) if isinstance(out, list) and "summary_text" in out[0]: return f"📝 Summary: {out[0]['summary_text']}" return "⚠️ Summarize failed (HF key missing or model error)."

========== SMALL TALK & FUN (single-best API per intent) ==========

def get_random_joke(): """Try Official Joke API, then fallback to a short local fallback.""" url = "https://official-joke-api.appspot.com/random_joke" res = _get(url) if isinstance(res, dict) and res.get("setup"): return f"{res['setup']}\n{res.get('punchline','')}" # fallback jokes = [ "Why don’t scientists trust atoms? Because they make up everything!", "I told my computer I needed a break, and it said: 'No problem — I'll go to sleep.'", ] return random.choice(jokes)

def get_random_fact(): # prefer cat facts when user asks for 'fact' about animals, otherwise use numbersapi for trivia res = _get("https://catfact.ninja/fact") if isinstance(res, dict) and res.get("fact"): return res["fact"] # fallback to numbers trivia res2 = _get("http://numbersapi.com/random/trivia") if isinstance(res2, str): return res2 return "Fun fact: Honey never spoils!"

========== NAME GUESS APIS (agify/genderize/nationalize) ==========

def guess_name_attributes(name: str): name = name.split()[0] apis = [ lambda: _get(f"https://api.agify.io?name={name}"), lambda: _get(f"https://api.genderize.io?name={name}"), lambda: _get(f"https://api.nationalize.io?name={name}"), ] # try each API in order and return the first useful result result = {} for call in apis: r = call() if not r: continue # merge results into a friendly string if 'age' in r: result['age'] = r.get('age') if 'gender' in r: result['gender'] = r.get('gender') if 'country' in r: if isinstance(r['country'], list) and r['country']: result['country'] = r['country'][0].get('country_id') if result: out = [] if 'age' in result: out.append(f"Age guess: {result['age']}") if 'gender' in result: out.append(f"Gender guess: {result['gender']}") if 'country' in result: out.append(f"Likely country: {result['country']}") return ' | '.join(out) return f"Couldn't guess attributes for {name}."

========== RANDOM USER / PLACEHOLDER APIS ==========

def get_random_user_profile(): res = _get("https://randomuser.me/api/") if isinstance(res, dict) and res.get('results'): u = res['results'][0] return f"{u['name']['title']} {u['name']['first']} {u['name']['last']}, {u.get('email')} - {u['location']['country']}" return "Random user could not be fetched."

========== IP / LOCATION / UNIVERSITIES / ZIP ==========

def get_my_ip(): res = _get("https://api.ipify.org?format=json") if isinstance(res, dict) and res.get('ip'): return f"Your IP is: {res['ip']}" return None

def lookup_universities(country: str): res = _get(f"http://universities.hipolabs.com/search?country={country}") if isinstance(res, list) and res: first = res[0] return f"{len(res)} universities found. Example: {first.get('name')} ({first.get('web_pages', ['n/a'])[0]})" return None

def lookup_zip_us(zipcode: str): res = _get(f"https://api.zippopotam.us/us/{zipcode}") if isinstance(res, dict): places = res.get('places', [{}]) p = places[0] return f"{p.get('place name')}, {p.get('state')}" return None

========== PROTOTYPE / FAKE DATA (JSONPlaceholder) ==========

def jsonplaceholder_post(post_id=1): res = _get(f"https://jsonplaceholder.typicode.com/posts/{post_id}") if isinstance(res, dict): return f"Post {res['id']}: {res['title']}\n{res['body']}" return None

========== MUSIC & FOOD ==========

def musicbrainz_artist(query: str): res = _get(f"https://musicbrainz.org/ws/2/artist/?query=artist:{query}&fmt=json") if isinstance(res, dict) and res.get('artists'): a = res['artists'][0] return f"Artist: {a.get('name')} — Score: {a.get('score')}" return None

def openfoodfacts_search(query: str): params = {"search_terms": query, "search_simple": 1, "json": 1} res = _get("https://world.openfoodfacts.org/cgi/search.pl", params=params) if isinstance(res, dict) and res.get('products'): prod = res['products'][0] return f"{prod.get('product_name', 'Unnamed')} — brands: {prod.get('brands') or 'n/a'}" return None

========== COUNTRY DATA ==========

def restcountries_lookup(name: str): res = _get(f"https://restcountries.com/v3.1/name/{name}") if isinstance(res, list) and res: c = res[0] return f"{c.get('name',{}).get('common')} — Capital: {c.get('capital', ['n/a'])[0]} — Population: {c.get('population')}" return None

========== WEATHER (OpenWeatherMap OR Open-Meteo) ==========

def openweather_city(city: str): if not OPENWEATHER_KEY: return None try: url = "https://api.openweathermap.org/data/2.5/weather" params = {"q": city, "appid": OPENWEATHER_KEY, "units": "metric"} r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT) r.raise_for_status() j = r.json() temp = j['main']['temp'] desc = j['weather'][0]['description'] return f"Weather in {city.title()}: {temp}°C — {desc}" except Exception: return None

def open_meteo_city(city: str): # geocode via open-meteo geocoding (no key), then current_weather g = _get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1") if not g or not g.get('results'): return None r = g['results'][0] lat = r.get('latitude') lon = r.get('longitude') if lat is None or lon is None: return None w = _get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true") if not w or not w.get('current_weather'): return None cw = w['current_weather'] temp = cw.get('temperature') wind = cw.get('windspeed') return f"Weather in {r.get('name')}: {temp}°C — wind {wind} m/s"

def get_weather_info(city: str): # prefer OpenWeather if key present, otherwise Open-Meteo for fn in (lambda: openweather_city(city), lambda: open_meteo_city(city)): resp = fn() if resp: return resp return f"❌ Couldn't fetch weather for {city}."

========== NEWS (single best API) ==========

def reddit_top_news(subreddit='news'): url = f"https://www.reddit.com/r/{subreddit}/top.json" params = {"limit": 1, "t": "day"} res = _get(url, params=params) if isinstance(res, dict) and res.get('data'): children = res['data'].get('children') if children: art = children[0]['data'] return f"{art.get('title')}\nhttps://reddit.com{art.get('permalink')}" return None

def get_news(): # try reddit then the gnews demo (if reddit fails) for fn in (lambda: reddit_top_news('news'), lambda: None): r = fn() if r: return f"📰 {r}" return "❌ Couldn't fetch news."

========== MEDICINE / DRUG INFO (tries prioritized list) ==========

def openfda_drug(drug: str): q = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug}&limit=1" res = _get(q) if isinstance(res, dict) and res.get('results'): first = res['results'][0] usage = first.get('indications_and_usage') if usage: return "OpenFDA: " + (usage[0] if isinstance(usage, list) else str(usage)) return None

def rxnav_drug(drug: str): res = _get(f"https://rxnav.nlm.nih.gov/REST/drugs.json?name={drug}") if isinstance(res, dict) and res.get('drugGroup'): return f"RxNav: Found matches for {drug}." return None

def dailymed_drug(drug: str): res = _get(f"https://dailymed.nlm.nih.gov/dailymed/services/v2/drugnames.json?drug_name={drug}") if isinstance(res, dict) and res.get('data'): return f"DailyMed: Registered medicine — sample name: {res['data'][0].get('name') if res['data'] else ''}" return None

def get_drug_info(drug: str): # priority: OpenFDA -> RxNav -> DailyMed for fn in (lambda: openfda_drug(drug), lambda: rxnav_drug(drug), lambda: dailymed_drug(drug)): r = fn() if r: return r return f"❌ Sorry, no info found for {drug}."

========== WIKI/SEARCH ==========

def search_wikipedia(query: str): # prefer Wikipedia summary then DuckDuckGo instant answer then dictionary q = query.replace(' ', '_') res = _get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}") if isinstance(res, dict) and res.get('extract'): return f"📘 {res.get('extract')}" res2 = _get(f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1") if isinstance(res2, dict) and res2.get('AbstractText'): return f"🔎 {res2.get('AbstractText')}" # dictionary res3 = _get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{query}") if isinstance(res3, list): try: return f"📖 {res3[0]['meanings'][0]['definitions'][0]['definition']}" except Exception: pass return f"❌ No results for '{query}'"

========== MAIN ROUTER (keeps high-level behavior) ==========

def chatbot_response(message: str) -> str: msg = message.lower().strip()

# greetings / goodbye
if any(word in msg for word in ["hello", "hi", "hey", "good morning", "good evening"]):
    return random.choice(["👋 Hey there!", "🌟 Hi! What’s up?", "Hello friend — what can I help you with?"])
if any(word in msg for word in ["bye", "goodbye", "see you", "later"]):
    return random.choice(["👋 Goodbye!", "✨ See you later!", "🚀 Bye-bye!"])
if "thank" in msg:
    return "🙏 You’re welcome!"
# overview/help
if msg in ("/start", "help", "menu", "overview"):
    return get_app_overview()
# fun
if msg == "joke" or msg.startswith("tell me a joke"):
    return get_random_joke()
if msg == "fact" or msg.startswith("fun fact"):
    return get_random_fact()

# nlp (HF) - only run if key present
if msg.startswith("summarize "):
    return summarize_text(message[len("summarize "):].strip())

# drug info
if msg.startswith("drug "):
    target = message[len("drug "):].strip()
    return get_drug_info(target)

# wiki/search
if msg.startswith("wiki ") or msg.startswith("search "):
    q = message.split(' ', 1)[1].strip() if ' ' in message else message
    return search_wikipedia(q)

# weather
if msg.startswith("weather "):
    city = message[len("weather "):].strip()
    return get_weather_info(city)

# name guess (agify/genderize/nationalize)
if msg.startswith("guess ") or msg.startswith("who is " ):
    # e.g. 'guess john' -> pass john
    name = message.split(' ', 1)[1] if ' ' in message else message
    return guess_name_attributes(name)

# misc quick helpers
if msg.startswith("ip") or msg == "my ip":
    return get_my_ip() or "Couldn't get your IP."
if msg.startswith("random user"):
    return get_random_user_profile()
if msg.startswith("university") or msg.startswith("universities"):
    # 'universities in United States' -> extract country
    parts = message.split(' in ')
    country = parts[1] if len(parts) > 1 else 'United States'
    return lookup_universities(country)
if msg.startswith("zip ") or msg.startswith("zipcode "):
    z = message.split(' ',1)[1]
    return lookup_zip_us(z)
if msg.startswith("post "):
    pid = message.split(' ',1)[1] if ' ' in message else '1'
    return jsonplaceholder_post(pid)

if 'news' in msg:
    return get_news()

# default fallback
return random.choice([
    "I didn't fully catch that — can you try rephrasing?",
    "Hmm, not sure. Try 'wiki <topic>' or 'drug <name>' or 'weather <city>'.",
])

========== TELEGRAM HELPERS (optional) ==========

def send_telegram_message(chat_id: int, text: str, parse_mode: str = 'HTML'): if not TELEGRAM_TOKEN: raise RuntimeError('TELEGRAM_TOKEN not set') url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage" payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode} try: requests.post(url, json=payload, timeout=6) except Exception: pass

def process_message(chat_id: int, text: str): """High-level helper you can call from your webhook/poller. It shows typing, computes response, and sends it back to Telegram (if token provided). """ # Use TypingIndicator so the user sees continuous three-dots while we compute with TypingIndicator(TELEGRAM_TOKEN, chat_id): resp = chatbot_response(text) try: if TELEGRAM_TOKEN: send_telegram_message(chat_id, resp) else: # if no Telegram token is set, simply return the response value return resp except Exception: # last-resort return resp

========== END OF FILE ==========

if name == 'main': # small quick local test print('Local quick tests:') print('Joke ->', chatbot_response('joke')) print('Fact ->', chatbot_response('fact')) print('Weather Lagos ->', chatbot_response('weather Lagos')) print('Drug aspirin ->', chatbot_response('drug aspirin')) print('Wiki Python ->', chatbot_response('wiki Python (programming language)'))

