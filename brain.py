# brain.py
"""
PharmaCare Bot brain module.

Exports:
 - HELP_TEXT (string) for help command
 - chatbot_response(user_text: str) -> str : main entrypoint used by bot.py

Environment variables:
 - HF_API_KEY (optional)   : Hugging Face API key (if you want HF features)
 - HF_MODEL (optional)     : Hugging Face model for generation/planning (default: google/flan-t5-small)
 - WEATHER_API_KEY or OPENWEATHER_KEY (optional) : OpenWeatherMap key
"""

import os
import re
import time
import json
import random
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime
from urllib.parse import quote_plus

# ---------------- CONFIG & KEYS ----------------
HF_KEY = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")
OWM_KEY = os.getenv("WEATHER_API_KEY") or os.getenv("OPENWEATHER_KEY")
CACHE_TTL = int(os.getenv("BRAIN_CACHE_TTL", "300"))
DEFAULT_TIMEOUT = 8
MAX_REPLY_CHARS = int(os.getenv("BRAIN_MAX_REPLY_CHARS", "3000"))

HF_HEADERS = {"Authorization": f"Bearer {HF_KEY}"} if HF_KEY else {}

# ---------------- HELP TEXT (imported by bot.py) ----------------

# Long welcome/write-up shown on /start
START_TEXT = (
    "👋 *Welcome to PharmaCare Bot!* \n\n"
    "I’m your friendly health & knowledge assistant. You can talk to me naturally —\n"
    "for example: *Tell me about malaria*, *Is it raining in Lagos?*, or *drug ibuprofen*.\n\n"
    "Below is a short guide so you know everything I can do and how to ask.\n\n"
    "— *Quick example commands*\n"
    "`/help` — show the full command list\n"
    "`wiki <topic>` — get a short summary from Wikipedia\n"
    "`drug <name>` — fetch drug info from medical APIs (OpenFDA / RxNav / DailyMed)\n"
    "`weather <city>` — get current weather for a city\n"
    "`time` — show current time (UTC/local)\n"
    "`news` — top headline\n"
    "`summarize <text>` — short summary (uses Hugging Face if configured)\n"
    "`expand <text>` — explain or expand text (uses Hugging Face if configured)\n"
    "`paraphrase <text>` — rephrase text (HF if available)\n"
    "`joke`, `cat fact`, `activity` — fun quick endpoints\n\n"
    "— *How I choose an API*\n"
    "For each question I try a single best API for that type of request (for speed and reliability).\n"
    "For example:\n"
    "• Weather → OpenWeather (if API key configured) → Open-Meteo → wttr.in\n"
    "• Drug info → OpenFDA → RxNav → DailyMed\n"
    "• Knowledge → Wikipedia → DuckDuckGo → Dictionary API\n\n"
    "— *Hugging Face (optional)*\n"
    "If the bot owner has set a Hugging Face API key, I will use it to:\n"
    "• understand complex English requests,\n"
    "• decide which tool to call when ambiguous,\n"
    "• rewrite long tool output into a natural human reply,\n"
    "• summarize/expand/paraphrase text.\n\n"
    "If HF is *not* configured, I still try the best web API and give helpful fallbacks.\n\n"
    "— *Privacy & Tokens*\n"
    "I never ask for tokens in chat. If an API key isn't working, the owner must set it in the server environment and restart the bot.\n\n"
    "If you want help or usage examples, type `/help`.\n"
)

# Detailed /help text (commands reference + API quick examples)
HELP_TEXT = (
    "📜 *PharmaCare Bot — Commands & Examples*\n\n"
    "💬 *Chat naturally* – just write a full sentence: \n"
    "  • \"How are you?\"\n"
    "  • \"Tell me about malaria\"\n\n"
    "💊 *Drug information* — medical APIs\n"
    "  • `drug paracetamol` — find indications/warnings (OpenFDA / RxNav / DailyMed)\n\n"
    "📘 *Knowledge & definitions* — wiki/search/dictionary\n"
    "  • `wiki diabetes` — short encyclopedia summary\n"
    "  • `define anemia` — dictionary definition\n\n"
    "🌦 *Weather & time* — live data\n"
    "  • `weather Lagos` — current weather\n"
    "  • `is it raining in Lagos?` — natural language works\n"
    "  • `time` — current UTC/local time\n\n"
    "📰 *News* — headlines\n"
    "  • `news` — top headline (reddit/gnews fallback)\n\n"
    "🧠 *NLP (Hugging Face)* — requires HF API key to be configured\n"
    "  • `summarize <text>` — concise summary\n"
    "  • `expand <text>` — expand/explain in plain English\n"
    "  • `paraphrase <text>` — rewrite\n\n"
    "🎲 *Fun & utilities*:\n"
    "  • `joke` — random joke\n"
    "  • `cat fact` — random cat fact\n"
    "  • `activity` — bored API suggestion\n"
    "  • `random user` — fake profile\n"
    "  • `number <n>` — number trivia\n\n"
    "🗺 *Maps & location*:\n"
    "  • `map Lagos` — static map link (OpenStreetMap)\n"
    "  • `zip 90210` — US zip lookup\n\n"
    "⚙️ *Notes for power users (owner only)*:\n"
    "  • HF key and model in environment: `HF_API_KEY` and `HF_MODEL`.\n"
    "  • OpenWeather key: `WEATHER_API_KEY` or `OPENWEATHER_KEY`.\n"
    "  • After changing env vars, *restart the bot process*.\n\n"
    "If something doesn't work, try again or ask the owner to check logs and env variables."
)

# ---------------- TTL cache ----------------
class TTLCache:
    def __init__(self, ttl: int = CACHE_TTL):
        self.ttl = ttl
        self.d: Dict[str, Any] = {}

    def get(self, k: str):
        v = self.d.get(k)
        if not v:
            return None
        exp, val = v
        if time.time() > exp:
            self.d.pop(k, None)
            return None
        return val

    def set(self, k: str, val: Any):
        # simple eviction strategy: keep insertion order small
        if len(self.d) > 1000:
            self.d.pop(next(iter(self.d)))
        self.d[k] = (time.time() + self.ttl, val)

_cache = TTLCache()

# ---------------- HTTP helper ----------------
def _safe_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None,
              timeout: int = DEFAULT_TIMEOUT, retries: int = 2) -> Optional[requests.Response]:
    headers = dict(headers or {})
    if "User-Agent" not in headers:
        headers["User-Agent"] = "PharmaCareBot/1.0 (+https://example.com)"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            # don't retry on 4xx besides maybe transient
        except requests.RequestException:
            pass
        time.sleep(0.25 * (attempt + 1))
    return None

def _shorten(text: str, limit: int = MAX_REPLY_CHARS) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit-3].rstrip() + "..."

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

# ---------------- Hugging Face helpers (optional) ----------------
def hf_query_raw(model: str, payload: dict, retries: int = 2, timeout: int = 20) -> Any:
    if not HF_KEY:
        return {"error": "hf_key_missing"}
    url = f"https://api-inference.huggingface.co/models/{model}"
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=HF_HEADERS, json=payload, timeout=timeout)
            r.raise_for_status()
            try:
                return r.json()
            except Exception:
                return r.text
        except Exception as e:
            if attempt + 1 == retries:
                return {"error": str(e)}
            time.sleep(0.5 + attempt * 0.25)
    return {"error": "unknown"}

def _extract_generated_text(hf_resp: Any) -> str:
    try:
        if isinstance(hf_resp, list) and hf_resp:
            first = hf_resp[0]
            if isinstance(first, dict):
                if "generated_text" in first:
                    return first["generated_text"]
                if "summary_text" in first:
                    return first["summary_text"]
            return str(first)
        if isinstance(hf_resp, dict) and "generated_text" in hf_resp:
            return hf_resp["generated_text"]
        return str(hf_resp)
    except Exception:
        return str(hf_resp)

# ---------------- Tools (single best) ----------------
# Weather tools
def tool_openweather(city: str) -> Optional[str]:
    if not OWM_KEY:
        return None
    r = _safe_get("https://api.openweathermap.org/data/2.5/weather",
                  params={"q": city, "appid": OWM_KEY, "units": "metric"})
    if r:
        j = r.json()
        if "main" in j:
            return f"🌦 {city.title()}: {j['main']['temp']}°C, {j['weather'][0]['description']}"
    return None

def tool_open_meteo(city: str) -> Optional[str]:
    g = _safe_get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1})
    if g:
        gj = g.json()
        if gj.get("results"):
            loc = gj["results"][0]
            lat, lon = loc["latitude"], loc["longitude"]
            r = _safe_get("https://api.open-meteo.com/v1/forecast",
                          params={"latitude": lat, "longitude": lon, "current_weather": True})
            if r and r.json().get("current_weather"):
                cw = r.json()["current_weather"]
                return f"🌦 {loc.get('name')}: {cw.get('temperature')}°C, wind {cw.get('windspeed')} m/s"
    return None

def tool_wttr_in(city: str) -> Optional[str]:
    r = _safe_get(f"https://wttr.in/{quote_plus(city)}", params={"format": "3"})
    if r:
        return r.text
    return None

# Knowledge tools
def tool_wikipedia(query: str) -> Optional[str]:
    q = quote_plus(query.replace(" ", "_"))
    r = _safe_get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}")
    if r:
        j = r.json()
        if j.get("extract"):
            return f"📘 {j.get('title','')}: {_shorten(j['extract'], 1200)}"
    return None

def tool_duckduckgo(query: str) -> Optional[str]:
    r = _safe_get("https://api.duckduckgo.com/", params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1})
    if r:
        j = r.json()
        text = j.get("AbstractText") or j.get("Definition") or ""
        if text:
            return f"🔎 {_shorten(text, 1200)}"
    return None

def tool_dictionary(query: str) -> Optional[str]:
    r = _safe_get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{quote_plus(query)}")
    if r:
        try:
            j = r.json()
            if isinstance(j, list) and j:
                d = j[0]["meanings"][0]["definitions"][0]["definition"]
                return f"📖 {query.title()}: {d}"
        except Exception:
            pass
    return None

# Drug / medicine tools
def tool_openfda(drug: str) -> Optional[str]:
    r = _safe_get("https://api.fda.gov/drug/label.json",
                  params={"search": f"openfda.brand_name:{quote_plus(drug)}", "limit": 1})
    if r:
        j = r.json()
        if j.get("results"):
            d = j["results"][0]
            usage = d.get("indications_and_usage", ["No usage info"])[0]
            warnings = d.get("warnings", [""])[0]
            return f"💊 {drug.title()} (OpenFDA): {_shorten(usage,700)}\nWarnings: {_shorten(warnings,300)}"
    return None

def tool_rxnav(drug: str) -> Optional[str]:
    r = _safe_get("https://rxnav.nlm.nih.gov/REST/drugs.json", params={"name": drug})
    if r and r.json().get("drugGroup", {}).get("conceptGroup"):
        return f"💊 {drug.title()} — found in RxNav"
    return None

def tool_dailymed(drug: str) -> Optional[str]:
    r = _safe_get("https://dailymed.nlm.nih.gov/dailymed/services/v2/drugnames.json", params={"drug_name": drug})
    if r and r.json().get("data"):
        return f"💊 {drug.title()} — listed in DailyMed"
    return None

# News / misc tools
def tool_reddit_top(_: str = "news") -> Optional[str]:
    r = _safe_get("https://www.reddit.com/r/news/top.json", params={"limit": 1, "t": "day"})
    if r:
        try:
            children = r.json().get("data", {}).get("children") or []
            if children:
                art = children[0]["data"]
                return f"📰 {art.get('title')}\nhttps://reddit.com{art.get('permalink')}"
        except Exception:
            pass
    return None

def tool_gnews_demo(topic: Optional[str] = None) -> Optional[str]:
    base = "https://gnews.io/api/v4/top-headlines"
    params = {"token": "demo", "lang": "en"}
    if topic:
        base = "https://gnews.io/api/v4/search"
        params["q"] = topic
    r = _safe_get(base, params=params)
    if r and r.json().get("articles"):
        a = r.json()["articles"][0]
        return f"📰 {a.get('title')}\n{a.get('url')}"
    return None

# Fun / misc
def tool_joke(_: str = "") -> Optional[str]:
    r = _safe_get("https://official-joke-api.appspot.com/random_joke")
    if r:
        j = r.json()
        return f"🤣 {j.get('setup')} — {j.get('punchline')}"
    return None

def tool_catfact(_: str = "") -> Optional[str]:
    r = _safe_get("https://catfact.ninja/fact")
    if r:
        return f"🐱 {r.json().get('fact')}"
    return None

def tool_bored(_: str = "") -> Optional[str]:
    r = _safe_get("https://www.boredapi.com/api/activity")
    if r:
        return f"🎯 {r.json().get('activity')}"
    return None

def tool_dog(_: str = "") -> Optional[str]:
    r = _safe_get("https://dog.ceo/api/breeds/image/random")
    if r:
        return r.json().get("message")
    return None

def tool_numbers(num: str = "random") -> Optional[str]:
    r = _safe_get(f"http://numbersapi.com/{quote_plus(num)}/trivia?json")
    if r:
        return f"🔢 {r.json().get('text')}"
    return None

def tool_random_user(_: str = "") -> Optional[str]:
    r = _safe_get("https://randomuser.me/api/")
    if r:
        u = r.json().get("results", [])[0]
        return f"{u['name']['first']} {u['name']['last']} — {u.get('email')} ({u['location']['country']})"
    return None

def tool_ip(_: str = "") -> Optional[str]:
    r = _safe_get("https://api.ipify.org", params={"format": "json"})
    if r:
        return f"Your IP: {r.json().get('ip')}"
    return None

def tool_universities(country: str) -> Optional[str]:
    r = _safe_get("http://universities.hipolabs.com/search", params={"country": country})
    if r and isinstance(r.json(), list):
        data = r.json()
        if not data:
            return None
        return f"Universities found: {len(data)} — example: {data[0].get('name')}"
    return None

def tool_ziplookup(zipcode: str) -> Optional[str]:
    r = _safe_get(f"https://api.zippopotam.us/us/{quote_plus(zipcode)}")
    if r:
        j = r.json()
        p = j.get('places', [{}])[0]
        return f"{p.get('place name')}, {p.get('state')}"
    return None

def tool_restcountries(name: str) -> Optional[str]:
    r = _safe_get(f"https://restcountries.com/v3.1/name/{quote_plus(name)}")
    if r and isinstance(r.json(), list):
        c = r.json()[0]
        return f"{c.get('name',{}).get('common')} — Capital: {c.get('capital',['n/a'])[0]} — Population: {c.get('population')}"
    return None

def tool_openfoodfacts(q: str) -> Optional[str]:
    params = {"search_terms": q, "search_simple": 1, "json": 1}
    r = _safe_get("https://world.openfoodfacts.org/cgi/search.pl", params=params)
    if r and r.json().get("products"):
        prod = r.json()["products"][0]
        return f"{prod.get('product_name','Unnamed')} — {prod.get('brands')}"
    return None

def tool_map(place: str) -> Optional[str]:
    r = _safe_get("https://nominatim.openstreetmap.org/search",
                  params={"q": place, "format": "json", "limit": 1},
                  headers={"User-Agent": "PharmaCareBot/1.0"})
    if r and r.json():
        loc = r.json()[0]
        lat, lon = loc["lat"], loc["lon"]
        static = f"https://staticmap.openstreetmap.de/staticmap.php?center={lat},{lon}&zoom=12&size=600x300&markers={lat},{lon}"
        return f"🗺 {place.title()} — {static}"
    return None

# ---------------- Tool registry ----------------
TOOL_REGISTRY = {
    "openweather": tool_openweather,
    "open_meteo": tool_open_meteo,
    "wttr": tool_wttr_in,
    "wikipedia": tool_wikipedia,
    "duckduckgo": tool_duckduckgo,
    "dictionary": tool_dictionary,
    "openfda": tool_openfda,
    "rxnav": tool_rxnav,
    "dailymed": tool_dailymed,
    "news_reddit": tool_reddit_top,
    "news_gnews": tool_gnews_demo,
    "joke": tool_joke,
    "catfact": tool_catfact,
    "bored": tool_bored,
    "dog": tool_dog,
    "numbers": tool_numbers,
    "randomuser": tool_random_user,
    "ip": tool_ip,
    "universities": tool_universities,
    "zip": tool_ziplookup,
    "country": tool_restcountries,
    "openfood": tool_openfoodfacts,
    "map": tool_map,
}

# ---------------- Heuristics (fast) ----------------
def heuristic_intent(user_text: str) -> Optional[Dict[str, Any]]:
    """
    Improved heuristic intent detection.
    Catches natural questions and routes them to the right tool.
    """
    t = user_text.lower().strip()
    if not t:
        return None

    # --- Weather ---
    if any(k in t for k in ["weather", "temperature", "forecast", "is it raining", "climate", "hot", "cold"]):
        m = re.search(r"(?:in|for)\s+([a-z\s]+)", t)
        city = m.group(1).strip() if m else (t.split()[-1] if len(t.split()) > 1 else "London")
        if OWM_KEY:
            return {"action": "call_tool", "tool": "weather_openweather", "args": city}
        return {"action": "call_tool", "tool": "weather_open_meteo", "args": city}

    # --- Time ---
    if any(k in t for k in ["what time", "current time", "time now", "date now", "clock"]):
        return {"action": "call_tool", "tool": "time", "args": ""}

    # --- Drug info (side effects, usage, warnings, about, etc.) ---
    if any(k in t for k in ["drug", "tablet", "pill", "medicine", "side effects", "indication", "treat", "dosage"]):
        m = re.search(r"(?:drug|tablet|pill|medicine|about|of)\s+(.+)", t)
        drug = m.group(1).strip() if m else t.split()[-1]
        return {"action": "call_tool", "tool": "drug_openfda", "args": drug}

    # --- Knowledge / Wiki (what is, who is, tell me about, do you know, explain, define) ---
    if any(k in t for k in ["what is", "who is", "tell me about", "define", "explain", "meaning of", "do you know"]):
        m = re.search(r"(?:about|is|define|explain|meaning of|know)\s+(.+)", t)
        topic = m.group(1).strip() if m else t.split()[-1]
        return {"action": "call_tool", "tool": "wikipedia", "args": topic}

    # --- News ---
    if "news" in t or "headline" in t:
        return {"action": "call_tool", "tool": "news_reddit", "args": ""}

    # --- Fun ---
    if "joke" in t:
        return {"action": "call_tool", "tool": "joke", "args": ""}
    if "cat fact" in t or "catfact" in t:
        return {"action": "call_tool", "tool": "catfact", "args": ""}
    if "activity" in t or "bored" in t:
        return {"action": "call_tool", "tool": "bored", "args": ""}

    # --- Numbers ---
    m = re.match(r"(?:number|num|fact)\s+(\d+|random)", t)
    if m:
        return {"action": "call_tool", "tool": "numbers", "args": m.group(1)}

    # --- Map / Location ---
    if "map" in t or "where is" in t or "show me map" in t:
        m = re.search(r"(?:map|where is|show me map of)\s+(.+)", t)
        place = m.group(1).strip() if m else t.split()[-1]
        return {"action": "call_tool", "tool": "map", "args": place}

    # --- Universities ---
    if "universities in" in t:
        country = t.split("universities in", 1)[1].strip()
        return {"action": "call_tool", "tool": "universities", "args": country}

    # --- Country info ---
    if "country" in t:
        name = t.replace("country", "").strip()
        return {"action": "call_tool", "tool": "country", "args": name}

    # --- Random user ---
    if "random user" in t:
        return {"action": "call_tool", "tool": "randomuser", "args": ""}

    # --- IP address ---
    if t in ("ip", "my ip", "what is my ip"):
        return {"action": "call_tool", "tool": "ip", "args": ""}

    # --- Fallback: search ---
    if any(k in t for k in ["search", "look up", "find info on"]):
        return {"action": "call_tool", "tool": "duckduckgo", "args": user_text}

    return None

# ---------------- Planner (HF optional) ----------------
def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    try:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        candidate = match.group(0)
        candidate = candidate.replace("'", '"')
        candidate = re.sub(r",\s*(\]|})", r"\1", candidate)
        return json.loads(candidate)
    except Exception:
        return None

def planner_intent(user_text: str) -> Dict[str, Any]:
    if not HF_KEY:
        return {"action": "respond", "args": user_text}
    prompt = (
        "You are a short planner. Given a user question, decide whether to call an external tool. "
        "Return ONLY a JSON object with keys: action (call_tool|respond), tool (one of the tool names), args (string).\n"
        f"User question: '''{user_text}'''\n\nJSON:"
    )
    out = hf_query_raw(HF_MODEL, {"inputs": prompt, "parameters": {"max_new_tokens": 120, "temperature": 0.0}})
    txt = _extract_generated_text(out)
    j = _extract_json_from_text(txt)
    if j and "action" in j:
        return j
    return {"action": "respond", "args": user_text}

# ---------------- Blending (HF optional) ----------------
def blend_tool_result(user_text: str, tool_name: str, tool_output: str, history: Optional[List[Dict[str,str]]] = None) -> str:
    short_tool_out = _shorten(tool_output or "", 1600)
    if HF_KEY:
        history_text = ""
        if history:
            for turn in history[-6:]:
                history_text += f"User: {turn.get('user')}\nAssistant: {turn.get('bot')}\n"
        prompt = (
            "You are a concise assistant. Use the tool output to answer the user's question. "
            "If the tool output is an error, apologize and answer best-effort. Keep reply friendly and short.\n\n"
            f"Chat history:\n{history_text}\nUser question: {user_text}\nTool called: {tool_name}\nTool output:\n'''{short_tool_out}'''\n\nReply:"
        )
        out = hf_query_raw(HF_MODEL, {"inputs": prompt, "parameters": {"max_new_tokens": 220, "temperature": 0.2}})
        txt = _extract_generated_text(out).strip()
        if txt:
            return _shorten(txt, MAX_REPLY_CHARS)
    # non-HF fallback templated reply
    if not tool_output:
        return "Sorry — external service returned no result."
    return _shorten(f"I looked this up for you (source: {tool_name}):\n\n{tool_output}", 1200)

# ---------------- Public entrypoint ----------------
_chat_history: List[Dict[str,str]] = []

def chatbot_response(user_text: str) -> str:
    """
    Main brain entrypoint used by your bot.py.
    """
    text = _clean(user_text or "")
    if not text:
        return "Say something so I can help 😊"

    low = text.lower().strip()

    # start/help
    if low in ("/start", "start"):
        return START_TEXT = ( "😊Welcome to PharmaCare Bot!* \n\n"
    "I’m your friendly health, knowledge, and utility assistant. You can chat with me in plain English, "
    "and I’ll do my best to understand, fetch info from trusted free APIs, and reply like a real human.\n\n"

    "✨ *How to talk to me*\n"
    "You don’t need special commands — just ask naturally. For example:\n"
    "• \"Tell me about malaria\"\n"
    "• \"Is it raining in Lagos?\"\n"
    "• \"drug ibuprofen\"\n\n"

    "📜 *Here’s what I can do for you:*\n\n"

    "💊 *Drug Information*\n"
    "I use medical APIs (OpenFDA, RxNav, DailyMed) to give you indications, usage, and warnings for medicines.\n"
    "_Example_: `drug paracetamol`, `tell me about ibuprofen`\n\n"

    "📘 *Knowledge & Definitions*\n"
    "I search Wikipedia, DuckDuckGo, and dictionary APIs to explain terms, people, and concepts.\n"
    "_Example_: `wiki diabetes`, `define anemia`, `what is hypertension`\n\n"

    "🌦 *Weather & Time*\n"
    "I can check the weather using OpenWeather (if key is set), Open-Meteo, or wttr.in. "
    "I can also tell you the current time.\n"
    "_Example_: `weather Lagos`, `is it raining in Abuja?`, `time`\n\n"

    "📰 *News*\n"
    "I fetch top headlines from Reddit and GNews.\n"
    "_Example_: `news`, `news technology`\n\n"

    "🧠 *Smart Text Tools* (powered by Hugging Face if enabled)\n"
    "I can process text to summarize, expand, or paraphrase it.\n"
    "_Example_: `summarize The sun is a star...`, `expand What is gravity?`, `paraphrase This is good`\n\n"

    "🗺 *Maps & Location*\n"
    "I can show static map links and search locations.\n"
    "_Example_: `map Lagos`, `where is Abuja`, `zip 90210`\n\n"

    "🎲 *Fun & Random*\n"
    "For fun or quick info I use joke, cat fact, random activity, random user, and number trivia APIs.\n"
    "_Example_: `joke`, `cat fact`, `activity`, `random user`, `number 7`\n\n"

    "🌍 *Countries & Food*\n"
    "I can fetch data about countries (capital, population) and look up food items.\n"
    "_Example_: `country Japan`, `food chocolate`\n\n"

    "⚙️ *How I work*\n"
    "1. I first check if your request matches one of my special APIs (weather, drugs, wiki, etc.).\n"
    "2. If not clear, I use Hugging Face AI (if configured) to understand and decide which tool to call.\n"
    "3. I reply in plain English with friendly answers.\n\n"

    "💡 *Tip*: If I don’t get it right the first time, try rephrasing. You can always type `/help` "
    "to see the full command list.\n\n"

    "Now go ahead — ask me anything! 🚀"
)