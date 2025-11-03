# brain.py
"""
PharmaCare Bot brain.

Exports:
 - START_TEXT (string) - long welcome shown on /start
 - HELP_TEXT (string)  - detailed help shown on /help
 - chatbot_response(user_text: str) -> str  - main entrypoint used by bot.py

Behavior:
 - fast heuristics route queries to prioritized groups of free APIs
 - uses Hugging Face (optional, via HF_API_KEY + HF_MODEL) as planner and blender
 - TTL cache to reduce repeated external calls
 - always tries fallbacks: group APIs -> search -> HF response -> friendly fallback
"""

import os
import re
import json
import math
import time
import html
import random
import requests
from functools import wraps
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from typing import Any, Dict, Optional  # ✅ Needed for TTLCache
from shared_state import update_state, get_state
from shared_state import shared_data, lock


# Example chatbot function
def chatbot_response(message: str) -> str:
    # Example response logic
    response = f"I got your message: {message}"
    
    # Update shared state
    update_state("last_user_message", message)
    update_state("last_bot_response", response)
    
    return response
HELP_TEXT = """
📜 *PharmaCare Bot — Commands & Examples*

💬 Chat naturally. Examples:
- "How are you?"
- "Tell me about malaria"

💊 Drug info: `drug <name>` — OpenFDA, RxNav, DailyMed.
📘 Knowledge: `wiki <topic>`, `define <word>` — Wikipedia, DuckDuckGo, Dictionary.
🌦 Weather: `weather <city>` — multiple providers.
📰 News: `news` — top headline.
🧠 NLP: `summarize <text>`, `expand <text>`, `paraphrase <text>` (needs HF key).
🎲 Fun: `joke`, `cat fact`, `activity`, `random user`, `number <n>`.
🗺 Map: `map <place>` — static OpenStreetMap link.

I can answer pharmacy-related questions, help with calculations, explain ingredients, and guide you through pharmaceutical methods.

-----------------------
🧮 **Dosage & Calculation Help**
You can ask questions like:
- "How do I calculate a child’s dose for paracetamol 15mg/kg if the child weighs 20kg?"
- "What’s the infusion rate for 500mg in 100mL over 30 minutes?"
- "How many tablets of 250mg do I need to make 1g?"
- "What’s the formula for dilution when making 5mg/mL from 20mg/mL?"
- "Explain the method for reconstituting a 1g vial with 10mL water."

I’ll show the step-by-step formula and the correct method.

-----------------------
⚗️ **Pharmacy Methods & Formulation**
You can ask things like:
- "What’s the method for preparing calamine lotion?"
- "Explain the levigation process in compounding."
- "How is an emulsion different from a suspension?"
- "Tell me the general steps in making a syrup."
- "What’s the role of preservatives in eye drops?"

I’ll explain the purpose of each step and ingredient.

-----------------------
💉 **Prescription Interpretation**
Try asking:
- "What does this prescription mean: T. Amox 500mg tds x 5/7?"
- "Interpret: Inj. Gentamicin 80mg IM stat, then bd x 5 days."
- "What’s the duration and frequency of this prescription?"
- "Explain abbreviations like bd, tds, stat, prn, od."

I’ll interpret and explain the meaning clearly.

-----------------------
🏷️ **Labeling Guidance**
Ask for help like:
- "What label should I use for eye drops?"
- "Show auxiliary labels for antibiotics."
- "How should I label a syrup given tds?"
- "What’s the correct label for external-use creams?"

I’ll give standard labeling text and cautions based on guidelines.

-----------------------
🧪 **Ingredients & Use**
You can ask:
- "What’s the function of methylparaben?"
- "Why is glycerin used in cough syrups?"
- "List the ingredients and their uses in calamine lotion."
- "What’s the role of lactose in tablets?"

I’ll explain their category (e.g., preservative, binder, humectant) and their importance.

-----------------------
📚 **Drug Information**
Ask me to find:
- "Show me the drug info for ibuprofen."
- "What’s the ATC classification of omeprazole?"
- "What are the contraindications of metformin?"
- "Get the PubChem data for paracetamol."

I’ll pull info from trusted sources like OpenFDA, RxNorm, DailyMed, PubChem, and WHO.

-----------------------
🧠 **Clinical & Research Insight**
Ask:
- "Find clinical trials on insulin therapy."
- "What’s the mechanism of action of metoprolol?"
- "Any study about herbal cough remedies?"
- "What are the common adverse effects of ACE inhibitors?"

-----------------------
🧾 **Regulatory & Product Lookup**
Try:
- "Get NAFDAC info for Augmentin 625mg."
- "Check if amlodipine is registered in Nigeria."
- "Show FDA warning updates on ranitidine."

-----------------------
💬 **Bonus Tips**
You can start questions with:
- "Explain…"
- "Calculate…"
- "Find…"
- "Interpret…"
- "Show…"
- "What’s the formula for…"
- "How to prepare…"

-----------------------
🩺 **Example Full Prompts**
- "Calculate IV infusion rate for 1g ceftriaxone diluted in 100mL over 1 hour."
- "Explain the compounding steps for a cream."
- "What’s the role of alcohol in hand sanitizers?"
- "Interpret this: Tab Amoxicillin 500mg tds x 7 days."
- "List ingredients and uses in oral rehydration salt."

-----------------------
💡 **Note**
I provide educational and reference guidance only — not a replacement for professional medical advice.

Type `/help` anytime to see this guide again.

-----------------------
⚙️ Owner notes: set env vars `HF_API_KEY`, `HF_MODEL`, `WEATHER_API_KEY` etc., then restart the bot.
"""

# ---------------- CONFIG & KEYS ----------------
HF_KEY = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")
OWM_KEY = os.getenv("WEATHER_API_KEY") or os.getenv("OPENWEATHER_KEY")
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY")  # optional
WEATHERBIT_KEY = os.getenv("WEATHERBIT_KEY")  # optional
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")        # optional
CACHE_TTL = int(os.getenv("BRAIN_CACHE_TTL", "300"))
DEFAULT_TIMEOUT = int(os.getenv("BRAIN_HTTP_TIMEOUT", "8"))
MAX_REPLY_CHARS = int(os.getenv("BRAIN_MAX_REPLY_CHARS", "3000"))

HF_HEADERS = {"Authorization": f"Bearer {HF_KEY}"} if HF_KEY else {}

# ---------------- START/HELP (imported by bot.py) ----------------
START_TEXT = (
    "👋 *Welcome to PharmaCare Bot!* \n\n"
    "I’m your friendly health, knowledge, and utility assistant. You can chat with me in plain English, "
    "and I’ll do my best to understand, fetch info from trusted free APIs, and reply like a real human.\n\n"

    "✨ *How to talk to me*\n"
    "You don’t need special commands — just ask naturally. For example:\n"
    "• \"Tell me about malaria\"\n"
    "• \"Is it raining in Lagos?\"\n"
    "• \"drug ibuprofen\"\n\n"

    "📜 *Here’s what I can do:*\n\n"

    "💊 *Drug Information* — OpenFDA, RxNav, DailyMed.\n"
    "_Example_: `drug paracetamol`, `tell me about ibuprofen`\n\n"

    "📘 *Knowledge & Definitions* — Wikipedia, DuckDuckGo, Dictionary.\n"
    "_Example_: `wiki diabetes`, `define anemia`, `what is hypertension`\n\n"

    "🌦 *Weather & Time* — OpenWeather (if configured), Open-Meteo, wttr.in, WeatherAPI, Weatherbit.\n"
    "_Example_: `weather Lagos`, `is it raining in Abuja?`, `time`\n\n"

    "📰 *News* — Reddit, GNews, BBC RSS.\n"
    "_Example_: `news`, `news technology`\n\n"

    "🧠 *Smart Text Tools* — summarize/expand/paraphrase (Hugging Face if configured).\n"
    "_Example_: `summarize <text>`, `expand <text>`, `paraphrase <text>`\n\n"

    "🎲 *Fun & Utilities* — jokes, cat facts, bored activities, number trivia, random user and more.\n\n"

    "⚙️ *How I work*:\n"
    "1. I try a single best API per intent (fast & reliable).\n"
    "2. If ambiguous, I use Hugging Face planner (if configured) to choose.\n"
    "3. If API output needs polishing, I use HF to rewrite it (if configured), otherwise return a readable template.\n\n"

    "💡 Tip: If I don’t get it right, try rephrasing. Type `/help` for command examples."
)

HELP_TEXT = """
📜 *PharmaCare Bot — Commands & Examples*

💬 Chat naturally. Examples:
- "How are you?"
- "Tell me about malaria"

💊 Drug info: `drug <name>` — OpenFDA, RxNav, DailyMed.
📘 Knowledge: `wiki <topic>`, `define <word>` — Wikipedia, DuckDuckGo, Dictionary.
🌦 Weather: `weather <city>` — multiple providers.
📰 News: `news` — top headline.
🧠 NLP: `summarize <text>`, `expand <text>`, `paraphrase <text>` (needs HF key).
🎲 Fun: `joke`, `cat fact`, `activity`, `random user`, `number <n>`.
🗺 Map: `map <place>` — static OpenStreetMap link.

I can answer pharmacy-related questions, help with calculations, explain ingredients, and guide you through pharmaceutical methods.

-----------------------
🧮 **Dosage & Calculation Help**
You can ask questions like:
- "How do I calculate a child’s dose for paracetamol 15mg/kg if the child weighs 20kg?"
- "What’s the infusion rate for 500mg in 100mL over 30 minutes?"
- "How many tablets of 250mg do I need to make 1g?"
- "What’s the formula for dilution when making 5mg/mL from 20mg/mL?"
- "Explain the method for reconstituting a 1g vial with 10mL water."

I’ll show the step-by-step formula and the correct method.

-----------------------
⚗️ **Pharmacy Methods & Formulation**
You can ask things like:
- "What’s the method for preparing calamine lotion?"
- "Explain the levigation process in compounding."
- "How is an emulsion different from a suspension?"
- "Tell me the general steps in making a syrup."
- "What’s the role of preservatives in eye drops?"

I’ll explain the purpose of each step and ingredient.

-----------------------
💉 **Prescription Interpretation**
Try asking:
- "What does this prescription mean: T. Amox 500mg tds x 5/7?"
- "Interpret: Inj. Gentamicin 80mg IM stat, then bd x 5 days."
- "What’s the duration and frequency of this prescription?"
- "Explain abbreviations like bd, tds, stat, prn, od."

I’ll interpret and explain the meaning clearly.

-----------------------
🏷️ **Labeling Guidance**
Ask for help like:
- "What label should I use for eye drops?"
- "Show auxiliary labels for antibiotics."
- "How should I label a syrup given tds?"
- "What’s the correct label for external-use creams?"

I’ll give standard labeling text and cautions based on guidelines.

-----------------------
🧪 **Ingredients & Use**
You can ask:
- "What’s the function of methylparaben?"
- "Why is glycerin used in cough syrups?"
- "List the ingredients and their uses in calamine lotion."
- "What’s the role of lactose in tablets?"

I’ll explain their category (e.g., preservative, binder, humectant) and their importance.

-----------------------
📚 **Drug Information**
Ask me to find:
- "Show me the drug info for ibuprofen."
- "What’s the ATC classification of omeprazole?"
- "What are the contraindications of metformin?"
- "Get the PubChem data for paracetamol."

I’ll pull info from trusted sources like OpenFDA, RxNorm, DailyMed, PubChem, and WHO.

-----------------------
🧠 **Clinical & Research Insight**
Ask:
- "Find clinical trials on insulin therapy."
- "What’s the mechanism of action of metoprolol?"
- "Any study about herbal cough remedies?"
- "What are the common adverse effects of ACE inhibitors?"

-----------------------
🧾 **Regulatory & Product Lookup**
Try:
- "Get NAFDAC info for Augmentin 625mg."
- "Check if amlodipine is registered in Nigeria."
- "Show FDA warning updates on ranitidine."

-----------------------
💬 **Bonus Tips**
You can start questions with:
- "Explain…"
- "Calculate…"
- "Find…"
- "Interpret…"
- "Show…"
- "What’s the formula for…"
- "How to prepare…"

-----------------------
🩺 **Example Full Prompts**
- "Calculate IV infusion rate for 1g ceftriaxone diluted in 100mL over 1 hour."
- "Explain the compounding steps for a cream."
- "What’s the role of alcohol in hand sanitizers?"
- "Interpret this: Tab Amoxicillin 500mg tds x 7 days."
- "List ingredients and uses in oral rehydration salt."

-----------------------
💡 **Note**
I provide educational and reference guidance only — not a replacement for professional medical advice.

Type `/help` anytime to see this guide again.

-----------------------
⚙️ Owner notes: set env vars `HF_API_KEY`, `HF_MODEL`, `WEATHER_API_KEY` etc., then restart the bot.
"""
class TTLCache:
    """
    Simple in-memory cache with TTL (time-to-live) per key.
    """
    def __init__(self, ttl: int = 60):
        self.ttl = ttl
        self.d: Dict[str, tuple[float, Any]] = {}

    def set(self, k: str, val: Any):
        self.d[k] = (time.time() + self.ttl, val)

    def get(self, k: str) -> Optional[Any]:
        if k in self.d:
            expiry, val = self.d[k]
            if expiry > time.time():
                return val
            else:
                del self.d[k]  # remove expired
        return None

    def delete(self, k: str):
        if k in self.d:
            del self.d[k]

    def clear(self):
        self.d.clear()

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

# ---------------- Tools (many providers) ----------------

# --- WEATHER PROVIDERS (priorities: OpenWeather, WeatherAPI, Weatherbit, Open-Meteo, wttr.in) ---
def tool_openweather(city: str) -> Optional[str]:
    if not OWM_KEY:
        return None
    r = _safe_get("https://api.openweathermap.org/data/2.5/weather",
                  params={"q": city, "appid": OWM_KEY, "units": "metric"})
    if r and r.ok:
        try:
            j = r.json()
            return f"🌦 {city.title()}: {j['main']['temp']}°C — {j['weather'][0]['description']}"
        except Exception:
            return None
    return None

def tool_weatherapi(city: str) -> Optional[str]:
    if not WEATHERAPI_KEY:
        return None
    r = _safe_get("https://api.weatherapi.com/v1/current.json", params={"key": WEATHERAPI_KEY, "q": city})
    if r and r.ok:
        j = r.json()
        return f"🌦 {city.title()}: {j['current']['temp_c']}°C — {j['current']['condition']['text']}"
    return None

def tool_weatherbit(city: str) -> Optional[str]:
    if not WEATHERBIT_KEY:
        return None
    r = _safe_get("https://api.weatherbit.io/v2.0/current", params={"city": city, "key": WEATHERBIT_KEY, "units": "M"})
    if r and r.ok:
        j = r.json()
        if j.get("data"):
            d = j["data"][0]
            return f"🌦 {city.title()}: {d['temp']}°C — {d['weather']['description']}"
    return None

def tool_open_meteo(city: str) -> Optional[str]:
    g = _safe_get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1})
    if g and g.ok:
        gj = g.json()
        if gj.get("results"):
            loc = gj["results"][0]
            lat, lon = loc["latitude"], loc["longitude"]
            r = _safe_get("https://api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "current_weather": True})
            if r and r.ok:
                j = r.json()
                cw = j.get("current_weather")
                if cw:
                    return f"🌦 {loc.get('name')}: {cw.get('temperature')}°C — wind {cw.get('windspeed')} m/s"
    return None

def tool_wttr_in(city: str) -> Optional[str]:
    r = _safe_get(f"https://wttr.in/{quote_plus(city)}", params={"format": "3"})
    if r and r.ok:
        return r.text
    return None

# --- KNOWLEDGE / SEARCH PROVIDERS ---
def tool_wikipedia(query: str) -> Optional[str]:
    q = quote_plus(query.replace(" ", "_"))
    r = _safe_get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}")
    if r and r.ok:
        try:
            j = r.json()
            if j.get("extract"):
                return f"📘 {j.get('title','')}: {_shorten(j['extract'], 1200)}"
        except Exception:
            pass
    return None

def tool_duckduckgo(query: str) -> Optional[str]:
    r = _safe_get("https://api.duckduckgo.com/", params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1})
    if r and r.ok:
        try:
            j = r.json()
            text = j.get("AbstractText") or j.get("Definition") or ""
            if text:
                return f"🔎 {_shorten(text, 1200)}"
        except Exception:
            pass
    return None

def tool_dictionary(query: str) -> Optional[str]:
    r = _safe_get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{quote_plus(query)}")
    if r and r.ok:
        try:
            j = r.json()
            if isinstance(j, list) and j:
                d = j[0]["meanings"][0]["definitions"][0]["definition"]
                return f"📖 {query.title()}: {d}"
        except Exception:
            pass
    return None

# --- DRUG / MEDICAL PROVIDERS ---
def tool_openfda(drug: str) -> Optional[str]:
    r = _safe_get("https://api.fda.gov/drug/label.json", params={"search": f"openfda.brand_name:{quote_plus(drug)}", "limit": 1})
    if r and r.ok:
        try:
            j = r.json()
            if j.get("results"):
                d = j["results"][0]
                usage = d.get("indications_and_usage", ["No usage info"])[0]
                warnings = d.get("warnings", [""])[0]
                return f"💊 {drug.title()} (OpenFDA): {_shorten(usage,700)}\nWarnings: {_shorten(warnings,300)}"
        except Exception:
            pass
    return None

def tool_rxnav(drug: str) -> Optional[str]:
    r = _safe_get("https://rxnav.nlm.nih.gov/REST/drugs.json", params={"name": drug})
    if r and r.ok:
        try:
            j = r.json()
            if j.get("drugGroup", {}).get("conceptGroup"):
                return f"💊 {drug.title()} — matches found in RxNav"
        except Exception:
            pass
    return None

def tool_dailymed(drug: str) -> Optional[str]:
    r = _safe_get("https://dailymed.nlm.nih.gov/dailymed/services/v2/drugnames.json", params={"drug_name": drug})
    if r and r.ok:
        try:
            j = r.json()
            if j.get("data"):
                return f"💊 {drug.title()} — listed in DailyMed"
        except Exception:
            pass
    return None

def tool_rximage(drug: str) -> Optional[str]:
    r = _safe_get("https://rximage.nlm.nih.gov/api/rximage/1/rxbase", params={"name": drug})
    if r and r.ok:
        try:
            j = r.json()
            if j.get("nlmRxImages"):
                return f"💊 {drug.title()} — images available (RxImage)"
        except Exception:
            pass
    return None

# --- NEWS / HEADLINES ---
def tool_reddit_top(_: str = "news") -> Optional[str]:
    r = _safe_get("https://www.reddit.com/r/news/top.json", params={"limit": 1, "t": "day"})
    if r and r.ok:
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
    if r and r.ok:
        try:
            j = r.json()
            if j.get("articles"):
                a = j["articles"][0]
                return f"📰 {a.get('title')}\n{a.get('url')}"
        except Exception:
            pass
    return None

def tool_bbc_rss(_: str = "") -> Optional[str]:
    r = _safe_get("http://feeds.bbci.co.uk/news/rss.xml")
    if r and r.ok:
        text = r.text
        if "<title>" in text:
            try:
                # pick the 2nd title (the first is feed name)
                title = text.split("<title>")[2].split("</title>")[0]
                return f"📰 BBC: {title}"
            except Exception:
                pass
    return None

# --- FUN / MISC UTILITIES ---
def tool_joke(_: str = "") -> Optional[str]:
    r = _safe_get("https://official-joke-api.appspot.com/random_joke")
    if r and r.ok:
        try:
            j = r.json()
            return f"🤣 {j.get('setup')} — {j.get('punchline')}"
        except Exception:
            pass
    return None

def tool_catfact(_: str = "") -> Optional[str]:
    r = _safe_get("https://catfact.ninja/fact")
    if r and r.ok:
        try:
            return f"🐱 {r.json().get('fact')}"
        except Exception:
            pass
    return None

def tool_bored(_: str = "") -> Optional[str]:
    r = _safe_get("https://www.boredapi.com/api/activity")
    if r and r.ok:
        try:
            return f"🎯 {r.json().get('activity')}"
        except Exception:
            pass
    return None

def tool_dog(_: str = "") -> Optional[str]:
    r = _safe_get("https://dog.ceo/api/breeds/image/random")
    if r and r.ok:
        try:
            return r.json().get("message")
        except Exception:
            pass
    return None

def tool_numbers(num: str = "random") -> Optional[str]:
    r = _safe_get(f"http://numbersapi.com/{quote_plus(num)}/trivia?json")
    if r and r.ok:
        try:
            return f"🔢 {r.json().get('text')}"
        except Exception:
            pass
    return None

def tool_random_user(_: str = "") -> Optional[str]:
    r = _safe_get("https://randomuser.me/api/")
    if r and r.ok:
        try:
            u = r.json().get("results", [])[0]
            return f"{u['name']['first']} {u['name']['last']} — {u.get('email')} ({u['location']['country']})"
        except Exception:
            pass
    return None

def tool_ip(_: str = "") -> Optional[str]:
    r = _safe_get("https://api.ipify.org", params={"format": "json"})
    if r and r.ok:
        try:
            return f"Your IP: {r.json().get('ip')}"
        except Exception:
            pass
    return None

def tool_universities(country: str) -> Optional[str]:
    r = _safe_get("http://universities.hipolabs.com/search", params={"country": country})
    if r and r.ok:
        j = r.json()
        if isinstance(j, list) and j:
            return f"Universities found: {len(j)} — example: {j[0].get('name')}"
    return None

def tool_ziplookup(zipcode: str) -> Optional[str]:
    r = _safe_get(f"https://api.zippopotam.us/us/{quote_plus(zipcode)}")
    if r and r.ok:
        j = r.json()
        p = j.get("places", [{}])[0]
        return f"{p.get('place name')}, {p.get('state')}"
    return None

def tool_restcountries(name: str) -> Optional[str]:
    r = _safe_get(f"https://restcountries.com/v3.1/name/{quote_plus(name)}")
    if r and r.ok:
        try:
            j = r.json()
            if isinstance(j, list) and j:
                c = j[0]
                return f"{c.get('name',{}).get('common')} — Capital: {c.get('capital',['n/a'])[0]} — Population: {c.get('population')}"
        except Exception:
            pass
    return None

def tool_openfoodfacts(q: str) -> Optional[str]:
    params = {"search_terms": q, "search_simple": 1, "json": 1}
    r = _safe_get("https://world.openfoodfacts.org/cgi/search.pl", params=params)
    if r and r.ok:
        try:
            j = r.json()
            if j.get("products"):
                prod = j["products"][0]
                return f"{prod.get('product_name','Unnamed')} — {prod.get('brands')}"
        except Exception:
            pass
    return None

def tool_map(place: str) -> Optional[str]:
    r = _safe_get("https://nominatim.openstreetmap.org/search", params={"q": place, "format": "json", "limit": 1},
                  headers={"User-Agent": "PharmaCareBot/1.0"})
    if r and r.ok:
        try:
            j = r.json()
            if j:
                loc = j[0]
                lat, lon = loc["lat"], loc["lon"]
                static = f"https://staticmap.openstreetmap.de/staticmap.php?center={lat},{lon}&zoom=12&size=600x300&markers={lat},{lon}"
                return f"🗺 {place.title()} — {static}"
        except Exception:
            pass
    return None
# Example in brain.py
def chatbot_response(message: str) -> str:
    response = f"Echo: {message}"  # your actual response logic

    with lock:
        shared_data["last_user_message"] = message
        shared_data["last_bot_response"] = response

    return response
# ---------------- Tool registry (single name keys) ----------------
TOOL_REGISTRY: Dict[str, Callable[[str], Optional[str]]] = {
    # weather family (note keys are simple)
    "openweather": tool_openweather,
    "weatherapi": tool_weatherapi,
    "weatherbit": tool_weatherbit,
    "open_meteo": tool_open_meteo,
    "wttr": tool_wttr_in,
    # knowledge/search family
    "wikipedia": tool_wikipedia,
    "duckduckgo": tool_duckduckgo,
    "dictionary": tool_dictionary,
    # drugs
    "openfda": tool_openfda,
    "rxnav": tool_rxnav,
    "dailymed": tool_dailymed,
    "rximage": tool_rximage,
    # news
    "news_reddit": tool_reddit_top,
    "news_gnews": tool_gnews_demo,
    "news_bbc": tool_bbc_rss,
    # fun / misc
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

# ---------------- Utility: try sequence of callables ----------------
def try_tools_sequence(key: str, arg: str, candidates: List[Callable[[str], Optional[str]]]) -> Optional[str]:
    """
    Try each candidate function in order until one returns a non-empty result.
    Cache the final result under a composite key.
    """
    cache_key = f"{key}:{arg}".lower()
    cached = _cache.get(cache_key)
    if cached:
        return cached
    for fn in candidates:
        try:
            out = fn(arg)
        except Exception:
            out = None
        if out:
            _cache.set(cache_key, out)
            return out
    _cache.set(cache_key, None)
    return None

# ---------------- Heuristics (fast) ----------------
def heuristic_intent(user_text: str) -> Optional[Dict[str, Any]]:
    t = user_text.lower().strip()
    if not t:
        return None

    # weather
    if any(k in t for k in ["weather", "temperature", "forecast", "is it raining", "raining", "rain"]):
        m = re.search(r"(?:in|for)\s+([a-zA-Z\s\-]+)", t)
        city = (m.group(1).strip() if m else (t.split()[-1] if len(t.split()) > 0 else "London"))
        # prefer keys when configured
        if OWM_KEY:
            return {"action": "call_tool_group", "group": "weather", "args": city}
        # else use open_meteo/wttr
        return {"action": "call_tool_group", "group": "weather", "args": city}

    # time
    if any(k in t for k in ["what time", "current time", "time now", "date now", "clock"]):
        return {"action": "call_tool", "tool": "time", "args": ""}

    # news
    if "news" in t or "headline" in t:
        return {"action": "call_tool_group", "group": "news", "args": ""}

    # joke / fun
    if "joke" in t:
        return {"action": "call_tool", "tool": "joke", "args": ""}
    if "cat fact" in t or "catfact" in t:
        return {"action": "call_tool", "tool": "catfact", "args": ""}
    if "activity" in t or "bored" in t:
        return {"action": "call_tool", "tool": "bored", "args": ""}

    # numbers
    m = re.match(r"(?:number|num|fact)\s+(\d+|random)", t)
    if m:
        return {"action": "call_tool", "tool": "numbers", "args": m.group(1)}

    # map / location
    if t.startswith("map ") or "where is " in t or t.startswith("show me map"):
        m = re.search(r"(?:map|where is|show me map of)\s+(.+)", t)
        place = (m.group(1).strip() if m else t.split()[-1])
        return {"action": "call_tool", "tool": "map", "args": place}

    # drug/medicine
    if any(k in t for k in ["drug", "medicine", "tablet", "pill", "side effects", "dosage", "indication", "treat"]):
        m = re.search(r"(?:drug|about|tell me about|medicine|tablet|pill|what is)\s+(.+)", t)
        drug = (m.group(1).strip() if m else t.split()[-1])
        return {"action": "call_tool_group", "group": "drug", "args": drug}

    # wiki / define / explain
    if any(k in t for k in ["what is", "who is", "tell me about", "define", "explain", "meaning of", "wiki"]):
        m = re.search(r"(?:what is|who is|tell me about|define|explain|meaning of|wiki)\s+(.+)", t)
        topic = (m.group(1).strip() if m else t)
        return {"action": "call_tool_group", "group": "knowledge", "args": topic}

    # universities
    if t.startswith("universities in ") or t.startswith("universities "):
        country = t.split(" in ",1)[1] if " in " in t else t.replace("universities", "").strip()
        return {"action": "call_tool", "tool": "universities", "args": country}

    # zip lookup
    if t.startswith("zip ") or t.startswith("zipcode "):
        z = t.split(" ",1)[1]
        return {"action": "call_tool", "tool": "zip", "args": z}

    # random user, ip, country, food
    if "random user" in t:
        return {"action": "call_tool", "tool": "randomuser", "args": ""}
    if t in ("ip", "my ip", "what is my ip"):
        return {"action": "call_tool", "tool": "ip", "args": ""}
    if t.startswith("country "):
        return {"action": "call_tool", "tool": "country", "args": t.split(" ",1)[1]}
    if t.startswith("food ") or t.startswith("openfood "):
        return {"action": "call_tool", "tool": "openfood", "args": t.split(" ",1)[1]}

    # fallback: search
    if any(k in t for k in ["search", "look up", "find info", "who is", "what is"]):
        return {"action": "call_tool_group", "group": "knowledge", "args": t}

    return None

# ---------------- Planner (HF) ----------------
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
    """
    Ask HF to plan whether to call a tool. Returns a dict with action/tool/args.
    If HF not configured or planner fails, returns {'action':'respond','args':user_text}.
    """
    if not HF_KEY:
        return {"action": "respond", "args": user_text}
    prompt = (
        "You are a short planner. Given the user question, decide whether to call an external tool. "
        "Return ONLY a JSON object with keys: action (call_tool|call_tool_group|respond), tool (tool key), group (optional), args (string).\n"
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
        out = hf_query_raw(HF_MODEL, {"inputs": prompt, "parameters": {"max_new_tokens": 220, "temperature": 0.15}})
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
    Main brain entrypoint used by bot.py.
    """
    text = _clean(user_text or "")
    if not text:
        return "Say something so I can help 😊"

    low = text.lower().strip()

    # --- start/help ---
    if low in ("/start", "start"):
        return START_TEXT
    if low in ("/help", "help", "commands"):
        return HELP_TEXT

    # --- explicit NLP commands ---
    m = re.match(r"^(summarize|summarise|shorten)\s+(.+)$", text, flags=re.I)
    if m:
        if not HF_KEY:
            # if no HF, try a fallback: short wikipedia or duckduckgo snippet
            topic = m.group(2).strip()
            snippet = tool_wikipedia(topic) or tool_duckduckgo(topic)
            return snippet or "⚠️ Summarize requires Hugging Face key (owner must set HF_API_KEY)."
        return _extract_generated_text(hf_query_raw(HF_MODEL, {"inputs": f"Summarize:\n\n{m.group(2).strip()}", "parameters": {"max_new_tokens": 200}}))

    m = re.match(r"^(expand|explain)\s+(.+)$", text, flags=re.I)
    if m:
        if not HF_KEY:
            return "⚠️ Expansion requires Hugging Face API key (owner must set HF_API_KEY)."
        return _extract_generated_text(hf_query_raw(HF_MODEL, {"inputs": f"Explain in simple terms:\n\n{m.group(2).strip()}", "parameters": {"max_new_tokens": 300}}))

    m = re.match(r"^(paraphrase|rephrase)\s+(.+)$", text, flags=re.I)
    if m:
        if not HF_KEY:
            return "⚠️ Paraphrase requires Hugging Face API key."
        return _extract_generated_text(hf_query_raw(HF_MODEL, {"inputs": f"Paraphrase:\n\n{m.group(2).strip()}", "parameters": {"max_new_tokens": 220}}))

    # --- explicit quick commands ---
    if re.search(r"\bjoke\b", low):
        return tool_joke() or random.choice(["🤣 No jokes now — try later!", "😅 I'm out of jokes!"])
    if "cat fact" in low or "catfact" in low:
        return tool_catfact() or "🐱 No cat facts now."
    if "activity" in low or "bored" in low:
        return tool_bored() or "🎯 No activity now."
    if "random user" in low:
        return tool_random_user() or "Couldn't fetch a random user."
    m = re.match(r"^(?:number|num|fact)\s+(\d+|random)$", low)
    if m:
        return tool_numbers(m.group(1))

    # --- explicit domain commands ---
    m = re.match(r"^(?:drug|tell me about)\s+(.+)$", text, flags=re.I)
    if m:
        drug_name = _clean(m.group(1))
        # try prioritized drug tools
        res = try_tools_sequence("drug", drug_name, [tool_openfda, tool_rxnav, tool_dailymed, tool_rximage])
        if res:
            _chat_history.append({"user": text, "bot": res})
            return res
        # fallback: try planner or search
        search = tool_wikipedia(drug_name) or tool_duckduckgo(drug_name)
        if search:
            _chat_history.append({"user": text, "bot": search})
            return search
        return f"❌ Sorry, no drug info found for *{drug_name}*."

    m = re.match(r"^(?:wiki|define|what is|who is|tell me about|explain)\s+(.+)$", text, flags=re.I)
    if m:
        topic = _clean(m.group(1))
        res = try_tools_sequence("knowledge", topic, [tool_wikipedia, tool_duckduckgo, tool_dictionary])
        if res:
            _chat_history.append({"user": text, "bot": res})
            return res
        return f"❌ No results for *{topic}*."

    m = re.match(r"^(?:weather|forecast|is it raining|temperature)\s*(?:in|for)?\s*(.*)$", text, flags=re.I)
    if m and m.group(1).strip():
        city = _clean(m.group(1))
        # prioritized weather group
        res = try_tools_sequence("weather", city, [
            tool_openweather, tool_weatherapi, tool_weatherbit, tool_open_meteo, tool_wttr_in
        ])
        if res:
            _chat_history.append({"user": text, "bot": res})
            return res
        return f"❌ Couldn't fetch weather for *{city}*."

    m = re.match(r"^(?:map|where is|show me map of)\s+(.+)$", text, flags=re.I)
    if m:
        place = _clean(m.group(1))
        res = tool_map(place)
        if res:
            _chat_history.append({"user": text, "bot": res})
            return res
        return f"❌ Map not found for *{place}*."

    if "news" in low or "headline" in low:
        res = try_tools_sequence("news", "", [tool_reddit_top, tool_gnews_demo, tool_bbc_rss])
        if res:
            _chat_history.append({"user": text, "bot": res})
            return res
        return "❌ Couldn't fetch news right now."

    # --- heuristics / planner ---
    intent = heuristic_intent(text)
    if not intent:
        intent = planner_intent(text)

    # handle intent
    action = intent.get("action")
    if action == "call_tool":
        tool_name = intent.get("tool")
        args = intent.get("args", "")
        fn = TOOL_REGISTRY.get(tool_name)
        if not fn:
            # try fuzzy match
            lk = (tool_name or "").lower()
            for k in TOOL_REGISTRY:
                if lk and lk in k.lower():
                    fn = TOOL_REGISTRY[k]
                    tool_name = k
                    break
        if tool_name == "time" or (not fn and intent.get("tool") == "time"):
            now = datetime.utcnow().isoformat() + "Z"
            reply = f"🕒 Current UTC time: {now}"
            _chat_history.append({"user": text, "bot": reply})
            return reply
        if not fn:
            # fallback to search
            fallback = tool_duckduckgo(args or text) or tool_wikipedia(args or text)
            if fallback:
                _chat_history.append({"user": text, "bot": fallback})
                return fallback
            return f"⚠️ I couldn't find a tool for `{tool_name}`."
        try:
            tool_out = fn(args or "")
        except Exception as e:
            tool_out = f"Tool call failed: {e}"
        reply = blend_tool_result(text, tool_name, tool_out or "", _chat_history)
        _chat_history.append({"user": text, "bot": reply})
        return reply

    if action == "call_tool_group":
        group = intent.get("group")
        arg = intent.get("args", "")
        if group == "weather":
            res = try_tools_sequence("weather", arg, [tool_openweather, tool_weatherapi, tool_weatherbit, tool_open_meteo, tool_wttr_in])
            if res:
                _chat_history.append({"user": text, "bot": res})
                return res
            return f"❌ Couldn't fetch weather for *{arg}*."
        if group == "drug":
            res = try_tools_sequence("drug", arg, [tool_openfda, tool_rxnav, tool_dailymed, tool_rximage])
            if res:
                _chat_history.append({"user": text, "bot": res})
                return res
            return f"❌ No drug info found for *{arg}*."
        if group == "knowledge":
            res = try_tools_sequence("knowledge", arg, [tool_wikipedia, tool_duckduckgo, tool_dictionary])
            if res:
                _chat_history.append({"user": text, "bot": res})
                return res
            return f"❌ No results for *{arg}*."
        if group == "news":
            res = try_tools_sequence("news", arg, [tool_reddit_top, lambda q: tool_gnews_demo(arg if arg else None), tool_bbc_rss])
            if res:
                _chat_history.append({"user": text, "bot": res})
                return res
            return "❌ No news right now."

    # action 'respond' or nothing: try to answer conversationally
    if HF_KEY:
        history_text = ""
        for turn in _chat_history[-6:]:
            history_text += f"User: {turn.get('user')}\nAssistant: {turn.get('bot')}\n"
        prompt = (
            "You are a helpful, concise assistant. Use the conversation history below to answer simply.\n\n"
            f"{history_text}\nUser: {text}\nAssistant:"
        )
        out = hf_query_raw(HF_MODEL, {"inputs": prompt, "parameters": {"max_new_tokens": 220, "temperature": 0.4}})
        candidate = _extract_generated_text(out).strip()
        reply = _shorten(candidate or "Sorry — I couldn't craft a reply.", MAX_REPLY_CHARS)
    else:
        # non-HF fallback: try search providers for an answer
        search = tool_duckduckgo(text) or tool_wikipedia(text) or tool_dictionary(text)
        if search:
            reply = search
        else:
            # final friendly fallback (guarantee some reply)
            reply = random.choice([
                "🤔 I didn’t quite get that. Try '/help' or rephrase your question.",
                "🧠 I'm learning — can you ask in another way?",
                "🙂 I'm here! Could you be more specific?"
            ])

    _chat_history.append({"user": text, "bot": reply})
    return reply