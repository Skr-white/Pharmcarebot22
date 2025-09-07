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
import time
import json
import random
import requests
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from urllib.parse import quote_plus

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

HELP_TEXT = (
    "📜 *PharmaCare Bot — Commands & Examples*\n\n"
    "💬 Chat naturally. Examples:\n"
    "  • \"How are you?\"  • \"Tell me about malaria\"\n\n"
    "💊 Drug info: `drug <name>` — OpenFDA, RxNav, DailyMed.\n"
    "📘 Knowledge: `wiki <topic>`, `define <word>` — Wikipedia, DuckDuckGo, Dictionary.\n"
    "🌦 Weather: `weather <city>` — multiple providers.\n"
    "📰 News: `news` — top headline.\n"
    "🧠 NLP: `summarize <text>`, `expand <text>`, `paraphrase <text>` (needs HF key).\n"
    "🎲 Fun: `joke`, `cat fact`, `activity`, `random user`, `number <n>`.\n"
    "🗺 Map: `map <place>` — static OpenStreetMap link.\n"
    "⚙️ Owner notes: set env vars `HF_API_KEY`, `HF_MODEL`, `WEATHER_API_KEY` etc., then restart the bot.\n"
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
        if len(self.d) > 2000:
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
            return {"action": "call_tool_group", "gr