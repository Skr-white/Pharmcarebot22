# brain.py
"""
PharmaCare bot brain (single-file).
- Entry: chatbot_response(user_text) -> str
- Uses prioritized free APIs per intent (OpenFDA, RxNav, DailyMed, Wikipedia, DuckDuckGo, Open-Meteo, wttr.in, etc.)
- Optionally uses Hugging Face (HF_API_KEY + HF_MODEL) for planning, blending, and NLP tasks.
- Keeps responses short and uses caching (TTL) to reduce API calls.
- Optimized HF wrapper (hf_fast_reply) for faster replies: small history, short tokens, low temperature.
"""

import os
import re
import time
import json
import random
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List
from urllib.parse import quote_plus

# ------------------ CONFIG / KEYS ------------------
HF_KEY = os.getenv("HF_API_KEY")                   # optional: Hugging Face API key
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")  # optional: HF model (instruction)
OWM_KEY = os.getenv("OPENWEATHER_KEY") or os.getenv("WEATHER_API_KEY")
CACHE_TTL = int(os.getenv("BRAIN_CACHE_TTL", "300"))
DEFAULT_TIMEOUT = 8
MAX_REPLY_CHARS = 3000

HF_HEADERS = {"Authorization": f"Bearer {HF_KEY}"} if HF_KEY else {}

# ------------------ SIMPLE TTL CACHE ------------------
class TTLCache:
    def __init__(self, ttl: int = CACHE_TTL):
        self.ttl = ttl
        self.data: Dict[str, Any] = {}

    def get(self, key: str):
        v = self.data.get(key)
        if not v:
            return None
        expiry, val = v
        if time.time() > expiry:
            try:
                del self.data[key]
            except KeyError:
                pass
            return None
        return val

    def set(self, key: str, value: Any):
        # simple eviction: if too many items, pop arbitrary oldest
        if len(self.data) > 1000:
            self.data.pop(next(iter(self.data)))
        self.data[key] = (time.time() + self.ttl, value)

_cache = TTLCache()

# ------------------ HTTP HELPER ------------------
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
        time.sleep(0.3 * (attempt + 1))
    return None

def _shorten(text: str, limit: int = MAX_REPLY_CHARS) -> str:
    if not text:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit - 3].rstrip() + "..."

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

# ------------------ HUGGING FACE HELPERS ------------------
def hf_query_raw(model: str, payload: dict, retries: int = 2, timeout: int = 15) -> Any:
    """Low-level HF call. Returns parsed JSON or raw text or {'error':...}"""
    if not HF_KEY:
        return {"error": "HF key missing"}
    url = f"https://api-inference.huggingface.co/models/{model}"
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=HF_HEADERS, json=payload, timeout=timeout)
            r.raise_for_status()
            # try json, else text
            try:
                return r.json()
            except Exception:
                return r.text
        except Exception as e:
            if attempt + 1 == retries:
                return {"error": str(e)}
            time.sleep(0.4)
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
        if isinstance(hf_resp, str):
            return hf_resp
    except Exception:
        pass
    return str(hf_resp)

def hf_fast_reply(prompt: str, max_tokens: int = 160, temp: float = 0.35, model: Optional[str] = None) -> str:
    """
    Fast wrapper for HF replies:
    - uses HF_MODEL by default
    - short max_tokens, low temp, short timeout to avoid long waits
    """
    if not HF_KEY:
        return ""
    model = model or HF_MODEL
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": temp}}
    out = hf_query_raw(model, payload, retries=2, timeout=12)
    txt = _extract_generated_text(out)
    return _shorten(txt, MAX_REPLY_CHARS).strip()

# ------------------ TOOLS (single-best per intent) ------------------
# WEATHER: OpenWeather (key) -> Open-Meteo (no key) -> wttr.in
def tool_openweather(city: str) -> Optional[str]:
    if not OWM_KEY:
        return None
    r = _safe_get("https://api.openweathermap.org/data/2.5/weather",
                  params={"q": city, "appid": OWM_KEY, "units": "metric"})
    if r:
        try:
            j = r.json()
            if "main" in j:
                return f"🌦 {city.title()}: {j['main']['temp']}°C, {j['weather'][0]['description']}"
        except Exception:
            pass
    return None

def tool_open_meteo(city: str) -> Optional[str]:
    g = _safe_get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1})
    if not g:
        return None
    try:
        gj = g.json()
        if gj.get("results"):
            loc = gj["results"][0]
            lat, lon = loc.get("latitude"), loc.get("longitude")
            r = _safe_get("https://api.open-meteo.com/v1/forecast",
                          params={"latitude": lat, "longitude": lon, "current_weather": True})
            if r and r.json().get("current_weather"):
                cw = r.json()["current_weather"]
                return f"🌦 {loc.get('name')}: {cw.get('temperature')}°C, wind {cw.get('windspeed')} m/s"
    except Exception:
        pass
    return None

def tool_wttr_in(city: str) -> Optional[str]:
    r = _safe_get(f"https://wttr.in/{quote_plus(city)}", params={"format": "3"})
    if r:
        return r.text
    return None

# WIKI / SEARCH / DICTIONARY
def tool_wikipedia(query: str) -> Optional[str]:
    q = quote_plus(query.replace(" ", "_"))
    r = _safe_get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}")
    if r:
        try:
            j = r.json()
            if j.get("extract"):
                return f"📘 {j.get('title','')}: {_shorten(j['extract'], 1200)}"
        except Exception:
            pass
    return None

def tool_duckduckgo(query: str) -> Optional[str]:
    r = _safe_get("https://api.duckduckgo.com/", params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1})
    if r:
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
    if r:
        try:
            j = r.json()
            if isinstance(j, list) and j:
                defi = j[0]["meanings"][0]["definitions"][0]["definition"]
                return f"📖 {query.title()}: {defi}"
        except Exception:
            pass
    return None

# DRUGS: OpenFDA -> RxNav -> DailyMed
def tool_openfda(drug: str) -> Optional[str]:
    r = _safe_get("https://api.fda.gov/drug/label.json",
                  params={"search": f"openfda.brand_name:{quote_plus(drug)}", "limit": 1})
    if r:
        try:
            j = r.json()
            if j.get("results"):
                d = j["results"][0]
                usage = d.get("indications_and_usage", ["No usage info"])[0]
                warnings = d.get("warnings", [""])[0]
                return f"💊 {drug.title()} (OpenFDA): {_shorten(usage,700)}\nWarnings: {_shorten(warnings,500)}"
        except Exception:
            pass
    return None

def tool_rxnav(drug: str) -> Optional[str]:
    r = _safe_get("https://rxnav.nlm.nih.gov/REST/drugs.json", params={"name": drug})
    if r:
        try:
            if r.json().get("drugGroup", {}).get("conceptGroup"):
                return f"💊 {drug.title()} — found in RxNav"
        except Exception:
            pass
    return None

def tool_dailymed(drug: str) -> Optional[str]:
    r = _safe_get("https://dailymed.nlm.nih.gov/dailymed/services/v2/drugnames.json", params={"drug_name": drug})
    if r:
        try:
            if r.json().get("data"):
                return f"💊 {drug.title()} — listed in DailyMed"
        except Exception:
            pass
    return None

# NEWS
def tool_reddit_top(subreddit: str = "news") -> Optional[str]:
    r = _safe_get(f"https://www.reddit.com/r/{subreddit}/top.json", params={"limit": 1, "t": "day"})
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
    if r:
        try:
            j = r.json()
            if j.get("articles"):
                a = j["articles"][0]
                return f"📰 {a.get('title')}\n{a.get('url')}"
        except Exception:
            pass
    return None

# FUN & MISC
def tool_joke(_: str = "") -> Optional[str]:
    r = _safe_get("https://official-joke-api.appspot.com/random_joke")
    if r:
        try:
            j = r.json()
            return f"🤣 {j.get('setup')} — {j.get('punchline')}"
        except Exception:
            pass
    return None

def tool_catfact(_: str = "") -> Optional[str]:
    r = _safe_get("https://catfact.ninja/fact")
    if r:
        try:
            return f"🐱 {r.json().get('fact')}"
        except Exception:
            pass
    return None

def tool_bored(_: str = "") -> Optional[str]:
    r = _safe_get("https://www.boredapi.com/api/activity")
    if r:
        try:
            return f"🎯 {r.json().get('activity')}"
        except Exception:
            pass
    return None

def tool_dog(_: str = "") -> Optional[str]:
    r = _safe_get("https://dog.ceo/api/breeds/image/random")
    if r:
        try:
            return r.json().get("message")
        except Exception:
            pass
    return None

def tool_numbers(num: str = "random") -> Optional[str]:
    r = _safe_get(f"http://numbersapi.com/{quote_plus(num)}/trivia?json")
    if r:
        try:
            return f"🔢 {r.json().get('text')}"
        except Exception:
            pass
    return None

def tool_random_user(_: str = "") -> Optional[str]:
    r = _safe_get("https://randomuser.me/api/")
    if r:
        try:
            u = r.json().get("results", [])[0]
            return f"{u['name']['first']} {u['name']['last']} — {u.get('email')} ({u['location']['country']})"
        except Exception:
            pass
    return None

def tool_ip(_: str = "") -> Optional[str]:
    r = _safe_get("https://api.ipify.org", params={"format": "json"})
    if r:
        try:
            return f"Your IP: {r.json().get('ip')}"
        except Exception:
            pass
    return None

def tool_universities(country: str) -> Optional[str]:
    r = _safe_get("http://universities.hipolabs.com/search", params={"country": country})
    if r:
        try:
            data = r.json()
            if isinstance(data, list) and data:
                return f"Universities: {len(data)} — example: {data[0].get('name')}"
        except Exception:
            pass
    return None

def tool_ziplookup(zipcode: str) -> Optional[str]:
    r = _safe_get(f"https://api.zippopotam.us/us/{quote_plus(zipcode)}")
    if r:
        try:
            j = r.json()
            p = j.get("places", [{}])[0]
            return f"{p.get('place name')}, {p.get('state')}"
        except Exception:
            pass
    return None

def tool_restcountries(name: str) -> Optional[str]:
    r = _safe_get(f"https://restcountries.com/v3.1/name/{quote_plus(name)}")
    if r:
        try:
            j = r.json()
            if isinstance(j, list) and j:
                c = j[0]
                return f"{c.get('name',{}).get('common','') } — Capital: {c.get('capital',['n/a'])[0]} — Population: {c.get('population')}"
        except Exception:
            pass
    return None

def tool_openfoodfacts(q: str) -> Optional[str]:
    params = {"search_terms": q, "search_simple": 1, "json": 1}
    r = _safe_get("https://world.openfoodfacts.org/cgi/search.pl", params=params)
    if r:
        try:
            j = r.json()
            if j.get("products"):
                prod = j["products"][0]
                return f"{prod.get('product_name','Unnamed')} — {prod.get('brands')}"
        except Exception:
            pass
    return None

def tool_map(place: str) -> Optional[str]:
    r = _safe_get("https://nominatim.openstreetmap.org/search",
                  params={"q": place, "format": "json", "limit": 1},
                  headers={"User-Agent": "PharmaCareBot/1.0"})
    if r:
        try:
            j = r.json()
            if j:
                lat = j[0]["lat"]; lon = j[0]["lon"]
                static = f"https://staticmap.openstreetmap.de/staticmap.php?center={lat},{lon}&zoom=12&size=600x300&markers={lat},{lon}"
                return f"🗺 {place.title()} — {static}"
        except Exception:
            pass
    return None

# ------------------ TOOL REGISTRY ------------------
TOOL_REGISTRY = {
    "weather_openweather": tool_openweather,
    "weather_open_meteo": tool_open_meteo,
    "wttr": tool_wttr_in,
    "wikipedia": tool_wikipedia,
    "duckduckgo": tool_duckduckgo,
    "dictionary": tool_dictionary,
    "drug_openfda": tool_openfda,
    "drug_rxnav": tool_rxnav,
    "drug_dailymed": tool_dailymed,
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

# ------------------ HEURISTICS (fast rules) ------------------
def heuristic_intent(user_text: str) -> Optional[Dict[str, Any]]:
    t = user_text.lower().strip()
    if not t:
        return None

    # Weather
    if any(k in t for k in ["weather", "temperature", "forecast", "is it raining"]):
        m = re.search(r"(?:weather|forecast|temperature)\s+(?:in|for)?\s*(.*)", t)
        city = (m.group(1).strip() if m and m.group(1).strip() else (t.split()[-1] if len(t.split()) > 1 else "London"))
        if OWM_KEY:
            return {"action": "call_tool", "tool": "weather_openweather", "args": city}
        return {"action": "call_tool", "tool": "weather_open_meteo", "args": city}

    # Time
    if any(k in t for k in ["what time", "current time", "time now", "date now"]):
        return {"action": "call_tool", "tool": "time", "args": ""}

    # News
    if "news" in t:
        return {"action": "call_tool", "tool": "news_reddit", "args": ""}

    # Fun
    if "joke" in t:
        return {"action": "call_tool", "tool": "joke", "args": ""}
    if "cat fact" in t or "catfact" in t:
        return {"action": "call_tool", "tool": "catfact", "args": ""}
    if "activity" in t or "bored" in t:
        return {"action": "call_tool", "tool": "bored", "args": ""}
    if "dog" in t and ("image" in t or "photo" in t or "picture" in t):
        return {"action": "call_tool", "tool": "dog", "args": ""}

    # Number fact
    m = re.match(r"(?:number|num|fact)\s+(\d+|random)", t)
    if m:
        return {"action": "call_tool", "tool": "numbers", "args": m.group(1)}

    # Map
    if t.startswith("map ") or "where is " in t:
        m = re.search(r"map\s+(.+)", t) or re.search(r"where is\s+(.+)", t)
        place = m.group(1).strip() if m else t.split()[-1]
        return {"action": "call_tool", "tool": "map", "args": place}

    # Drug detection
    if any(k in t for k in ["drug ", "medicine ", "tablet ", "pill ", "paracetamol", "ibuprofen", "amoxicillin", "aspirin", "acetaminophen"]):
        m = re.search(r"(?:drug|medicine|tablet|pill|about)\s+(.+)", t)
        drug = m.group(1).strip() if m else t.split()[-1]
        return {"action": "call_tool", "tool": "drug_openfda", "args": drug}

    # Wiki-like
    if any(k in t for k in ["what is", "who is", "tell me about", "explain", "wiki", "define"]):
        m = re.search(r"(?:about|is|what is|who is|define|explain)\s+(.+)", t)
        topic = m.group(1).strip() if m else t
        return {"action": "call_tool", "tool": "wikipedia", "args": topic}

    # Universities
    if t.startswith("universities in ") or t.startswith("universities "):
        country = t.split(" in ", 1)[1] if " in " in t else t.replace("universities", "").strip()
        return {"action": "call_tool", "tool": "universities", "args": country}

    # Zip
    if t.startswith("zip ") or t.startswith("zipcode "):
        z = t.split(" ", 1)[1]
        return {"action": "call_tool", "tool": "zip", "args": z}

    # Country
    if t.startswith("country "):
        name = t.split(" ",1)[1]
        return {"action": "call_tool", "tool": "country", "args": name}

    # random user
    if "random user" in t:
        return {"action": "call_tool", "tool": "randomuser", "args": ""}

    # ip
    if t in ("ip", "my ip", "what is my ip"):
        return {"action": "call_tool", "tool": "ip", "args": ""}

    # fallback: search
    if any(k in t for k in ["search", "look up", "who is", "what is", "tell me about"]):
        return {"action": "call_tool", "tool": "duckduckgo", "args": t}

    return None

# ------------------ PLANNER (HF) ------------------
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
    Ask HF planner to decide action. Returns {"action": "call_tool"|"respond", "tool": "...", "args": "..."}
    If HF not available, returns respond fallback.
    """
    if not HF_KEY:
        return {"action": "respond", "args": user_text}
    prompt = (
        "You are a concise planner. Given a user question, decide if we should call an external tool. "
        "Return ONLY a JSON object with keys: action (call_tool|respond), tool (one of known tool keys), args (string).\n"
        f"User question: '''{user_text}'''\n\nJSON:"
    )
    # Use hf_fast_reply but ask for a JSON like output with low tokens and very low temperature
    raw = hf_fast_reply(prompt, max_tokens=120, temp=0.0)
    j = _extract_json_from_text(raw)
    if j and "action" in j:
        return j
    # fallback
    return {"action": "respond", "args": user_text}

# ------------------ BLENDING ------------------
def blend_tool_result(