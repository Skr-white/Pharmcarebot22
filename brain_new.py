# brain_new.py
"""
brain_new.py
Extended PharmaCare Bot Brain (Advanced Edition)

This module powers the advanced reasoning layer of PharmaCare Bot.
It extends brain.py with deep pharmaceutical knowledge, chemical intelligence,
and educational support — using layered APIs and rule-based logic.

Features:
 - Advanced drug information (OpenFDA, RxNorm, DailyMed)
 - Chemical and pharmacological data (PubChem, ATC/DDD)
 - Clinical trials (ClinicalTrials.gov)
 - Regulatory and product details (NAFDAC Greenbook, NAPAMS)
 - Optional expansions (DrugBank, Elsevier Drug Info APIs)
 - WHO, EMA, NIH lookups and Wikipedia fallback for education
 - Pharmaceutical dose/concentration calculations
 - Formula and compounding reference assistance
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
from typing import Any, Dict, Optional, List, Callable

from shared_state import update_state, get_state, shared_data, lock

# ---------------- IMPORT/INHERIT CONFIG FROM brain.py ----------------
brain = None
try:
    import brain as base_brain
    brain = base_brain
except Exception:
    # brain.py not present or not loadable — fallback
    brain = None
#-------------- CHATBOT FUNCTION ----------------
def chatbot_response(message: str) -> str:
    response = f"Echo from brain_new: {message}"  # Example logic
    update_state("last_user_message", message)
    update_state("last_bot_response", response)
    return response
# Inherit configuration and keys from brain.py if available, else from env
HF_KEY = getattr(brain, "HF_KEY", None) or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL = getattr(brain, "HF_MODEL", None) or os.getenv("HF_MODEL", "google/flan-t5-small")
HF_HEADERS = getattr(brain, "HF_HEADERS", {}) if getattr(brain, "HF_HEADERS", None) else ({"Authorization": f"Bearer {HF_KEY}"} if HF_KEY else {})
DEFAULT_TIMEOUT = getattr(brain, "DEFAULT_TIMEOUT", None) or int(os.getenv("BRAIN_HTTP_TIMEOUT", "8"))
CACHE_TTL = getattr(brain, "CACHE_TTL", None) or int(os.getenv("BRAIN_CACHE_TTL", "300"))
MAX_REPLY_CHARS = getattr(brain, "MAX_REPLY_CHARS", None) or int(os.getenv("BRAIN_MAX_REPLY_CHARS", "3000"))

# Optional API Keys (DrugBank, Elsevier)
DRUGBANK_KEY = getattr(brain, "DRUGBANK_KEY", None) or os.getenv("DRUGBANK_KEY")
ELSEVIER_DI_KEY = getattr(brain, "831ed530dcd31b028ccc09a4e3712978", None) or os.getenv("ELSEVIER_DI_KEY")

# Telegram token (used in main if running this file directly)
TELEGRAM_TOKEN = getattr(brain, "TELEGRAM_TOKEN", None) or os.getenv("TELEGRAM_TOKEN")

# ---------------- API ENDPOINTS ----------------
OPENFDA_URL = "https://api.fda.gov/drug/label.json"
RXNORM_FIND = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
RXNORM_RELATED = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json"
DAILYMED_SPLS = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
DAILYMED_NAMES = "https://dailymed.nlm.nih.gov/dailymed/services/v2/drugnames.json?drug_name={}"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/JSON"
CLINICALTRIALS_SEARCH = "https://clinicaltrials.gov/api/query/study_fields"
ATC_INDEX = "https://atcddd.fhi.no/atc_ddd_index/"
NAFDAC_GREENBOOK = "https://greenbook.nafdac.gov.ng/api/products?search={}"
NAPAMS_URL = "https://napams.nafdac.gov.ng/api/product/search?term={}"
WIKIPEDIA_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
DRUGBANK_LOOKUP = "https://api.drugbankplus.com/v1/us/drugs/{identifier}"  # placeholder (requires key & correct endpoint)
ELSEVIER_LOOKUP = "https://api.elsevier.com/content/abstract/scopus_id/{id}"  # placeholder

# ---------------- SIMPLE CACHE ----------------
_cache = {}

def cached(ttl=CACHE_TTL):
    """Simple in-memory TTL cache for API calls."""
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                key = (func.__name__, json.dumps(args, default=str), json.dumps(kwargs, default=str))
            except Exception:
                key = (func.__name__, str(args), str(kwargs))
            entry = _cache.get(key)
            now = time.time()
            if entry and now - entry["ts"] < ttl:
                return entry["val"]
            val = func(*args, **kwargs)
            _cache[key] = {"ts": now, "val": val}
            return val
        return wrapper
    return deco
# ---------------- TTL CACHE EXAMPLE ----------------
class TTLCache:
    def __init__(self, ttl: int = 60):
        self.ttl = ttl
        self.d: Dict[str, Any] = {}

    def set(self, k: str, val: Any):
        self.d[k] = (time.time() + self.ttl, val)

    def get(self, k: str) -> Optional[Any]:
        item = self.d.get(k)
        if not item:
            return None
        expire, val = item
        if time.time() > expire:
            del self.d[k]
            return None
        return val


# ---------------- SAFETY FILTERS ----------------
MANUFACTURING_PATTERNS = [
    r"how to (synthesize|produce|manufactur|make) ",
    r"step[\s-]?by[\s-]?step (recipe|protocol|procedure)",
    r"process to produce",
]

RISKY_WORDS = ["synthesis", "synthesize", "lab protocol", "ferment", "batch recipe", "reactor", "how to make", "manufactur"]

def is_unsafe_request(text: str) -> bool:
    t = text.lower()
    for p in MANUFACTURING_PATTERNS:
        if re.search(p, t):
            return True
    for w in RISKY_WORDS:
        if w in t:
            return True
    return False

def safety_response() -> str:
    return ("I can't provide step-by-step manufacturing, synthesis, or lab protocols. "
            "I can provide ingredients, official label information, dosing calculations, general pharmaceutics education, and regulatory references.")

# ---------------- HELPERS ----------------
def shorten(text: str, length: int = 900) -> str:
    if not text:
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= length else text[:length-1] + "…"

def safe_get(url: str, params: dict = None, headers: dict = None, timeout: int = DEFAULT_TIMEOUT):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                return r.text
    except Exception:
        return None
    return None

# ---------------- API CLIENTS (cached) ----------------

@cached()
def openfda_search_label(drug_name: str, limit: int = 3):
    """Search openFDA drug label endpoint for a drug name."""
    queries = [
        f'openfda.generic_name:"{drug_name}"',
        f'openfda.brand_name:"{drug_name}"',
        f'other_name:"{drug_name}"'
    ]
    results = []
    for q in queries:
        try:
            data = safe_get(OPENFDA_URL, params={"search": q, "limit": limit})
            if not data:
                continue
            for item in data.get("results", []):
                openfda = item.get("openfda", {}) or {}
                title = openfda.get("brand_name", openfda.get("generic_name", ["Unknown"])) 
                if isinstance(title, list):
                    title = title[0] if title else "Unknown"
                rmeta = {
                    "title": title,
                    "generic_name": openfda.get("generic_name", ["Unknown"])[0] if openfda.get("generic_name") else "Unknown",
                    "indications": shorten(item.get("indications_and_usage", [""])[0]),
                    "active_ingredients": item.get("active_ingredient", []),
                    "inactive_ingredients": item.get("inactive_ingredient", ["Not listed"]),
                    "warnings": shorten(item.get("warnings", [""])[0]),
                    "raw": item
                }
                results.append(rmeta)
        except Exception:
            continue
    return results

@cached()
def rxnorm_name_to_ingredients(drug_name: str):
    """Find RxNorm rxcui and ingredient mapping."""
    try:
        r = safe_get(RXNORM_FIND, params={"name": drug_name})
        if not r:
            return None
        ids = r.get("idGroup", {}).get("rxnormId", [])
        if not ids:
            return None
        rxcui = ids[0]
        rel = safe_get(RXNORM_RELATED.format(rxcui=rxcui))
        ingredients = []
        if rel:
            for relset in rel.get("relatedGroup", {}).get("conceptGroup", []):
                for c in relset.get("conceptProperties", []):
                    if c.get("tty") in ("IN", "PIN", "MIN"):
                        ingredients.append({"name": c.get("name"), "rxcui": c.get("rxcui"), "tty": c.get("tty")})
        return {"rxcui": rxcui, "ingredients": ingredients}
    except Exception:
        return None

@cached()
def dailymed_search(drug_name: str, limit: int = 3):
    """Search DailyMed SPLs for matching drug_name (lightweight implementation)."""
    try:
        data = safe_get(DAILYMED_SPLS)
        if not data:
            return []
        found = []
        for entry in data.get("data", [])[:500]:
            title = entry.get("title", "") or entry.get("setid", "")
            if drug_name.lower() in title.lower():
                found.append({
                    "title": title,
                    "setid": entry.get("setid"),
                    "spl_url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={entry.get('setid')}"
                })
                if len(found) >= limit:
                    break
        return found
    except Exception:
        return []

@cached()
def pubchem_summary(chemical_name: str):
    """Return PubChem JSON summary for a chemical name."""
    try:
        name = requests.utils.quote(chemical_name)
        data = safe_get(PUBCHEM_URL.format(name= name))
        return data
    except Exception:
        return None

@cached()
def clinicaltrials_search(drug_name: str, max_studies: int = 3):
    """Query ClinicalTrials.gov study_fields for a drug term."""
    try:
        params = {
            "expr": drug_name,
            "fmt": "JSON",
            "fields": "NCTId,Condition,BriefTitle,OverallStatus,StartDate",
            "min_rnk": 1,
            "max_rnk": max_studies
        }
        data = safe_get(CLINICALTRIALS_SEARCH, params=params)
        return data.get("StudyFieldsResponse", {}).get("StudyFields", []) if data else []
    except Exception:
        return []

@cached()
def atc_lookup(substance: str):
    """Quick ATC/DDD index pointer."""
    try:
        data = safe_get(ATC_INDEX)
        if data and substance.lower() in str(data).lower():
            return {"found": True, "note": "See ATC/DDD index at https://atcddd.fhi.no/atc_ddd_index/ for formal codes."}
    except Exception:
        pass
    return {"found": False}

@cached()
def nafdac_search_local(name: str):
    """NAFDAC Greenbook pointer (no public API widely documented)."""
    try:
        data = safe_get(NAFDAC_GREENBOOK.format(quote_plus(name)))
        # If the Greenbook returns structured JSON, parse it; else return pointer
        if isinstance(data, dict) and data.get("data"):
            return data
    except Exception:
        pass
    return {"note": "Use the NAFDAC Greenbook (https://greenbook.nafdac.gov.ng/) or NAPAMS registration portal for official checks."}

# ---------------- Optional advanced APIs (DrugBank, Elsevier) ----------------
@cached()
def drugbank_lookup(name_or_id: str):
    """Placeholder DrugBank lookup — requires a proper API key and final endpoint."""
    if not DRUGBANK_KEY:
        return {"note": "DrugBank API key not set."}
    try:
        # NOTE: Actual DrugBank API endpoints/authorization differ; this is a placeholder
        headers = {"Authorization": f"Bearer {DRUGBANK_KEY}"}
        # If you have an identifier, use it; otherwise you would use a search endpoint.
        url = DRUGBANK_LOOKUP.format(identifier=quote_plus(name_or_id))
        data = safe_get(url, headers=headers)
        return data or {"note": "No DrugBank data returned."}
    except Exception:
        return {"note": "DrugBank lookup error."}

@cached()
def elsevier_lookup(identifier: str):
    """Placeholder for Elsevier / Clinical/Drug Info. Requires API key and correct endpoint."""
    if not ELSEVIER_DI_KEY:
        return {"note": "Elsevier DI API key not set."}
    try:
        headers = {"X-ELS-APIKey": ELSEVIER_DI_KEY}
        url = ELSEVIER_LOOKUP.format(id=quote_plus(identifier))
        data = safe_get(url, headers=headers)
        return data or {"note": "No Elsevier data returned."}
    except Exception:
        return {"note": "Elsevier lookup error."}

# ---------------- SMART SEARCH CORE ----------------
def pharma_search(query: str) -> str:
    """Aggregate multi-API response for a drug or chemical query."""
    q = quote_plus(query)
    info = []

    # openFDA
    try:
        data = openfda_search_label(query, limit=1)
        if data:
            item = data[0]
            brand = item.get("title") or item.get("generic_name")
            purpose = item.get("indications", "") or ""
            warnings = item.get("warnings", "") or ""
            info.append(f"💊 {brand}\nIndications: {shorten(purpose, 600)}\nWarnings: {shorten(warnings, 600)}")
    except Exception:
        pass

    # RxNorm
    try:
        rx = rxnorm_name_to_ingredients(query)
        if rx and rx.get("ingredients"):
            names = [i.get("name") for i in rx.get("ingredients", [])][:8]
            info.append(f"🧩 RxNorm ingredients: {', '.join(names)}")
    except Exception:
        pass

    # PubChem (chemical)
    try:
        pub = pubchem_summary(query)
        if pub and isinstance(pub, dict) and pub.get("PC_Compounds"):
            # Try to find formula or CID
            cid = None
            try:
                cid = pub["PC_Compounds"][0]["id"]["id"]["cid"]
            except Exception:
                cid = None
            if cid:
                info.append(f"⚗️ PubChem CID: {cid} — https://pubchem.ncbi.nlm.nih.gov/compound/{cid}")
    except Exception:
        pass

    # DailyMed
    try:
        dail = dailymed_search(query, limit=1)
        if dail:
            info.append(f"📘 DailyMed record: {dail[0].get('title', 'record found')}")
    except Exception:
        pass

    # ClinicalTrials
    try:
        trials = clinicaltrials_search(query, max_studies=3)
        if trials:
            first = trials[0]
            title = first.get("BriefTitle") or first.get("brief_title") or (first.get("BriefTitle", [""])[0] if isinstance(first.get("BriefTitle"), list) else "")
            if isinstance(title, list):
                title = title[0] if title else ""
            if title:
                info.append(f"🧪 Clinical trial example: {shorten(title, 300)}")
    except Exception:
        pass

    # NAFDAC (pointer)
    try:
        naf = nafdac_search_local(query)
        if naf and naf.get("data"):
            # basic parsing
            p = naf.get("data")[0]
            info.append(f"🇳🇬 NAFDAC product: {p.get('product_name','N/A')} — holder: {p.get('holder_name','N/A')}")
        elif naf and naf.get("note"):
            info.append(f"🇳🇬 NAFDAC: {naf.get('note')}")
    except Exception:
        pass

    # Wikipedia fallback
    try:
        wiki = safe_get(WIKIPEDIA_SUMMARY.format(quote_plus(query)))
        if isinstance(wiki, dict) and wiki.get("extract"):
            info.append(f"📚 {shorten(wiki.get('extract'), 700)}")
    except Exception:
        pass

    # DrugBank / Elsevier extras (optional)
    try:
        if DRUGBANK_KEY:
            db = drugbank_lookup(query)
            if isinstance(db, dict) and db.get("note"):
                info.append(f"🔎 DrugBank: {db.get('note')}")
            else:
                info.append("🔎 DrugBank results available (see full JSON).")
    except Exception:
        pass
    try:
        if ELSEVIER_DI_KEY:
            ev = elsevier_lookup(query)
            if isinstance(ev, dict) and ev.get("note"):
                info.append(f"🔎 Elsevier: {ev.get('note')}")
            else:
                info.append("🔎 Elsevier results available (see full JSON).")
    except Exception:
        pass

    if not info:
        return "No matching pharmaceutical data found right now. Try rephrasing or using a generic name."
    return "\n\n".join(info)

# ---------------- CALCULATION ENGINE ----------------
def to_number(s):
    try:
        return float(s)
    except:
        s2 = re.sub(r'[^\d\.\-\/]', '', str(s))
        if '/' in s2:
            try:
                a, b = s2.split('/', 1)
                return float(a) / float(b)
            except Exception:
                return None
        try:
            return float(s2)
        except Exception:
            return None

def mg_per_kg_to_total(mg_per_kg, weight_kg):
    a = float(mg_per_kg)
    b = float(weight_kg)
    total_mg = a * b
    return total_mg

def c1v1_to_c2v2(c1, v1, c2=None, v2=None):
    known = {"c1": c1, "v1": v1, "c2": c2, "v2": v2}
    missing = [k for k,v in known.items() if v is None]
    if len(missing) != 1:
        raise ValueError("Exactly one variable must be unknown")
    if "c1" in missing:
        return ("c1", (c2 * v2) / v1)
    if "v1" in missing:
        return ("v1", (c2 * v2) / c1)
    if "c2" in missing:
        return ("c2", (c1 * v1) / v2)
    if "v2" in missing:
        return ("v2", (c1 * v1) / c2)

def aliquot_method(amount_needed, available_amount, aliquot_power=10):
    if available_amount <= 0:
        raise ValueError("available_amount must be > 0")
    parts = math.ceil((available_amount / amount_needed) * aliquot_power)
    return {"parts": parts, "aliquot_power": aliquot_power, "note": "Illustrative calculation — check pharmacopeia rules."}

# ---------------- LOCAL CODEX ----------------
CODEX = {
    "labeling_rules": "Labels should include drug name, strength, route, frequency, patient name, prescriber, date, expiry, and storage.",
    "trituration": "Trituration: grinding powders to reduce particle size using mortar and pestle.",
    "levigation": "Levigation: moistening a powder with a levigating agent to make a smooth paste, then incorporating into an ointment base.",
    "suspension_definition": "A suspension is a liquid with insoluble solid drug particles dispersed throughout the vehicle."
}

def codex_lookup(term: str):
    t = term.lower()
    matches = {k: v for k, v in CODEX.items() if t in k or t in v.lower()}
    if matches:
        return matches
    return {"note": "No exact match found in local codex. Try a more general query (e.g., 'trituration', 'labeling_rules')."}

# ---------------- TELEGRAM HANDLERS ----------------
def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Hi — I'm Pharmabot. I can give ingredients, labels, dosing math, ATC classes, clinical trial references, and general pharmaceutics education. "
        "I won't provide step-by-step manufacturing or synthesis instructions. Try: 'ingredients aspirin', 'dose 5 mg/kg for 70 kg', 'ATC ibuprofen', 'pubchem paracetamol', or 'label amoxicillin'."
    )

def help_cmd(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Commands: /start /help\n\nJust ask: 'ingredients <drug>', 'label <drug>', 'dose <mg/kg> <weight_kg>', 'calculate c1 v1 c2 v2', 'pubchem <chemical>', 'trials <drug>', 'codex <term>'"
    )

# format_response improved to present dicts & lists neatly for Telegram
def format_response(resp):
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        if resp.get("error"):
            return "*Error*: " + resp["error"]
        if resp.get("note"):
            return resp["note"]
        if resp.get("auto"):
            data = resp.get("data")
            if isinstance(data, list):
                out = []
                for d in data[:3]:
                    title = d.get("title", "Unknown")
                    indications = d.get("indications", "") or ""
                    actives = ', '.join(d.get('active_ingredients', [])) if d.get('active_ingredients') else 'N/A'
                    out.append(f"*{title}* — {shorten(indications, 300)}\nIngredients (active): {actives}")
                return "\n\n".join(out)
        # pretty-print rest as JSON block
        try:
            return "```\n" + json.dumps(resp, indent=2, default=str) + "\n```"
        except Exception:
            return str(resp)
    if isinstance(resp, list):
        return "\n".join([shorten(str(x), 800) for x in resp])
    return str(resp)
# Example in brain_new.py
def chatbot_response_new(message: str) -> str:
    response = f"Brain_new processed: {message}"  # your logic

    with lock:
        shared_data["last_user_message"] = message
        shared_data["last_bot_response"] = response

    return response

# ---------------- 