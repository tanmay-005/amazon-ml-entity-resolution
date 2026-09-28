"""
Cleans up business names and addresses so that different spellings of the same
business end up looking alike.

- Works the same way for every country, so France (not in training) goes through
  exactly the same steps.
- Removes accents ("Société Générale" becomes "societe generale").
- Turns common abbreviations into one standard form, so "Pvt. Ltd." = "Private Limited",
  "Rd" = "Road", "Bd" = "Boulevard", and so on.
- Produces several versions of each record (full name, name without legal words,
  trade name, address without landmarks, house numbers, sound-alike key) that the
  blocking and feature steps use.
"""
from __future__ import annotations

import math
import re
import unicodedata

import numpy as np
import pandas as pd

from translit import has_indic, transliterate

# generic helpers
def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _clean_base(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = transliterate(str(s))                 # Indic scripts -> Latin (must precede NFKD)
    s = strip_accents(s).lower()
    s = s.replace("\u2019", "'").replace("`", "'").replace("\u00b4", "'")
    s = s.replace("\u00ba", " ").replace("\u00b0", " ")   # º °
    return s


_COUNTRY_ALIASES = {
    "us": "us", "usa": "us", "u s": "us", "u s a": "us", "united states": "us",
    "united states of america": "us", "america": "us",
    "india": "india", "in": "india", "ind": "india", "bharat": "india",
    "france": "france", "fr": "france", "fra": "france", "republique francaise": "france",
}


def norm_country(c) -> str:
    """Open-set country label normaliser (unknown labels pass through lower-cased)."""
    c = re.sub(r"[^a-z]+", " ", _clean_base(c)).strip()
    return _COUNTRY_ALIASES.get(c, c)


def _merge_initials(toks):
    """['i','b','m','corp'] -> ['ibm','corp'] (handles I.B.M., L.L.C., A B C)."""
    out, buf = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
            continue
        if buf:
            out.append("".join(buf)) if len(buf) > 1 else out.extend(buf)
            buf = []
        out.append(t)
    if buf:
        out.append("".join(buf)) if len(buf) > 1 else out.extend(buf)
    return out


_SKEL_RULES = [("ph", "f"), ("gh", "g"), ("kh", "k"), ("bh", "b"), ("dh", "d"),
               ("th", "t"), ("sh", "s"), ("ch", "c"), ("ck", "k"), ("qu", "k"),
               ("q", "k"), ("w", "v"), ("z", "s"), ("x", "ks"), ("y", "i"),
               ("tion", "sn"), ("b", "p"), ("d", "t"), ("g", "k"), ("c", "k")]


def skeleton(s: str) -> str:
    """Light phonetic key that is robust to Indian/French transliteration variants
    (e.g. 'Shree/Shri/Sri', 'Bazaar/Bazar', 'Choudhary/Chaudhari')."""
    out = []
    for tok in s.split():
        if tok.isdigit():
            out.append(tok)
            continue
        t = re.sub(r"c(?=[eiy])", "s", tok)          # soft c: "consultancy" ~ "kansaltensi"
        for a, b in _SKEL_RULES:
            t = t.replace(a, b)
        t = re.sub(r"(.)\1+", r"\1", t)
        if t:
            t = t[0] + re.sub(r"[aeiou]", "", t[1:])
        out.append(t)
    return " ".join(out)


# business names
NAME_MAP = {
    "corporation": "corp", "incorporated": "inc", "incorporation": "inc",
    "limited": "ltd", "private": "pvt", "pte": "pvt", "pvtltd": "pvt ltd",
    "privatelimited": "pvt ltd", "company": "co", "cie": "compagnie",
    "international": "intl", "internationale": "intl", "intnl": "intl", "int'l": "intl",
    "manufacturing": "mfg", "manufacturers": "mfg", "manufacturer": "mfg",
    "services": "svc", "service": "svc", "svcs": "svc", "srvcs": "svc",
    "technologies": "tech", "technology": "tech", "technologie": "tech", "techs": "tech",
    "systems": "sys", "system": "sys", "systemes": "sys",
    "associates": "assoc", "association": "assoc",
    "brothers": "bros", "brother": "bros",
    "enterprises": "ent", "enterprise": "ent", "entreprise": "ent", "entreprises": "ent",
    "industries": "inds", "industry": "inds", "industrie": "inds", "industrial": "inds",
    "management": "mgmt", "national": "natl", "nationale": "natl",
    "university": "univ", "hospital": "hosp", "hopital": "hosp",
    "centre": "ctr", "center": "ctr", "cntr": "ctr",
    "solutions": "soln", "solution": "soln",
    "laboratories": "labs", "laboratory": "labs", "lab": "labs",
    "laboratoire": "labs", "laboratoires": "labs",
    "pharmaceuticals": "pharma", "pharmaceutical": "pharma",
    "engineering": "engg", "engineers": "engg", "eng": "engg",
    "consultancy": "consulting", "consultants": "consulting",
    "department": "dept", "societe": "ste", "saint": "st",
    "et": "and", "n": "and",
    "shree": "shri", "sri": "shri", "shre": "shri",
    # transliterated (Indic-script) legal words
    "praivet": "pvt", "prayvet": "pvt", "privet": "pvt", "limitad": "ltd", "limited": "ltd",
    "kampani": "co", "kampni": "co", "elaelapi": "llp", "elelpi": "llp", "elalpi": "llp",
    "aiti": "it", "aaiti": "it", "aitee": "it", "korporeshan": "corp", "enterprij": "ent", "entarapraij": "ent", "entarprais": "ent",
}

LEGAL = {"inc", "corp", "co", "ltd", "llc", "llp", "lp", "plc", "pvt", "pllc", "pc", "public",
         "gmbh", "ag", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ste",
         "bv", "nv", "opc", "lllp", "pty", "srl", "spa"}
NAME_STOP = {"the", "and", "of", "a", "an", "de", "la", "le", "les", "du", "des", "l", "d"}

_DBA_RE = re.compile(
    r"\b(?:d\s*/\s*b\s*/\s*a|d\s*\.\s*b\s*\.\s*a\.?|dba|doing business as|trading as|"
    r"f\s*/\s*k\s*/\s*a|n\s*/\s*k\s*/\s*a|nka|"
    r"t\s*/\s*a|a\s*\.\s*k\s*\.\s*a\.?|aka|also known as|formerly known as|fka|formerly)\b")
_ID_RE = re.compile(r"[(\[]?\s*\b(?:id|ref|reg|no)\s*[:#.]\s*[a-z0-9\-]+\s*[)\]]?")
_WEB_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:co\.in|org\.in|net\.in|c[o0]m|in|n[e3]t|[o0]rg|co|io|biz|fr|us|info|app|shop|store)\b")
_HANDLE_RE = re.compile(r"(?:^|\s)@([a-z0-9_.]+)")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"})


def _deleet(t):
    """'5upertech' -> 'supertech', 'ch1ts' -> 'chits' (exactly one digit inside a word)."""
    digits = sum(ch.isdigit() for ch in t)
    if digits == 1 and len(t) >= 4 and sum(ch.isalpha() for ch in t) >= 3:
        return t.translate(_LEET)
    return t


def _name_tokens(s: str):
    s = s.replace("&", " and ").replace("+", " and ")
    s = re.sub(r"(?<=[a-z0-9])'(?=[a-z])", "", s)          # mcdonald's -> mcdonalds
    s = re.sub(r"[^a-z0-9]+", " ", s)
    out, seen = [], set()
    for t in _merge_initials(s.split()):
        for u in NAME_MAP.get(_deleet(t), _deleet(t)).split():
            if u not in seen:                                  # drop repeated words ("vii vii")
                seen.add(u); out.append(u)
    return out


def _core(toks):
    core = [t for t in toks if t not in LEGAL and t not in NAME_STOP]
    return core if core else [t for t in toks if t not in NAME_STOP] or toks


def normalize_name(raw):
    indic = has_indic(str(raw))
    s = _clean_base(raw)
    s = _ID_RE.sub(" ", s)
    web = 0
    if _WEB_RE.search(s) or _HANDLE_RE.search(s):
        web = 1
        s = _WEB_RE.sub(lambda m: " " + m.group(1).replace("-", " ") + " ", s)
        s = _HANDLE_RE.sub(lambda m: " " + m.group(1).replace("_", " ").replace(".", " ") + " ", s)
    s = re.sub(r"[()\[\]{}<>|*~^\"]+", " ", s)                # wrappers / junk symbols, keep content
    parts = _DBA_RE.split(s)
    alts = [p for p in parts[1:] if p.strip()]
    full = _name_tokens(_DBA_RE.sub(" ", s))
    core = _core(_name_tokens(parts[0]))
    if not core:
        core = _core(full)
    alt = _core(_name_tokens(" ".join(alts))) if alts else []
    acr = "".join(t[0] for t in core if t and not t.isdigit()) if len(core) >= 2 else ""
    core_s = " ".join(core)
    return {
        "n_full": " ".join(full),
        "n_core": core_s,
        "n_alt": " ".join(alt),
        "n_acr": acr,
        "n_skel": skeleton(core_s),
        "n_nums": " ".join(t for t in full if t.isdigit()),
        "n_web": web,
        "n_indic": int(indic),
    }


# addresses
ADDR_MAP = {
    # street types
    "street": "st", "str": "st", "strt": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "avn": "ave", "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "bvd": "blvd",
    "drive": "dr", "lane": "ln", "place": "pl", "plaza": "plz", "court": "ct",
    "square": "sq", "highway": "hwy", "parkway": "pkwy", "expressway": "expy",
    "terrace": "ter", "circle": "cir", "crescent": "cres", "route": "rte",
    "suite": "ste", "apartment": "apt", "building": "bldg", "bldng": "bldg",
    "floor": "fl", "flr": "fl", "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "mount": "mt", "fort": "ft", "saint": "st",
    "number": "", "no": "", "num": "", "nos": "", "city": "",
    # India
    "ngr": "nagar", "nagr": "nagar", "mrg": "marg", "sector": "sec", "sect": "sec",
    "phase": "ph", "clny": "colony", "bazaar": "bazar", "bzr": "bazar",
    "chauk": "chowk", "chok": "chowk", "galli": "gali", "mohala": "mohalla",
    "crs": "cross", "mn": "main", "extension": "extn", "ext": "extn",
    "industrial": "indl", "estate": "est", "complex": "cmplx", "opposite": "opp",
    "nr": "near", "district": "dist", "distt": "dist", "village": "vill",
    "vpo": "vill", "post": "po", "tehsil": "teh", "taluka": "tal", "taluk": "tal",
    "bangalore": "bengaluru", "bombay": "mumbai", "madras": "chennai",
    "calcutta": "kolkata", "gurgaon": "gurugram", "pondicherry": "puducherry",
    "trivandrum": "thiruvananthapuram", "baroda": "vadodara", "allahabad": "prayagraj",
    "mysore": "mysuru", "mangalore": "mangaluru", "poona": "pune",
    "rajasthan": "rj", "raj": "rj", "maharashtra": "mh", "karnataka": "ka",
    "gujarat": "gj", "punjab": "pb", "haryana": "hr", "kerala": "kl", "delhi": "dl",
    "telangana": "ts", "bihar": "br", "odisha": "od", "orissa": "od", "assam": "as",
    "chandigarh": "ch", "goa": "ga", "uttarakhand": "uk", "jharkhand": "jh",
    "tg": "ts", "chhattisgarh": "cg", "chattisgarh": "cg", "puducherry": "py",
    "manipur": "mn", "meghalaya": "ml", "tripura": "tr", "nagaland": "nl", "sikkim": "sk",
    "mizoram": "mz", "gurgaon": "gurugram", "cuddapah": "kadapa", "trichy": "tiruchirappalli",
    "hno": "", "khno": "kh", "khasra": "kh",
    # France
    "chemin": "chem", "allee": "all", "impasse": "imp", "quai": "qu",
    "faubourg": "fbg", "residence": "res", "cedex": "",
    # connectors
    "et": "and",
}
ADDR_PHRASES = {
    "tamil nadu": "tn", "uttar pradesh": "up", "madhya pradesh": "mp",
    "andhra pradesh": "ap", "himachal pradesh": "hp", "west bengal": "wb",
    "new delhi": "delhi", "jammu and kashmir": "jk", "arunachal pradesh": "ar",
    "uttar pradesh": "up", "utar pradesh": "up",
    "new york": "ny", "new jersey": "nj", "new mexico": "nm", "new hampshire": "nh",
    "north carolina": "nc", "south carolina": "sc", "north dakota": "nd",
    "south dakota": "sd", "west virginia": "wv", "rhode island": "ri",
    "district of columbia": "dc", "los angeles": "la", "san francisco": "sf",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl",
    "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in",
    "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
}
ADDR_MAP.update(US_STATES)
_IN_STATES = {"maharashtra": "mh", "karnataka": "ka", "rajasthan": "rj", "gujarat": "gj", "punjab": "pb",
              "haryana": "hr", "kerala": "kl", "telangana": "ts", "bihar": "br", "odisha": "od",
              "assam": "as", "uttarakhand": "uk", "jharkhand": "jh", "chhattisgarh": "cg",
              "tamilnadu": "tn", "bengal": "wb", "pradesh": "pradesh", "delhi": "dl"}
_STATE_SKEL = {}   # filled after skeleton() is defined (see bottom of file)
ADDR_STOP = {"the", "and", "of", "de", "la", "le", "les", "du", "des", "l", "d", "at", ""}

_ORD_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
              "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
              "premier": "1", "premiere": "1", "deuxieme": "2", "troisieme": "3"}

_LANDMARK_RE = re.compile(
    r"\b(?:near|nr|opp|opposite|behind|beside|besides|next to|adjacent to|adj to|"
    r"in front of|close to|across from|facing|landmark|pres de|a cote de|en face de|face a)"
    r"\b[^,;]*")
_REL_RE = re.compile(r"\b(?:c|s|r|w|d|h)\s*/\s*o\b\.?|\bcare of\b")
_PIN_RE1 = re.compile(r"\b(?:pin\s*code|pincode|pin|zip|zip\s*code|postal\s*code|code\s*postal|cp)"
                      r"\s*[:\-.]?\s*(\d{3})\s?(\d{3})\b")
_PIN_RE2 = re.compile(r"[-,]\s*(\d{3})\s(\d{3})\s*$")
_ORD_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|ere|e|eme|me)\b")
_PHRASE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, ADDR_PHRASES), key=len, reverse=True)) + r")\b")


def _addr_tokens(s: str):
    s = _PHRASE_RE.sub(lambda m: " " + ADDR_PHRASES[m.group(1)] + " ", s)
    s = _ORD_RE.sub(r"\1", s)
    s = s.replace("&", " and ").replace("#", " ")
    s = re.sub(r"(?<=[a-z0-9])'(?=[a-z])", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", s)
    toks = _merge_initials(s.split())
    out = []
    for i, t in enumerate(toks):
        t = _ORD_WORDS.get(t, t)
        if t.isdigit():
            t = t.lstrip("0") or "0"                        # "00140" == "140"
        # contextual street-type abbreviations: "12 r victor hugo" -> rue
        if t in _CTX_ABBR and i > 0 and (toks[i - 1].isdigit() or toks[i - 1] in ("bis", "ter")):
            t = _CTX_ABBR[t]
        m = ADDR_MAP.get(t)
        if m is None:
            m = _STATE_SKEL.get(skeleton(t), t) if len(t) >= 5 else t
        out.extend(m.split())
    return out


_CTX_ABBR = {"r": "rue", "fg": "fbg", "che": "chem", "rte": "rte", "all": "all"}
# street-type / generic tokens: dropped in the abbreviation-agnostic address key
ADDR_GENERIC = {
    # canonical street types / unit words (values produced by ADDR_MAP) + raw forms
    "st", "rd", "ave", "blvd", "dr", "ln", "pl", "plz", "ct", "sq", "hwy", "pkwy", "expy",
    "ter", "cir", "cres", "rte", "ste", "apt", "unit", "bldg", "fl", "n", "s", "e", "w",
    "ne", "nw", "se", "sw", "rue", "chem", "all", "imp", "qu", "fbg", "res", "bis",
    "road", "street", "avenue", "shop", "plot", "flat", "h", "gali", "marg", "nagar",
    "chowk", "bazar", "colony", "extn", "indl", "est", "cmplx", "area",
    "sec", "ph", "po", "vill", "dist", "teh", "tal", "mohalla", "near", "opp", "behind"}


def address_key(tokens):
    """Abbreviation-agnostic key: numbers + distinctive words (len>2, not a street type).
    Makes the matcher robust to abbreviation styles never seen in training (e.g. French 'R.')."""
    return " ".join(t for t in tokens if t.isdigit() or (len(t) > 2 and t not in ADDR_GENERIC))


_NULL_RE = re.compile(r"<\s*null\s*>|\bnull\b|\bn\s*/\s*a\b|\bnone\b|\bnot available\b")


def normalize_address(raw):
    s = _clean_base(raw)
    s = _NULL_RE.sub(" ", s)
    s = _REL_RE.sub(" ", s)
    m = _PIN_RE1.search(s)                   # postal codes are almost never present in this data:
    labelled_postal = (m.group(1) + m.group(2)) if m else ""   # only trust explicitly labelled ones
    s = _PIN_RE1.sub(" ", s)
    has_lm = 1 if _LANDMARK_RE.search(s) else 0
    full = _addr_tokens(s)
    nolm = _addr_tokens(_LANDMARK_RE.sub(" ", s))
    if not nolm:
        nolm = full
    digits = [t for t in full if t.isdigit()]
    postal = labelled_postal
    nums = " ".join(dict.fromkeys(t.lstrip("0") or "0" for t in digits if t != postal and len(t) <= 6))
    nolm_s = " ".join(nolm)
    return {
        "a_full": " ".join(full),
        "a_nolm": nolm_s,
        "a_tok": " ".join(dict.fromkeys(t for t in nolm if t not in ADDR_STOP and not t.isdigit())),
        "a_postal": postal,
        "a_nums": nums,
        "a_first_num": next(((t.lstrip("0") or "0") for t in nolm if t.isdigit() and t != postal), ""),
        "a_skel": skeleton(nolm_s),
        "a_key": address_key(nolm),
        "a_lm": has_lm,
    }


# dataframe-level
def _norm_chunk(args):
    names, addrs = args
    return [normalize_name(v) for v in names], [normalize_address(v) for v in addrs]


def normalize_records(df: pd.DataFrame, n_jobs: int = 1, chunk: int = 200_000) -> pd.DataFrame:
    """Adds all normalised columns. Works on unique raw strings; parallel when n_jobs > 1."""
    df = df.copy()
    for col in ("business_name", "business_address", "country"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    un = pd.unique(df["business_name"]).tolist()
    ua = pd.unique(df["business_address"]).tolist()
    jobs = []
    for i in range(0, max(len(un), len(ua)), chunk):
        jobs.append((un[i:i + chunk], ua[i:i + chunk]))
    if n_jobs > 1 and len(jobs) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        # spawn: fresh workers, immune to the fork-after-threads deadlock (LightGBM/OpenMP)
        ex = ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("spawn"))
        try:
            res = list(ex.map(_norm_chunk, jobs, timeout=5400))   # give up after 90 min
            ex.shutdown(wait=True)
        except Exception as e:                         # stuck or crashed -> kill workers, go serial
            print(f"parallel normalisation failed ({e!r}); using 1 process", flush=True)
            for proc in list(getattr(ex, "_processes", {}).values()):
                proc.kill()
            ex.shutdown(wait=False, cancel_futures=True)
            res = [_norm_chunk(j) for j in jobs]
    else:
        res = [_norm_chunk(j) for j in jobs]
    rn = [r for part in res for r in part[0]]
    ra = [r for part in res for r in part[1]]

    code_n = pd.Index(un).get_indexer(df["business_name"])
    code_a = pd.Index(ua).get_indexer(df["business_address"])
    for k in rn[0].keys() if rn else []:
        col = np.array([r[k] for r in rn], dtype=object if isinstance(rn[0][k], str) else np.int8)
        df[k] = col[code_n]
    for k in ra[0].keys() if ra else []:
        col = np.array([r[k] for r in ra], dtype=object if isinstance(ra[0][k], str) else np.int8)
        df[k] = col[code_a]
    df["country_n"] = pd.Series([norm_country(c) for c in pd.unique(df["country"])],
                                index=pd.unique(df["country"])).reindex(df["country"]).values
    return df.reset_index(drop=True)


_STATE_SKEL.update({skeleton(k): v for k, v in _IN_STATES.items() if len(k) >= 5})
_STATE_SKEL.update({skeleton(k): v for k, v in US_STATES.items() if len(k) >= 6})


# word segmentation of concatenated website names ("dentistrycenterpediatric")
def build_vocab(cores, min_count=2):
    from collections import Counter
    c = Counter(t for x in cores for t in x.split() if len(t) >= 2 and not t.isdigit())
    tot = sum(c.values()) or 1
    return {w: math.log(n / tot) for w, n in c.items() if n >= min_count}


def segment(word, vocab, max_len=20):
    """Viterbi unigram segmentation; returns the input unchanged if it cannot be improved."""
    n = len(word)
    if n < 6 or word in vocab:
        return word
    unk = min(vocab.values(), default=-20.0) - 5.0
    best = [0.0] + [-1e18] * n
    back = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):
            w = word[j:i]
            sc = vocab.get(w, unk * len(w) if len(w) > 1 else unk * 3)
            if best[j] + sc > best[i]:
                best[i], back[i] = best[j] + sc, j
    parts, i = [], n
    while i > 0:
        parts.append(word[back[i]:i]); i = back[i]
    parts = parts[::-1]
    return " ".join(parts) if len(parts) > 1 and all(p in vocab for p in parts) else word


def segment_web_names(df, vocab):
    """Split concatenated single-token names (mostly websites/handles) into words."""
    core = df["n_core"].values.copy()
    mask = np.array([(" " not in c) and len(c) >= 8 for c in core]) & (df["n_web"].values == 1)
    for i in np.where(mask)[0]:
        seg = segment(core[i], vocab)
        if seg != core[i]:
            core[i] = seg
    df["n_core"] = core
    df["n_skel"] = [skeleton(c) for c in core] if mask.any() else df["n_skel"]
    return int(mask.sum())
