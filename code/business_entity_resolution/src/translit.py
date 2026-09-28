"""
Converts text in the 9 main Indian scripts into Latin letters, using simple rules.

Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and
Malayalam all use the same layout inside their Unicode blocks, so a letter at
a given position sounds the same in every script. That means one lookup
table covers all nine.

The output is an approximate spelling of how the text sounds
("प्राइवेट लिमिटेड" -> "praivet limited", "ఏస్ పవర్" -> "es pavar").
It is then compared with English names using the sound-based and fuzzy
features. No outside data or library is used.
"""
import re

BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
INDIC_RE = re.compile(r"[\u0900-\u0D7F]")
_ZW_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff\u00ad]")

VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
          0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
          0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li"}
CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
        0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
        0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
        0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
        0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
        0x38: "s", 0x39: "h",
        0x58: "k", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
          0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
          0x4C: "au", 0x62: "li", 0x63: "li", 0x57: "au", 0x56: "ai", 0x55: ""}
VIRAMA, NUKTA = 0x4D, 0x3C
NUKTA_MAP = {"j": "z", "ph": "f", "s": "sh", "k": "k", "kh": "kh", "g": "g", "d": "r", "dh": "rh"}
SIGNS = {0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n", 0x71: ""}          
EXTRA = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k",  
         0x09CE: "t"}                                                                   
_LABIALS = set("pbm")


def _decode(ch):
    o = ord(ch)
    if o in EXTRA:
        return "X", EXTRA[o]
    for b in BLOCKS:
        if b <= o < b + 0x80:
            off = o - b
            if off in CONS:
                return "C", CONS[off]
            if off in VOWELS:
                return "V", VOWELS[off]
            if off in MATRAS:
                return "M", MATRAS[off]
            if off == VIRAMA:
                return "H", ""
            if off == NUKTA:
                return "N", ""
            if off in SIGNS:
                return "S", SIGNS[off]
            if 0x66 <= off <= 0x6F:
                return "D", str(off - 0x66)
            return "O", ""
    return "L", ch


def transliterate(text: str) -> str:
    """Leaves Latin text as it is, converts Indian-script text by sound, and drops the silent 'a' at the end of words."""
    if not text or not INDIC_RE.search(text):
        return _ZW_RE.sub("", text or "")
    text = _ZW_RE.sub("", text)
    items = [_decode(c) for c in text]
    out = []
    i, n = 0, len(items)
    while i < n:
        kind, val = items[i]
        if kind == "C":
            j = i + 1
            if j < n and items[j][0] == "N":
                val = NUKTA_MAP.get(val, val)
                j += 1
            nxt = items[j][0] if j < n else None
            if nxt == "H":
                out.append(val); i = j + 1; continue
            if nxt == "M":
                out.append(val + items[j][1]); i = j + 1; continue
           
            end = j >= n or items[j][0] not in ("C", "S", "V", "M", "N", "H", "X")
            out.append(val if end else val + "a")
            i = j
            continue
        if kind == "S" and val == "h" and i + 1 < n and items[i + 1][0] == "C" and items[i + 1][1] in ("p", "ph"):
            items[i + 1] = ("C", "f"); i += 1; continue       
        if kind == "S" and val == "n":
            nxt = items[i + 1][1][:1] if i + 1 < n and items[i + 1][0] == "C" else ""
            out.append("m" if nxt in _LABIALS else "n")
        elif kind in ("V", "M", "X", "D", "S", "L"):
            out.append(val)
        i += 1
    return "".join(out)


def has_indic(text: str) -> bool:
    return bool(text) and bool(INDIC_RE.search(text))
