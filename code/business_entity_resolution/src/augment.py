"""
Adds noise to the training data so the model copes better with France.

France only appears in the test set, and French addresses use short forms we never
see in training (like 'R.' for Rue, 'Bd', 'Imp.'). If the model only ever sees street
words match on true pairs, it starts treating any mismatch there as a sign of a
different business. To avoid that, we randomly shorten or drop street words and
legal suffixes in some S2/S3 training records. The labels stay the same, so the model
learns to lean on house numbers and the important name words instead.
This is only applied to the training data, never to the test data.
"""
import random
import re

from normalize import ADDR_GENERIC, ADDR_MAP, LEGAL, NAME_MAP, _clean_base

_NO_TOUCH = {"near", "opp", "behind", "n", "s", "e", "w", "ne", "nw", "se", "sw", "h"}
_STREETISH = (ADDR_GENERIC | {k for k, v in ADDR_MAP.items() if v in ADDR_GENERIC}) - _NO_TOUCH
_STREETISH -= {k for k, v in ADDR_MAP.items() if v in _NO_TOUCH}
_LEGALISH = LEGAL | {k for k, v in NAME_MAP.items() if set(v.split()) & LEGAL}
_WORD = re.compile(r"[A-Za-z\u00C0-\u017F]+\.?")


def _abbrev(word, rng):
    w = word.rstrip(".")
    r = rng.random()
    if r < 0.45:
        return w[:1] + "."
    if r < 0.75:
        return w[:2] + "." if len(w) > 2 else w
    if r < 0.9:
        return ""
    return w.upper()


def augment_address(addr, rng, p_word=0.7):
    def rep(m):
        w = m.group(0)
        if _clean_base(w.rstrip(".")) in _STREETISH and rng.random() < p_word:
            return _abbrev(w, rng)
        return w
    return re.sub(r"\s{2,}", " ", _WORD.sub(rep, addr)).strip()


def augment_name(name, rng, p_word=0.5):
    def rep(m):
        w = m.group(0)
        if _clean_base(w.rstrip(".")) in _LEGALISH and rng.random() < p_word:
            return "" if rng.random() < 0.5 else w[:2].upper()
        return w
    return re.sub(r"\s{2,}", " ", _WORD.sub(rep, name)).strip() or name


def augment_targets(tg, frac=0.3, seed=13):
    rng = random.Random(seed)
    tg = tg.copy()
    addrs, names = tg["business_address"].tolist(), tg["business_name"].tolist()
    for i in range(len(tg)):
        if rng.random() < frac:
            addrs[i] = augment_address(addrs[i], rng)
        if rng.random() < frac / 2:
            names[i] = augment_name(names[i], rng)
    tg["business_address"], tg["business_name"] = addrs, names
    return tg
