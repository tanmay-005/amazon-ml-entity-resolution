"""
Finds a short list of likely matches (candidates) for every S1 record.

Comparing every S1 record with every S2/S3 record is impossible at this size
(millions x millions), so instead each record is turned into a handful of short
"keys", and two records only become candidates if they share some keys:

  w|<word>          a distinctive word from the name
  t|<skel>          the sound-alike version of a name word (catches typos and transliteration)
  b|<skelA>|<skelB> two name words together, in any order
  ns|<nospace>      the whole name without spaces ("acepower" = "ace power"), or its acronym
  n6|<prefix>       the first 6 letters of the name without spaces
  an|<num>|<word>   a house number plus an address word (helps with renamed businesses)
  aw|<wordA>|<wordB> two address words together

Very common keys (shared by more than `key_cap` records, e.g. "shri", "sai") are
ignored because they don't tell businesses apart. Each pair gets a score based on
how many rare keys it shares; the best 40 per source are then re-scored with fuzzy
string matching and the top `k_per_source` are kept. The work is done in chunks so
memory use stays under control.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from normalize import skeleton


@lru_cache(maxsize=1 << 20)
def _sk(t):
    return skeleton(t)


def _record_keys(core, alt, nums, akey):
    ks = set()
    toks = [t for t in (core + " " + alt).split() if len(t) >= 2 and not t.isdigit()]
    for t in toks:
        if len(t) >= 3:
            ks.add("w|" + t)                            # exact word (rare words stay selective)
    sk = list(dict.fromkeys(_sk(t) for t in toks))
    for x, t in zip(sk, toks):
        if len(t) >= 3 and len(x) >= 2:
            ks.add("t|" + x)
    u = sk[:4]
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            a, b = (u[i], u[j]) if u[i] < u[j] else (u[j], u[i])
            ks.add("b|" + a + "|" + b)
    ns = core.replace(" ", "")
    if len(ns) >= 4:
        ks.add("ns|" + ns)
    ct = core.split()
    if len(ct) >= 3:                                    # acronym: "southern ram care" <-> "src"
        ks.add("ns|" + "".join(t[0] for t in ct))
    if len(ns) >= 7:
        ks.add("n6|" + ns[:6])
    words = list(dict.fromkeys(_sk(w) for w in akey.split() if len(w) >= 4 and not w.isdigit()))[:4]
    for n in nums.split()[:3]:                       # every house/unit number x address word
        for w in words:
            ks.add("an|" + n + "|" + w)
    for i in range(len(words)):                      # address word pairs: number-independent
        for j in range(i + 1, len(words)):
            a, b = (words[i], words[j]) if words[i] < words[j] else (words[j], words[i])
            ks.add("aw|" + a + "|" + b)
    return ks


def make_keys(df: pd.DataFrame, chunk=500_000):
    """Returns (keys uint64, owner int64) for all rows of df, hashed per chunk to bound RAM."""
    out_k, out_o = [], []
    cols = [df[c].values for c in ("n_core", "n_alt", "a_nums", "a_key")]
    for st in range(0, len(df), chunk):
        ks, ow = [], []
        for i in range(st, min(st + chunk, len(df))):
            for k in _record_keys(cols[0][i], cols[1][i], cols[2][i], cols[3][i]):
                ks.append(k)
                ow.append(i)
        if ks:
            out_k.append(pd.util.hash_array(np.array(ks, dtype=object)))
            out_o.append(np.array(ow, dtype=np.int64))
    if not out_k:
        return np.empty(0, np.uint64), np.empty(0, np.int64)
    return np.concatenate(out_k), np.concatenate(out_o)


def build_target_index(tg: pd.DataFrame, key_cap: int):
    tk, tt = make_keys(tg)
    uk, inv, cnt = np.unique(tk, return_inverse=True, return_counts=True)
    keep = (cnt <= key_cap)[inv]
    tk, tt = tk[keep], tt[keep]
    o = np.argsort(tk, kind="stable")
    return {"k": tk[o], "t": tt[o], "n_keys": len(uk), "n_dropped": int((cnt > key_cap).sum()),
            "N": len(tg)}


def _rerank(cand, s1, tg, k):
    """Stage-2 blocking: cheap fuzzy score on the key-based pool, keep top-k per (S1, source)."""
    from rapidfuzz import fuzz, process
    def sim(a, b, scorer):
        return process.cpdist(list(a), list(b), scorer=scorer, workers=-1, dtype=np.float32) / 100.0
    qi, ti = cand["qi"].values, cand["ti"].values
    ac, bc = s1["n_core"].values[qi], tg["n_core"].values[ti]
    aa, ba = s1["n_alt"].values[qi], tg["n_alt"].values[ti]
    name = np.maximum.reduce([
        sim(ac, bc, fuzz.token_set_ratio),
        sim([x.replace(" ", "") for x in ac], [y.replace(" ", "") for y in bc], fuzz.ratio),
        sim(s1["n_skel"].values[qi], tg["n_skel"].values[ti], fuzz.token_set_ratio),
        sim(np.where(aa != "", aa, ac), bc, fuzz.token_set_ratio),
        sim(ac, np.where(ba != "", ba, bc), fuzz.token_set_ratio)])
    ak, bk = s1["a_key"].values[qi], tg["a_key"].values[ti]
    addr = sim(ak, bk, fuzz.token_set_ratio)
    addr[(ak == "") | (bk == "")] = 0.35
    af = sim(s1["a_nolm"].values[qi], tg["a_nolm"].values[ti], fuzz.token_set_ratio)
    af[(ak == "") | (bk == "")] = 0.0
    same_site = np.minimum(addr, af)                   # renamed business at the same address
    cand["rr_score"] = np.maximum(0.6 * name + 0.35 * addr + 0.05 * np.minimum(cand["k_score"].values / 20.0, 1.0),
                                  0.85 * same_site)
    cand = cand.sort_values(["qi", "src", "rr_score"], ascending=[True, True, False])
    cand["k_rank"] = cand.groupby(["qi", "src"]).cumcount().astype(np.float32)
    return cand[cand["k_rank"] < k].reset_index(drop=True)


def generate_candidates(s1: pd.DataFrame, tg: pd.DataFrame, cfg, log=print):
    """s1, tg: normalised frames of ONE country partition. Returns candidate frame."""
    idx = build_target_index(tg, cfg.key_cap)
    log(f"    target keys: {idx['n_keys']:,} distinct, {idx['n_dropped']:,} dropped (> cap {cfg.key_cap})")
    tk, tt, N = idx["k"], idx["t"], idx["N"]
    tsrc = (tg["src"].values == "S3").astype(np.int8)

    qk, qq = make_keys(s1)
    lo = np.searchsorted(tk, qk, "left")
    hi = np.searchsorted(tk, qk, "right")
    cnt = hi - lo
    hit = cnt > 0
    qq, lo, cnt = qq[hit], lo[hit], cnt[hit]
    w = np.log((N + 1.0) / (cnt + 1.0)).astype(np.float32)

    # chunk S1 records so that each join stays under `join_budget` rows
    per_rec = np.bincount(qq, weights=cnt, minlength=len(s1))
    cum = np.cumsum(per_rec)
    cut = np.searchsorted(cum, np.arange(cfg.join_budget, cum[-1] + cfg.join_budget, cfg.join_budget)) if len(cum) and cum[-1] > 0 else np.array([])
    edges = np.unique(np.r_[0, np.minimum(cut + 1, len(s1)), len(s1)])

    parts, n_pool = [], 0
    for a, b in zip(edges[:-1], edges[1:]):
        ks, ke = np.searchsorted(qq, a, "left"), np.searchsorted(qq, b, "left")
        if ke <= ks:
            continue
        c = cnt[ks:ke]
        total = int(c.sum())
        q_rep = np.repeat(qq[ks:ke], c)
        w_rep = np.repeat(w[ks:ke], c)
        offs = np.repeat(lo[ks:ke] - (np.cumsum(c) - c), c) + np.arange(total)
        t_rep = tt[offs]
        pid = q_rep * N + t_rep
        upid, inv = np.unique(pid, return_inverse=True)
        score = np.bincount(inv, weights=w_rep).astype(np.float32)
        nk = np.bincount(inv).astype(np.float32)
        q, t = upid // N, upid % N
        s = tsrc[t]
        o = np.lexsort((-score, s, q))
        q, t, s, score, nk = q[o], t[o], s[o], score[o], nk[o]
        g = q * 2 + s
        first = np.r_[True, g[1:] != g[:-1]]
        ar = np.arange(len(g))
        rank = ar - np.maximum.accumulate(np.where(first, ar, 0))
        keep = rank < cfg.k_pool
        n_pool += int(keep.sum())
        part = pd.DataFrame({"qi": q[keep], "ti": t[keep], "src": np.where(s[keep] == 1, "S3", "S2"),
                             "k_score": score[keep], "k_nkeys": nk[keep]})
        parts.append(_rerank(part, s1, tg, cfg.k_per_source))       # re-rank inside the chunk: bounded RAM

    cols = ["qi", "ti", "src", "k_score", "k_nkeys", "rr_score", "k_rank"]
    cand = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    log(f"    pool {n_pool:,} -> re-ranked top {cfg.k_per_source}/source")
    g = cand.groupby(["qi", "src"])["rr_score"]
    cand["k_gap_s1"] = cand["rr_score"] - g.transform("max")
    cand["k_rel_s1"] = cand["k_score"] / cand.groupby(["qi", "src"])["k_score"].transform("max").clip(lower=1e-6)
    gt = cand.groupby("ti")["rr_score"]
    cand["k_rank_t"] = gt.rank(ascending=False, method="first").astype(np.float32)
    cand["k_gap_t"] = cand["rr_score"] - gt.transform("max")
    cand["k_ncand_t"] = gt.transform("size").astype(np.float32)
    cand["n_cand_s1src"] = cand.groupby(["qi", "src"])["ti"].transform("size").astype(np.float32)
    n_none = len(s1) - cand["qi"].nunique()
    log(f"    candidates: {len(cand):,} ({len(cand) / max(len(s1), 1):.1f} per S1), S1 with none: {n_none:,}")
    return cand.sort_values(["qi", "src", "k_rank"]).reset_index(drop=True)
