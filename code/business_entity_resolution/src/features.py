"""
Turns each (S1 record, candidate) pair into a row of numbers the model can learn from.

None of the features depend on the country, so the model can also be used on France,
which it never saw in training. When information is missing (e.g. no address), the
value is left as NaN instead of pretending it's a similarity score; LightGBM handles
missing values on its own.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein


def _pd(a, b, scorer, scale=1.0):
    """Element-wise similarity of two equal-length string arrays (multithreaded)."""
    a, b = list(a), list(b)
    try:
        out = process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
    except Exception:                                  # older rapidfuzz
        out = np.array([scorer(x, y) for x, y in zip(a, b)], dtype=np.float32)
    out = out.astype(np.float32) / scale
    empty = np.array([(not x) or (not y) for x, y in zip(a, b)])
    out[empty] = np.nan
    return out


def build_idf(texts):
    """IDF over whitespace tokens of an iterable of strings."""
    df = Counter()
    n = 0
    for s in texts:
        df.update(set(s.split()))
        n += 1
    n = max(n, 1)
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    return idf, math.log(n + 1) + 1.0


def _sets(arr):
    return [frozenset(x.split()) for x in arr]


def _set_feats(A, B, idf, default_idf):
    A, B = _sets(A), _sets(B)
    n = len(A)
    jac = np.full(n, np.nan, np.float32)
    idf_ov = np.full(n, np.nan, np.float32)
    miss_max = np.full(n, np.nan, np.float32)
    miss_sum = np.full(n, np.nan, np.float32)
    n_common = np.zeros(n, np.float32)
    for i, (x, y) in enumerate(zip(A, B)):
        if not x or not y:
            continue
        inter, uni = x & y, x | y
        jac[i] = len(inter) / len(uni)
        w = [idf.get(t, default_idf) for t in uni]
        wi = sum(idf.get(t, default_idf) for t in inter)
        idf_ov[i] = wi / sum(w)
        miss = [idf.get(t, default_idf) for t in uni - inter]
        miss_max[i] = max(miss) if miss else 0.0
        miss_sum[i] = sum(miss)
        n_common[i] = len(inter)
    return jac, idf_ov, miss_max, miss_sum, n_common


def _num_feats(A, B):
    A, B = _sets(A), _sets(B)
    n = len(A)
    jac = np.full(n, np.nan, np.float32)
    conflict = np.full(n, np.nan, np.float32)
    for i, (x, y) in enumerate(zip(A, B)):
        if x and y:
            inter = len(x & y)
            jac[i] = inter / len(x | y)
            conflict[i] = 1.0 if inter == 0 else 0.0
    return jac, conflict


def _num_struct(A, B):
    """House-number structure. Harmless noise: truncation/padding ('a207'->'a20', '550'->'550 552').
    Sibling businesses: a single changed digit ('617' vs '618', '1068/118/1' vs '1068/118/12')."""
    n = len(A)
    out = np.full((n, 5), np.nan, np.float32)
    for i, (x, y) in enumerate(zip(A, B)):
        xa, yb = x.split(), y.split()
        if not xa or not yb:
            continue
        sx, sy = set(xa), set(yb)
        ox, oy = sx - sy, sy - sx
        prefix = near = 0.0
        for u in ox:
            for v in oy:
                if u.startswith(v) or v.startswith(u):
                    prefix = 1.0
                elif len(u) == len(v) and sum(c1 != c2 for c1, c2 in zip(u, v)) == 1:
                    near = 1.0
        out[i] = (len(ox), len(oy), prefix, near, 1.0 if xa == yb else 0.0)
    return out


def _eq_or_nan(a, b):
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    out = (a == b).astype(np.float32)
    out[(a == "") | (b == "")] = np.nan
    return out


def build_features(cand: pd.DataFrame, s1: pd.DataFrame, tg: pd.DataFrame, idf_n, idf_a):
    qi, ti = cand["qi"].values, cand["ti"].values
    A = s1.iloc[qi].reset_index(drop=True)
    B = tg.iloc[ti].reset_index(drop=True)
    F = pd.DataFrame(index=cand.index)

    # name similarity
    ac, bc = A["n_core"].values, B["n_core"].values
    F["n_ratio"] = _pd(ac, bc, fuzz.ratio, 100)
    F["n_tsort"] = _pd(ac, bc, fuzz.token_sort_ratio, 100)
    F["n_tset"] = _pd(ac, bc, fuzz.token_set_ratio, 100)
    F["n_partial"] = _pd(ac, bc, fuzz.partial_ratio, 100)
    F["n_wratio"] = _pd(ac, bc, fuzz.WRatio, 100)
    F["n_jw"] = _pd(ac, bc, JaroWinkler.normalized_similarity)
    F["n_lev"] = _pd(ac, bc, Levenshtein.normalized_similarity)
    F["n_full_tset"] = _pd(A["n_full"].values, B["n_full"].values, fuzz.token_set_ratio, 100)
    F["n_full_ratio"] = _pd(A["n_full"].values, B["n_full"].values, fuzz.ratio, 100)
    F["n_skel_ratio"] = _pd(A["n_skel"].values, B["n_skel"].values, fuzz.ratio, 100)
    F["n_skel_tset"] = _pd(A["n_skel"].values, B["n_skel"].values, fuzz.token_set_ratio, 100)
    # compare without spaces, so "Mc Donalds" and "McDonalds" match
    F["n_nospace"] = _pd([x.replace(" ", "") for x in ac], [y.replace(" ", "") for y in bc], fuzz.ratio, 100)

    # if either side has a trade name (d/b/a), use whichever name version matches best
    aa = np.where(A["n_alt"].values != "", A["n_alt"].values, ac)
    ba = np.where(B["n_alt"].values != "", B["n_alt"].values, bc)
    combos = [F["n_tset"].values,
              _pd(aa, bc, fuzz.token_set_ratio, 100),
              _pd(ac, ba, fuzz.token_set_ratio, 100),
              _pd(aa, ba, fuzz.token_set_ratio, 100)]
    F["n_alt_best"] = np.nanmax(np.vstack(combos), axis=0)
    F["n_has_alt"] = ((A["n_alt"].values != "") | (B["n_alt"].values != "")).astype(np.float32)

    j, ov, mm, ms, nc = _set_feats(ac, bc, *idf_n)
    F["n_jac"], F["n_idf_ov"], F["n_miss_max"], F["n_miss_sum"], F["n_common"] = j, ov, mm, ms, nc
    F["n_exact_core"] = (ac == bc).astype(np.float32)
    F["n_exact_full"] = (A["n_full"].values == B["n_full"].values).astype(np.float32)
    F["n_first_tok_eq"] = np.array([(x.split()[:1] == y.split()[:1]) if x and y else np.nan
                                    for x, y in zip(ac, bc)], dtype=np.float32)
    ans, bns = [x.replace(" ", "") for x in ac], [y.replace(" ", "") for y in bc]
    F["n_acr_match"] = np.array(
        [1.0 if (xa and xa == yn) or (ya and ya == xn) else 0.0
         for xa, ya, xn, yn in zip(A["n_acr"].values, B["n_acr"].values, ans, bns)], np.float32)
    nj, ncf = _num_feats(A["n_nums"].values, B["n_nums"].values)
    F["n_num_jac"], F["n_num_conflict"] = nj, ncf
    la = np.array([len(x) for x in ac], np.float32); lb = np.array([len(y) for y in bc], np.float32)
    F["n_len_ratio"] = np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)
    F["n_ntok_diff"] = np.abs(np.array([x.count(" ") for x in ac]) - np.array([y.count(" ") for y in bc])).astype(np.float32)
    F["n_freq_a"] = A["n_core_freq"].values.astype(np.float32)     # how common this exact name is among S1 records
    F["n_freq_b"] = B["n_core_freq"].values.astype(np.float32)     # how common it is among S2/S3 records
    F["n_web_b"] = B["n_web"].values.astype(np.float32)
    F["n_indic_b"] = B["n_indic"].values.astype(np.float32)
    F["n_empty_b"] = (bc == "").astype(np.float32)

    # compare names after removing words that also appear in the address (like the city)
    # e.g. "bordeaux compagnie" vs "bordeaux theatre" only share the city, which proves nothing
    atok_a, atok_b = _sets(A["a_tok"].values), _sets(B["a_tok"].values)
    ca, cb = _sets(ac), _sets(bc)
    da = [" ".join(sorted(x - (u | v))) for x, u, v in zip(ca, atok_a, atok_b)]
    db = [" ".join(sorted(x - (u | v))) for x, u, v in zip(cb, atok_a, atok_b)]
    F["n_dist_tset"] = _pd(da, db, fuzz.token_set_ratio, 100)
    F["n_dist_ratio"] = _pd(da, db, fuzz.token_sort_ratio, 100)
    F["n_dist_empty"] = np.array([(not x) + (not y) for x, y in zip(da, db)], np.float32)
    F["n_city_share_a"] = np.array([len(x & (u | v)) / max(len(x), 1) for x, u, v in zip(ca, atok_a, atok_b)], np.float32)
    F["n_city_share_b"] = np.array([len(x & (u | v)) / max(len(x), 1) for x, u, v in zip(cb, atok_a, atok_b)], np.float32)

    # address similarity
    F["a_ratio"] = _pd(an, bn, fuzz.ratio, 100)
    F["a_tsort"] = _pd(an, bn, fuzz.token_sort_ratio, 100)
    F["a_tset"] = _pd(an, bn, fuzz.token_set_ratio, 100)
    F["a_partial"] = _pd(an, bn, fuzz.partial_ratio, 100)
    F["a_jw"] = _pd(an, bn, JaroWinkler.normalized_similarity)
    F["a_full_tset"] = _pd(A["a_full"].values, B["a_full"].values, fuzz.token_set_ratio, 100)
    ak, bk = A["a_key"].values, B["a_key"].values
    F["a_key_tset"] = _pd(ak, bk, fuzz.token_set_ratio, 100)
    F["a_key_tsort"] = _pd(ak, bk, fuzz.token_sort_ratio, 100)
    F["a_skel_tset"] = _pd(A["a_skel"].values, B["a_skel"].values, fuzz.token_set_ratio, 100)
    j, ov, mm, ms, nc = _set_feats(A["a_tok"].values, B["a_tok"].values, *idf_a)
    F["a_jac"], F["a_idf_ov"], F["a_miss_max"], F["a_miss_sum"], F["a_common"] = j, ov, mm, ms, nc
    F["a_postal_eq"] = _eq_or_nan(A["a_postal"].values, B["a_postal"].values)
    F["a_postal3_eq"] = _eq_or_nan([p[:3] for p in A["a_postal"].values], [p[:3] for p in B["a_postal"].values])
    F["a_postal_missing"] = ((A["a_postal"].values == "").astype(np.float32)
                             + (B["a_postal"].values == "").astype(np.float32))
    nj, ncf = _num_feats(A["a_nums"].values, B["a_nums"].values)
    F["a_num_jac"], F["a_num_conflict"] = nj, ncf
    NS = _num_struct(A["a_nums"].values, B["a_nums"].values)
    for j, c in enumerate(["a_num_only_a", "a_num_only_b", "a_num_prefix", "a_num_1digit", "a_num_seq_eq"]):
        F[c] = NS[:, j]
    F["a_first_num_eq"] = _eq_or_nan(A["a_first_num"].values, B["a_first_num"].values)
    F["a_lm_any"] = (A["a_lm"].values + B["a_lm"].values).astype(np.float32)
    F["a_empty_any"] = ((an == "") | (bn == "")).astype(np.float32)
    la = np.array([len(x) for x in an], np.float32); lb = np.array([len(y) for y in bn], np.float32)
    F["a_len_ratio"] = np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)

    # combined name + address signals
    F["x_name_addr"] = np.nan_to_num(F["n_tset"].values) * np.nan_to_num(F["a_tset"].values)
    F["x_name_akey"] = np.nan_to_num(F["n_tset"].values) * np.nan_to_num(F["a_key_tset"].values)
    F["x_name_postal"] = np.nan_to_num(F["n_tset"].values) * np.nan_to_num(F["a_postal_eq"].values, nan=0.5)

    # scores carried over from the blocking step
    for c in ("k_score", "k_nkeys", "rr_score", "k_rank", "k_gap_s1", "k_rel_s1",
              "k_rank_t", "k_gap_t", "k_ncand_t", "n_cand_s1src"):
        F[c] = cand[c].values.astype(np.float32)
    F["is_s3"] = (cand["src"].values == "S3").astype(np.float32)
    return F.astype(np.float32)
