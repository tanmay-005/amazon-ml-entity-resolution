"""
Learns which word differences between two names matter, using the training labels.

Real matches often differ only by filler words like "center", "services",
"sri", "m/s" or "dr". Different businesses with similar names differ by words
that change the meaning, like "technologies", "ventures" or "jewellers".

For each word, we measure how much more often it shows up as a difference in
matching pairs than in non-matching pairs (a log-odds score). For each pair,
we then add up, and take the min and max of, the scores of the words that
appear in only one of the two names.

On training data, each half of the S1 records is scored using odds learned
from the other half, so a row never sees its own label.
"""
import math
from collections import Counter

import numpy as np


class TokenOdds:
    def __init__(self, min_count=20, alpha=1.0):
        self.min_count, self.alpha, self.lo = min_count, alpha, {}

    def fit(self, a_texts, b_texts, y):
        pos, neg = Counter(), Counter()
        for a, b, yy in zip(a_texts, b_texts, y):
            d = set(a.split()) ^ set(b.split())
            (pos if yy else neg).update(d)
        P, N = max(int(np.sum(y)), 1), max(len(y) - int(np.sum(y)), 1)
        al = self.alpha
        self.lo = {t: math.log((pos[t] + al) / (P + al)) - math.log((neg[t] + al) / (N + al))
                   for t in set(pos) | set(neg) if pos[t] + neg[t] >= self.min_count}
        return self

    def transform(self, a_texts, b_texts):
        n = len(a_texts)
        out = np.zeros((n, 5), np.float32)
        for i, (a, b) in enumerate(zip(a_texts, b_texts)):
            d = set(a.split()) ^ set(b.split())
            if not d:
                continue
            v = [self.lo[t] for t in d if t in self.lo]
            out[i] = (sum(v), min(v, default=0.0), max(v, default=0.0),
                      len(d) - len(v), len(d))
        return out


COLS = ["to_sum", "to_min", "to_max", "to_n_unknown", "to_n_diff"]


def cross_fit(a_texts, b_texts, y, groups):
    """Scores each half of the training rows using odds learned from the other half (split by S1 id)."""
    half = (groups % 2).astype(bool)
    out = np.zeros((len(y), 5), np.float32)
    for h in (False, True):
        tr, te = half != h, half == h
        m = TokenOdds().fit(a_texts[tr], b_texts[tr], y[tr])
        out[te] = m.transform(a_texts[te], b_texts[te])
    return out
