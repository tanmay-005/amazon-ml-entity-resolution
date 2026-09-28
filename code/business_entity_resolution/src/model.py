"""
Trains the LightGBM matcher and picks the decision rule.

The model gets one row of features per (S1, candidate) pair and predicts how likely
the two records are the same business. Training uses GroupKFold grouped by S1 record,
so all candidates of one business stay in the same fold and the validation score is
honest. A second-stage model (extra features based on the first model's scores) is
also included here, but it did not beat the first stage and is switched off.

Decision rule: a pair is accepted if its probability is at least t, at least r times
the best probability for that S1 record, and (optionally) the best S1 for that target.
t, r and the unique flag are chosen by directly maximising the competition metric
(macro F0.5, singletons included).
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.1, n_estimators=1000, num_leaves=63,
    min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_lambda=1.0, n_jobs=-1, verbose=-1, max_bin=63,
)


def cv_train(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_folds=5, seeds=(42,), log=print,
             n_estimators=None):
    warnings.filterwarnings("ignore", category=Warning, module="lightgbm")
    params = dict(LGB_PARAMS)
    if n_estimators:
        params["n_estimators"] = n_estimators
    oof = np.zeros(len(y), np.float64)
    models = []
    n_folds = max(2, min(n_folds, len(np.unique(groups))))
    for seed in seeds:
        gkf = GroupKFold(n_splits=n_folds)
        # shuffle groups deterministically per seed
        rng = np.random.RandomState(seed)
        uniq = np.unique(groups)
        perm = dict(zip(uniq, rng.permutation(len(uniq))))
        g2 = np.array([perm[g] for g in groups])
        for f, (tr, va) in enumerate(gkf.split(X, y, g2)):
            m = lgb.LGBMClassifier(random_state=seed + f, **params)
            m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
                  callbacks=[lgb.early_stopping(200, min_delta=1e-4, verbose=False)])
            oof[va] += m.predict_proba(X.iloc[va])[:, 1] / len(seeds)
            models.append(m)
            log(f"    seed {seed} fold {f}: best_iter={m.best_iteration_}")
    return oof, models


def fit_full(X, y, n_estimators, seed=42):
    """Trains one final model on all training pairs; faster to use on the test set than several fold models."""
    warnings.filterwarnings("ignore", category=Warning, module="lightgbm")
    params = dict(LGB_PARAMS); params["n_estimators"] = max(50, int(n_estimators))
    m = lgb.LGBMClassifier(random_state=seed, **params)
    m.fit(X, y)
    return m


def predict(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def context_features(cand: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"qi": cand["qi"].values, "ti": cand["ti"].values,
                      "src": cand["src"].values, "p": p})
    C = pd.DataFrame(index=d.index)
    C["p1"] = d["p"]
    g = d.groupby("qi")["p"]
    C["p1_max_s1"] = g.transform("max")
    C["p1_gap_s1"] = d["p"] - C["p1_max_s1"]
    C["p1_rank_s1"] = g.rank(ascending=False, method="min")
    C["p1_sum_s1"] = g.transform("sum")
    C["p1_cnt50_s1"] = (d["p"] > 0.5).groupby(d["qi"]).transform("sum")
    gs = d.groupby(["qi", "src"])["p"]
    C["p1_gap_s1src"] = d["p"] - gs.transform("max")
    C["p1_rank_s1src"] = gs.rank(ascending=False, method="min")
    gt = d.groupby("ti")["p"]
    C["p1_max_t"] = gt.transform("max")
    C["p1_gap_t"] = d["p"] - C["p1_max_t"]
    C["p1_rank_t"] = gt.rank(ascending=False, method="first")
    # best competing S1 record for the same target
    srt = d.sort_values(["ti", "p"], ascending=[True, False])
    second = srt.groupby("ti")["p"].nth(1)
    second = pd.Series(second.values, index=srt.loc[second.index, "ti"].values)
    top2 = d["ti"].map(second).fillna(0.0).values
    C["p1_margin_t"] = np.where(C["p1_rank_t"].values == 1, d["p"].values - top2,
                                d["p"].values - C["p1_max_t"].values)
    C["p1_cnt50_t"] = (d["p"] > 0.5).groupby(d["ti"]).transform("sum")
    return C.astype(np.float32)


# metric
def macro_f05_from_sets(pred: dict, gold: dict, s1_ids) -> float:
    tot = 0.0
    for s in s1_ids:
        P, G = pred.get(s, set()), gold.get(s, set())
        if not P and not G:
            tot += 1.0
            continue
        tp = len(P & G)
        if tp == 0:
            continue
        p, r = tp / len(P), tp / len(G)
        tot += 1.25 * p * r / (0.25 * p + r)
    return tot / max(len(s1_ids), 1)


def _macro_f05_vec(keep, y, qcode, n_gold, n_s1):
    npred = np.bincount(qcode[keep], minlength=n_s1)
    tp = np.bincount(qcode[keep & (y == 1)], minlength=n_s1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(npred > 0, tp / np.maximum(npred, 1), 0.0)
        r = np.where(n_gold > 0, tp / np.maximum(n_gold, 1), 0.0)
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f = np.where((n_gold == 0) & (npred == 0), 1.0, f)
    return float(f.mean())


def decision_mask(cand, p, t, r, unique):
    d = pd.DataFrame({"qi": cand["qi"].values, "ti": cand["ti"].values, "p": p})
    mx = d.groupby("qi")["p"].transform("max").values
    keep = (p >= t) & (p >= r * mx)
    if unique:
        keep &= p >= d.groupby("ti")["p"].transform("max").values
    return keep


def tune_decision(cand, p, y, n_gold_per_qi, n_s1, log=print, tol=2e-4):
    """Tries many (t, r, unique) combinations on the validation scores and keeps the one with the best
    macro F0.5, counting singletons and the true matches that blocking missed."""
    qcode = cand["qi"].values
    d = pd.DataFrame({"qi": qcode, "ti": cand["ti"].values, "p": p})
    mx = d.groupby("qi")["p"].transform("max").values
    is_best_t = p >= d.groupby("ti")["p"].transform("max").values
    results = []
    for unique in (False, True):
        for r in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
            base = (p >= r * mx) & (is_best_t if unique else True)
            for t in np.round(np.arange(0.05, 0.991, 0.01), 3):
                keep = base & (p >= t)
                results.append((_macro_f05_vec(keep, y, qcode, n_gold_per_qi, n_s1), float(t), float(r), bool(unique)))
    res = pd.DataFrame(results, columns=["f", "t", "r", "unique"])
    fmax = res["f"].max()
    # robustness: among settings within `tol` of the optimum, take the (r, unique) family
    # with the widest plateau and the median threshold of that plateau
    near = res[res["f"] >= fmax - tol]
    fam = near.groupby(["r", "unique"]).size().idxmax()
    plateau = near[(near["r"] == fam[0]) & (near["unique"] == fam[1])]
    t_star = float(np.median(plateau["t"]))
    t_star = float(plateau.iloc[(plateau["t"] - t_star).abs().argmin()]["t"])
    row = res[(res["t"] == t_star) & (res["r"] == fam[0]) & (res["unique"] == fam[1])].iloc[0]
    best = (float(row["f"]), dict(t=t_star, r=float(fam[0]), unique=bool(fam[1]), oof_max=float(fmax)))
    log(f"  best decision rule: {best[1]}  ->  OOF macro F0.5 = {best[0]:.5f}")
    return best
