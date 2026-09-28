"""
Trains the entity-matching model on the full dataset.

Steps: load the data, clean it (including converting Indian scripts to Latin
letters), split joined-up website names into words, find likely candidate
pairs within each country, compute about 100 comparison features, train
LightGBM with cross-validation, then pick the match threshold that gives
the best F0.5.

Run with --train_only to save the model, then run predict.py for the test set.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from itertools import chain

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from augment import augment_targets                                         
from blocking import generate_candidates                                    
from config import Config                                                   
from features import build_features, build_idf                              
from model import (context_features, cv_train, decision_mask, fit_full,     
                   macro_f05_from_sets, predict, tune_decision)
from tokenodds import COLS as TO_COLS, TokenOdds, cross_fit                 
from normalize import (ADDR_MAP, US_STATES, build_vocab, normalize_records,  
                       segment_web_names)

T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, encoding="utf-8")


def load_split(data_dir, split):
    d = os.path.join(data_dir, split)
    s1 = read_tsv(os.path.join(d, f"{split}_source1.tsv"))
    s2 = read_tsv(os.path.join(d, f"{split}_source2.tsv"))
    s3 = read_tsv(os.path.join(d, f"{split}_source3.tsv"))
    tg = pd.concat([s2, s3], ignore_index=True)
    tg["src"] = tg["entity_id"].str[:2]
    tg = tg[tg["src"].isin(["S2", "S3"])]
    return (s1.drop_duplicates("entity_id").reset_index(drop=True),
            tg.drop_duplicates("entity_id").reset_index(drop=True))


def load_gold(path):
    gt = read_tsv(path)
    return {s: {x.strip() for x in str(m).split(",") if x.strip()}
            for s, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}


_STATE_CODES = set(US_STATES.values()) | {v for v in ADDR_MAP.values() if len(v) == 2}


def region_of(a_nolm: str) -> str:
    """Coarse region key (state code) used to sample training S1 records by region."""
    toks = a_nolm.split()
    for t in reversed(toks):
        if t in _STATE_CODES:
            return t
    return toks[-1] if toks else ""


def sample_by_region(s1n, frac, seed=0):
    if frac >= 1.0:
        return s1n
    reg = s1n["country_n"] + "|" + s1n["a_nolm"].map(region_of)
    uniq = pd.unique(reg)
    rng = np.random.RandomState(seed)
    chosen = set(uniq[rng.rand(len(uniq)) < frac])
    return s1n[reg.isin(chosen).values].reset_index(drop=True)


def prepare(split, s1, tg, cfg):
    log(f"[{split}] normalising {len(s1):,} S1 + {len(tg):,} S2/S3 records ({cfg.n_jobs} procs)")
    s1n = normalize_records(s1, n_jobs=cfg.n_jobs)
    src = tg["src"].values
    tgn = normalize_records(tg, n_jobs=cfg.n_jobs)
    tgn["src"] = src
    voc = build_vocab(chain(s1n["n_core"], tgn["n_core"]))
    log(f"[{split}] segmented {segment_web_names(tgn, voc):,} concatenated website names")
    for d in (s1n, tgn):                                   # name frequency (uniqueness) features
        d["n_core_freq"] = d.groupby(["country_n", "n_core"])["n_core"].transform("size").astype(np.float32)
    idf_n = build_idf(chain(s1n["n_core"], tgn["n_core"]))
    idf_a = build_idf(chain(s1n["a_tok"], tgn["a_tok"]))
    return s1n, tgn, idf_n, idf_a


def partitions(s1n, tgn):
    """Open-set country partitions; targets with an unseen/empty country join every partition."""
    s1c, tgc = s1n["country_n"].values, tgn["country_n"].values
    cs = pd.unique(s1c)
    orphan = ~np.isin(tgc, cs) | (tgc == "")
    for c in cs:
        qi = np.where(s1c == c)[0]
        ti = np.arange(len(tgn)) if c == "" else np.where((tgc == c) | orphan)[0]
        yield c, qi, ti


def chunk_bounds(qi_sorted, size):
    """Chunk boundaries that never split one S1 record's candidates."""
    n, out, a = len(qi_sorted), [], 0
    while a < n:
        b = min(a + size, n)
        while b < n and qi_sorted[b] == qi_sorted[b - 1]:
            b += 1
        out.append((a, b)); a = b
    return out


def featurize(split, s1n, tgn, idf_n, idf_a, cfg, sink):
    """Blocking + features per country and chunk; `sink(meta, F)` consumes each chunk."""
    for c, qi, ti in partitions(s1n, tgn):
        log(f"[{split}] country={c!r}: {len(qi):,} S1 vs {len(ti):,} targets -> blocking")
        s1p = s1n.iloc[qi].reset_index(drop=True)
        tgp = tgn.iloc[ti].reset_index(drop=True)
        cand = generate_candidates(s1p, tgp, cfg, log=log)
        for a, b in chunk_bounds(cand["qi"].values, cfg.feat_chunk):
            cc = cand.iloc[a:b].reset_index(drop=True)
            F = build_features(cc, s1p, tgp, idf_n, idf_a)
            meta = pd.DataFrame({"qi": qi[cc["qi"].values], "ti": ti[cc["ti"].values],
                                 "src": cc["src"].values})
            sink(meta, F)
            log(f"[{split}]   features {b:,}/{len(cand):,}")
        del s1p, tgp, cand
        gc.collect()


def write_id_lists(path, header2, s1_ids, lists):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{header2}\n")
        for s in s1_ids:
            f.write(f"{s}\t{','.join(dict.fromkeys(lists.get(s, [])))}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--train_frac", type=float, default=0.15)
    ap.add_argument("--augment_frac", type=float, default=0.3)
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n_estimators", type=int, default=0)
    ap.add_argument("--k_per_source", type=int, default=6)
    ap.add_argument("--key_cap", type=int, default=3000)
    ap.add_argument("--join_budget", type=int, default=2_000_000)
    ap.add_argument("--train_only", action="store_true")
    a = ap.parse_args()
    cfg = Config(data_dir=a.data_dir, out_dir=a.out_dir, work_dir=a.work_dir, n_jobs=a.n_jobs,
                 train_frac=a.train_frac, augment_frac=a.augment_frac, n_folds=a.n_folds,
                 seeds=tuple(int(x) for x in a.seeds.split(",")), n_estimators=a.n_estimators,
                 k_per_source=a.k_per_source, key_cap=a.key_cap, join_budget=a.join_budget)
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.work_dir, exist_ok=True)
    report = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()}}
    n_est = cfg.n_estimators or None

    # ================================================================ TRAIN
    s1_tr, tg_tr = load_split(cfg.data_dir, "train")
    gold = load_gold(os.path.join(cfg.data_dir, "train", "train_ground_truth.tsv"))
    if cfg.augment_frac > 0:
        log(f"[train] augmenting {cfg.augment_frac:.0%} of S2/S3 records (unseen-country robustness)")
        tg_tr = augment_targets(tg_tr, frac=cfg.augment_frac)
    s1n, tgn, idf_n, idf_a = prepare("train", s1_tr, tg_tr, cfg)
    del s1_tr, tg_tr; gc.collect()
    s1n = sample_by_region(s1n, cfg.train_frac)
    log(f"[train] training on {len(s1n):,} S1 records (region sample {cfg.train_frac:.0%}) vs all targets")

    metas, Fs = [], []
    featurize("train", s1n, tgn, idf_n, idf_a, cfg, lambda m, F: (metas.append(m), Fs.append(F)))
    meta = pd.concat(metas, ignore_index=True); X = pd.concat(Fs, ignore_index=True)
    del metas, Fs; gc.collect()
    s1_ids, t_ids = s1n["entity_id"].values, tgn["entity_id"].values
    ps, pt = s1_ids[meta["qi"].values], t_ids[meta["ti"].values]
    y = np.fromiter((t in gold.get(s, ()) for s, t in zip(ps, pt)), dtype=np.int8, count=len(ps))
    n_gold_per_qi = np.array([len(gold.get(s, ())) for s in s1_ids])
    n_gold = int(n_gold_per_qi.sum())
    rec = y.sum() / max(n_gold, 1)
    oracle = {}
    for s, t, yy in zip(ps, pt, y):
        if yy:
            oracle.setdefault(s, set()).add(t)
    ceil = macro_f05_from_sets(oracle, gold, s1_ids)
    log(f"[train] BLOCKING RECALL = {rec:.4f} | F0.5 CEILING = {ceil:.5f} | pairs = {len(y):,} "
        f"({len(y) / len(s1_ids):.1f}/S1) | positives = {int(y.sum()):,}")
    report.update(blocking_recall=float(rec), f05_ceiling=float(ceil), n_pairs_train=int(len(y)),
                  n_s1_train=int(len(s1_ids)))

    a_txt, b_txt = s1n["n_full"].values[meta["qi"].values], tgn["n_full"].values[meta["ti"].values]
    X[TO_COLS] = cross_fit(a_txt, b_txt, y, meta["qi"].values)          # leakage-free on train
    odds = TokenOdds().fit(a_txt, b_txt, y)                               # full fit, used for test
    del a_txt, b_txt
    cand = meta
    groups = meta["qi"].values
    log("[train] stage-1 LightGBM")
    oof1, m1 = cv_train(X, y, groups, cfg.n_folds, cfg.seeds, log=log, n_estimators=n_est)
    f1 = tune_decision(cand, oof1, y, n_gold_per_qi, len(s1_ids), log=log)
    X2, C = X, None                                   # stage 2 disabled (time budget)
    oof2, m2, f2 = oof1, m1, (-1.0, None)
    use2 = f2[0] >= f1[0]
    best_f, rule = f2 if use2 else f1
    oof = oof2 if use2 else oof1
    log(f"[train] OOF macro F0.5 = {best_f:.5f} (stage {2 if use2 else 1}), rule = {rule}")
    report.update(oof_f05_stage1=f1[0], oof_f05_stage2=f2[0], used_stage=2 if use2 else 1, rule=rule)

    keep = decision_mask(cand, oof, rule["t"], rule["r"], rule["unique"])
    pred = {}
    for s, t in zip(ps[keep], pt[keep]):
        pred.setdefault(s, set()).add(t)
    by_c = {c: round(macro_f05_from_sets(pred, gold, s1_ids[s1n["country_n"].values == c]), 5)
            for c in pd.unique(s1n["country_n"])}
    log(f"[train] OOF F0.5 by country: {by_c}")
    report["oof_f05_by_country"] = by_c

    # error analysis files
    qn, tn = meta["qi"].values, meta["ti"].values
    err = pd.DataFrame({"s1_id": ps, "t_id": pt, "p": oof, "label": y, "pred": keep.astype(int),
                        "s1_name": s1n["business_name"].values[qn], "t_name": tgn["business_name"].values[tn],
                        "s1_addr": s1n["business_address"].values[qn], "t_addr": tgn["business_address"].values[tn]})
    err[(err.pred == 1) & (err.label == 0)].sort_values("p", ascending=False).head(20000) \
        .to_csv(os.path.join(cfg.work_dir, "errors_false_positives.tsv"), sep="\t", index=False)
    err[(err.pred == 0) & (err.label == 1)].sort_values("p").head(20000) \
        .to_csv(os.path.join(cfg.work_dir, "errors_false_negatives.tsv"), sep="\t", index=False)
    del err; gc.collect()

    # final models on all training rows (1 model per stage -> fast test inference)
    it1 = int(np.mean([m.best_iteration_ or m.n_estimators for m in m1]) * 1.1)
    it2 = int(np.mean([m.best_iteration_ or m.n_estimators for m in m2]) * 1.1)
    log(f"[train] refitting full models (stage1 {it1} trees, stage2 {it2} trees)")
    M1 = fit_full(X, y, it1)
    M2 = fit_full(X2, y, it2) if use2 else None
    import pickle
    with open(os.path.join(cfg.out_dir, "model_bundle.pkl"), "wb") as f:
        pickle.dump({"M1": M1, "rule": rule, "feat_cols": list(X.columns), "odds": odds}, f)
    log("saved model -> output/model_bundle.pkl")

    imp = pd.Series((M2 or M1).feature_importances_, index=(X2 if use2 else X).columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(cfg.work_dir, "feature_importance.tsv"), sep="\t", header=["importance"])
    if a.train_only:
        with open(os.path.join(cfg.work_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
        log("train_only: done -> now run predict.py")
        return
    feat_cols, feat_cols2 = list(X.columns), list(X2.columns)
    del X, X2, C, meta, cand, s1n, tgn, oof1, oof2; gc.collect()

    # ================================================================ TEST
    del gold
    gc.collect()
    s1_te, tg_te = load_split(cfg.data_dir, "test")
    s1t, tgt, idf_n, idf_a = prepare("test", s1_te, tg_te, cfg)
    del s1_te, tg_te; gc.collect()
    cache = os.path.join(cfg.work_dir, "test_feat_cache"); os.makedirs(cache, exist_ok=True)
    metas, p1s, files = [], [], []

    def sink(m, F):
        F[TO_COLS] = odds.transform(s1t["n_full"].values[m["qi"].values], tgt["n_full"].values[m["ti"].values])
        F = F[feat_cols]
        p1s.append(M1.predict_proba(F)[:, 1]); metas.append(m)
        if use2:
            fp = os.path.join(cache, f"F_{len(files):04d}.npy"); np.save(fp, F.values.astype(np.float32)); files.append(fp)

    featurize("test", s1t, tgt, idf_n, idf_a, cfg, sink)
    meta = pd.concat(metas, ignore_index=True); p = np.concatenate(p1s)
    if use2:
        C = context_features(meta, p)
        p2, off = [], 0
        for fp in files:
            F = np.load(fp); n = len(F)
            X2 = pd.concat([pd.DataFrame(F, columns=feat_cols), C.iloc[off:off + n].reset_index(drop=True)], axis=1)
            p2.append(M2.predict_proba(X2[feat_cols2])[:, 1]); off += n
            os.remove(fp)
        p = np.concatenate(p2)
    keep = decision_mask(meta, p, rule["t"], rule["r"], rule["unique"])
    pd.DataFrame({"source1_entity_id": s1t["entity_id"].values[meta["qi"].values],
                  "entity_id": tgt["entity_id"].values[meta["ti"].values],
                  "country": s1t["country_n"].values[meta["qi"].values],
                  "p": np.round(p, 4)}).to_csv(os.path.join(cfg.out_dir, "test_scores.tsv.gz"),
                                               sep="\t", index=False, compression="gzip")
    log("saved test probabilities -> output/test_scores.tsv.gz")

    s1_ids, t_ids = s1t["entity_id"].values, tgt["entity_id"].values
    cand_l, match_l = {}, {}
    for s, t, k in zip(s1_ids[meta["qi"].values], t_ids[meta["ti"].values], keep):
        cand_l.setdefault(s, []).append(t)
        if k:
            match_l.setdefault(s, []).append(t)
    write_id_lists(os.path.join(cfg.out_dir, "candidate_pairs.tsv"), "candidate_entity_ids", s1_ids, cand_l)
    write_id_lists(os.path.join(cfg.out_dir, "matching_results.tsv"), "matched_entity_ids", s1_ids, match_l)
    rate = pd.Series([bool(match_l.get(s)) for s in s1_ids]).groupby(s1t["country_n"].values).mean().round(4).to_dict()
    log(f"[test] wrote outputs: {len(s1_ids):,} S1 rows, {int(keep.sum()):,} matches; "
        f"share of S1 with >=1 match by country: {rate}")
    report.update(n_pairs_test=int(len(meta)), n_matches_test=int(keep.sum()), test_match_rate=rate,
                  runtime_min=round((time.time() - T0) / 60, 1))
    with open(os.path.join(cfg.work_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    log("done -> work/report.json")


if __name__ == "__main__":
    main()
