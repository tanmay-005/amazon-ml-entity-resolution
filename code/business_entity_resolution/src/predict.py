"""
Makes the predictions for the test set, as a separate program from training.

It loads the model saved by `run_pipeline.py --train_only` and writes
output/matching_results.tsv, output/candidate_pairs.tsv and output/test_scores.tsv.gz.
Running this on its own means it starts with empty memory; running it straight after
training in the same program used up all the available RAM.

    python src/predict.py --data_dir dataset --model_path output/model_bundle.pkl --out_dir output
"""
import argparse
import gc
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                          # noqa: E402
import pandas as pd                                                         # noqa: E402

from config import Config                                                   # noqa: E402
from model import decision_mask                                             # noqa: E402
from run_pipeline import featurize, load_split, log, prepare, write_id_lists  # noqa: E402
from tokenodds import COLS as TO_COLS                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--model_path", default="output/model_bundle.pkl")
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--k_per_source", type=int, default=6)
    ap.add_argument("--key_cap", type=int, default=3000)
    ap.add_argument("--join_budget", type=int, default=2_000_000)
    a = ap.parse_args()
    cfg = Config(data_dir=a.data_dir, out_dir=a.out_dir, n_jobs=a.n_jobs,
                 k_per_source=a.k_per_source, key_cap=a.key_cap, join_budget=a.join_budget)
    os.makedirs(cfg.out_dir, exist_ok=True)

    with open(a.model_path, "rb") as f:
        b = pickle.load(f)
    M1, rule, feat_cols, odds = b["M1"], b["rule"], b["feat_cols"], b["odds"]
    log(f"loaded model <- {a.model_path}; rule = {rule}")

    s1, tg = load_split(cfg.data_dir, "test")
    s1t, tgt, idf_n, idf_a = prepare("test", s1, tg, cfg)
    del s1, tg
    gc.collect()

    metas, ps = [], []

    def sink(m, F):
        F[TO_COLS] = odds.transform(s1t["n_full"].values[m["qi"].values],
                                    tgt["n_full"].values[m["ti"].values])
        ps.append(M1.predict_proba(F[feat_cols])[:, 1])
        metas.append(m)

    featurize("test", s1t, tgt, idf_n, idf_a, cfg, sink)
    meta = pd.concat(metas, ignore_index=True)
    p = np.concatenate(ps)
    keep = decision_mask(meta, p, rule["t"], rule["r"], rule["unique"])

    s1_ids, t_ids = s1t["entity_id"].values, tgt["entity_id"].values
    pd.DataFrame({"source1_entity_id": s1_ids[meta["qi"].values],
                  "entity_id": t_ids[meta["ti"].values],
                  "country": s1t["country_n"].values[meta["qi"].values],
                  "p": np.round(p, 4)}).to_csv(os.path.join(cfg.out_dir, "test_scores.tsv.gz"),
                                               sep="\t", index=False, compression="gzip")
    log("saved test probabilities -> test_scores.tsv.gz")

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
    log("done")


if __name__ == "__main__":
    main()