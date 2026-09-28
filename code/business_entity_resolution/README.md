# Business Entity Resolution — reproducible pipeline

Normalise (incl. 9 Indic scripts) -> key-based blocking + fuzzy re-rank (12 candidates per S1)
-> ~100 pair features -> LightGBM -> macro-F0.5-tuned decision rule.
No external data, APIs or look-ups. Libraries: numpy, pandas, scipy, scikit-learn (BSD), LightGBM, rapidfuzz (MIT).

**Results:** leaderboard macro F0.5 = 0.943 · out-of-fold F0.5 = 0.952 · blocking recall 0.906 · 12 candidates per S1

## Environment
Tested on a Kaggle CPU notebook (4 cores, ~30 GB RAM, Python 3.10+). Full run: ~2 h training + ~3 h prediction.

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

## Data layout (run from the folder that contains `dataset/`)
The competition data is not included in this repository; place it as shown below.
```
dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
dataset/test/{test_source1,test_source2,test_source3}.tsv
```

## Reproduce end-to-end (two separate processes)

Training and prediction run as **two separate Python processes**. The training process saves the
model and exits, which releases all its memory; prediction then starts with a clean memory
footprint. (Running both phases in one process exhausted the 30 GB RAM limit at full scale.)

**Step 1 — train and save the model**
```bash
python code/business_entity_resolution/src/run_pipeline.py \
    --data_dir dataset --out_dir output --work_dir work --n_jobs 4 --train_only
```
Writes `output/model_bundle.pkl` and `work/report.json` (blocking recall, F0.5 ceiling,
out-of-fold F0.5 per country, decision rule), `work/feature_importance.tsv`, error-analysis files.

**Step 2 — predict the test set**
```bash
python code/business_entity_resolution/src/predict.py \
    --data_dir dataset --model_path output/model_bundle.pkl --out_dir output --n_jobs 4
```
Writes:
- `output/matching_results.tsv` — final matches (leaderboard file)
- `output/candidate_pairs.tsv` — exactly the candidate set the model scores
- `output/test_scores.tsv.gz` — model probability of every candidate pair (diagnostics only)

**Step 3 — validate** (`validate_submission.py` is provided by the organisers and not included here)
```bash
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Settings used for the submission (all are the defaults)
| Setting | Value | Meaning |
|---|---|---|
| `--k_per_source` | 6 | candidates per S1 per source (12 per S1 in total) |
| `--key_cap` | 3000 | blocking keys shared by more targets are dropped |
| `--train_frac` | 0.15 | share of training regions (states) used to train |
| `--n_folds` | 3 | GroupKFold folds (grouped by S1 id) |
| `--augment_frac` | 0.3 | training-only abbreviation augmentation |
| `--join_budget` | 2,000,000 | max blocking join rows in memory at once |
| `--seeds` | 42 | random seed |

## Source files (`src/`)
| File | Role |
|---|---|
| `run_pipeline.py` | training pipeline; `--train_only` stops after saving the model |
| `predict.py` | test-set inference in a fresh process |
| `config.py` | default settings |
| `translit.py` | rule-based transliteration of Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam |
| `normalize.py` | name/address cleaning, abbreviations, DBA/FKA, phonetic skeleton, address key, word segmentation |
| `augment.py` | training-only abbreviation noise for robustness to the unseen country |
| `blocking.py` | inverted-index key blocking + fuzzy re-rank |
| `features.py` | ~100 pairwise features |
| `tokenodds.py` | cross-fitted word-difference log-odds |
| `model.py` | GroupKFold LightGBM, full refit, macro-F0.5 decision tuning |