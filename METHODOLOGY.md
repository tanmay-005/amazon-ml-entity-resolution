# Business Entity Resolution — Methodology

**Public leaderboard F0.5: 0.943** · Out-of-fold (validation) F0.5: 0.952 · Candidates per Source-1 entity: 12

Amazon ML Challenge 2026 · Team *Model Mavericks* · Code: [`code/business_entity_resolution`](code/business_entity_resolution)

---

## 1. Problem framing

Source 1 (S1) is a deduplicated reference list. For every S1 entity we must return all Source 2 / Source 3
records that describe the same real-world business (zero, one or many). There is no shared identifier, and
S2/S3 contain abbreviations, typos, DBA/former names, transliterations, website names, missing fields,
landmark references and reordered address components.

We treat the task as **pairwise classification over a small candidate set**, followed by a **set-level decision
rule tuned directly for the evaluation metric** (macro F0.5 per S1 entity, singletons included). Because
F0.5 weights precision twice as much as recall and a single false match on a singleton scores 0, the pipeline
is deliberately cautious.

Pipeline:

```
raw TSVs → normalisation → key blocking + fuzzy re-rank (12 candidates/S1) → candidate_pairs.tsv
        → ~100 pair features → LightGBM → F0.5-tuned decision rule → matching_results.tsv
```

## 2. Data findings that shaped the design

Measured on the provided training files (not assumptions):

| Finding | Example | Handling |
|---|---|---|
| Scale | 2.2M S1 vs 10.3M S2+S3 (train); 1.73M vs 9.97M (test) | inverted-index blocking, chunked processing |
| 9 Indic scripts (~13% of Indian target names) | `ఏస్ పవర్ ప్రైవేట్ లిమిటెడ్` | rule-based transliteration |
| Website / handle names (~4%) | `acepower.com`, `SUPERTECHCHITS.COM (ID: 50951)` | junk removal + word segmentation |
| Leetspeak, accents | `5upertech Chíts` | character mapping, accent stripping |
| Former / trade names | `X F/K/A First It Private Limited` | DBA/FKA split into an alternate name |
| Postal codes almost absent; 5-digit tokens are house numbers | `21758 TOVAR RD` | postal code accepted only when labelled |
| Placeholders, zero padding, relation markers | `<NULL>`, `N/A`, `00140`, `C/O`, `S/O` | removed / normalised |
| Sibling businesses (hard negatives) | `United Builders`, No 617 vs No 618 | house-number structure + word-odds features |
| Renamed businesses at the same address | random new name, identical address | address-only re-rank score |
| Ground truth | 5.6% singletons, 3.46 matches per S1 on average, no S2/S3 record linked to two S1 | unique-assignment option in the decision rule (used only because it improved validation F0.5) |

## 3. Normalisation (`normalize.py`, `translit.py`)

All normalisation is country-agnostic; country is treated as an open set of labels and never hard-coded.

- **Transliteration:** one rule table covers Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
  Kannada and Malayalam (these Unicode blocks share the same internal layout), with inherent-vowel handling and
  word-final schwa deletion. Transliterated legal words are mapped (`praivet` → pvt, `limited` → ltd).
- **Names:** accent stripping; removal of IDs, brackets and junk symbols; website/handle extraction; single-digit
  leetspeak repair; `&` → and; initials merged (`I.B.M.` → ibm); ~80 abbreviation canonicalisations
  (Private → pvt, Technologies → tech, Shree/Sri → shri, Compagnie kept as a distinctive word); legal suffixes
  removed for a *core* name; DBA/FKA/parenthesised parts kept as an *alternate* name; repeated words removed.
- **Addresses:** `C/O`, `S/O`, `R/O` markers, `<NULL>`, `N/A` removed; zero-padding stripped; ordinals
  (`3rd` → 3); street types (Road → rd, Boulevard/Bd → blvd, Rue, Nagar, Marg …); US and Indian state names to
  codes (including native-script and misspelled states via the phonetic key); landmark phrases (`near`, `opp.`,
  `behind`, `en face de` …) removed into a landmark-free view; an **abbreviation-agnostic address key**
  (numbers + distinctive words, street types removed).
- **Phonetic skeleton:** digraph folding (ph→f, sh→s), voicing merge (t/d, p/b, k/g — Tamil and Malayalam do not
  mark voicing), soft c, repeated letters collapsed, vowels after the first letter dropped
  (`consultancy` ≈ `kansaltensi` → `knsltns`).
- **Word segmentation:** concatenated website names are split with a unigram model built from the data
  (`urgentcareclinic` → `urgent care clinic`).

## 4. Candidate generation / blocking (`blocking.py`)

All-pairs comparison (1.73M × 9.97M ≈ 1.7×10¹³ pairs on test) is infeasible. Blocking runs **per country
partition** (records with an unseen or empty country label are added to every partition), using an inverted
index of short keys. Each record emits ~15 keys:

| Key | Purpose |
|---|---|
| exact distinctive name word | rare words are highly selective |
| phonetic skeleton of each name word | typos, transliteration |
| unordered pairs of name-word skeletons | word order, far more selective than single words |
| name without spaces; 6-character prefix | `ace power` = `acepower`, website names |
| acronym of the core name | `southern ram care` ↔ `src` |
| house/unit number × distinctive address word | renamed businesses, Indic-script names |
| pairs of distinctive address words | number-independent address match |

Keys shared by more than 3,000 target records are dropped (they identify nobody). Candidate pairs are
scored by the IDF-weighted count of shared keys; the top 40 per (S1, source) form a pool that is **re-ranked with
fast rapidfuzz similarity** (name, no-space name, phonetic name, alternate name, address key, plus an
address-only score for renamed businesses). The **top 6 per source (12 per S1)** are kept. All joins are numpy
(`searchsorted`) processed in memory-bounded chunks.

`candidate_pairs.tsv` is **exactly the set the model scores**; every matched ID is a subset of it.

| Blocking metric | Value |
|---|---|
| Candidates per S1 (train and test) | **12.0** |
| Test candidate pairs | 20,785,001 (US 7.95M, France 3.11M, India 9.72M) |
| Test S1 entities with no candidates | 4 of 1,732,544 |
| Reduction ratio (test) | **0.9999988** (1 − 20.8M / 1.73×10¹³) |
| Pair recall on training data | 0.906 |
| F0.5 ceiling with a perfect matcher on these candidates | 0.964 |

**Candidate-size trade-off.** We measured 20 candidates per S1 (recall 0.916, ceiling 0.967, OOF F0.5 ≈ 0.954)
against 12 (recall 0.906, ceiling 0.964, OOF F0.5 0.952). Reducing the candidate set by 40% cost only ~0.002
F0.5, so we chose the smaller set.

## 5. Features (`features.py`, `tokenodds.py`) — ~100 per pair

Missing information is encoded as NaN (never as a fake similarity); LightGBM learns how to treat it.

- **Name similarity:** rapidfuzz ratio / token_sort / token_set / partial / WRatio, Jaro-Winkler, Levenshtein on the
  core name; full-name and phonetic-skeleton scores; no-space ratio; best score across core/alternate (DBA)
  names; Jaccard and IDF-weighted overlap; IDF of the rarest unshared word; exact-match, first-token, acronym and
  name-number agreement; length ratios.
- **Distinctive name part:** name similarity after removing words that also appear in either address — the city
  in `Bordeaux Compagnie` vs `Bordeaux Theatre` is not evidence of identity.
- **Address similarity:** fuzzy scores on landmark-free, full, skeleton and abbreviation-agnostic key views;
  token overlap and IDF overlap; labelled postal agreement; house-number Jaccard, conflict and **structure**
  (a single changed digit such as 617 vs 618 signals a sibling business; truncation such as A207 vs A20 is noise).
- **Word-difference log-odds (`tokenodds.py`):** for every word, the log-odds of appearing in the symmetric
  difference of true matches vs non-matches, learned from training labels and **cross-fitted** (each half of the
  S1 entities is scored with odds fitted on the other half) to avoid leakage. Harmless words (center, services,
  sri) score high; meaningful words (technologies, jewellers) score low. Among the most important features.
- **Name frequency:** how many records share the exact core name — decisive when the target address is empty.
- **Record flags:** website name, Indic-script name, empty name/address.
- **Blocking signals:** key score, number of shared keys, re-rank score, rank and gap to the best candidate of
  the same S1 and of the same target, candidate counts.

## 6. Model (`model.py`)

- **LightGBM** binary classifier (MIT licence; a few MB, far below the 8B-parameter limit).
  `learning_rate=0.1`, up to 1,000 trees with early stopping, `num_leaves=63`, `max_bin=63`,
  `subsample=0.8`, `colsample_bytree=0.7`, `reg_lambda=1`.
- **Training data:** S1 entities from a random 15% of regions (states), 448,175 entities, matched against **all**
  10.3M training targets so that negatives are realistic (5.38M candidate pairs, 1.41M positives).
- **Validation:** 3-fold **GroupKFold grouped by S1 entity**, so all candidates of an entity stay in one fold
  (best iterations 909 / 966 / 925). Out-of-fold predictions are used for threshold tuning and score estimates.
- **Final model:** refit on all training pairs (1,026 trees).
- A second-stage listwise model was implemented but did not beat the first stage on validation and was disabled.

## 7. Decision rule

A candidate is accepted if (a) its probability ≥ t, (b) it is ≥ r × the best probability among that S1's
candidates, and optionally (c) it is the best-scoring S1 for that target record (unique assignment).
t, r and the unique flag are grid-searched on out-of-fold predictions for **macro F0.5 over all S1 entities**,
including singletons and true pairs lost at blocking. Among near-optimal settings the centre of the widest stable
plateau is chosen, for robustness on the private leaderboard.

**Selected rule:** t = 0.67, r = 0.7, unique assignment on.

## 8. Generalisation to the unseen country (France)

- No country-specific features or encodings; country labels are only used to partition blocking (open set).
- French accents, street types (Rue, Bd, R., Allée …) and legal forms (SARL, SAS, EURL) handled in normalisation;
  the distinctive-name features ignore city names embedded in business names.
- **Training-time augmentation (`augment.py`):** 30% of training S2/S3 addresses have street-type words
  randomly abbreviated, truncated or dropped, and 15% of names have legal suffixes altered, so unfamiliar
  abbreviation styles are not treated as evidence against a match. Test data is never modified.
- Outcome on test: share of S1 entities with at least one predicted match — **France 0.960, US 0.940,
  India 0.931** — France behaves like the training countries (no sign of systematic over-matching).

## 9. Results

| Metric | Value |
|---|---|
| **Public leaderboard F0.5** | **0.943** |
| Out-of-fold F0.5 (validation) | 0.952 |
| Out-of-fold F0.5 — US / India | 0.970 / 0.943 |
| F0.5 ceiling of the candidate set | 0.964 |
| Pair recall of blocking | 0.906 |
| Test predictions | 5,528,461 matches; 106,335 S1 entities (6.1%) predicted as singletons |

The main remaining loss is in blocking (true pairs never reaching the model), mostly renamed businesses whose
address also changed, and heavily transliterated Indian names.

## 10. Engineering notes

- **Memory:** at full scale, training fragments the Python heap (~20 GB not returned to the OS). Running the test
  phase in the same process exceeded the 30 GB limit. The final pipeline therefore runs **training and prediction
  as two separate processes**: `run_pipeline.py --train_only` saves `model_bundle.pkl` and exits, and
  `predict.py` loads it with a clean memory footprint.
- Normalisation uses a `spawn`-based process pool with a timeout and a serial fallback.
- Features are computed in chunks of 500k pairs; blocking joins are capped at 2M rows at a time.
- Runtime on a Kaggle CPU notebook (4 cores, 30 GB): ~2 h training, ~3 h prediction.

## 11. Competition rules compliance

- No external databases, APIs, geocoders or internet look-ups; only the provided training and test files.
  Transliteration, abbreviation maps and all vocabularies are rule-based code or fitted on the provided data.
- Libraries: numpy, pandas, scipy, scikit-learn (BSD), LightGBM (MIT), rapidfuzz (MIT). No pretrained models.
- Country is handled as an open set; no country is hard-coded, filtered or one-hot encoded.
- Fully reproducible with the two commands in `code/business_entity_resolution/README.md`.

## 12. Future improvements

The gap between the candidate-set ceiling (0.964) and the validation score (0.952), plus the blocking recall of
0.906, point to where the next gains are:

- **Blocking recall:** add keys for renamed-and-moved businesses and more transliteration variants; study the
  false-negative file to find which true pairs never become candidates.
- **More training data:** train on 30–50% of regions instead of 15% (needs more than 30 GB RAM).
- **Second-stage listwise model:** re-enable it with a longer time budget; it targets chain stores and
  near-duplicate siblings.
- **Model tuning:** lower learning rate with more trees, more folds and several seeds averaged.
- **Unseen-country handling:** richer rules for French abbreviations and stop-words.