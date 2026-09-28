   # Business Entity Resolution — Amazon ML Challenge 2026

   Matches business records across three sources (S1 → S2/S3) at the scale of millions of records,
   including 9 Indian scripts and a country (France) that appears only in the test set.

   **Results:** leaderboard macro F0.5 = **0.943** · validation 0.952 · 12 candidates per S1 record · reduction ratio 0.9999988

   **Pipeline:** text cleaning + rule-based transliteration of Indic scripts → inverted-index blocking with
   fuzzy re-ranking → ~100 pair features → LightGBM (GroupKFold) → decision rule tuned for macro F0.5.

   - Code and how to run it: [`code/business_entity_resolution`](code/business_entity_resolution)
   - Full methodology: [`METHODOLOGY.md`](METHODOLOGY.md)

   Team: Model Mavericks