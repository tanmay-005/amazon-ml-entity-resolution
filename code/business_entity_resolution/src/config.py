from dataclasses import dataclass, field


@dataclass
class Config:
    data_dir: str = "dataset"
    out_dir: str = "output"
    work_dir: str = "work"
    n_jobs: int = 4                    # processes for normalisation
    # blocking
    key_cap: int = 3000                # drop keys shared by more targets than this (generic words)
    k_pool: int = 40                   # key-score pool per (S1, source) before fuzzy re-rank
    k_per_source: int = 6              # final candidates per (S1, source) -> candidate_pairs.tsv
    join_budget: int = 5_000_000       # max join rows held in memory at once (~400 MB peak)
    # training
    train_frac: float = 0.25           # share of train S1 regions (states) used to train
    augment_frac: float = 0.3          # unseen-country robustness augmentation (train only)
    feat_chunk: int = 500_000        # pairs per feature chunk
    n_folds: int = 4
    seeds: tuple = field(default_factory=lambda: (42,))
    n_estimators: int = 0              # 0 = default with early stopping
