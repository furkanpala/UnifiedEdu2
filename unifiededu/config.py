"""config.py -- All hyperparameters for UnifiedEdu."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelGraphConfig:
    k_node: int = 8   # node groups for Theta
    k_edge: int = 8   # edge groups for Theta


@dataclass
class FederationConfig:
    num_rounds:   int   = 100
    t_ic:         int   = 5     # intra-cluster aggregation interval
    t_bc:         int   = 20    # inter-cluster aggregation interval
    T_init:       int   = 30    # warm-up rounds before inter-cluster agg
    t_update:     int   = 20    # re-clustering interval
    num_clients:  int   = 3     # number of participating institutions
    max_clusters: int   = -1    # -1 = ceil(num_clients/2) auto

    def effective_max_clusters(self, num_clients: int = None) -> int:
        """
        Return the maximum number of clusters to consider.

        Parameters
        ----------
        num_clients : int, optional
            Actual number of clients this round.  Overrides the config field
            when provided (the server passes self.num_clients here so the
            value stays consistent with reality even if the config field is
            left at its default).
        """
        nc = num_clients if num_clients is not None else self.num_clients
        if self.max_clusters > 0:
            return self.max_clusters
        return max(1, (nc + 1) // 2)  # ceiling: 3 clients -> 2 max clusters


@dataclass
class TrainingConfig:
    lr:                          float = 1e-4
    beta1:                       float = 0.9
    beta2:                       float = 0.999
    weight_decay:                float = 1e-2
    batch_size:                  int   = 8
    local_epochs:                int   = 1     # local SGD epochs per round (E in FedAvg)
    gradient_accumulation_steps: int   = 4
    max_seq_len:                 int   = 512
    chunk_stride:                int   = 128


@dataclass
class DataConfig:
    chunk_size:           int   = 512   # tokens
    chunk_stride:         int   = 128   # tokens
    train_ratio:          float = 0.80
    val_ratio:            float = 0.10
    test_ratio:           float = 0.10
    min_anchor_topics:    int   = 10
    min_anchor_chunks:    int   = 20
    embedding_model:      str   = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim:        int   = 384
    min_centroid_distance: float = 0.15  # non-IID go/no-go gate


@dataclass
class EvalConfig:
    bootstrap_samples: int   = 10_000
    alpha:             float = 0.05
    equity_epsilon:    float = 1e-6
    bloom_weights:     List[float] = field(
        default_factory=lambda: [0.33, 0.34, 0.33]  # alpha_1,2,3 for C_k
    )
    n_reverse_questions: int = 3   # for answer relevancy metric


@dataclass
class UnifiedEduConfig:
    model_graph: ModelGraphConfig = field(default_factory=ModelGraphConfig)
    federation:  FederationConfig = field(default_factory=FederationConfig)
    training:    TrainingConfig   = field(default_factory=TrainingConfig)
    data:        DataConfig       = field(default_factory=DataConfig)
    eval:        EvalConfig       = field(default_factory=EvalConfig)
    seed:        int              = 42
    device:      str              = "cuda"


# Singleton default config -- override fields as needed
DEFAULT_CONFIG = UnifiedEduConfig()
